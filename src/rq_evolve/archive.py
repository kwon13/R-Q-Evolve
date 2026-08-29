import ast
import collections
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from .code_utils import (
    TRUSTED_ASSEMBLER_VERSION,
    extract_problem_statement_template,
    lint_problem_instance,
    structural_inspiration_safety_reason,
    validated_domain_declaration,
)
from .constancy import check_constancy
from .concepts import DOMAINS, PROBLEM_TYPES, axis_index
from .program import ProblemProgram
from .problem_type import PROBLEM_TYPE_RULESET, problem_type_ruleset_sha256
from .prompts import DOMAIN_LABELING_METHOD, domain_labeling_ruleset_sha256
from .scoring import selection_priority
from .similarity import SimilarityTimeout, matched_size, sequence_ratio
from .structural_fingerprint import exact_top_level_build_instance

ARCHIVE_SCHEMA = "rq-evolve-domain-problem-type-v2"
ARCHIVE_SCHEMA_VERSION = 2
CONFIRMED_ARCHIVE_SCHEMA = "rq-evolve-domain-problem-type-v3"
CONFIRMED_ARCHIVE_SCHEMA_VERSION = 3
SOURCE_DOMAIN_AUTHORITY = "source_exact_one_literal"
CONFIRMED_DOMAIN_AUTHORITY = "manual_seed_or_policy_binary_label_v1"
GENERATED_DOMAIN_AUTHORITY = "local_policy_binary_label_v1"
MANUAL_DOMAIN_AUTHORITY = "manual_seed_source_literal_v1"


@dataclass
class Niche:
    domain_bin: int
    problem_type_bin: int
    champion: ProblemProgram | None = None
    champion_rq: float = -1.0
    selection_count: int = 0
    update_count: int = 0
    history: list[dict] = field(default_factory=list)
    # Cell-based MCE statistics survive champion replacement. Each row is one
    # resolved reproductive pull: {"iteration": int, "reward": 0|1}. Only
    # the configured sliding window is retained and checkpointed.
    mce_history: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class StructuralInspirationSelection:
    """One label-free donor assignment for a stage-1 mutation prompt.

    ``template`` is the only donor material allowed into the model context.
    ``provenance`` is audit data for logs/archive metadata and is never rendered
    into the prompt.
    """

    donor: ProblemProgram | None
    template: str | None
    provenance: dict


class ArchiveSchemaError(ValueError):
    """The snapshot cannot be interpreted as this archive schema."""


class MAPElitesArchive:
    """Complete DOMAIN x PROBLEM_TYPE MAP-Elites grid.

    Both coordinates are fixed behavioural descriptors. In production,
    ``DOMAIN`` is assigned from the fixed family by a label-blind local-policy
    binary labeler; Stage 2 emits no domain label. ``PROBLEM_TYPE`` is
    deterministically inferred from the visible output request and verifier
    across all verification seeds. The grid is always the complete 7 x 5
    Cartesian product. There is no supported-cell mask or mutation target.

    Uncertainty is NOT an axis. H stays in the fitness, ``R_Q = p(1-p)H``, which
    decides who holds a cell and which champion is sampled as a parent. Binning
    on H as well would add a coordinate no generator can control: a child lands
    in an H bin only as a side effect of how hard it turned out to be.
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        ucb_c: float = 1.0,
        selection_strategy: str = "random",
        mce_window_iterations: int = 50,
        select_ignores_uncertainty: bool = False,
        select_ignores_variance: bool = False,
        binning: str = "grid",
        require_domain_labeling: bool = False,
        domain_labeling_min_probability: float = 0.55,
        domain_labeling_min_logit_margin: float = 0.50,
    ) -> None:
        if selection_strategy not in {"ucb", "random", "sliding_window_mce"}:
            raise ValueError(f"unknown selection_strategy: {selection_strategy}")
        if int(mce_window_iterations) <= 0:
            raise ValueError("mce_window_iterations must be positive")
        if binning not in {"grid", "flat"}:
            raise ValueError(f"binning must be 'grid' or 'flat', got {binning!r}")
        if select_ignores_uncertainty and select_ignores_variance:
            raise ValueError(
                "select_ignores_uncertainty and select_ignores_variance are "
                "mutually exclusive: enabling both drops every R_Q factor and "
                "leaves a constant (signal-free) selection priority."
            )
        self.n_domain_bins = len(DOMAINS)
        self.n_problem_type_bins = len(PROBLEM_TYPES)
        self.epsilon = float(epsilon)
        self.ucb_c = float(ucb_c)
        self.selection_strategy = selection_strategy
        self.mce_window_iterations = int(mce_window_iterations)
        # The evolver announces the current outer iteration before drawing a
        # batch. Pending pulls are virtual losses in the UCB denominator, which
        # prevents one temporarily best cell from monopolising all 32 slots
        # before any result is available.
        self._mce_iteration = -1
        self._mce_pending_by_parent: dict[str, list[tuple[int, int]]] = {}
        self._mce_pending_by_cell: collections.Counter[tuple[int, int]] = (
            collections.Counter()
        )
        # Ablations: rank mutation parents by s(1-s) (ignore_uncertainty) or by H
        # (ignore_variance) instead of s(1-s)*H. Neither changes what is stored
        # or binned -- champion_rq and the cell labels stay real.
        self.select_ignores_uncertainty = bool(select_ignores_uncertainty)
        self.select_ignores_variance = bool(select_ignores_variance)
        # Ablation: "flat" keeps the same 35 slots, the same validity gates and
        # the same parent sampling, but a candidate no longer competes only
        # against the champion sharing its (DOMAIN, PROBLEM_TYPE). It takes any
        # free slot, and once full it competes against the weakest occupant. That
        # turns the MAP into a plain top-K pool and isolates what the grid --
        # reserving capacity per behaviour cell -- is actually buying.
        # Labels are still read and recorded, so coverage stays measurable.
        self.binning = binning
        self.require_domain_labeling = bool(require_domain_labeling)
        self.domain_labeling_min_probability = float(
            domain_labeling_min_probability
        )
        self.domain_labeling_min_logit_margin = float(
            domain_labeling_min_logit_margin
        )
        if not 0.0 <= self.domain_labeling_min_probability <= 1.0:
            raise ValueError("domain_labeling_min_probability must be in [0, 1]")
        if self.domain_labeling_min_logit_margin < 0.0:
            raise ValueError("domain_labeling_min_logit_margin must be >= 0")
        self.total_insertions = 0
        self.total_replacements = 0
        self.total_selections = 0
        # Manually certified structural donors live independently of MAP cell
        # competition. Otherwise a valid child replacing a seed champion would
        # silently turn off the treatment later in the same run.
        self.structural_donors: dict[str, ProblemProgram] = {}
        self.grid: dict[tuple[int, int], Niche] = {
            (d, t): Niche(domain_bin=d, problem_type_bin=t)
            for d in range(self.n_domain_bins)
            for t in range(self.n_problem_type_bins)
        }

    def program_to_cell(self, program: ProblemProgram) -> tuple[int, int] | None:
        """Grid coordinate from pinned descriptor provenance, else ``None``.

        None rather than a hashed fallback: a program whose DOMAIN or
        PROBLEM_TYPE is outside the vocabulary has no meaningful niche, and
        folding unknown labels into one bin would make a single cell the
        contest for every misclassified generator.
        """
        is_generated = (program.metadata or {}).get("op") == "mutate"
        if self.require_domain_labeling and is_generated:
            try:
                tree = ast.parse(program.source_code)
            except SyntaxError:
                return None
            if any(
                isinstance(node, ast.Name) and node.id == "DOMAIN"
                for node in ast.walk(tree)
            ):
                return None
            domain = (program.metadata or {}).get("domain")
        else:
            domain, domain_errors = validated_domain_declaration(program.source_code)
            if domain_errors:
                return None
        if re.search(
            r"\b(?:PROBLEM_TYPE|GROUP|SKILL)\s*[:=]", program.source_code
        ):
            return None
        contract = (program.metadata or {}).get("descriptor_contract")
        if contract is None or not self._descriptor_contract_matches(
            program, domain, contract
        ):
            return None
        domain_bin = axis_index("domain", domain)
        problem_type_bin = axis_index("problem_type", program.get_problem_type())
        if domain_bin is None or problem_type_bin is None:
            return None
        return (domain_bin, problem_type_bin)

    def _descriptor_contract_matches(
        self, program: ProblemProgram, domain: str | None, contract: object
    ) -> bool:
        """Whether cached descriptor provenance matches source and rules."""

        if not isinstance(contract, dict):
            return False
        is_generated = (program.metadata or {}).get("op") == "mutate"
        problem_type = (program.metadata or {}).get("problem_type")
        expected = {
            "problem_type_authority": "deterministic_statement_and_verifier",
            "problem_type_ruleset": PROBLEM_TYPE_RULESET,
            "problem_type_ruleset_sha256": problem_type_ruleset_sha256(),
            "domain": domain,
            "problem_type": problem_type,
            "source_sha256": hashlib.sha256(
                program.source_code.encode("utf-8")
            ).hexdigest(),
        }
        if not all(contract.get(key) == value for key, value in expected.items()):
            return False
        if not self.require_domain_labeling:
            return contract.get("domain_authority") == SOURCE_DOMAIN_AUTHORITY

        labeling = contract.get("domain_labeling")
        if not isinstance(labeling, dict):
            return False
        if is_generated:
            probabilities = labeling.get("probabilities")
            generator_contract = (program.metadata or {}).get("generator_contract")
            if not bool(
                contract.get("domain_authority") == GENERATED_DOMAIN_AUTHORITY
                and labeling
                == (program.metadata or {}).get("domain_labeling")
                and labeling.get("method") == DOMAIN_LABELING_METHOD
                and labeling.get("passed") is True
                and labeling.get("failure_kind") is None
                and labeling.get("predicted") == domain
                and (program.metadata or {}).get("domain") == domain
                and isinstance(probabilities, dict)
                and set(probabilities) == set(DOMAINS)
                and all(
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    and 0.0 <= float(value) <= 1.0
                    for value in probabilities.values()
                )
                and labeling.get("ruleset_sha256")
                == domain_labeling_ruleset_sha256()
                and isinstance(generator_contract, dict)
                and generator_contract.get("version") == TRUSTED_ASSEMBLER_VERSION
                and labeling.get("family_sha256")
                == generator_contract.get("family_sha256")
                and isinstance(generator_contract.get("family_sha256"), str)
                and len(generator_contract["family_sha256"]) == 64
                and not isinstance(labeling.get("policy_iteration"), bool)
                and isinstance(labeling.get("policy_iteration"), int)
                and labeling.get("policy_iteration") >= -1
                and program.generation >= 1
                and bool(program.parent_id)
            ):
                return False

            ranked = sorted(
                ((candidate, float(probabilities[candidate])) for candidate in DOMAINS),
                key=lambda item: item[1],
                reverse=True,
            )
            predicted, top_probability = ranked[0]
            runner_probability = ranked[1][1]
            try:
                recorded_top = float(labeling["top_probability"])
                recorded_margin = float(labeling["logit_margin"])
                min_probability = float(labeling["min_probability"])
                min_margin = float(labeling["min_logit_margin"])
            except (KeyError, TypeError, ValueError):
                return False
            if not all(
                math.isfinite(value)
                for value in (
                    recorded_top,
                    recorded_margin,
                    min_probability,
                    min_margin,
                )
            ):
                return False
            bounded_top = min(max(top_probability, 1e-6), 1.0 - 1e-6)
            bounded_runner = min(max(runner_probability, 1e-6), 1.0 - 1e-6)
            recomputed_margin = math.log(bounded_top / (1.0 - bounded_top)) - math.log(
                bounded_runner / (1.0 - bounded_runner)
            )
            return bool(
                predicted == domain
                and min_probability == self.domain_labeling_min_probability
                and min_margin == self.domain_labeling_min_logit_margin
                and math.isclose(recorded_top, top_probability, abs_tol=1e-12)
                and math.isclose(recorded_margin, recomputed_margin, abs_tol=1e-12)
                and top_probability >= min_probability
                and recomputed_margin >= min_margin
            )
        seed_source = (program.metadata or {}).get("manual_seed_source")
        return bool(
            contract.get("domain_authority") == MANUAL_DOMAIN_AUTHORITY
            and labeling.get("method") == "manual_seed_source_literal_v1"
            and labeling.get("passed") is True
            and labeling.get("declared") == domain
            and labeling.get("predicted") == domain
            and program.generation == 0
            and not program.parent_id
            and isinstance(seed_source, dict)
            and seed_source.get("method") == "loaded_seed_file_v1"
            and seed_source.get("source_file")
            == (program.metadata or {}).get("source_file")
            and seed_source.get("source_sha256")
            == hashlib.sha256(program.source_code.encode("utf-8")).hexdigest()
            and labeling.get("seed_source") == seed_source
        )

    def _schema_contract(self) -> dict:
        """Snapshot identity for legacy or independently labeled archives."""

        if self.require_domain_labeling:
            return {
                "schema": CONFIRMED_ARCHIVE_SCHEMA,
                "schema_version": CONFIRMED_ARCHIVE_SCHEMA_VERSION,
                "domain_authority": CONFIRMED_DOMAIN_AUTHORITY,
                "domain_labeling_required": True,
                "domain_labeling_method": DOMAIN_LABELING_METHOD,
                "domain_labeling_ruleset_sha256": (
                    domain_labeling_ruleset_sha256()
                ),
                "domain_labeling_min_probability": (
                    self.domain_labeling_min_probability
                ),
                "domain_labeling_min_logit_margin": (
                    self.domain_labeling_min_logit_margin
                ),
            }
        return {
            "schema": ARCHIVE_SCHEMA,
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "domain_authority": SOURCE_DOMAIN_AUTHORITY,
        }

    def _insert_cell(self, program: ProblemProgram) -> tuple[int, int] | None:
        """Slot the program competes for.

        Under ``binning="grid"`` that is its own (DOMAIN, PROBLEM_TYPE) cell.
        Under ``"flat"`` the grid is only storage: the program takes the first
        free slot, and once all 35 are occupied it challenges the weakest one.
        Same capacity, same gates, same sampling -- only the reservation of
        capacity per behaviour cell is removed.

        A program with no usable label is still rejected in both modes. Letting
        it in under "flat" would relax the label contract too, and the arm is
        meant to isolate the grid, not two changes at once.
        """
        if self.program_to_cell(program) is None:
            return None
        if self.binning == "grid":
            return self.program_to_cell(program)
        free = [k for k, n in sorted(self.grid.items()) if n.champion is None]
        if free:
            return free[0]
        return min(
            sorted(self.grid),
            key=lambda k: self._select_priority(self.grid[k].champion),
        )

    def cell_labels(self, cell: tuple[int, int]) -> tuple[str, str]:
        return (DOMAINS[cell[0]], PROBLEM_TYPES[cell[1]])

    def try_insert(
        self,
        program: ProblemProgram,
        u_value: float,
        rq_score: float,
    ) -> bool:
        cell = self._insert_cell(program)
        if cell is None:
            program.metadata["archive_status"] = "unlabelled_rejected"
            return False
        domain_bin, problem_type_bin = cell

        program.niche_domain = domain_bin
        program.niche_problem_type = problem_type_bin
        # U is not a coordinate, but it is the uncertainty factor of
        # R_Q = s(1-s)U, so it is recorded on the program exactly as before.
        program.u_score = float(u_value)
        program.rq_score = float(rq_score)

        # --- Safety gates (ported from evo-sample), applied at EVERY archive
        # entry, including champion re-evaluation. ---
        #
        # R_Q = 0 is NOT one of them. Classical MAP-Elites does not evict on
        # fitness alone: a cell holds the best thing found for it so far, and
        # "nothing yet" is strictly worse than a program the policy cannot
        # solve. The earlier gate here rejected every p=0 and p=1 candidate,
        # which can remove an axis value from the bootstrap before a single
        # mutation runs.
        #
        # Nothing downstream needs the gate: training examples are drained by
        # the frontier band in dataset.py (low < s_hat < high), which excludes
        # p=0 and p=1 on its own, and cell competition below is a strict `>`
        # so an R_Q=0 champion yields to the first scoring program that
        # arrives and never displaces one.
        #
        if not self.passes_admission_preflight(program):
            return False

        certification = (program.metadata or {}).get(
            "structural_donor_certification"
        ) or {}
        if certification.get("passed"):
            self.structural_donors[program.program_id] = program

        niche = self.grid[cell]
        # Champion competition ranks by selection priority: real R_Q in
        # production, s(1-s) under the select_ignores_uncertainty ablation
        # (program.rq_score / s_hat are already set above). The stored
        # champion_rq stays the real R_Q, so the MAP still logs true scores --
        # only the winner choice is H-blind under the ablation.
        new_priority = self._select_priority(program)
        if niche.champion is None or new_priority > self._select_priority(
            niche.champion
        ):
            event = "inserted" if niche.champion is None else "replaced"
            if niche.champion is not None:
                self.total_replacements += 1
            niche.champion = program
            niche.champion_rq = float(rq_score)
            program.metadata["archive_status"] = "champion"
            niche.update_count += 1
            niche.history.append(
                {
                    "event": event,
                    "program_id": program.program_id,
                    "rq_score": float(rq_score),
                }
            )
            self.total_insertions += 1
            # Archive-global uniqueness: one generator occupies one cell only.
            self._purge_program_from_other_cells(program.program_id, keep_cell=cell)
            return True
        program.metadata["archive_status"] = "lost_cell_contest"
        return False

    def passes_admission_preflight(self, program: ProblemProgram) -> bool:
        """Run score-independent archive gates without mutating the grid.

        Candidate generation used to pay for all solver rollouts before these
        deterministic checks ran inside :meth:`try_insert`. Running the same
        checks after DOMAIN labeling but before rollout scoring rejects an
        existing archive duplicate cheaply. ``try_insert`` intentionally runs
        them again: another child from the same flat batch may have entered the
        archive between preflight and final admission.
        """

        if self._insert_cell(program) is None:
            program.metadata["archive_status"] = "unlabelled_rejected"
            return False

        # 1. Strict seed-variation: block near-constant / thin-rewording
        #    generators before they pollute the archive and mutation chain.
        if not self._passes_seed_variation(program):
            program.metadata["archive_status"] = "seed_variation_rejected"
            return False
        # 2. Behavior duplicate: digit-identical clone already in the grid.
        dup = self._find_duplicate_behavior(program)
        if dup is not None:
            program.metadata["archive_status"] = "duplicate_behavior_rejected"
            program.metadata["duplicate_of"] = dup.program_id
            return False
        # 3. Template duplicate: same problem skeleton, different numbers.
        tdup = self._find_duplicate_template(program)
        if tdup is not None:
            program.metadata["archive_status"] = "duplicate_template_rejected"
            program.metadata["duplicate_of"] = tdup.program_id
            return False
        # 4. Near-duplicate: the same statement a few words apart. Exact hashes
        #    miss it, and a noisy descriptor can put the restatement in another
        #    cell where it would count as false coverage.
        try:
            near = self._find_near_duplicate_template(program)
        except SimilarityTimeout as exc:
            program.metadata["archive_status"] = "similarity_timeout_rejected"
            program.metadata["similarity_timeout"] = str(exc)
            return False
        if near is not None:
            other, ratio = near
            program.metadata["archive_status"] = "near_duplicate_template_rejected"
            program.metadata["duplicate_of"] = other.program_id
            program.metadata["duplicate_ratio"] = round(ratio, 3)
            return False
        # 5. Structural duplicate: a different statement over the same program.
        #    Runs last of the four because it is the only one that parses, and
        #    first among things that matter -- it is the gate the run needed and
        #    did not have.
        try:
            sdup = self._find_structural_duplicate(program)
        except SimilarityTimeout as exc:
            program.metadata["archive_status"] = "similarity_timeout_rejected"
            program.metadata["similarity_timeout"] = str(exc)
            return False
        if sdup is not None:
            other, ratio = sdup
            program.metadata["archive_status"] = "structural_duplicate_rejected"
            program.metadata["duplicate_of"] = other.program_id
            program.metadata["structural_ratio"] = round(ratio, 3)
            return False
        return True

    def placement_cell(self, program: ProblemProgram) -> tuple[int, int] | None:
        """Return the post-hoc cell a verified program would enter.

        This is insertion bookkeeping only: it neither selects nor communicates
        a mutation destination. None means the program has no usable
        DOMAIN/PROBLEM_TYPE pair, matching :meth:`try_insert`.
        """
        return self._insert_cell(program)

    def remove_program(self, program_id: str) -> list[tuple[int, int]]:
        """Remove one champion identity from the single live MAP archive."""
        removed: list[tuple[int, int]] = []
        for key, niche in self.grid.items():
            if niche.champion is not None and niche.champion.program_id == program_id:
                niche.champion = None
                niche.champion_rq = -1.0
                removed.append(key)
        return removed

    def restore_program_after_similarity_timeout(
        self,
        program: ProblemProgram,
        cells: list[tuple[int, int]],
        *,
        rq_score: float,
    ) -> list[tuple[int, int]]:
        """Put a re-evaluated incumbent back after a fail-closed timeout.

        Re-evaluation temporarily removes an incumbent before re-inserting it,
        otherwise it would compete against the same mutable object.  A timeout
        in a cross-cell novelty comparison is an infrastructure failure, not
        evidence that the already-admitted incumbent became invalid.  Preserve
        its old occupancy while retaining its freshly measured score.
        """

        restored: list[tuple[int, int]] = []
        for cell in cells:
            niche = self.grid.get(cell)
            if niche is None or niche.champion is not None:
                continue
            niche.champion = program
            niche.champion_rq = float(rq_score)
            program.niche_domain, program.niche_problem_type = cell
            niche.history.append(
                {
                    "event": "reevaluation_preserved_after_similarity_timeout",
                    "program_id": program.program_id,
                    "rq_score": float(rq_score),
                }
            )
            restored.append(cell)
        if restored:
            program.metadata["archive_status"] = (
                "champion_preserved_after_similarity_timeout"
            )
        return restored

    # ------------------------------------------------------------------
    # Safety gates (ported from evo-sample map_elites / mutation)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_sig_text(text: str) -> str:
        """Whitespace-collapsed, lowercased text for exact-clone signatures."""
        return " ".join(str(text or "").split()).lower()

    @staticmethod
    def _template_normalize_text(text: str) -> str:
        """Numeric-free skeleton: every int / decimal literal becomes 'N'."""
        skeleton = re.sub(r"-?\d+(?:\.\d+)?", "N", str(text or ""))
        return " ".join(skeleton.split()).lower()

    def program_behavior_signature(
        self, program: ProblemProgram, n_seeds: int = 5
    ) -> str | None:
        """Hash of the (problem, answer) sequence across seeds, id-independent.

        Catches near-clone generators that emit the identical problem/answer
        sequence for the verification seeds but land in different cells under
        noisy H/D estimates. Cached on program metadata.
        """
        cache_key = f"_behavior_sig_{n_seeds}"
        cached = (program.metadata or {}).get(cache_key)
        if cached:
            return str(cached)
        pairs = []
        for seed in range(n_seeds):
            inst = program.execute(seed=seed)
            if inst is None:
                return None
            pairs.append(
                (
                    self._normalize_sig_text(inst.problem),
                    self._normalize_sig_text(inst.answer),
                )
            )
        signature = hashlib.sha256(
            json.dumps(pairs, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        program.metadata[cache_key] = signature
        return signature

    def program_template_signature(
        self, program: ProblemProgram, n_seeds: int = 5
    ) -> str | None:
        """Hash of the numeric-free problem skeleton across seeds.

        Unlike the behavior signature (exact pairs -> digit-identical clones),
        this replaces every number with 'N', catching generators that emit the
        same sentence template with different sampled numbers.
        """
        cache_key = f"_template_sig_{n_seeds}"
        cached = (program.metadata or {}).get(cache_key)
        if cached:
            return str(cached)
        templates = []
        for seed in range(n_seeds):
            inst = program.execute(seed=seed)
            if inst is None:
                return None
            templates.append(self._template_normalize_text(inst.problem))
        signature = hashlib.sha256(
            json.dumps(sorted(set(templates)), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        program.metadata[cache_key] = signature
        return signature

    # A restatement is not "similar to" the original -- it CONTAINS it, plus a
    # few words. So the measure is containment (matched characters over the
    # shorter statement), not SequenceMatcher's ratio, which divides by the
    # combined length and so penalises exactly the extra words that make a
    # restatement a restatement: the live pair
    #
    #   Let n = N. How many distinct prime factors does n have?
    #   Let n = N be a positive integer. How many distinct prime factors ...
    #
    # scores 1.00 by containment and only 0.88 by ratio.
    #
    # Measured on a live 21-cell archive, containment separated the collisions
    # (1.000, 0.964, 0.901) from the merely related (0.855, 0.788, 0.720).
    # 0.85, lowered from 0.90 on measurement. Over the 137 parent/child pairs of
    # the 4B run the band this opens up (0.85-0.90, 17 pairs) has AST-skeleton
    # similarity median 1.000, and 15 of the 17 are also >= 0.95 structurally --
    # same statement AND same program, i.e. duplicates by both axes rather than
    # a legitimate refinement that happens to share vocabulary. Below 0.85 the
    # band stops being clean: the 0.85-and-under pairs are only 44% structural
    # matches, so that is where genuine variety starts.
    near_duplicate_template_ratio: float = 0.85
    # Containment alone would reject a genuinely bigger question that happens to
    # open with a smaller one ("... how many divisors?" vs "... how many
    # divisors? of those, how many are prime?"). Both statements must also be
    # comparable in size before one is called a restatement of the other.
    near_duplicate_length_ratio: float = 0.60
    # Absolute floor on the compared text, for the same reason the structural
    # gate has one: containment over a 20-character statement is not evidence.
    # "real: what is n + n?" and any other short question share most of their
    # characters, and at the 0.85 threshold that is a rejection. Real material is
    # nowhere near it -- champion templates of the 4B run run 114 to 1267
    # characters (median 283).
    #
    # 60. A numeric-free statement shorter than that is not a competition
    # problem, it is a fragment, and containment over it is dominated by the
    # common English words any two questions share.
    #
    # The bound is pinned from both sides by material that exists:
    #   20  chars  a bare toy ("real: what is n + n?")            -- must exempt
    #   45  chars  "algebra studied by counting, then add n to n" -- must exempt
    #   79  chars  "Let n = N. How many distinct prime factors
    #              does n have? State only the integer."          -- MUST gate
    #   114 chars  the shortest real champion template of the 4B run
    #
    # The 79-character case is the one the gate exists for: the same question
    # with "be a positive integer" inserted lands in a second cell and reports
    # as coverage. An earlier floor of 80 sat just above it and silently turned
    # the gate off for precisely that failure.
    near_duplicate_min_chars: int = 60
    # Structural duplicate: the SAME PROGRAM behind a different statement.
    #
    # The template gates above compare what the problem SAYS. This one compares
    # what the generator IS -- its AST reduced to the sequence of node types,
    # with every identifier, constant and string discarded. The two catch
    # opposite halves of the same failure, and the run's worst case slipped
    # through both because only the statement gate existed:
    #
    #   parent : "... Construct a specific coloring ... all the same color."
    #   child  : "... Prove that it is impossible to construct a specific
    #             coloring ... all the same color."   Descriptor relabelled.
    #
    # Byte-identical code. Template similarity 0.746 -- under the 0.90 bar.
    # Skeleton similarity 1.000. It entered a second cell and counted as
    # coverage, and 31 of 31 such pairs at source similarity >= 0.90 did the
    # same, which is how 48/48 coverage was reached with 37 dead cells.
    #
    # 0.90. Measured over 221 generated children against the 48-champion grid:
    #
    #   threshold   rejected   parent copies caught   resample multiplier
    #   0.96          19%           41 of 52               1.23
    #   0.95          24%           52 of 52               1.32
    #   0.90          35%           52 of 52               1.55
    #
    # Recall is already 1.00 at 0.95, so 0.90 was first read as 26 pointless
    # extra rejections. It is not: those 25 children sit at skeleton 0.933 to
    # their nearest champion AND 0.913 to their own parent, and 18 of 25 (72%)
    # are also >= 0.70 similar to that champion in raw source. They are
    # rewordings that fall just under the copy cutoff, not new material -- the
    # archive's own baseline between different-cell champions is 0.677, so 0.90
    # is still far above what unrelated programs score.
    #
    # The rejection is affordable because this gate runs BEFORE scoring: a
    # rejected child costs one stage-1 + stage-2 regeneration (~480 decode
    # tokens) and saves the 10 rollouts it would otherwise have been scored on.
    #
    # It is also the threshold ShinkaEvolve and Promptbreeder use, though both
    # apply it to embeddings of TEXT -- which cannot see this failure at all,
    # because the text genuinely differs. The measured copy pair scores 0.746 on
    # statement text and 1.000 on skeleton.
    structural_duplicate_ratio: float = 0.90
    # Below this many AST nodes the skeleton carries no information and the
    # comparison must not run. Measured: two UNRELATED 20-node generators (a
    # function that returns one f-string) score 0.905 against each other, which
    # this gate would call a duplicate. Over real material the picture inverts --
    # sampling unrelated champion pairs by the shorter skeleton's length gives
    # median 0.685 and only 2.7% at >= 0.90 in the 60-119 node band, and median
    # 0.453 at 120-179. Every champion of the 4B run is 99-604 nodes (median
    # 159) and every seed program is 147-604, so a floor of 60 exempts nothing
    # real while excluding exactly the degenerate range where the metric is
    # noise. tests/test_archive.py builds 20-node fixtures and is what surfaced
    # this.
    structural_min_nodes: int = 60

    def program_skeleton(self, program: ProblemProgram) -> tuple[str, ...] | None:
        """The generator's AST as a flat sequence of node type names.

        Cached on the program's metadata like ``template_text``, so the archive
        sweep costs one parse per program rather than one per comparison.
        """
        # v2 selects the model-owned build_instance subtree when the trusted
        # assembler contract is present.  Never reuse the pre-builder cache:
        # those entries flatten the whole module and are dominated by the common
        # canonical generate wrapper.
        cache_key = "_ast_skeleton_v2_builder_aware"
        cached = (program.metadata or {}).get(cache_key)
        if cached is not None:
            return tuple(cached) or None
        try:
            tree = ast.parse(program.source_code)
        except SyntaxError:
            program.metadata[cache_key] = []
            return None
        builder, builder_claimed = exact_top_level_build_instance(tree)
        if builder_claimed and builder is None:
            # Never fall through to the shared wrapper when the builder name is
            # duplicated/rebound or its signature is malformed.
            program.metadata[cache_key] = []
            return None
        seq: list[str] = []

        def walk(node: ast.AST) -> None:
            seq.append(type(node).__name__)
            for child in ast.iter_child_nodes(node):
                walk(child)

        # Legacy generate-only sources retain the old whole-module fingerprint.
        # Assembled sources intentionally exclude imports, the frozen family
        # template, DOMAIN, and the canonical generate shell.
        walk(builder or tree)
        program.metadata[cache_key] = seq
        return tuple(seq)

    def _find_structural_duplicate(
        self, program: ProblemProgram
    ) -> tuple[ProblemProgram, float] | None:
        """The closest champion whose generator has essentially this shape."""
        mine = self.program_skeleton(program)
        if not mine or len(mine) < self.structural_min_nodes:
            return None
        best: tuple[ProblemProgram, float] | None = None
        for niche in self.grid.values():
            existing = niche.champion
            if existing is None or existing.program_id == program.program_id:
                continue
            theirs = self.program_skeleton(existing)
            if not theirs or len(theirs) < self.structural_min_nodes:
                continue
            ratio = sequence_ratio(mine, theirs, autojunk=False)
            if ratio >= self.structural_duplicate_ratio and (
                best is None or ratio > best[1]
            ):
                best = (existing, ratio)
        return best

    def template_text(self, program: ProblemProgram, n_seeds: int = 5) -> str | None:
        """The numeric-free statement set, as text rather than a hash.

        ``program_template_signature`` compares hashes, so it only catches an
        exact match. That let "Let n = N. How many distinct prime factors does
        n have?" and "Let n = N be a positive integer. How many distinct prime
        factors does n have?" occupy two different cells: five words apart, and
        two different hashes. Comparing the text admits a threshold.
        """
        cache_key = f"_template_text_{n_seeds}"
        cached = (program.metadata or {}).get(cache_key)
        if cached is not None:
            return str(cached) or None
        templates = []
        for seed in range(n_seeds):
            inst = program.execute(seed=seed)
            if inst is None:
                return None
            templates.append(self._template_normalize_text(inst.problem))
        text = " ".join(sorted(set(templates)))
        program.metadata[cache_key] = text
        return text

    def _find_near_duplicate_template(
        self, program: ProblemProgram
    ) -> tuple[ProblemProgram, float] | None:
        """The closest champion whose statement is essentially this one.

        Archive-wide, and deliberately so: the labels are what decide the cell,
        and when they are unreliable the same problem lands in several cells and
        reports as coverage. A cell holding a restatement of another cell is a
        cell the curriculum does not actually have.
        """
        mine = self.template_text(program)
        if not mine or len(mine) < self.near_duplicate_min_chars:
            return None
        best: tuple[ProblemProgram, float] | None = None
        for niche in self.grid.values():
            existing = niche.champion
            if existing is None or existing.program_id == program.program_id:
                continue
            theirs = self.template_text(existing)
            if not theirs or len(theirs) < self.near_duplicate_min_chars:
                continue
            shorter, longer = sorted((len(mine), len(theirs)))
            if shorter / max(1, longer) < self.near_duplicate_length_ratio:
                continue
            # autojunk=False is load-bearing, not a style choice. difflib's
            # default heuristic drops any element appearing in more than 1% of a
            # sequence longer than 200 -- and these are CHARACTER sequences of
            # ~500, so it junks the common letters and reports two statements
            # that differ by one inserted clause as 0.746 similar. Measured over
            # the 137 parent/child pairs of the 4B run this gate rejected 0 of
            # them with the default and 23 (17%) without it; the pair it most
            # obviously had to catch -- identical source, "Construct a ..."
            # reworded to "Prove that it is impossible to construct a ...", the
            # descriptor relabelled, landing in a second cell -- scores 0.746 with
            # autojunk and 1.000 without. The gate had never once fired.
            matched = matched_size(mine, theirs, autojunk=False)
            ratio = matched / max(1, shorter)
            if ratio >= self.near_duplicate_template_ratio and (
                best is None or ratio > best[1]
            ):
                best = (existing, ratio)
        return best

    def _find_duplicate(self, program, signature_fn) -> ProblemProgram | None:
        signature = signature_fn(program)
        if not signature:
            return None
        for niche in self.grid.values():
            existing = niche.champion
            if existing is None or existing.program_id == program.program_id:
                continue
            if signature_fn(existing) == signature:
                return existing
        return None

    def _find_duplicate_behavior(
        self, program: ProblemProgram
    ) -> ProblemProgram | None:
        return self._find_duplicate(program, self.program_behavior_signature)

    def _find_duplicate_template(
        self, program: ProblemProgram
    ) -> ProblemProgram | None:
        return self._find_duplicate(program, self.program_template_signature)

    def _purge_program_from_other_cells(
        self, program_id: str, keep_cell: tuple[int, int]
    ) -> int:
        """Archive-global uniqueness: a generator is champion in one cell only.

        Clears ``program_id`` from every cell except ``keep_cell`` (e.g. an
        older champion seat left behind after H/D rebinning). Returns the count
        purged (logging only).
        """
        purged = 0
        for key, niche in self.grid.items():
            if key == keep_cell:
                continue
            if niche.champion is not None and niche.champion.program_id == program_id:
                niche.champion = None
                niche.champion_rq = -1.0
                purged += 1
        return purged

    def _passes_seed_variation(self, program: ProblemProgram, n_seeds: int = 5) -> bool:
        """Require valid execution and visible variation across verification seeds.

        Constant answers can be mathematically legitimate (for example an
        invariant or feasibility family), so answer diversity is diagnostic
        rather than a validity requirement. Hidden seed-dependent answers behind
        one unchanged problem are rejected because the visible instances must
        themselves vary.
        """
        rq_now = float(getattr(program, "rq_score", 0.0) or 0.0)
        cache = (program.metadata or {}).get("validity_check")
        if cache is not None and cache.get("rq_score_at_check") == rq_now:
            return bool(cache.get("passed"))

        problems: list[str] = []
        answers: list[str] = []
        n_broken = 0
        for seed in range(n_seeds):
            inst = program.execute(seed=seed)
            if inst is None:
                n_broken += 1
                continue
            answer = (inst.answer or "").strip()
            if not answer or lint_problem_instance(inst):
                n_broken += 1
                continue
            problems.append(" ".join((inst.problem or "").split()))
            answers.append(answer)

        n_total = n_seeds
        passed = (
            n_broken == 0
            and len(answers) == n_total
            and (n_total <= 1 or len(set(problems)) >= 2)
        )

        # Structural gates against a generator that ignores its seed. These are
        # pass/fail and not a fitness term on purpose: rewarding consistency
        # would hand a free maximum to exactly the generator being excluded,
        # and evolution would have a gradient to climb toward it. A gate has no
        # such gradient. (Design note: Corollary 2.2.)
        verdict = None
        if passed:
            verdict = check_constancy(program.source_code, problems, answers)
            passed = verdict.passed

        program.metadata["validity_check"] = {
            "passed": passed,
            "n_distinct_problems": len(set(problems)),
            "n_distinct_answers": len(set(answers)),
            "n_valid": len(answers),
            "n_total": n_total,
            "rq_score_at_check": rq_now,
            **(
                {
                    "constancy_passed": verdict.passed,
                    "constancy_reason": verdict.reason,
                    "canonical_templates": verdict.templates,
                    "distinct_answers": verdict.answers,
                    "z_sensitive_fraction": round(verdict.z_sensitive_fraction, 3),
                }
                if verdict is not None
                else {}
            ),
        }
        return passed

    def champions(self) -> list[ProblemProgram]:
        return [n.champion for n in self.grid.values() if n.champion is not None]

    def sample_structural_inspiration(
        self,
        parent: ProblemProgram,
        *,
        rng: random.Random,
        max_template_chars: int = 1600,
        selection_strategy: str = "cross_lineage_random",
        require_certified: bool = False,
        require_positive_rq: bool = False,
    ) -> StructuralInspirationSelection:
        """Uniformly sample one usable donor outside the parent's lineage.

        The archive contains no ground-truth semantic-family identifier.  A
        primary-parent lineage is therefore the reproducible proxy: every seed
        starts a root and every child inherits its primary parent's root.  The
        archive's existing template/near-template/structural duplicate gates
        provide the complementary content-level separation.

        No UCB/selection counter is touched here.  Those counters describe
        reproductive parent selection; an inspiration is context only.
        """
        if selection_strategy != "cross_lineage_random":
            raise ValueError(
                "structural inspiration selection must be "
                f"'cross_lineage_random', got {selection_strategy!r}"
            )
        if max_template_chars < 1:
            raise ValueError("max_template_chars must be >= 1")

        candidates = (
            list(self.structural_donors.values())
            if require_certified
            else self.champions()
        )
        pool = [p for p in candidates if p.program_id != parent.program_id]
        quality_pool: list[ProblemProgram] = []
        uncertified_count = 0
        nonpositive_rq_count = 0
        for donor in pool:
            certified = bool(
                (
                    (donor.metadata or {}).get("structural_donor_certification") or {}
                ).get("passed")
            )
            if require_certified and not certified:
                uncertified_count += 1
                continue
            if require_positive_rq and not float(donor.rq_score or 0.0) > 0.0:
                nonpositive_rq_count += 1
                continue
            quality_pool.append(donor)
        parent_root = parent.lineage_root_id()
        cross_lineage = [p for p in quality_pool if p.lineage_root_id() != parent_root]

        usable: list[tuple[ProblemProgram, str]] = []
        missing_template = 0
        oversized_template = 0
        unsafe_template = 0
        unsafe_reasons: dict[str, int] = {}
        for donor in cross_lineage:
            template = extract_problem_statement_template(donor.source_code)
            if not template:
                missing_template += 1
                continue
            if len(template) > max_template_chars:
                oversized_template += 1
                continue
            unsafe_reason = structural_inspiration_safety_reason(template)
            if unsafe_reason is not None:
                unsafe_template += 1
                unsafe_reasons[unsafe_reason] = unsafe_reasons.get(unsafe_reason, 0) + 1
                continue
            usable.append((donor, template))

        base = {
            "enabled": True,
            "selection_strategy": selection_strategy,
            "pool_size": len(pool),
            "quality_eligible_pool_size": len(quality_pool),
            "uncertified_donor_count": uncertified_count,
            "nonpositive_rq_donor_count": nonpositive_rq_count,
            "require_certified": bool(require_certified),
            "require_positive_rq": bool(require_positive_rq),
            "cross_lineage_pool_size": len(cross_lineage),
            "eligible_count": len(usable),
            "missing_template_count": missing_template,
            "oversized_template_count": oversized_template,
            "unsafe_template_count": unsafe_template,
            "unsafe_template_reasons": dict(sorted(unsafe_reasons.items())),
            "max_template_chars": int(max_template_chars),
        }
        if not usable:
            if not pool:
                reason = "no_other_champion"
            elif not quality_pool:
                reason = "no_quality_eligible_donor"
            elif not cross_lineage:
                reason = "no_cross_lineage_champion"
            elif oversized_template and not missing_template and not unsafe_template:
                reason = "all_cross_lineage_templates_oversized"
            elif unsafe_template and not missing_template and not oversized_template:
                reason = "all_cross_lineage_templates_unsafe"
            else:
                reason = "no_usable_cross_lineage_template"
            return StructuralInspirationSelection(
                donor=None,
                template=None,
                provenance={**base, "attached": False, "omitted_reason": reason},
            )

        # Directionless mutation means descriptors cannot influence which
        # inspiration is shown. Lineage and safety gates define eligibility;
        # the donor is otherwise uniform over the complete usable pool.
        donor, template = rng.choice(usable)
        provenance = {
            **base,
            "attached": True,
            "selection_tier": "cross_lineage_uniform",
            "selection_pool_size": len(usable),
            "program_id": donor.program_id,
            "lineage_root_id": donor.lineage_root_id(),
            # Descriptors stay out of the prompt and out of selection, but are
            # persisted for audit after donor eviction.
            "domain": donor.get_domain(),
            "problem_type": donor.get_problem_type(),
            "donor_rq_score": float(donor.rq_score),
            "donor_certification_source": str(
                (
                    (donor.metadata or {}).get("structural_donor_certification") or {}
                ).get("source", "")
            ),
            "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "template_chars": len(template),
            # The donor may be evicted before the post-iteration archive is
            # written. Persist this already-sanitized <=max_chars statement so
            # claimed transfers and copy-gate decisions remain auditable.
            "statement_template": template,
        }
        return StructuralInspirationSelection(
            donor=donor,
            template=template,
            provenance=provenance,
        )

    def compare_with_structural_inspiration(
        self,
        child: ProblemProgram,
        donor: ProblemProgram,
        *,
        max_token_jaccard: float | None = None,
        near_duplicate_min_chars: int | None = None,
    ) -> dict:
        """Apply the archive's copy gates directly to one assigned donor.

        The live MAP is allowed to replace a donor before this child eventually
        competes for a cell.  Comparing against the frozen donor object here
        makes "inspiration, not imitation" invariant to that replacement and
        moves a deterministic rejection ahead of expensive solver rollouts.
        """
        exact_source = child.program_id == donor.program_id
        child_behavior = self.program_behavior_signature(child)
        donor_behavior = self.program_behavior_signature(donor)
        exact_behavior = bool(
            child_behavior and donor_behavior and child_behavior == donor_behavior
        )
        child_template = self.program_template_signature(child)
        donor_template = self.program_template_signature(donor)
        exact_template = bool(
            child_template and donor_template and child_template == donor_template
        )

        token_jaccard: float | None = None
        child_text = self.template_text(child)
        donor_text = self.template_text(donor)
        if child_text and donor_text:
            child_tokens = set(re.findall(r"[a-z]+(?:_[a-z]+)*", child_text.lower()))
            donor_tokens = set(re.findall(r"[a-z]+(?:_[a-z]+)*", donor_text.lower()))
            union = child_tokens | donor_tokens
            if union:
                token_jaccard = len(child_tokens & donor_tokens) / len(union)

        comparison_min_chars = (
            self.near_duplicate_min_chars
            if near_duplicate_min_chars is None
            else max(0, int(near_duplicate_min_chars))
        )
        near_template_ratio: float | None = None
        similarity_timed_out = False
        if (
            child_text
            and donor_text
            and len(child_text) >= comparison_min_chars
            and len(donor_text) >= comparison_min_chars
        ):
            shorter, longer = sorted((len(child_text), len(donor_text)))
            if shorter / max(1, longer) >= self.near_duplicate_length_ratio:
                try:
                    matched = matched_size(
                        child_text, donor_text, autojunk=False
                    )
                    near_template_ratio = matched / max(1, shorter)
                except SimilarityTimeout:
                    near_template_ratio = None
                    similarity_timed_out = True

        structural_ratio: float | None = None
        child_skeleton = self.program_skeleton(child)
        donor_skeleton = self.program_skeleton(donor)
        if (
            child_skeleton
            and donor_skeleton
            and len(child_skeleton) >= self.structural_min_nodes
            and len(donor_skeleton) >= self.structural_min_nodes
        ):
            try:
                structural_ratio = sequence_ratio(
                    child_skeleton, donor_skeleton, autojunk=False
                )
            except SimilarityTimeout:
                structural_ratio = None
                similarity_timed_out = True

        reason = None
        if similarity_timed_out:
            reason = "similarity_timeout"
        elif exact_source:
            reason = "exact_source"
        elif exact_behavior:
            reason = "duplicate_behavior"
        elif exact_template:
            reason = "duplicate_template"
        elif (
            max_token_jaccard is not None
            and token_jaccard is not None
            and token_jaccard >= float(max_token_jaccard)
        ):
            reason = "donor_token_jaccard"
        elif (
            near_template_ratio is not None
            and near_template_ratio >= self.near_duplicate_template_ratio
        ):
            reason = "near_duplicate_template"
        elif (
            structural_ratio is not None
            and structural_ratio >= self.structural_duplicate_ratio
        ):
            reason = "structural_duplicate"

        return {
            "checked": True,
            "rejected": reason is not None,
            "reason": reason,
            "exact_source": exact_source,
            "exact_behavior": exact_behavior,
            "exact_template": exact_template,
            "token_jaccard": (
                round(token_jaccard, 6) if token_jaccard is not None else None
            ),
            "max_token_jaccard": (
                float(max_token_jaccard) if max_token_jaccard is not None else None
            ),
            "near_duplicate_min_chars": comparison_min_chars,
            "near_template_ratio": (
                round(near_template_ratio, 6)
                if near_template_ratio is not None
                else None
            ),
            "structural_ratio": (
                round(structural_ratio, 6) if structural_ratio is not None else None
            ),
            "similarity_timed_out": bool(similarity_timed_out),
        }

    def _select_priority(self, program: ProblemProgram | None) -> float:
        """Selection priority for a champion: full R_Q, or s(1-s) under the
        select_ignores_uncertainty ablation, or H under select_ignores_variance."""
        if program is None:
            return 0.0
        return selection_priority(
            float(getattr(program, "s_hat", 0.0) or 0.0),
            float(getattr(program, "rq_score", 0.0) or 0.0),
            float(getattr(program, "u_score", 0.0) or 0.0),
            ignore_uncertainty=self.select_ignores_uncertainty,
            ignore_variance=self.select_ignores_variance,
        )

    def _is_learnable(self, program: ProblemProgram | None) -> bool:
        """Priority > 0, i.e. 0 < s_hat < 1 with a non-zero uncertainty term.

        NOT a parent filter -- see ``sample_parent``. Kept as a description of
        which champions are learnable RIGHT NOW, which is a different question
        from which ones are worth mutating.
        """
        return program is not None and self._select_priority(program) > 0.0

    def sample_parent(self, rng: random.Random | None = None) -> ProblemProgram | None:
        rng = rng or random
        occupied = [(key, n) for key, n in self.grid.items() if n.champion is not None]
        if not occupied:
            return None
        # EVERY occupied niche is a parent, R_Q = 0 included.
        #
        # This used to keep only champions with R_Q > 0. That silently undid the
        # decision to let R_Q = 0 champions stay on the map: they held a cell and
        # never reproduced, so the cells that most need new blood were exactly
        # the ones mutation could not start from. Measured on a 4B probe, the
        # pool was 5 of 10 champions and every excluded cell was s_hat = 0 --
        # casework, induction, invariant, extremal_principle, inequality/counting
        # -- the five the policy cannot solve yet. Two of the five survivors were
        # `counting`, so 40% of parents came from one descriptor column and both
        # evolved children landed there.
        #
        # R_Q = 0 covers two opposite situations and neither is a reason to
        # sterilise a niche. s_hat = 0 means the policy cannot solve it YET, and
        # a stronger policy should get another look; s_hat = 1 means it is solved
        # outright, and its structure is still a fine starting point for a
        # harder variant. Fitness decides which champion holds a cell, not which
        # cells may reproduce -- that is the MAP-Elites separation.
        pool = occupied

        if self.selection_strategy == "random":
            key, niche = rng.choice(pool)
        elif self.selection_strategy == "sliding_window_mce":
            key, niche = self._sample_sliding_window_mce(pool, rng=rng)
        elif rng.random() < self.epsilon:
            key, niche = rng.choice(pool)
        else:
            key, niche = self._sample_ucb(pool)
        if self.selection_strategy == "sliding_window_mce":
            assert niche.champion is not None
            self._register_mce_pull(niche.champion.program_id, key)
        niche.selection_count += 1
        self.total_selections += 1
        assert niche.champion is not None
        return niche.champion

    def begin_selection_iteration(self, iteration: int) -> None:
        """Start one SW-MCE accounting window at an outer-iteration boundary.

        Every pull from the preceding batch must have received its binary
        survival outcome before a new iteration starts. A non-empty pending set
        means selection and reward credit have diverged, which must fail loudly
        instead of silently biasing future UCB scores.
        """

        iteration = int(iteration)
        if self._mce_pending_by_cell:
            raise RuntimeError("unresolved SW-MCE parent selections at iteration boundary")
        self._mce_iteration = iteration
        self._prune_mce_history(iteration)

    def finalize_parent_outcomes(self) -> None:
        """Assert that every SW-MCE batch pull received exactly one reward."""

        if self.selection_strategy == "sliding_window_mce" and (
            self._mce_pending_by_cell
        ):
            raise RuntimeError("unresolved SW-MCE parent selections after batch")

    def record_parent_outcome(
        self,
        parent_id: str | None,
        survived: bool,
        *,
        iteration: int | None = None,
    ) -> None:
        """Resolve one SW-MCE pull with the original binary survival reward.

        ``survived`` is true exactly when the offspring was admitted to the
        archive, whether by filling an empty cell or replacing its incumbent.
        All compile, verification, rollout, novelty and cell-contest failures
        receive zero, matching Monte Carlo Elites' offspring-survival signal.
        """

        if self.selection_strategy != "sliding_window_mce":
            return
        pending = self._mce_pending_by_parent.get(str(parent_id))
        if not pending:
            raise RuntimeError(
                f"no pending SW-MCE selection for parent_id={parent_id!r}"
            )
        cell = pending.pop(0)
        if not pending:
            self._mce_pending_by_parent.pop(str(parent_id), None)
        self._mce_pending_by_cell[cell] -= 1
        if self._mce_pending_by_cell[cell] <= 0:
            del self._mce_pending_by_cell[cell]

        event_iteration = self._mce_iteration if iteration is None else int(iteration)
        if event_iteration < 0:
            event_iteration = 0
        self.grid[cell].mce_history.append(
            {"iteration": event_iteration, "reward": int(bool(survived))}
        )
        self._prune_mce_history(event_iteration)

    def _register_mce_pull(self, parent_id: str, cell: tuple[int, int]) -> None:
        self._mce_pending_by_parent.setdefault(parent_id, []).append(cell)
        self._mce_pending_by_cell[cell] += 1

    def _prune_mce_history(self, iteration: int) -> None:
        cutoff = int(iteration) - self.mce_window_iterations + 1
        for niche in self.grid.values():
            if niche.mce_history:
                niche.mce_history = [
                    row
                    for row in niche.mce_history
                    if int(row["iteration"]) >= cutoff
                ]

    def _sample_sliding_window_mce(
        self,
        occupied: list[tuple[tuple[int, int], Niche]],
        *,
        rng: random.Random,
    ) -> tuple[tuple[int, int], Niche]:
        """Cell-based MCE UCB over recent binary survival outcomes.

        Pending selections count as pulls with an as-yet unknown (therefore
        zero) reward. This virtual-loss convention is only for allocating a
        parallel batch; the real binary result replaces it once reported.
        """

        if self._mce_iteration >= 0:
            self._prune_mce_history(self._mce_iteration)
        counts: dict[tuple[int, int], int] = {}
        rewards: dict[tuple[int, int], int] = {}
        for key, niche in occupied:
            counts[key] = len(niche.mce_history) + self._mce_pending_by_cell[key]
            rewards[key] = sum(int(row["reward"]) for row in niche.mce_history)
        total = sum(counts.values())

        best_score = -math.inf
        best: list[tuple[tuple[int, int], Niche]] = []
        for item in occupied:
            key, _niche = item
            pulls = counts[key]
            if pulls <= 0:
                score = math.inf
            else:
                survival_rate = rewards[key] / pulls
                exploration = self.ucb_c * math.sqrt(
                    math.log(max(total, 1)) / pulls
                )
                score = survival_rate + exploration
            if score > best_score:
                best_score = score
                best = [item]
            elif score == best_score:
                best.append(item)
        return rng.choice(best)

    def _sample_ucb(self, occupied: list[tuple[tuple[int, int], Niche]]):
        # Exploitation term ranks by selection priority: real R_Q in production,
        # s(1-s) under the select_ignores_uncertainty ablation. champion_rq stays
        # the real stored R_Q; we recompute priority from each champion.
        priorities = {key: self._select_priority(n.champion) for key, n in occupied}
        sorted_rqs = sorted(set(priorities.values()))
        denom = max(len(sorted_rqs) - 1, 1)
        total = self.total_selections + 1

        best_score = -math.inf
        best = occupied[0]
        for item in occupied:
            key, niche = item
            rank = sorted_rqs.index(priorities[key]) / denom
            if niche.selection_count <= 0:
                exploration = math.inf
            else:
                exploration = self.ucb_c * math.sqrt(
                    math.log(total + 1) / niche.selection_count
                )
            score = rank + exploration
            if score > best_score:
                best_score = score
                best = item
        return best

    def stats(self) -> dict[str, float | int]:
        champions = self.champions()
        rqs = [p.rq_score for p in champions]
        total = self.n_domain_bins * self.n_problem_type_bins
        domains_hit = {p.get_domain() for p in champions if p.get_domain()}
        problem_types_hit = {
            p.get_problem_type() for p in champions if p.get_problem_type()
        }
        stats: dict[str, float | int] = {
            "num_champions": len(champions),
            "total_niches": total,
            "coverage": len(champions) / total if total else 0.0,
            "domain_coverage": len(domains_hit) / self.n_domain_bins,
            "problem_type_coverage": (
                len(problem_types_hit) / self.n_problem_type_bins
            ),
            "mean_rq": sum(rqs) / len(rqs) if rqs else 0.0,
            "max_rq": max(rqs) if rqs else 0.0,
            "total_insertions": self.total_insertions,
            "total_replacements": self.total_replacements,
            "total_selections": self.total_selections,
            "num_structural_donors": len(self.structural_donors),
        }
        if self.selection_strategy == "sliding_window_mce":
            window_events = [
                event
                for niche in self.grid.values()
                for event in niche.mce_history
            ]
            survivals = sum(int(event["reward"]) for event in window_events)
            stats.update(
                {
                    "mce_window_pulls": len(window_events),
                    "mce_window_survivals": survivals,
                    "mce_window_survival_rate": (
                        survivals / len(window_events) if window_events else 0.0
                    ),
                }
            )
        return stats

    def to_payload(self) -> dict:
        """In-memory snapshot of the archive (same structure written to
        ``archive.json``). Used to embed the live MAP into the verl ``data.pt``
        checkpoint so the grid is restored atomically with the weights."""
        schema_contract = self._schema_contract()
        return {
            "meta": {
                **schema_contract,
                "axes": ["domain", "problem_type"],
                "domain_labels": list(DOMAINS),
                "problem_type_labels": list(PROBLEM_TYPES),
                "problem_type_authority": "deterministic_statement_and_verifier",
                "problem_type_ruleset": PROBLEM_TYPE_RULESET,
                "problem_type_ruleset_sha256": problem_type_ruleset_sha256(),
                "binning": self.binning,
                "epsilon": self.epsilon,
                "ucb_c": self.ucb_c,
                "selection_strategy": self.selection_strategy,
                "mce_window_iterations": self.mce_window_iterations,
                "stats": self.stats(),
            },
            "champions": [p.to_dict() for p in self.champions()],
            "structural_donors": [
                p.to_dict() for _, p in sorted(self.structural_donors.items())
            ],
            "niches": [
                {
                    "domain_bin": niche.domain_bin,
                    "problem_type_bin": niche.problem_type_bin,
                    "selection_count": niche.selection_count,
                    "update_count": niche.update_count,
                    "history": list(niche.history),
                    "mce_history": list(niche.mce_history),
                }
                for _, niche in sorted(self.grid.items())
                if (
                    niche.selection_count
                    or niche.update_count
                    or niche.history
                    or niche.mce_history
                )
            ],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "archive.json").write_text(
            json.dumps(self.to_payload(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_payload(self, payload: dict) -> int:
        """Restore champions from an in-memory payload (see :meth:`to_payload`).

        Snapshot compatibility is exact and checked before the live archive is
        changed. Old GROUP x SKILL archives, reordered vocabularies, and a
        snapshot from the other binning arm all raise :class:`ArchiveSchemaError`
        instead of being partially resumed or silently rebinned.
        """
        if not isinstance(payload, dict):
            raise ArchiveSchemaError("archive payload must be a mapping")
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ArchiveSchemaError("archive payload has no metadata mapping")

        expected = {
            **self._schema_contract(),
            "axes": ["domain", "problem_type"],
            "domain_labels": list(DOMAINS),
            "problem_type_labels": list(PROBLEM_TYPES),
            "problem_type_authority": "deterministic_statement_and_verifier",
            "problem_type_ruleset": PROBLEM_TYPE_RULESET,
            "problem_type_ruleset_sha256": problem_type_ruleset_sha256(),
            "binning": self.binning,
        }
        mismatches = [
            f"{key}={meta.get(key)!r} (expected {value!r})"
            for key, value in expected.items()
            if meta.get(key) != value
        ]
        if mismatches:
            raise ArchiveSchemaError(
                "incompatible archive snapshot: " + "; ".join(mismatches)
            )

        champion_rows = payload.get("champions", [])
        donor_rows = payload.get("structural_donors", [])
        niche_rows = payload.get("niches", [])
        if not isinstance(champion_rows, list):
            raise ArchiveSchemaError("archive champions must be a list")
        if not isinstance(donor_rows, list):
            raise ArchiveSchemaError("archive structural_donors must be a list")
        if not isinstance(niche_rows, list):
            raise ArchiveSchemaError("archive niches must be a list")
        if len(champion_rows) > len(self.grid):
            raise ArchiveSchemaError(
                f"archive has {len(champion_rows)} champions for "
                f"{len(self.grid)} cells"
            )

        # Decode and validate everything first. A failed load leaves the live
        # archive untouched, which is the fail-closed contract used on resume.
        decoded_champions: list[tuple[ProblemProgram, tuple[int, int]]] = []
        descriptor_cells: set[tuple[int, int]] = set()
        champion_ids: set[str] = set()
        for i, row in enumerate(champion_rows):
            try:
                program = ProblemProgram.from_dict(row)
                descriptor_cell = self.program_to_cell(program)
                float(program.s_hat)
                float(program.u_score)
                float(program.rq_score)
            except Exception as exc:
                raise ArchiveSchemaError(f"invalid champion row {i}: {exc}") from exc
            if descriptor_cell is None:
                raise ArchiveSchemaError(
                    f"champion row {i} has no valid DOMAIN/PROBLEM_TYPE pair"
                )
            contract = (program.metadata or {}).get("descriptor_contract")
            if contract is None:
                raise ArchiveSchemaError(
                    f"champion row {i} has no deterministic descriptor contract"
                )
            if program.program_id in champion_ids:
                raise ArchiveSchemaError(
                    f"duplicate champion program_id {program.program_id!r}"
                )
            champion_ids.add(program.program_id)
            if self.binning == "grid" and descriptor_cell in descriptor_cells:
                raise ArchiveSchemaError(
                    f"multiple champions claim grid cell {descriptor_cell}"
                )
            descriptor_cells.add(descriptor_cell)
            decoded_champions.append((program, descriptor_cell))

        decoded_donors: list[ProblemProgram] = []
        donor_ids: set[str] = set()
        for i, row in enumerate(donor_rows):
            try:
                donor = ProblemProgram.from_dict(row)
                donor_cell = self.program_to_cell(donor)
                float(donor.rq_score)
            except Exception as exc:
                raise ArchiveSchemaError(
                    f"invalid structural donor row {i}: {exc}"
                ) from exc
            if donor_cell is None:
                raise ArchiveSchemaError(
                    f"structural donor row {i} has no valid " "DOMAIN/PROBLEM_TYPE pair"
                )
            certification = (donor.metadata or {}).get(
                "structural_donor_certification"
            )
            if certification is not None and not isinstance(certification, dict):
                raise ArchiveSchemaError(
                    f"invalid structural donor certification in row {i}: "
                    "expected a mapping"
                )
            contract = (donor.metadata or {}).get("descriptor_contract")
            if contract is None:
                raise ArchiveSchemaError(
                    f"structural donor row {i} has no deterministic "
                    "descriptor contract"
                )
            if donor.program_id in donor_ids:
                raise ArchiveSchemaError(
                    f"duplicate structural donor program_id {donor.program_id!r}"
                )
            donor_ids.add(donor.program_id)
            decoded_donors.append(donor)

        decoded_niches: list[tuple[tuple[int, int], int, int, list, list]] = []
        seen_niche_cells: set[tuple[int, int]] = set()
        for i, row in enumerate(niche_rows):
            if not isinstance(row, dict):
                raise ArchiveSchemaError(f"invalid niche row {i}: not a mapping")
            try:
                cell = (
                    int(row["domain_bin"]),
                    int(row["problem_type_bin"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ArchiveSchemaError(
                    f"invalid niche coordinates in row {i}"
                ) from exc
            if cell not in self.grid:
                raise ArchiveSchemaError(
                    f"niche row {i} is outside the 7 x 5 grid: {cell}"
                )
            if cell in seen_niche_cells:
                raise ArchiveSchemaError(f"duplicate niche row for cell {cell}")
            seen_niche_cells.add(cell)
            try:
                selection_count = int(row.get("selection_count", 0))
                update_count = int(row.get("update_count", 0))
            except (TypeError, ValueError) as exc:
                raise ArchiveSchemaError(f"invalid counters in niche row {i}") from exc
            if selection_count < 0 or update_count < 0:
                raise ArchiveSchemaError(f"negative counters in niche row {i}")
            history = row.get("history") or []
            if not isinstance(history, list):
                raise ArchiveSchemaError(
                    f"invalid history in niche row {i}: expected a list"
                )
            mce_history = row.get("mce_history") or []
            if not isinstance(mce_history, list):
                raise ArchiveSchemaError(
                    f"invalid mce_history in niche row {i}: expected a list"
                )
            decoded_mce_history = []
            for j, event in enumerate(mce_history):
                if not isinstance(event, dict):
                    raise ArchiveSchemaError(
                        f"invalid mce_history event {j} in niche row {i}"
                    )
                try:
                    event_iteration = int(event["iteration"])
                    event_reward = int(event["reward"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ArchiveSchemaError(
                        f"invalid mce_history event {j} in niche row {i}"
                    ) from exc
                if event_iteration < 0 or event_reward not in {0, 1}:
                    raise ArchiveSchemaError(
                        f"invalid mce_history values in niche row {i}, event {j}"
                    )
                decoded_mce_history.append(
                    {"iteration": event_iteration, "reward": event_reward}
                )
            decoded_niches.append(
                (
                    cell,
                    selection_count,
                    update_count,
                    list(history),
                    decoded_mce_history,
                )
            )

        saved_stats = meta.get("stats") or {}
        if not isinstance(saved_stats, dict):
            raise ArchiveSchemaError("archive stats must be a mapping")
        try:
            saved_totals = (
                int(saved_stats.get("total_insertions", 0)),
                int(saved_stats.get("total_replacements", 0)),
                int(saved_stats.get("total_selections", 0)),
            )
        except (TypeError, ValueError) as exc:
            raise ArchiveSchemaError("archive totals must be integers") from exc

        for niche in self.grid.values():
            niche.champion = None
            niche.champion_rq = -1.0
            niche.selection_count = 0
            niche.update_count = 0
            niche.history = []
            niche.mce_history = []
        self._mce_pending_by_parent = {}
        self._mce_pending_by_cell = collections.Counter()
        self.structural_donors = {}
        for donor in decoded_donors:
            certified = (
                (donor.metadata or {}).get("structural_donor_certification") or {}
            ).get("passed")
            if certified:
                self.structural_donors[donor.program_id] = donor

        placed = 0
        for program, _descriptor_cell in decoded_champions:
            # _insert_cell, NOT program_to_cell: it is the only thing that reads
            # self.binning. Under "grid" the two are identical, so this changes
            # nothing for a grid archive; under "flat" it is the difference
            # between restoring the archive and collapsing it, because every
            # champion sharing a descriptor pair may occupy a separate slot.
            cell = self._insert_cell(program)
            assert cell is not None  # validated above
            # The saved coordinates are re-derived rather than trusted: both
            # axes are pure functions of the program's own labels, so a stored
            # coordinate can only ever agree or be stale.
            if (
                program.niche_domain,
                program.niche_problem_type,
            ) != cell:
                program.niche_domain, program.niche_problem_type = cell
            niche = self.grid[cell]
            incumbent = niche.champion
            if incumbent is not None and self._select_priority(
                incumbent
            ) >= self._select_priority(program):
                continue
            niche.champion = program
            niche.champion_rq = float(program.rq_score)
            niche.update_count += 1
            placed += 1
        for cell, selection_count, update_count, history, mce_history in decoded_niches:
            niche = self.grid[cell]
            niche.selection_count = selection_count
            niche.update_count = update_count
            niche.history = history
            niche.mce_history = mce_history
        (
            self.total_insertions,
            self.total_replacements,
            self.total_selections,
        ) = saved_totals
        return placed

    def load(self, path: str | Path) -> int:
        """Restore champions written by :meth:`save`.

        Returns the number of champions placed. Schema, ordered vocabularies,
        and every champion's DOMAIN/PROBLEM_TYPE assignment are validated before
        the live archive is cleared. Validity and R_Q gates are not re-applied;
        this restores a compatible saved state exactly.
        """
        path = Path(path)
        archive_file = path / "archive.json" if path.is_dir() else path
        if not archive_file.exists():
            raise FileNotFoundError(f"no archive snapshot at {archive_file}")
        payload = json.loads(archive_file.read_text(encoding="utf-8"))
        return self.load_payload(payload)


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest(), 16)
