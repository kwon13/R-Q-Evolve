import hashlib
from difflib import SequenceMatcher
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from .code_utils import lint_problem_instance
from .constancy import check_constancy
from .concepts import GROUPS, SKILLS, axis_index
from .program import ProblemProgram
from .scoring import selection_priority


@dataclass
class Niche:
    group_bin: int
    skill_bin: int
    champion: ProblemProgram | None = None
    champion_rq: float = -1.0
    selection_count: int = 0
    update_count: int = 0
    history: list[dict] = field(default_factory=list)


class MAPElitesArchive:
    """GROUP x SKILL MAP-Elites grid.

    Both axes are behavioural descriptors read off a program's own declared
    labels: GROUP is the mathematical domain, SKILL the reasoning the visible
    problem forces. The grid is therefore a fixed ``len(GROUPS) x len(SKILLS)``
    and needs no bin-count configuration -- a cell is a (domain, reasoning)
    pair, and the two mutation operators each move exactly one coordinate.

    Uncertainty is NOT an axis. H stays in the fitness, ``R_Q = p(1-p)H``, which
    decides who holds a cell and which champion is sampled as a parent. Binning
    on H as well would have meant one grid coordinate the mutation operators
    cannot aim at: a child lands in an H bin only as a side effect of how hard
    it turned out to be.
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        ucb_c: float = 1.0,
        selection_strategy: str = "random",
        select_ignores_uncertainty: bool = False,
        select_ignores_variance: bool = False,
        binning: str = "grid",
    ) -> None:
        if selection_strategy not in {"ucb", "random"}:
            raise ValueError(f"unknown selection_strategy: {selection_strategy}")
        if binning not in {"grid", "flat"}:
            raise ValueError(f"binning must be 'grid' or 'flat', got {binning!r}")
        if select_ignores_uncertainty and select_ignores_variance:
            raise ValueError(
                "select_ignores_uncertainty and select_ignores_variance are "
                "mutually exclusive: enabling both drops every R_Q factor and "
                "leaves a constant (signal-free) selection priority."
            )
        self.n_group_bins = len(GROUPS)
        self.n_skill_bins = len(SKILLS)
        self.epsilon = float(epsilon)
        self.ucb_c = float(ucb_c)
        self.selection_strategy = selection_strategy
        # Ablations: rank mutation parents by s(1-s) (ignore_uncertainty) or by H
        # (ignore_variance) instead of s(1-s)*H. Neither changes what is stored
        # or binned -- champion_rq and the cell labels stay real.
        self.select_ignores_uncertainty = bool(select_ignores_uncertainty)
        self.select_ignores_variance = bool(select_ignores_variance)
        # Ablation: "flat" keeps the same 48 slots, the same validity gates and
        # the same parent sampling, but a candidate no longer competes only
        # against the champion sharing its (GROUP, SKILL). It takes any free
        # slot, and once full it competes against the weakest occupant. That
        # turns the MAP into a plain top-K pool and isolates what the grid --
        # reserving capacity per behaviour cell -- is actually buying.
        # Labels are still read and recorded, so coverage stays measurable.
        self.binning = binning
        self.total_insertions = 0
        self.total_replacements = 0
        self.total_selections = 0
        self.grid: dict[tuple[int, int], Niche] = {
            (g, s): Niche(group_bin=g, skill_bin=s)
            for g in range(self.n_group_bins)
            for s in range(self.n_skill_bins)
        }

    def program_to_cell(self, program: ProblemProgram) -> tuple[int, int] | None:
        """Grid coordinate from the program's own labels, None if unlabelled.

        None rather than a hashed fallback: a program whose GROUP or SKILL is
        outside the vocabulary has no meaningful niche, and folding those into
        one bin would make a single cell the contest for every mislabelled
        generator.
        """
        group_bin = axis_index("group", program.get_group())
        skill_bin = axis_index("skill", program.get_skill())
        if group_bin is None or skill_bin is None:
            return None
        return (group_bin, skill_bin)

    def _insert_cell(self, program: ProblemProgram) -> tuple[int, int] | None:
        """Slot the program competes for.

        Under ``binning="grid"`` that is its own (GROUP, SKILL) cell. Under
        ``"flat"`` the grid is only storage: the program takes the first free
        slot, and once all 48 are occupied it challenges the weakest one. Same
        capacity, same gates, same sampling -- only the reservation of capacity
        per behaviour cell is removed.

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
        return (GROUPS[cell[0]], SKILLS[cell[1]])

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
        group_bin, skill_bin = cell

        program.niche_group = group_bin
        program.niche_skill = skill_bin
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
        # which cost the bootstrap 3 of 8 seeds and with them the ONLY
        # representative of geometry, inequality, induction and
        # extremal_principle -- the grid lost two GROUPs and three SKILLs
        # before a single mutation ran.
        #
        # Nothing downstream needs the gate: training examples are drained by
        # the frontier band in dataset.py (low < s_hat < high), which excludes
        # p=0 and p=1 on its own, and cell competition below is a strict `>`
        # so an R_Q=0 champion yields to the first scoring program that
        # arrives and never displaces one.
        #
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
        #    miss it, and with SKILL labels only ~22% accurate the restatement
        #    lands in a different cell and counts as new coverage.
        near = self._find_near_duplicate_template(program)
        if near is not None:
            other, ratio = near
            program.metadata["archive_status"] = "near_duplicate_template_rejected"
            program.metadata["duplicate_of"] = other.program_id
            program.metadata["duplicate_ratio"] = round(ratio, 3)
            return False

        niche = self.grid[cell]
        # Champion competition ranks by selection priority: real R_Q in
        # production, s(1-s) under the select_ignores_uncertainty ablation
        # (program.rq_score / s_hat are already set above). The stored
        # champion_rq stays the real R_Q, so the MAP still logs true scores --
        # only the winner choice is H-blind under the ablation.
        new_priority = self._select_priority(program)
        if niche.champion is None or new_priority > self._select_priority(niche.champion):
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
            self._purge_program_from_other_cells(
                program.program_id, keep_cell=cell
            )
            return True
        return False

    def target_cell(self, program: ProblemProgram) -> tuple[int, int] | None:
        """Return the cell a program would target without mutating the MAP.

        None when the program carries no usable GROUP/SKILL pair, matching what
        :meth:`try_insert` would do with it -- including under
        ``binning="flat"``, where the slot is chosen by occupancy rather than
        by label, so the telemetry names the champion actually challenged.
        """
        return self._insert_cell(program)

    def remove_program(self, program_id: str) -> list[tuple[int, int]]:
        """Remove one champion identity from the single live MAP archive."""
        removed: list[tuple[int, int]] = []
        for key, niche in self.grid.items():
            if (
                niche.champion is not None
                and niche.champion.program_id == program_id
            ):
                niche.champion = None
                niche.champion_rq = -1.0
                removed.append(key)
        return removed

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
    near_duplicate_template_ratio: float = 0.90
    # Containment alone would reject a genuinely bigger question that happens to
    # open with a smaller one ("... how many divisors?" vs "... how many
    # divisors? of those, how many are prime?"). Both statements must also be
    # comparable in size before one is called a restatement of the other.
    near_duplicate_length_ratio: float = 0.60

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
        if not mine:
            return None
        best: tuple[ProblemProgram, float] | None = None
        for niche in self.grid.values():
            existing = niche.champion
            if existing is None or existing.program_id == program.program_id:
                continue
            theirs = self.template_text(existing)
            if not theirs:
                continue
            shorter, longer = sorted((len(mine), len(theirs)))
            if shorter / max(1, longer) < self.near_duplicate_length_ratio:
                continue
            matched = sum(
                block.size
                for block in SequenceMatcher(None, mine, theirs).get_matching_blocks()
            )
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

    def _find_duplicate_behavior(self, program: ProblemProgram) -> ProblemProgram | None:
        return self._find_duplicate(program, self.program_behavior_signature)

    def _find_duplicate_template(self, program: ProblemProgram) -> ProblemProgram | None:
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

    def sample_parent(self) -> ProblemProgram | None:
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
        # `counting`, so 40% of parents came from one skill column and both
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
            key, niche = random.choice(pool)
        elif random.random() < self.epsilon:
            key, niche = random.choice(pool)
        else:
            key, niche = self._sample_ucb(pool)
        niche.selection_count += 1
        self.total_selections += 1
        assert niche.champion is not None
        return niche.champion

    def sample_target_cell(self) -> tuple[str, str]:
        """A niche for the next child to aim at: uniform over the whole grid.

        NOT uniform over the EMPTY cells. Measured over 7,776 candidates of the
        4B run, only 16 (0.2%) ever declared a cell that was free, and the
        archive gained exactly the 11 that got in -- mutation was a random walk
        in descriptor space with no goal, so 26 of 48 cells were never even
        aimed at. Naming a target fixes that (empty-cell survival 1.0% -> 17%,
        distinct cells reached 3 -> 18-21 per 400 attempts).

        Restricting the draw to empty cells would buy more of that -- 69% of
        draws land empty instead of 54% -- and cost the thing MAP-Elites is for:
        an occupied cell would never be challenged again. This grid has cells
        that need challenging. Eight to nine champions sit at s_hat = 1 (solved,
        R_Q = 0, contributing nothing to the frontier) and two are ill-posed
        (`number_theory/construction` asks about a collection its statement
        never defines). Under an empty-only draw none of them can ever be
        replaced. Measured, the whole-grid draw reaches the same 17 distinct
        cells and the policy follows the target BETTER (GROUP compliance
        66% -> 78%), because it is no longer being pushed only into the
        combinations it was avoiding for a reason.

        Uniform over SKILL then over GROUP is exactly uniform over the 48 cells,
        the grid being a complete product -- the spelling below is the one that
        stays uniform per axis if either vocabulary ever changes length.
        """
        return (random.choice(GROUPS), random.choice(SKILLS))

    def _sample_ucb(self, occupied: list[tuple[tuple[int, int], Niche]]):
        # Exploitation term ranks by selection priority: real R_Q in production,
        # s(1-s) under the select_ignores_uncertainty ablation. champion_rq stays
        # the real stored R_Q; we recompute priority from each champion.
        priorities = {
            key: self._select_priority(n.champion) for key, n in occupied
        }
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
        total = self.n_group_bins * self.n_skill_bins
        # Per-axis coverage separates "we only ever work in two domains" from
        # "we only ever exercise two reasoning moves" -- the two failure modes
        # the operators are meant to fix, and they look identical in the single
        # coverage number.
        groups_hit = {p.get_group() for p in champions if p.get_group()}
        skills_hit = {p.get_skill() for p in champions if p.get_skill()}
        return {
            "num_champions": len(champions),
            "total_niches": total,
            "coverage": len(champions) / total if total else 0.0,
            "group_coverage": len(groups_hit) / self.n_group_bins,
            "skill_coverage": len(skills_hit) / self.n_skill_bins,
            "mean_rq": sum(rqs) / len(rqs) if rqs else 0.0,
            "max_rq": max(rqs) if rqs else 0.0,
            "total_insertions": self.total_insertions,
            "total_replacements": self.total_replacements,
            "total_selections": self.total_selections,
        }

    def to_payload(self) -> dict:
        """In-memory snapshot of the archive (same structure written to
        ``archive.json``). Used to embed the live MAP into the verl ``data.pt``
        checkpoint so the grid is restored atomically with the weights."""
        return {
            "meta": {
                "axes": ["group", "skill"],
                "group_labels": list(GROUPS),
                "skill_labels": list(SKILLS),
                "epsilon": self.epsilon,
                "ucb_c": self.ucb_c,
                "selection_strategy": self.selection_strategy,
                "stats": self.stats(),
            },
            "champions": [p.to_dict() for p in self.champions()],
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

        Shares all placement/re-binning logic with :meth:`load`; the file-based
        ``load`` is a thin reader that delegates here.
        """
        meta = payload.get("meta", {})
        pre_migration = meta.get("axes") != ["group", "skill"]
        if pre_migration and payload.get("champions"):
            print(
                "[archive.load] snapshot predates the GROUP x SKILL grid "
                f"(axes={meta.get('axes') or 'h x diversity'}). Its champions "
                "carry no SKILL label, so they cannot be placed on the skill "
                "axis and are dropped; the run bootstraps from seed_programs."
            )

        for niche in self.grid.values():
            niche.champion = None
            niche.champion_rq = -1.0

        placed = 0
        unlabelled = 0
        for champ_dict in payload.get("champions", []):
            program = ProblemProgram.from_dict(champ_dict)
            # _insert_cell, NOT program_to_cell: it is the only thing that reads
            # self.binning. Under "grid" the two are identical, so this changes
            # nothing for a grid archive; under "flat" it is the difference
            # between restoring the archive and collapsing it, because every
            # champion sharing a (GROUP, SKILL) pair would otherwise land on one
            # cell and overwrite the others. A flat run that resumed from a
            # snapshot silently became a partly-grid run.
            cell = self._insert_cell(program)
            if cell is None:
                unlabelled += 1
                continue
            # The saved coordinates are re-derived rather than trusted: both
            # axes are pure functions of the program's own labels, so a stored
            # coordinate can only ever agree or be stale.
            if (
                program.niche_group,
                program.niche_skill,
            ) != cell:
                program.niche_group, program.niche_skill = cell
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
        if unlabelled:
            print(
                f"[archive.load] dropped {unlabelled} champion(s) without a "
                "usable GROUP/SKILL pair"
            )
        return placed

    def load(self, path: str | Path) -> int:
        """Restore champions written by :meth:`save`.

        Returns the number of champions placed. Every niche is cleared first so
        the restored grid reflects exactly the saved state. Each champion's cell
        is re-derived from its own GROUP/SKILL labels rather than trusted from
        the snapshot, because both coordinates are pure functions of those
        labels. Validity/RQ gates are not re-applied -- the saved state is
        reproduced as-is -- and a champion with no usable label pair is dropped.
        """
        path = Path(path)
        archive_file = path / "archive.json" if path.is_dir() else path
        if not archive_file.exists():
            raise FileNotFoundError(f"no archive snapshot at {archive_file}")
        payload = json.loads(archive_file.read_text(encoding="utf-8"))
        return self.load_payload(payload)


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest(), 16)
