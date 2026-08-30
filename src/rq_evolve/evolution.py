import collections
import ast
import copy
import concurrent.futures
import hashlib
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .archive import (
    GENERATED_DOMAIN_AUTHORITY,
    MANUAL_DOMAIN_AUTHORITY,
    SOURCE_DOMAIN_AUTHORITY,
    MAPElitesArchive,
    StructuralInspirationSelection,
)
from .ast_contract import check_generator_contract, check_problem_text
from .backends import EvolutionBackend, RolloutRecord
from .code_utils import (
    answer_is_bare_draw,
    answer_leaks_in_every_instance,
    compile_stage2_reply,
    extract_generator_code,
    lint_generator_source,
    lint_compiled_stage2_semantics,
    lint_mutation_generator_source,
    lint_problem_instance,
    structural_inspiration_safety_reason,
    TRUSTED_ASSEMBLER_VERSION,
    validated_domain_declaration,
)
from .concepts import DOMAINS, validate_label_decl
from .config import EvolutionConfig, TrainingDataConfig
from .constancy import canonical_template
from .dataset import (
    DynamicProblemDataset,
    build_replay_training_examples,
    build_training_examples,
)
from .problem_type import (
    PROBLEM_TYPE_RULESET,
    annotate_problem_type,
    problem_type_contract_errors,
    problem_type_ruleset_sha256,
)
from .reward import answers_match, grader_stats
from .program import ProblemInstance, ProblemProgram, configure_sandbox_workers
from .replay import PreviousRQScoreboard, RolloutReplayBuffer
from .seed_stream import SeedStream
from .prompts import (
    DOMAIN_LABELING_METHOD,
    MutationTask,
    MUTATION_OP,
    build_domain_labeling_task,
    build_family_task,
    build_generator_task,
    domain_labeling_ruleset_sha256,
    parse_family_plan,
)
from .scoring import RQResult, compute_rq_program, is_frontier, score_seed
from .similarity import SimilarityTimeout, sequence_ratio
from .solver_trace import clean_and_grade_solver_rollout


# A rejected child is discarded the moment its report is written, so without
# this the source behind "execute failed at seed=0" is gone. Across six runs
# that left ~7,000 rejections whose programs cannot be recovered, and no way to
# measure a new gate's false-positive rate against the population it rejects.
_MAX_LOGGED_SOURCE_CHARS = 8000


@dataclass(slots=True)
class CandidateReport:
    status: str
    op: str
    child_id: str | None = None
    rq_score: float = 0.0
    s_hat: float = 0.0
    u_score: float = 0.0
    reason: str | None = None
    source_code: str | None = None
    ast_findings: list[str] = field(default_factory=list)
    # Reproductive ancestry and context provenance are logged for every
    # outcome, including stage-1 parse failures. Without them only champions
    # could be audited and the donor's effect on the rejection funnel would be
    # unknowable.
    parent_id: str | None = None
    inspiration: dict | None = None
    # Full one-vs-rest DOMAIN score audit on labeler rejection. Accepted rows
    # retain the same mapping in child/archive metadata.
    domain_labeling: dict | None = None
    # Detailed prompt-copy comparison for rejected Stage-2 one-shot copies.
    copy_audit: dict | None = None
    # The one model-visible Stage-2 example selected for this proposal. Keeping
    # it on every outcome makes per-shot compile/verify survival measurable.
    stage2_one_shot: dict | None = None
    # Stage 1 has already fixed these before Stage 2 writes code. Keeping the
    # label-free family and its deterministic output type on compile/verify
    # failures tells us whether code validity is blocking a potentially useful
    # MAP column without pretending to know DOMAIN before the policy labeler
    # has actually run.
    child_family: str | None = None
    prospective_problem_type: str | None = None
    # Stage-1 plans copied from a concrete model-visible family example are
    # rejected before the Stage-2 code call.  Keep the comparison so an empty
    # downstream output is not mistaken for a sampling/parse failure.
    stage1_copy_audit: dict | None = None
    # Exact post-label archive decision for candidates that reach admission.
    # This is per-candidate rather than an aggregate metric: it records the
    # target cell, incumbent and whether novelty or cell competition refused it.
    archive_decision: dict | None = None


def _logged_source(source: str | None) -> str | None:
    if source is None:
        return None
    return source[:_MAX_LOGGED_SOURCE_CHARS]


def _ast_findings(program: "ProblemProgram | None") -> list[str]:
    if program is None:
        return []
    verdict = program.metadata.get("ast_contract") or {}
    return list(verdict.get("findings") or [])


def _report_context(task: "MutationTask | None") -> dict:
    if task is None:
        return {
            "parent_id": None,
            "inspiration": None,
            "stage2_one_shot": None,
            "child_family": None,
            "prospective_problem_type": None,
            "stage1_copy_audit": None,
        }
    parent = getattr(task, "parent", None)
    provenance = dict(getattr(task, "provenance", {}) or {})
    inspiration = provenance.get("structural_inspiration")
    return {
        "parent_id": getattr(parent, "program_id", None),
        "inspiration": dict(inspiration) if inspiration is not None else None,
        "stage2_one_shot": (
            dict(provenance["stage2_one_shot"])
            if provenance.get("stage2_one_shot") is not None
            else None
        ),
        "child_family": provenance.get("child_family"),
        "prospective_problem_type": provenance.get("prospective_problem_type"),
        "stage1_copy_audit": (
            dict(provenance["stage1_copy_audit"])
            if provenance.get("stage1_copy_audit") is not None
            else None
        ),
    }


_FAMILY_COPY_PLACEHOLDER_RE = re.compile(r"\[\[[a-z][a-z0-9_]*\]\]")
_FAMILY_COPY_TOKEN_RE = re.compile(r"[a-z0-9]+|\[\[slot\]\]")


def _family_copy_form(family: str) -> str:
    """Placeholder-insensitive text used only by the Stage-1 copy gate."""

    text = _FAMILY_COPY_PLACEHOLDER_RE.sub("[[slot]]", str(family or "").lower())
    return " ".join(_FAMILY_COPY_TOKEN_RE.findall(text))


def _stage1_prompt_copy_audit(task: "MutationTask", child_family: str) -> dict:
    """Compare a Stage-1 plan with the exact examples visible on that call."""

    candidate = _family_copy_form(child_family)
    candidate_tokens = set(candidate.split())
    comparisons = []
    for example in tuple((task.provenance or {}).get("stage1_visible_examples") or ()):
        example_family = str((example or {}).get("family", ""))
        reference = _family_copy_form(example_family)
        reference_tokens = set(reference.split())
        similarity_timed_out = False
        try:
            # Preserve SequenceMatcher's historical default (autojunk=True),
            # but keep model-controlled text off the trainer driver.
            ratio = sequence_ratio(candidate, reference, autojunk=True)
        except SimilarityTimeout:
            ratio = None
            similarity_timed_out = True
        union = candidate_tokens | reference_tokens
        jaccard = (
            len(candidate_tokens & reference_tokens) / len(union) if union else 0.0
        )
        exact = bool(candidate and candidate == reference)
        # The high near-copy thresholds deliberately avoid treating a shared
        # output phrase ("Compute ...", "Find all ...") as imitation.  Their
        # purpose is to catch a concrete worked family with placeholders merely
        # renamed or lightly paraphrased, not to police mathematical topics.
        rejected = (
            similarity_timed_out
            or exact
            or (ratio is not None and ratio >= 0.92)
            or jaccard >= 0.90
        )
        comparisons.append(
            {
                "visible_index": int((example or {}).get("visible_index", 0)),
                "family_sha256": str((example or {}).get("family_sha256", "")),
                "exact": exact,
                "sequence_ratio": (
                    round(float(ratio), 6) if ratio is not None else None
                ),
                "token_jaccard": round(float(jaccard), 6),
                "similarity_timed_out": similarity_timed_out,
                "rejected": bool(rejected),
            }
        )
    rejection = next((item for item in comparisons if item["rejected"]), None)
    return {
        "checked": bool(comparisons),
        "rejected": rejection is not None,
        "reason": (
            "stage1_similarity_timeout"
            if rejection is not None and rejection.get("similarity_timed_out")
            else "stage1_visible_example_copy"
            if rejection is not None
            else None
        ),
        "visible_index": (
            rejection.get("visible_index") if rejection is not None else None
        ),
        "comparisons": comparisons,
    }


def _binary_p_yes(
    pair: tuple[int, float] | None,
    yes_id: int | None,
    no_id: int | None,
) -> float | None:
    """Recover exact P(YES) from one restricted-token sampled log-prob.

    Decoded text is deliberately not a fallback: without the sampled token's
    processed log-prob, a greedy YES cannot distinguish 0.51 from 0.99 and
    would fabricate confidence for the authoritative DOMAIN labeler.
    """

    if pair is not None and yes_id is not None and no_id is not None:
        try:
            token_id, log_probability = pair
            probability = math.exp(float(log_probability))
            token_id = int(token_id)
            if 0.0 <= probability <= 1.0 and token_id in {yes_id, no_id}:
                return probability if token_id == yes_id else 1.0 - probability
        except (TypeError, ValueError, OverflowError):
            pass
    return None


def _probability_logit(probability: float) -> float:
    bounded = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(bounded / (1.0 - bounded))


def _child_metadata(task: "MutationTask") -> dict:
    """Persistent ancestry/plan metadata for a generated child."""
    metadata = {
        "op": task.op,
        "lineage_root_id": task.parent.lineage_root_id(),
    }
    provenance = dict(task.provenance or {})
    if provenance.get("structural_inspiration") is not None:
        metadata["structural_inspiration"] = dict(provenance["structural_inspiration"])
    if provenance.get("family_plan") is not None:
        metadata["family_plan"] = dict(provenance["family_plan"])
    if provenance.get("stage2_one_shot") is not None:
        metadata["stage2_one_shot"] = dict(provenance["stage2_one_shot"])
    return metadata


def _record_inspiration_copy_gate(
    task: "MutationTask", child: "ProblemProgram", verdict: dict
) -> None:
    """Attach donor-copy telemetry to both the report path and child snapshot."""
    provenance = dict(task.provenance or {})
    inspiration = dict(provenance.get("structural_inspiration") or {})
    inspiration["copy_gate"] = dict(verdict)
    provenance["structural_inspiration"] = inspiration
    task.provenance = provenance
    if "structural_inspiration" in child.metadata:
        child.metadata["structural_inspiration"] = dict(inspiration)


@dataclass
class RQEvolver:
    """Owns one archive, one backend, and one dynamic training dataset."""

    archive: MAPElitesArchive
    backend: EvolutionBackend
    evolution_config: EvolutionConfig = field(default_factory=EvolutionConfig)
    training_config: TrainingDataConfig = field(default_factory=TrainingDataConfig)
    dataset: DynamicProblemDataset = field(default_factory=DynamicProblemDataset)
    used_seeds: dict[str, set[int]] = field(default_factory=dict)
    # The never-reused seed source every evaluation draws from. Persisted with
    # the archive so a resumed run does not re-issue seeds the pre-restart run
    # already graded on -- which would reopen exactly the overfitting hole that
    # fresh seeds close.
    seed_stream: SeedStream = field(default_factory=SeedStream)
    # This iteration's re-scoring rollouts, kept so the solver update trains on
    # them instead of paying for a second sampling pass over the same programs.
    replay: RolloutReplayBuffer = field(default_factory=RolloutReplayBuffer)
    # Which elites train is decided by PAST scores; what they train on is this
    # iteration's rollouts. Scoring and selecting on the same draw would keep
    # whichever elite's measurement noise landed high.
    previous_rq: PreviousRQScoreboard = field(default_factory=PreviousRQScoreboard)
    # program_id -> why it was rejected, for every child that ever failed a gate.
    #
    # Mutation is a near-deterministic function of (prompt, parent) and parents
    # are drawn with replacement from a small archive, so the same source comes
    # back repeatedly: one program was regenerated 13 times in 32 slots, and 34%
    # of a two-iteration probe went on children already rejected. Re-running them
    # cannot teach us anything -- the source and deterministic gates are
    # unchanged -- while each one still costs a 5-seed execution. Keyed by
    # program_id, which IS
    # md5(source_code), so a hit means byte-identical code.
    #
    # Rejections only. Accepted children live in the archive, which has its own
    # behaviour/template duplicate check.
    rejected_children: dict[str, str] = field(default_factory=dict)

    events: list[dict] = field(default_factory=list)
    # Candidate reports from the most recent run_outer_iteration, exposed so the
    # sampler can persist them to the per-step evolution log.
    last_reports: list[CandidateReport] = field(default_factory=list)
    # Number of refresh_dataset() calls so far. Only used to vary the seed of the
    # select_random_order ablation shuffle across outer iterations (reproducible).
    dataset_refresh_count: int = 0
    # Per-champion frontier decision from the most recent refresh_dataset():
    # {program_id, s_hat, rq_score, decision: in_frontier | s_hat_out_of_range}.
    # Persisted into evolution_log.jsonl so the learnability/curriculum filter
    # is observable (which champions fed the training set, and why not).
    last_frontier: list[dict] = field(default_factory=list)
    current_iteration: int = -1
    # Separate from the process-global RNG so enabling donor context does not
    # change baseline parent draws or few-shot rotation. Persisted on disk and
    # in verl's data checkpoint for exact resume at a weight-aligned point.
    inspiration_draw_count: int = 0
    mutation_prompt_draw_count: int = 0
    search_draw_count: int = 0
    # Process-local by design: a resumed process audits the restored snapshot
    # once under the code/rules it actually loaded, while a live process does
    # not repay the deterministic audit every outer iteration.
    strict_champion_audit_completed: bool = field(
        default=False, init=False, repr=False
    )
    strict_champion_audit_stats: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        configure_sandbox_workers(self.evolution_config.program_verify_workers)
        expected = bool(self.evolution_config.independent_domain_labeling)
        actual = bool(self.archive.require_domain_labeling)
        if expected != actual:
            raise ValueError(
                "evolution.independent_domain_labeling must match "
                "archive.require_domain_labeling so direct insertion and "
                "snapshot resume cannot bypass the admission gate"
            )
        if expected:
            evolution_probability = float(
                self.evolution_config.domain_labeling_min_probability
            )
            evolution_margin = float(
                self.evolution_config.domain_labeling_min_logit_margin
            )
            if (
                evolution_probability
                != self.archive.domain_labeling_min_probability
                or evolution_margin
                != self.archive.domain_labeling_min_logit_margin
            ):
                raise ValueError(
                    "evolution and archive DOMAIN-label thresholds must match "
                    "exactly so snapshots cannot weaken the admission contract"
                )

    def load_seed_programs(self, seed_dir: str | Path) -> list[ProblemProgram]:
        programs: list[ProblemProgram] = []
        certified_files = set(self.evolution_config.manual_certified_seed_files)
        for path in sorted(Path(seed_dir).glob("*.py")):
            program = ProblemProgram.from_file(path, generation=0)
            program.metadata["manual_seed_source"] = {
                "method": "loaded_seed_file_v1",
                "source_file": path.name,
                "source_sha256": hashlib.sha256(
                    program.source_code.encode("utf-8")
                ).hexdigest(),
            }
            # Each authored seed begins one primary-parent lineage. Descendants
            # inherit this id; structural inspirations never create a second
            # ancestry edge.
            program.metadata.setdefault("lineage_root_id", program.program_id)
            if path.name in certified_files:
                program.metadata["structural_donor_certification"] = {
                    "passed": True,
                    "source": "manual_seed_allowlist",
                    "seed_file": path.name,
                }
            inst, reason = self.verify_program(program)
            if inst is None:
                self.events.append(
                    {
                        "event": "seed_rejected",
                        "file": path.name,
                        "reason": reason,
                    }
                )
                continue
            programs.append(program)
            self.events.append(
                {
                    "event": "seed_loaded",
                    "file": path.name,
                    "program_id": program.program_id,
                    "problem": inst.problem,
                }
            )
        return programs

    def initialize_archive(self, seed_dir: str | Path) -> int:
        """Load seeds, evaluate them once, and place them in MAP-Elites."""
        inserted = 0
        for program in self.load_seed_programs(seed_dir):
            inst, reason = self.verify_program(program)
            if inst is None:
                self.events.append(
                    {
                        "event": "seed_verify_failed",
                        "program_id": program.program_id,
                        "reason": reason,
                    }
                )
                continue
            result = self.evaluate_programs([program], store_replay=True)[0]
            if result is None:
                continue
            # Bootstrap counts as iteration -1. Without it every seed would be
            # "first scored this iteration" at t=0, the previous-score filter would
            # exclude all of them, and the first training batch would be empty.
            self.previous_rq.record(program.program_id, -1, result.rq_score)
            if self.archive.try_insert(
                program=program,
                u_value=result.u_score,
                rq_score=result.rq_score,
            ):
                inserted += 1
        self.refresh_dataset()
        return inserted

    def verify_program(
        self,
        program: ProblemProgram,
        n_seeds: int | None = None,
        *,
        reserve_seed_stream: bool = True,
    ) -> tuple[ProblemInstance | None, str | None]:
        """Multi-seed validity plus deterministic descriptor verification.

        In production, generated Stage-2 source contains no DOMAIN. A separate
        batched, label-blind policy readback assigns DOMAIN from the already
        fixed family after this local verification and before solver rollouts.
        Hand-authored bootstrap seeds retain an exact source DOMAIN tied to
        their loaded-file hash. PROBLEM_TYPE is inferred from each rendered
        request and cross-checked with its verifier on every verification seed.
        """
        n = n_seeds or self.evolution_config.verify_seeds
        source_errors = lint_generator_source(program.source_code)
        mode = self.evolution_config.ast_contract
        is_generated_mutation = program.metadata.get("op") == MUTATION_OP
        requires_labeler = self.archive.require_domain_labeling
        manual_seed_source = program.metadata.get("manual_seed_source")
        is_loaded_manual_seed = bool(
            not is_generated_mutation
            and program.generation == 0
            and not program.parent_id
            and isinstance(manual_seed_source, dict)
            and manual_seed_source.get("method") == "loaded_seed_file_v1"
            and manual_seed_source.get("source_file")
            == program.metadata.get("source_file")
            and manual_seed_source.get("source_sha256")
            == hashlib.sha256(program.source_code.encode("utf-8")).hexdigest()
        )
        if requires_labeler and is_generated_mutation:
            domain = None
            try:
                source_tree = ast.parse(program.source_code)
            except SyntaxError:
                source_tree = None
            if source_tree is None or any(
                isinstance(node, ast.Name) and node.id == "DOMAIN"
                for node in ast.walk(source_tree)
            ):
                source_errors.append(
                    "generated Stage-2 source must not declare or read DOMAIN; "
                    "the local family labeler assigns it later"
                )
        else:
            domain, domain_errors = validated_domain_declaration(program.source_code)
            source_errors.extend(domain_errors)
            if requires_labeler and not is_loaded_manual_seed:
                source_errors.append(
                    "confirmed archives accept source DOMAIN only from a seed "
                    "loaded by load_seed_programs"
                )
        if re.search(
            r"\b(?:PROBLEM_TYPE|GROUP|SKILL)\s*[:=]", program.source_code
        ):
            source_errors.append(
                "program may not contain PROBLEM_TYPE/GROUP/SKILL declarations "
                "or field markers"
            )
        # Stale metadata must never override the current source/rules verdict.
        for key in (
            "domain",
            "problem_type",
            "descriptor_contract",
            "domain_labeling",
        ):
            program.metadata.pop(key, None)
        if is_generated_mutation:
            source_errors.extend(
                lint_compiled_stage2_semantics(program.source_code)
            )
            # Determinism only: a seeded `rng`, no direct `random.*` calls, and
            # sorted() around dict/set iteration, so a program yields the same
            # instance in the verifier and in every later rollout process.
            #
            # Every notation flag stays OFF. MAX_ATTEMPTS = 200, the canonical
            # `instance_data` assignment and the `str(sympy.Integer(answer))`
            # wrapper were contracts a registered family compiler satisfied by
            # construction; free-form generation does not. Enforcing them here
            # rejected mathematically sound programs on shape alone -- observed
            # on 20/20 candidates for instance_data and on 9 of 14 rejections
            # for the notation pair. Execution-level checks carry the guarantee
            # instead: every verified seed must satisfy its declarative verifier
            # and the deterministic descriptor contract below.
            source_errors.extend(
                lint_mutation_generator_source(
                    program.source_code,
                    reject_descriptor_markers=False,
                    require_assert=False,
                    reject_trivial_assert=False,
                    reject_unbounded_sampling=False,
                    require_answer_routes=False,
                    require_canonical_instance_data=False,
                    require_mechanical_shape=False,
                )
            )
            # The structural contract on the child's own cross-check. This is a
            # dataflow predicate, not one of the shape predicates above, and it
            # lives in its own module for that reason: the flags overhead are
            # all OFF and must stay that way, and a single ON flag among six OFF
            # ones invites the next reader to flip them as a batch.
            #
            # The verdict is always recorded so a shadow run costs no plumbing;
            # only "enforce" turns it into a rejection.
            if mode != "off":
                findings = check_generator_contract(program.source_code)
                program.metadata["ast_contract"] = {
                    "verdict": "flagged" if findings else "clean",
                    "findings": [str(f) for f in findings],
                }
                if findings and mode == "enforce":
                    source_errors.extend(str(f) for f in findings)
            # The statement/code decoupling the AST contract does not cover:
            # `answer` being a sampled name unchanged. 21% of the 4B run's
            # champions are built this way, and the assert cannot catch it
            # because both routes trivially yield the same draw.
            bare = answer_is_bare_draw(program.source_code)
            if bare:
                source_errors.append(bare)
        if source_errors:
            return None, "; ".join(source_errors[:3])

        first: ProblemInstance | None = None
        # statement -> the answers it was rendered with. A set would only
        # answer "does the program vary?"; the mapping additionally proves
        # ill-posedness when it does not vary the way it should.
        seen_problems: dict[str, set[tuple[str, str]]] = {}
        rendered: list[ProblemInstance] = []
        type_annotations = []
        for seed in range(n):
            inst = program.execute(seed=seed)
            if inst is None:
                # Name the failure. "execute failed" was the single largest
                # rejection reason in a run where 58% of candidates died here,
                # and it cannot distinguish the child's own AssertionError --
                # its cross-check working as designed -- from broken code.
                why = program.last_execution_error or "unknown"
                return None, f"execute failed at seed={seed}: {why}"
            if is_generated_mutation:
                # Static RNG rules are necessary but not sufficient: aliases
                # such as ``from random import randint`` and process-global
                # state have both produced different instances for the same
                # seed while satisfying the historical shape lint.  Re-run the
                # exact source/seed before trusting any descriptor or answer.
                repeated = program.execute(seed=seed)
                if repeated is None:
                    why = program.last_execution_error or "unknown"
                    return None, (
                        f"repeat execute failed at seed={seed}: {why}"
                    )
                first_wire = (
                    inst.problem,
                    inst.answer,
                    json.dumps(
                        inst.verifier,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
                repeated_wire = (
                    repeated.problem,
                    repeated.answer,
                    json.dumps(
                        repeated.verifier,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
                if first_wire != repeated_wire:
                    return None, (
                        "generator is nondeterministic for the same seed "
                        f"{seed}: first={first_wire!r} repeat={repeated_wire!r}"
                    )
            instance_errors = lint_problem_instance(inst)
            if instance_errors:
                return None, "; ".join(instance_errors[:3])
            if is_generated_mutation:
                unsafe_reason = structural_inspiration_safety_reason(inst.problem)
                if unsafe_reason:
                    return (
                        None,
                        "rendered problem contains unsafe prompt-control text: "
                        f"{unsafe_reason}",
                    )
            if is_generated_mutation and mode == "enforce":
                # A statement that names its own technique makes the declared
                # SKILL untrue: the reasoning is quoted, not forced. Needs the
                # rendered text, so it cannot live with the source rules.
                handed_over = check_problem_text(inst.problem)
                if handed_over:
                    return None, str(handed_over[0])
            if not answers_match(inst.answer, inst.answer, inst.verifier):
                return None, (
                    "reference answer is not self-consistent with verifier: "
                    f"answer={inst.answer!r} verifier={inst.verifier!r}"
                )
            annotation = annotate_problem_type(inst.problem)
            type_errors = problem_type_contract_errors(
                annotation, inst.verifier, inst.answer
            )
            if type_errors:
                return None, (
                    f"problem-type contract failed at seed={seed}: "
                    + "; ".join(type_errors)
                )
            type_annotations.append(annotation)
            if inst.verifier.get("mode") == "one_of":
                # A generated child must not widen reward by listing the gold
                # beside arbitrary wrong answers. This pipeline has no trusted
                # witness predicate, so every accepted spelling must be
                # mathematically equivalent to the auditable reference. Each
                # comparison uses the ordinary expression grader and inherits
                # its hard-kill worker budget.
                for alternative in inst.verifier.get("answers", []):
                    if not answers_match(
                        str(alternative),
                        inst.answer,
                        {"mode": "expression"},
                    ):
                        return None, (
                            "one_of verifier contains an alternative that is "
                            "not equivalent to the reference answer: "
                            f"{alternative!r} vs {inst.answer!r}"
                        )
            first = first or inst
            rendered.append(inst)
            key = " ".join(inst.problem.split())
            answers = seen_problems.setdefault(key, set())
            verifier_json = json.dumps(
                inst.verifier, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            answers.add((str(inst.answer).strip(), verifier_json))
            if len(answers) > 1:
                # The one ill-posedness result that is a PROOF rather than a
                # heuristic: the same question, word for word, graded against
                # two different answers. Whatever the statement determines, it
                # is not the value being scored.
                #
                # The old check -- at least two distinct statements over n seeds
                # -- cannot see it. Champion 3cedc80f7798 of the 4B run renders
                # three distinct statements across five seeds and so passed,
                # while "Let n = 1 be a positive integer. Determine the number
                # of ways to partition n into distinct prime factors." appeared
                # twice with answers 1 and 2.
                return None, (
                    "the same problem text is graded against two answer/verifier "
                    f"contracts ({sorted(answers)}): the statement does not determine it"
                )

        if n > 1 and len(seen_problems) <= 1:
            return None, "program does not vary its visible problem across seeds"
        inferred_types = {annotation.problem_type for annotation in type_annotations}
        if len(inferred_types) != 1:
            return None, (
                "one problem family must have one PROBLEM_TYPE across all "
                f"verification seeds; found {sorted(str(x) for x in inferred_types)}"
            )
        problem_type = next(iter(inferred_types))
        descriptor_domain = domain if domain is not None else DOMAINS[0]
        label_errors = validate_label_decl(descriptor_domain, problem_type)
        if label_errors:
            return None, "; ".join(label_errors)
        family_contract = None
        if is_generated_mutation:
            canonical_templates = {
                canonical_template(instance.problem) for instance in rendered
            }
            verifier_modes = {
                str(instance.verifier.get("mode", "")) for instance in rendered
            }
            family_contract = {
                "canonical_template_count": len(canonical_templates),
                "verifier_mode_count": len(verifier_modes),
                "verified_seeds": len(rendered),
            }
            if len(canonical_templates) != 1:
                return None, (
                    "generated mutation must implement one problem family: "
                    f"found {len(canonical_templates)} canonical templates "
                    "across verification seeds"
                )
            if len(verifier_modes) != 1:
                return None, (
                    "generated mutation must use one verifier mode across the "
                    f"problem family: found {sorted(verifier_modes)}"
                )
            canonical = next(iter(canonical_templates))
            family_contract.update(
                {
                    "template_sha256": hashlib.sha256(
                        canonical.encode("utf-8")
                    ).hexdigest(),
                    "verifier_mode": next(iter(verifier_modes)),
                }
            )
        if is_generated_mutation and mode == "enforce":
            # Needs every rendered instance, so it cannot live in the per-seed
            # loop above: one coincidental appearance of the answer in the text
            # is common, the same one on all n seeds is the answer being handed
            # to the solver. 25% of the 4B run's champions leak this way.
            leaked = answer_leaks_in_every_instance(rendered)
            if leaked:
                return None, leaked
        # Stamp authoritative metadata only after every admission gate has
        # passed. A rejected object must not retain a contract that would let a
        # later direct archive call bypass the failed family/leak check.
        if domain is not None:
            program.metadata["domain"] = domain
        program.metadata["problem_type"] = problem_type
        for instance in rendered:
            instance.domain = domain
            instance.problem_type = problem_type
        if self.archive.require_domain_labeling:
            if is_generated_mutation:
                domain_authority = "pending_local_policy_binary_label_v1"
                domain_labeling = None
            else:
                domain_authority = MANUAL_DOMAIN_AUTHORITY
                domain_labeling = {
                    "method": "manual_seed_source_literal_v1",
                    "passed": True,
                    "declared": domain,
                    "predicted": domain,
                    "seed_source": dict(manual_seed_source),
                }
        else:
            domain_authority = SOURCE_DOMAIN_AUTHORITY
            domain_labeling = None
        descriptor_contract = {
            "domain_authority": domain_authority,
            "problem_type_authority": "deterministic_statement_and_verifier",
            "problem_type_ruleset": PROBLEM_TYPE_RULESET,
            "problem_type_ruleset_sha256": problem_type_ruleset_sha256(),
            "verified_seeds": len(rendered),
            "domain": domain,
            "problem_type": problem_type,
            "type_evidence": sorted(
                {annotation.evidence for annotation in type_annotations}
            ),
            "source_sha256": hashlib.sha256(
                program.source_code.encode("utf-8")
            ).hexdigest(),
        }
        if domain_labeling is not None:
            descriptor_contract["domain_labeling"] = domain_labeling
        program.metadata["descriptor_contract"] = descriptor_contract
        if family_contract is not None:
            program.metadata["family_contract"] = family_contract
        # Move past verification seeds so the first scored instance is fresh.
        # Doing this after every gate also leaves a rejected candidate with no
        # certified metadata and no consumed scoring cursor.
        if reserve_seed_stream:
            self.seed_stream.reserve_through(program.program_id, n - 1)
        return first, None

    def audit_champions_strict_once(self) -> dict[str, int]:
        """Evict snapshot champions that fail today's deterministic verifier.

        The probe is a deep copy so ``verify_program`` can clear and rebuild
        descriptor metadata without mutating a valid incumbent.  DOMAIN is not
        re-labeled here: this audit checks source, statement, answer/check and
        PROBLEM_TYPE consistency only, preserving the deliberately final-only
        DOMAIN labeling stage.
        """

        if self.strict_champion_audit_completed:
            return dict(self.strict_champion_audit_stats)
        self.strict_champion_audit_completed = True
        champions = list(self.archive.champions())
        checked = evicted = 0
        for champion in champions:
            checked += 1
            probe = ProblemProgram(
                source_code=champion.source_code,
                program_id=champion.program_id,
                parent_id=champion.parent_id,
                generation=champion.generation,
                metadata=copy.deepcopy(champion.metadata),
            )
            _instance, reason = self.verify_program(
                probe,
                n_seeds=self.evolution_config.verify_seeds,
                reserve_seed_stream=False,
            )
            if reason is None:
                continue
            removed_cells = self.archive.remove_program(champion.program_id)
            if not removed_cells:
                continue
            evicted += 1
            self.events.append(
                {
                    "event": "strict_champion_audit_evicted",
                    "iteration": self.current_iteration,
                    "program_id": champion.program_id,
                    "cells": [list(cell) for cell in removed_cells],
                    "reason": reason,
                }
            )
        self.strict_champion_audit_stats = {
            "strict_champion_audit_checked": checked,
            "strict_champion_audit_evicted": evicted,
        }
        return dict(self.strict_champion_audit_stats)

    def run_outer_iteration(self, outer_iteration: int) -> dict:
        self.current_iteration = int(outer_iteration)
        self.archive.begin_selection_iteration(self.current_iteration)
        self._domain_labeling_stats = collections.Counter()
        attempted = 0
        inserted = 0
        reports: list[CandidateReport] = []
        # Rollouts are on-policy for exactly one update; carrying them across an
        # iteration would need an importance-ratio correction this design does
        # not have, so the buffer starts empty every time.
        self.replay.begin_iteration(outer_iteration)

        if self.evolution_config.strict_champion_audit:
            self.audit_champions_strict_once()

        # Push current actor weights into vLLM ONCE for the whole evolve phase.
        # Evolve runs no optimizer step, so weights are static throughout:
        # reevaluate + every inner batch reuse the resident model (no per-session
        # re-push). The backend's begin_session no longer pushes weights when the
        # rollout is resident (sleep mode off).
        self.backend.sync_weights()
        # Ablation: with re-measurement off, a champion keeps the R_Q it was
        # admitted with and is never rescored or evicted. It is the archive's
        # dominant sink -- across three 4B arms the eviction/insertion ratio sat
        # at 0.86-0.94, so 268 insertions over 77 iterations left only +19 net
        # champions and half of them lived 3 iterations or fewer. Turning it off
        # asks whether the MAP is a curriculum that has to track the policy or a
        # conveyor belt paying for itself.
        if self.evolution_config.reevaluate_champions:
            self.reevaluate_champions()

        # Freeze the population whose CURRENT rollouts may train this update
        # before mutation changes the archive.  Selection is based on scores
        # known before those rollouts (PreviousRQScoreboard), the current s_hat is
        # only a zero-advantage gate, and the payload is this iteration's replay.
        # A child inserted below therefore becomes eligible at the next outer
        # iteration instead of displacing a measured incumbent from the batch
        # that is about to update theta_t -> theta_{t+1}.
        training_pool = list(self.archive.champions())

        cfg = self.evolution_config
        batch_size = cfg.inner_iteration_batch_size
        minimum_slots = int(cfg.inner_iterations)
        maximum_slots = (
            int(cfg.mutation_refill_max_iterations)
            if cfg.adaptive_mutation_refill
            else minimum_slots
        )
        sampled_slots = 0
        mutation_batches = 0
        mutation_frontier_candidates = 0
        mutation_frontier_insertions = 0
        mutation_refill_stop_reason = (
            "fixed_budget" if not cfg.adaptive_mutation_refill else "maximum_slots"
        )
        low, high = cfg.frontier_s_hat_range
        while sampled_slots < maximum_slots:
            current_batch = min(batch_size, maximum_slots - sampled_slots)
            mutation_batches += 1
            batch_started = time.monotonic()
            print(
                "[RQ-Evolve] mutation batch begin: "
                f"outer_iteration={self.current_iteration} "
                f"batch={mutation_batches} slots={current_batch} "
                f"sampled_before={sampled_slots} maximum_slots={maximum_slots}",
                flush=True,
            )
            batch_reports = self.inner_iteration_batch(current_batch)
            sampled_slots += current_batch
            reports.extend(batch_reports)
            attempted += sum(1 for r in batch_reports if r.status != "no_parent")
            inserted += sum(1 for r in batch_reports if r.status == "inserted")
            batch_frontier = [
                r
                for r in batch_reports
                if is_frontier(float(r.s_hat), low, high)
            ]
            mutation_frontier_candidates += len(batch_frontier)
            mutation_frontier_insertions += sum(
                1 for r in batch_frontier if r.status == "inserted"
            )
            batch_statuses = collections.Counter(r.status for r in batch_reports)
            print(
                "[RQ-Evolve] mutation batch end: "
                f"outer_iteration={self.current_iteration} "
                f"batch={mutation_batches} duration_s="
                f"{time.monotonic() - batch_started:.3f} "
                f"reports={len(batch_reports)} statuses={dict(batch_statuses)}",
                flush=True,
            )

            # ``inner_iterations`` is a strict minimum, so enabling refill
            # cannot silently make an existing experiment cheaper. Targets are
            # OR conditions: insertion says useful MAP supply was restored;
            # measured frontier candidates say further brute force is unlikely
            # to solve a competition/novelty bottleneck this iteration.
            if not cfg.adaptive_mutation_refill or sampled_slots < minimum_slots:
                continue
            if (
                cfg.mutation_refill_target_frontier_insertions > 0
                and mutation_frontier_insertions
                >= cfg.mutation_refill_target_frontier_insertions
            ):
                mutation_refill_stop_reason = "frontier_insertion_target"
                break
            if (
                cfg.mutation_refill_target_frontier_candidates > 0
                and mutation_frontier_candidates
                >= cfg.mutation_refill_target_frontier_candidates
            ):
                mutation_refill_stop_reason = "frontier_candidate_target"
                break

        self.refresh_dataset(training_champions=training_pool)
        stats = self.archive.stats()
        frontier_in = sum(
            1 for f in self.last_frontier if f["decision"] == "in_frontier"
        )
        # Dispersion is a diagnostic, never a fitness term (rewarding
        # consistency would hand a free maximum to a seed-ignoring generator),
        # but it is the signal that says whether a champion's R_Q is an average
        # over comparable instances or over a trivial/impossible mixture.
        dispersions = [
            float(c.metadata.get("dispersion", 0.0))
            for c in self.archive.champions()
            if "dispersion" in (c.metadata or {})
        ]
        dispersion_metrics = (
            {
                "mean_dispersion": sum(dispersions) / len(dispersions),
                "max_dispersion": max(dispersions),
            }
            if dispersions
            else {}
        )
        # Replay accounting. `replay_degenerate_frac` is the share of the batch
        # that can produce no gradient at all (all-correct or all-wrong groups
        # self-neutralise under the per-instance LOO baseline), so it is the
        # honest measure of how much of the update is wasted.
        replay_metrics = self.replay.stats()
        # Why candidates died, not just how many. accept_rate alone cannot tell
        # source/descriptor failures from rollout or cell competition failures.
        status_counts: dict[str, int] = {}
        for report in reports:
            key = f"status_{report.status}"
            status_counts[key] = status_counts.get(key, 0) + 1

        inspiration_reports = [
            report for report in reports if report.inspiration is not None
        ]
        attached_inspirations = [
            report
            for report in inspiration_reports
            if bool((report.inspiration or {}).get("attached"))
        ]
        omitted_reasons = collections.Counter(
            str((report.inspiration or {}).get("omitted_reason"))
            for report in inspiration_reports
            if not bool((report.inspiration or {}).get("attached"))
        )
        selection_tiers = collections.Counter(
            str((report.inspiration or {}).get("selection_tier"))
            for report in attached_inspirations
        )
        inspiration_metrics = {
            "inspiration_enabled": int(self.evolution_config.structural_inspiration),
            "inspiration_attempted": len(inspiration_reports),
            "inspiration_attached": len(attached_inspirations),
            "inspiration_attached_rate": (
                len(attached_inspirations) / len(inspiration_reports)
                if inspiration_reports
                else 0.0
            ),
            "inspiration_unique_donors": len(
                {
                    str((report.inspiration or {}).get("program_id"))
                    for report in attached_inspirations
                }
            ),
            "inspiration_inserted": sum(
                1 for report in attached_inspirations if report.status == "inserted"
            ),
            **{
                f"inspiration_omitted_{reason}": count
                for reason, count in omitted_reasons.items()
            },
            **{
                f"inspiration_tier_{tier}": count
                for tier, count in selection_tiers.items()
            },
        }
        copy_checked = [
            report
            for report in attached_inspirations
            if bool(((report.inspiration or {}).get("copy_gate") or {}).get("checked"))
        ]
        copy_rejected = [
            report
            for report in copy_checked
            if bool(((report.inspiration or {}).get("copy_gate") or {}).get("rejected"))
        ]
        copy_rejection_reasons = collections.Counter(
            str(((report.inspiration or {}).get("copy_gate") or {}).get("reason"))
            for report in copy_rejected
        )
        inspiration_metrics.update(
            {
                "inspiration_copy_checked": len(copy_checked),
                "inspiration_copy_rejected": len(copy_rejected),
                "inspiration_copy_rejected_rate": (
                    len(copy_rejected) / len(copy_checked) if copy_checked else 0.0
                ),
                "inspiration_cross_descriptor_rate": (
                    selection_tiers.get("cross_lineage_cross_descriptor", 0)
                    / len(attached_inspirations)
                    if attached_inspirations
                    else 0.0
                ),
                **{
                    f"inspiration_copy_rejected_{reason}": count
                    for reason, count in copy_rejection_reasons.items()
                },
            }
        )

        grader_metrics = {
            # How often grading had to be killed. Non-zero means math_verify met
            # an input it cannot finish (`51!!` -> factorial(factorial(51))) and
            # the answer was scored non-match without wedging the run. Silence
            # here used to mean the same thing had leaked a thread instead.
            **{f"grader_{k}": v for k, v in grader_stats().items()},
        }

        # Why inserts were refused, split by cause. The novelty gates and
        # ordinary cell competition used to be one indistinguishable bucket.
        insert_rejects = collections.Counter(
            e.get("reason") for e in self.events if e.get("event") == "insert_rejected"
        )
        insert_metrics = {f"insert_rejected_{k}": v for k, v in insert_rejects.items()}
        insert_metrics["insert_rejected"] = sum(insert_rejects.values())
        preflight_rejects = collections.Counter(
            e.get("reason")
            for e in self.events
            if e.get("event") == "archive_preflight_rejected"
        )
        preflight_metrics = {
            f"archive_preflight_rejected_{k}": v
            for k, v in preflight_rejects.items()
        }
        preflight_metrics["archive_preflight_rejected"] = sum(
            preflight_rejects.values()
        )
        domain_labeling_metrics = dict(
            getattr(self, "_domain_labeling_stats", {}) or {}
        )

        result = {
            "outer_iteration": outer_iteration,
            "attempted": attempted,
            "inserted": inserted,
            "accept_rate": inserted / attempted if attempted else 0.0,
            "dataset_size": len(self.dataset),
            "frontier_in": frontier_in,
            "frontier_out": len(self.last_frontier) - frontier_in,
            "group_size": self.evolution_config.group_size,
            "train_batch_target": self.evolution_config.train_batch_target,
            **dict(self.strict_champion_audit_stats),
            "mutation_refill_enabled": int(cfg.adaptive_mutation_refill),
            "program_verify_workers": int(cfg.program_verify_workers),
            "verify_seeds": int(cfg.verify_seeds),
            "archive_preflight_enabled": int(
                cfg.archive_preflight_before_rollout
            ),
            "mutation_sampled_slots": sampled_slots,
            "mutation_batches": mutation_batches,
            "mutation_frontier_candidates": mutation_frontier_candidates,
            "mutation_frontier_insertions": mutation_frontier_insertions,
            "mutation_refill_stop_reason": mutation_refill_stop_reason,
            **dispersion_metrics,
            **replay_metrics,
            **grader_metrics,
            **insert_metrics,
            **preflight_metrics,
            **domain_labeling_metrics,
            **inspiration_metrics,
            **status_counts,
            **stats,
        }
        self.last_reports = reports
        self.events.append({"event": "outer_iteration_done", **result})
        return result

    def inner_iteration_batch(self, batch_size: int) -> list[CandidateReport]:
        """Run one batch: mutate -> verify -> evaluate -> rollout -> report.

        ``entries`` is a ``list[dict]`` where an entry's *state is which key it
        holds* (there is no status field). Each entry moves through three shapes:

          * ``{"task", "child", "inst"}``   -- verified child, alive for rollouts;
          * ``{"_retry": {...}}``           -- parsed but failed verify; eligible
                                               for one Reflexion self-fix;
          * ``{"report": CandidateReport}`` -- terminal; carries the outcome.

        Flow: mutate() -> deterministic local verification -> split into the
        three shapes -> _resolve_retries -> independent DOMAIN labeling ->
        generate_rollouts on survivors -> each child scored and archived into
        a terminal CandidateReport. See docs/PIPELINE.md
        ("Evolution Candidate State") for the diagram and full status vocabulary.
        """
        parents: list[ProblemProgram] = []
        for _ in range(batch_size):
            search_rng = self._next_search_rng()
            parent = self.archive.sample_parent(rng=search_rng)
            if parent is None:
                return [CandidateReport(status="no_parent", op="none")]
            parents.append(parent)

        # Draw donors only AFTER every primary-parent draw is complete,
        # and with an independent deterministic RNG. Inspiration therefore
        # changes only the prompt context -- it cannot shift the baseline arm's
        # subsequent parent choices through Python's global RNG state.
        inspirations = self._sample_structural_inspirations(parents)

        # One vLLM wake for the whole batch: the mutation generate and every
        # solver generate run while vLLM is awake; entropy (actor forward) is
        # deferred until after end_session (vLLM asleep) inside finalize_rollouts.
        self.backend.begin_session()
        try:
            if not self.evolution_config.two_stage_mutation:
                raise RuntimeError(
                    "single-stage mutation was removed along with its prompt "
                    "templates (mutation_*_prompt.txt); set "
                    "evolution.two_stage_mutation: true"
                )
            tasks, outputs, mutation_strategies = self._mutate_in_two_stages(
                parents, inspirations=inspirations
            )
            entries: list[dict] = []
            # Repeats also happen WITHIN one batch: 32 parents are drawn with
            # replacement from a ~10-cell archive and mutation is near
            # deterministic, so a measured 11 of 32 slots in one iteration were
            # the same few programs. self.rejected_children cannot catch those --
            # it is written when reports are finalized, once the batch is over --
            # so the first occurrence in a batch claims the id and the rest are
            # reported against it. They would have produced the same verdict
            # because the source and deterministic verification seeds match.
            in_flight: set[str] = set()
            built_children = self._make_children_from_outputs(tasks, outputs)
            for task, output, mutation_strategy, built_child in zip(
                tasks, outputs, mutation_strategies, built_children
            ):
                child, inst, reason, source = built_child
                if child is not None:
                    child.metadata["mutation_strategy"] = mutation_strategy
                if child is not None and child.program_id in in_flight:
                    # The source is already known viable, but copy status is a
                    # property of this proposal's (child, assigned donor) pair.
                    # Record it before collapsing the repeated source so every
                    # attached donor assignment remains visible in telemetry.
                    self._check_structural_inspiration_copy(task, child)
                    entries.append(
                        {
                            "report": CandidateReport(
                                status="already_rejected",
                                op=task.op,
                                child_id=child.program_id,
                                reason="duplicate of an earlier candidate in this batch",
                                **_report_context(task),
                            )
                        }
                    )
                elif child is not None and child.program_id in self.rejected_children:
                    entries.append(
                        {
                            "report": CandidateReport(
                                status="already_rejected",
                                op=task.op,
                                child_id=child.program_id,
                                reason=reason,
                                **_report_context(task),
                            )
                        }
                    )
                elif inst is not None:
                    prompt_copy = self._check_prompt_example_copy(task, child)
                    if prompt_copy and prompt_copy["rejected"]:
                        comparison = next(
                            (
                                item
                                for item in prompt_copy["comparisons"]
                                if item.get("rejected")
                            ),
                            {},
                        )
                        copy_reason = (
                            f"example={prompt_copy.get('example_index')} "
                            f"reason={prompt_copy.get('reason')} "
                            f"near={comparison.get('near_template_ratio')} "
                            f"structural={comparison.get('structural_ratio')}"
                        )
                        entries.append(
                            {
                                "report": CandidateReport(
                                    status="prompt_example_copy_rejected",
                                    op=task.op,
                                    child_id=child.program_id,
                                    reason=copy_reason,
                                    source_code=_logged_source(child.source_code),
                                    ast_findings=_ast_findings(child),
                                    copy_audit=prompt_copy,
                                    **_report_context(task),
                                )
                            }
                        )
                        continue
                    copy_verdict = self._check_structural_inspiration_copy(task, child)
                    if copy_verdict and copy_verdict["rejected"]:
                        entries.append(
                            {
                                "report": CandidateReport(
                                    status="inspiration_copy_rejected",
                                    op=task.op,
                                    child_id=child.program_id,
                                    reason=str(copy_verdict["reason"]),
                                    source_code=_logged_source(child.source_code),
                                    ast_findings=_ast_findings(child),
                                    **_report_context(task),
                                )
                            }
                        )
                        continue
                    # A donor-copy verdict is relative to (child, donor), not
                    # to child source alone. Claim the batch-wide source id only
                    # after that pair passes, so the same source can still be
                    # checked against a different assigned donor later in this
                    # batch.
                    in_flight.add(child.program_id)
                    entries.append({"task": task, "child": child, "inst": inst})
                elif source is not None:
                    in_flight.add(child.program_id)
                    # Parses but failed verification -> eligible for one
                    # self-fix. Keep the raw output: it becomes the assistant
                    # turn in the multi-turn fix prompt.
                    # ``source`` and ``child_id`` ride INSIDE the payload on
                    # purpose. _resolve_retries writes e["report"] without
                    # calling e.clear(), so a top-level "child" key would
                    # incorrectly survive into rollout selection.
                    entries.append(
                        {
                            "_retry": {
                                "task": task,
                                "output": output,
                                "reason": reason,
                                "source": source,
                                "child_id": child.program_id if child else None,
                                "ast_findings": (
                                    list(
                                        (child.metadata.get("ast_contract") or {}).get(
                                            "findings", []
                                        )
                                    )
                                    if child
                                    else []
                                ),
                            }
                        }
                    )
                else:
                    compile_status = "mutation_failed" if not output else "no_code"
                    if (
                        not output
                        and ((task.provenance or {}).get("stage1_copy_audit") or {}).get(
                            "rejected"
                        )
                    ):
                        compile_status = "stage1_example_copy_rejected"
                    lowered_reason = str(reason or "").lower()
                    if output and "invalid:" in lowered_reason:
                        compile_status = "stage2_invalid"
                    elif output and task.stage == "generator":
                        compile_status = "stage2_compile_failed"
                    entries.append(
                        {
                            "report": CandidateReport(
                                status=compile_status,
                                op=task.op,
                                reason=reason,
                                # A malformed/abstaining core has no assembled
                                # child source, so retain the bounded raw reply
                                # in the same audit field. Without this, a
                                # compiler rejection leaves only a terse reason
                                # and the exact model failure is unrecoverable.
                                source_code=_logged_source(output),
                                **_report_context(task),
                            )
                        }
                    )

            # One-shot Reflexion self-fix: show the model its rejected program +
            # reason and re-verify. Runs inside the open vLLM session so the extra
            # generate reuses the already-awake rollout worker.
            self._resolve_retries(entries)
            self._apply_domain_labeling(entries)

            # DOMAIN and PROBLEM_TYPE are now authoritative, so every
            # score-independent archive gate can run before solver rollout.
            # Final admission repeats the check because an earlier child from
            # this flat batch may enter the archive in the meantime.
            if self.evolution_config.archive_preflight_before_rollout:
                for entry in entries:
                    if "child" not in entry:
                        continue
                    task, child = entry["task"], entry["child"]
                    passed, archive_decision = (
                        self._archive_preflight_with_telemetry(child)
                    )
                    if passed:
                        continue
                    report = CandidateReport(
                        status="archive_preflight_rejected",
                        op=task.op,
                        child_id=child.program_id,
                        reason=str(archive_decision.get("reason") or "rejected"),
                        source_code=_logged_source(child.source_code),
                        ast_findings=_ast_findings(child),
                        domain_labeling=dict(
                            (child.metadata or {}).get("domain_labeling") or {}
                        ),
                        archive_decision=archive_decision,
                        **_report_context(task),
                    )
                    entry.clear()
                    entry["report"] = report

            to_eval = [e for e in entries if "child" in e]
            # A candidate is graded on fresh seeds beyond those used by the
            # deterministic verifier.
            for e in to_eval:
                e["eval_instances"] = self.draw_instances(e["child"])
            pending = self.backend.generate_rollouts(
                [inst for e in to_eval for inst in e["eval_instances"]],
                n_rollouts=self.evolution_config.group_size,
            )
        finally:
            self.backend.end_session()

        grouped = self.backend.finalize_rollouts(pending)
        rollouts_by_child: dict[int, list] = {}
        cursor = 0
        for e in to_eval:
            take = len(e["eval_instances"])
            rollouts_by_child[id(e["child"])] = list(
                zip(e["eval_instances"], grouped[cursor : cursor + take])
            )
            cursor += take

        reports: list[CandidateReport] = []
        for entry in entries:
            if "report" in entry:
                reports.append(entry["report"])
                continue
            task, child, inst = entry["task"], entry["child"], entry["inst"]
            scored = rollouts_by_child.get(id(child), [])
            stats = []
            for eval_inst, rollouts in scored:
                stat = self._seed_stat(child, eval_inst, rollouts)
                if stat is not None:
                    stats.append(stat)
            if not stats:
                # Every seed lost its rollouts (timeout / worker error / stale).
                # Report the dominant reason instead of letting it masquerade as
                # an ordinary s_hat_zero.
                reasons: dict[str, int] = {}
                for _, rollouts in scored:
                    for r in rollouts:
                        key = r.reject_reason or "unknown"
                        reasons[key] = reasons.get(key, 0) + 1
                reports.append(
                    CandidateReport(
                        status="rollout_failed",
                        op=task.op,
                        child_id=child.program_id,
                        reason=(
                            max(reasons, key=reasons.get) if reasons else "no seeds"
                        ),
                        source_code=_logged_source(child.source_code),
                        ast_findings=_ast_findings(child),
                        **_report_context(task),
                    )
                )
                continue
            result = self._store_result(
                child,
                compute_rq_program(
                    stats,
                    fitness_mode=self.evolution_config.rq_fitness_mode,
                    reverse_u_constant=(
                        self.evolution_config.rq_reverse_u_constant
                    ),
                ),
            )
            # An R_Q of 0 no longer ends the candidate's run. It still cannot
            # take a cell from a scoring champion (competition is a strict `>`)
            # and it still contributes no training examples (the frontier band
            # in dataset.py drops p=0 and p=1), but it can hold an otherwise
            # empty cell and be drawn as a mutation parent from there.
            inserted, archive_decision = self._try_insert_with_telemetry(
                program=child,
                u_value=result.u_score,
                rq_score=result.rq_score,
            )
            if inserted:
                # The candidate was measured under theta_t.  Recording that
                # score now makes it a PAST score at t+1, so a new child waits
                # exactly one update before it can train (not two).
                self.previous_rq.record(
                    child.program_id, self.current_iteration, result.rq_score
                )
            if inserted:
                status = "inserted"
            elif result.s_hat <= 0.0:
                status = "s_hat_zero"
            elif result.rq_score <= 0.0:
                status = "rq_zero"
            else:
                status = "rejected_non_elite"
            reports.append(
                CandidateReport(
                    status=status,
                    op=task.op,
                    child_id=child.program_id,
                    rq_score=result.rq_score,
                    s_hat=result.s_hat,
                    u_score=result.u_score,
                    source_code=_logged_source(child.source_code),
                    ast_findings=_ast_findings(child),
                    domain_labeling=dict(
                        (child.metadata or {}).get("domain_labeling") or {}
                    ),
                    archive_decision=archive_decision,
                    **_report_context(task),
                )
            )
        for report in reports:
            self.archive.record_parent_outcome(
                report.parent_id,
                report.status == "inserted",
                iteration=self.current_iteration,
            )
        self.archive.finalize_parent_outcomes()
        self._memoize_rejections(reports)
        return reports

    def _next_search_rng(self) -> random.Random:
        """One restart-stable RNG for a reproductive proposal."""
        material = (
            f"{self.evolution_config.search_seed}:{self.search_draw_count}"
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        self.search_draw_count += 1
        return random.Random(seed)

    def _check_structural_inspiration_copy(
        self, task: MutationTask, child: ProblemProgram
    ) -> dict | None:
        donor = getattr(task, "inspiration_donor", None)
        if donor is None:
            return None
        copy_kwargs = {}
        max_jaccard = self.evolution_config.structural_inspiration_max_token_jaccard
        if max_jaccard is not None:
            copy_kwargs["max_token_jaccard"] = max_jaccard
        verdict = self.archive.compare_with_structural_inspiration(
            child, donor, **copy_kwargs
        )
        _record_inspiration_copy_gate(task, child, verdict)
        return verdict

    def _check_prompt_example_copy(
        self, task: MutationTask, child: ProblemProgram
    ) -> dict | None:
        """Reject a child copied from a concrete example visible to Stage 2."""

        examples = tuple(getattr(task, "copy_exclusion_examples", ()) or ())
        if not examples:
            return None
        verdicts = []
        for example in examples:
            verdict = self.archive.compare_with_structural_inspiration(
                child,
                example,
                # The accepted interface example is intentionally concise.
                # The archive-wide floor protects arbitrary short champions;
                # here the comparison target is a known, exact prompt source.
                near_duplicate_min_chars=40,
            )
            verdict["example_index"] = int(
                (example.metadata or {}).get("prompt_copy_exclusion_index", 0)
            )
            verdict["family_sha256"] = str(
                (example.metadata or {}).get("family_sha256", "")
            )
            # A bare AST-shape collision is too broad for a short prompt
            # example: unrelated product/counting families can share the same
            # anonymized skeleton. Require some textual overlap when structure
            # is the sole signal; exact behavior/template gates remain strict.
            if verdict.get("reason") == "structural_duplicate" and not (
                float(verdict.get("near_template_ratio") or 0.0) >= 0.55
                or float(verdict.get("token_jaccard") or 0.0) >= 0.35
            ):
                verdict["raw_structural_rejected"] = True
                verdict["rejected"] = False
                verdict["reason"] = None
            verdicts.append(verdict)
        rejected = next(
            (verdict for verdict in verdicts if verdict.get("rejected")), None
        )
        audit = {
            "checked": True,
            "rejected": rejected is not None,
            "reason": (rejected or {}).get("reason"),
            "example_index": (rejected or {}).get("example_index"),
            "comparisons": verdicts,
        }
        child.metadata["prompt_example_copy_gate"] = audit
        return audit

    def _domain_labeling_token_ids(self) -> tuple[int | None, int | None]:
        """Exact single-token ids for YES/NO, or no restriction if unavailable."""

        cached = getattr(self, "_domain_labeling_ids", None)
        if cached is not None:
            return cached
        ids: tuple[int | None, int | None] = (None, None)
        tokenizer = getattr(self.backend, "tokenizer", None)
        if tokenizer is not None:
            try:
                yes = tokenizer.encode("YES", add_special_tokens=False)
                no = tokenizer.encode("NO", add_special_tokens=False)
                if len(yes) == len(no) == 1 and int(yes[0]) != int(no[0]):
                    ids = (int(yes[0]), int(no[0]))
            except Exception:
                ids = (None, None)
        self._domain_labeling_ids = ids
        return ids

    def _apply_domain_labeling(self, entries: list[dict]) -> None:
        """Assign generated DOMAIN with the local seven-arm binary labeler.

        Stage 2 supplies no DOMAIN.  Each fixed family is paired with every
        Omni top-level domain, and the current local policy emits a token from
        the restricted vocabulary {YES, NO}.  The one-vs-rest argmax is the
        authoritative label—not a check of a generator-proposed label—only
        when both confidence thresholds pass.
        """

        if not self.evolution_config.independent_domain_labeling:
            return
        live = [entry for entry in entries if "child" in entry and "inst" in entry]
        if not live:
            return

        yes_id, no_id = self._domain_labeling_token_ids()
        token_contract_error = (
            None
            if yes_id is not None and no_id is not None
            else "tokenizer does not encode YES and NO as two distinct tokens"
        )
        allowed = [yes_id, no_id] if token_contract_error is None else None
        tasks: list[MutationTask] = []
        ownership: list[tuple[int, str]] = []
        families: list[str] = []
        for index, entry in enumerate(live):
            family = str(
                ((entry["task"].provenance or {}).get("family_plan") or {}).get(
                    "CHILD FAMILY", ""
                )
            ).strip()
            families.append(family)
            if family and token_contract_error is None:
                for domain in DOMAINS:
                    tasks.append(
                        build_domain_labeling_task(
                            parent=entry["task"].parent,
                            child_family=family,
                            domain=domain,
                            allowed_token_ids=allowed,
                        )
                    )
                    ownership.append((index, domain))

        if tasks:
            self.backend.mutate(tasks)
        pairs = getattr(self.backend, "last_mutation_logprobs", None) or []
        by_child: dict[int, dict[str, float]] = collections.defaultdict(dict)
        for offset, (child_index, domain) in enumerate(ownership):
            probability = _binary_p_yes(
                pairs[offset] if offset < len(pairs) else None,
                yes_id,
                no_id,
            )
            if probability is not None:
                by_child[child_index][domain] = float(probability)

        stats = getattr(self, "_domain_labeling_stats", None)
        if stats is None:
            stats = collections.Counter()
            self._domain_labeling_stats = stats
        ruleset_hash = domain_labeling_ruleset_sha256()
        for index, entry in enumerate(live):
            child = entry["child"]
            task = entry["task"]
            probabilities = by_child.get(index, {})
            ranked = sorted(
                probabilities.items(), key=lambda item: item[1], reverse=True
            )
            predicted = ranked[0][0] if ranked else None
            top_probability = ranked[0][1] if ranked else 0.0
            margin = (
                _probability_logit(ranked[0][1])
                - _probability_logit(ranked[1][1])
                if len(ranked) >= 2
                else 0.0
            )
            failure = None
            failure_kind = None
            if not families[index]:
                failure_kind = "missing_family"
                failure = "DOMAIN labeler has no fixed CHILD FAMILY"
            elif token_contract_error is not None:
                failure_kind = "tokenizer_contract"
                failure = token_contract_error
            elif set(probabilities) != set(DOMAINS):
                failure_kind = "incomplete"
                missing = sorted(set(DOMAINS) - set(probabilities))
                failure = "DOMAIN labeler returned no score for " + ", ".join(
                    missing
                )
            elif top_probability < float(
                self.evolution_config.domain_labeling_min_probability
            ):
                failure_kind = "low_probability"
                failure = (
                    f"top domain probability {top_probability:.4f} is below "
                    f"{self.evolution_config.domain_labeling_min_probability:.4f}"
                )
            elif margin < float(
                self.evolution_config.domain_labeling_min_logit_margin
            ):
                failure_kind = "low_margin"
                failure = (
                    f"top-vs-runner logit margin {margin:.4f} is below "
                    f"{self.evolution_config.domain_labeling_min_logit_margin:.4f}"
                )
            family_sha = str(
                ((child.metadata or {}).get("generator_contract") or {}).get(
                    "family_sha256", ""
                )
            )
            labeling = {
                "method": DOMAIN_LABELING_METHOD,
                "passed": failure is None,
                "predicted": predicted,
                "probabilities": {
                    domain: float(probabilities[domain])
                    for domain in DOMAINS
                    if domain in probabilities
                },
                "top_probability": float(top_probability),
                "logit_margin": float(margin),
                "min_probability": float(
                    self.evolution_config.domain_labeling_min_probability
                ),
                "min_logit_margin": float(
                    self.evolution_config.domain_labeling_min_logit_margin
                ),
                "ruleset_sha256": ruleset_hash,
                "family_sha256": family_sha,
                "policy_iteration": int(self.current_iteration),
                "failure_kind": failure_kind,
            }
            child.metadata["domain_labeling"] = labeling
            stats["domain_labeling_attempted"] += 1
            if failure is not None:
                stats["domain_labeling_rejected"] += 1
                stats[f"domain_labeling_rejected_{failure_kind}"] += 1
                report = CandidateReport(
                    status="domain_labeling_failed",
                    op=task.op,
                    child_id=child.program_id,
                    reason=failure,
                    source_code=_logged_source(child.source_code),
                    ast_findings=_ast_findings(child),
                    domain_labeling=labeling,
                    **_report_context(task),
                )
                entry.clear()
                entry["report"] = report
                continue

            child.metadata["domain"] = predicted
            descriptor = dict(child.metadata.get("descriptor_contract") or {})
            descriptor["domain_authority"] = GENERATED_DOMAIN_AUTHORITY
            descriptor["domain"] = predicted
            descriptor["domain_labeling"] = labeling
            child.metadata["descriptor_contract"] = descriptor
            entry["inst"].domain = predicted
            stats["domain_labeling_passed"] += 1

    # Only rejections that are a deterministic property of the source and its
    # verification seeds, and so cannot come out differently later.
    #
    # Deliberately excluded: s_hat_zero, rq_zero, rejected_non_elite and
    # rollout_failed. Those are verdicts about the CURRENT policy and the
    # CURRENT occupant of the cell -- a program nobody can solve today is
    # exactly the program a stronger policy should get another look at, and one
    # that lost its cell should be reconsidered when the incumbent changes.
    # Memoizing them would quietly make the archive monotone in a way the
    # design does not intend.
    _DETERMINISTIC_REJECTIONS = frozenset(
        {
            "verify_failed",
            "no_code",
            "mutation_failed",
        }
    )

    def _memoize_rejections(self, reports: list[CandidateReport]) -> None:
        for report in reports:
            if report.child_id and report.status in self._DETERMINISTIC_REJECTIONS:
                self.rejected_children.setdefault(
                    report.child_id, f"{report.status}: {report.reason}"
                )

    def _sample_structural_inspirations(
        self, parents: list[ProblemProgram]
    ) -> list[StructuralInspirationSelection | None]:
        """One cross-lineage donor per parent, or ``None`` when disabled.

        The SHA-derived local seeds are stable across Python processes (unlike
        ``hash()``) and the draw counter is checkpointed. Each proposal gets a
        distinct draw even when the same parent appears repeatedly in a batch.
        """
        cfg = self.evolution_config
        if not cfg.structural_inspiration:
            return [None] * len(parents)

        selections: list[StructuralInspirationSelection] = []
        for parent in parents:
            material = (
                f"{cfg.structural_inspiration_seed}:"
                f"{self.inspiration_draw_count}:{parent.program_id}"
            ).encode("utf-8")
            local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            self.inspiration_draw_count += 1
            selections.append(
                self.archive.sample_structural_inspiration(
                    parent,
                    rng=random.Random(local_seed),
                    max_template_chars=cfg.structural_inspiration_max_chars,
                    selection_strategy=cfg.structural_inspiration_selection,
                    require_certified=(
                        cfg.structural_inspiration_require_certified_donor
                    ),
                    require_positive_rq=(
                        cfg.structural_inspiration_require_positive_rq
                    ),
                )
            )
        return selections

    def _mutate_in_two_stages(self, parents, targets=None, inspirations=None):
        """Problem first, program second: two calls instead of one.

        The two stages are PIPELINED per parent when the backend supports it
        (``mutate_pipelined``): a parent's stage-2 generator call goes out the
        moment its own stage-1 plan lands, rather than after the slowest plan
        in the batch. Both stages are also output-capped
        (``evolution.family_max_output_tokens`` / ``generator_max_output_tokens``)
        -- uncapped, each stage ran to data.max_response_length and a 32-parent
        batch paid its worst sample twice per iteration.

        Returns ``(tasks, outputs)`` in the shape the single-call path returns,
        so everything downstream -- retries, scoring, and reporting -- is
        untouched. A parent whose stage-1 reply does not parse yields an empty
        output, which ``_make_child_from_output`` already reports as a failed
        mutation.

        Stage 1 receives no descriptor names, definitions, parent labels, or a
        desired cell. Stage 2 receives the descriptor-free parent program and
        parent family only as transformation context, plus the immutable child
        family implementation contract. It receives neither the DOMAIN
        vocabulary nor a destination and cannot change the child family. The
        later label-blind readback assigns DOMAIN. PROBLEM_TYPE is derived by
        verification code. ``targets`` is retained only to fail loudly for
        stale callers.
        """
        cfg = self.evolution_config
        rotate = cfg.rotate_few_shots
        if targets not in (None, []):
            raise ValueError(
                "target-directed mutation is retired; targets must be omitted"
            )
        inspirations = inspirations or [None] * len(parents)
        if len(inspirations) != len(parents):
            raise ValueError(
                "one structural-inspiration selection is required per parent"
            )
        prompt_rngs: list[tuple[random.Random, random.Random]] = []
        for parent in parents:
            draw = self.mutation_prompt_draw_count
            self.mutation_prompt_draw_count += 1

            def _prompt_rng(stage: str) -> random.Random:
                material = (
                    f"{cfg.mutation_prompt_seed}:{draw}:" f"{parent.program_id}:{stage}"
                ).encode("utf-8")
                seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
                return random.Random(seed)

            prompt_rngs.append((_prompt_rng("family"), _prompt_rng("generator")))
        family_tasks = [
            build_family_task(
                parent,
                temperature=cfg.code_temperature,
                top_p=cfg.code_top_p,
                max_output_tokens=cfg.family_max_output_tokens,
                rotate_shots=rotate,
                rng=prompt_rngs[index][0],
                inspiration_template=(selection.template if selection else None),
                inspiration_donor=(selection.donor if selection else None),
                provenance=(
                    {"structural_inspiration": dict(selection.provenance)}
                    if selection is not None
                    else None
                ),
            )
            for index, (parent, selection) in enumerate(
                zip(parents, inspirations)
            )
        ]

        def _stage_two(index: int, family_reply: str | None):
            """Stage-1 reply -> (plan, stage-2 task), or (None, None)."""
            plan = parse_family_plan(family_reply)
            if plan is None:
                return None, None
            family_task = family_tasks[index]
            family_task.provenance["child_family"] = plan["CHILD FAMILY"]
            family_task.provenance["prospective_problem_type"] = (
                annotate_problem_type(plan["CHILD FAMILY"]).problem_type
            )
            stage1_copy = _stage1_prompt_copy_audit(
                family_task, plan["CHILD FAMILY"]
            )
            family_task.provenance["stage1_copy_audit"] = stage1_copy
            if stage1_copy["rejected"]:
                family_task.provenance["stage1_rejection_reason"] = (
                    "Stage-1 CHILD FAMILY copies a visible worked example: "
                    f"example={stage1_copy.get('visible_index')}"
                )
                return None, None
            provenance = dict(family_tasks[index].provenance)
            provenance["family_plan"] = {
                key: plan[key]
                for key in (
                    "STRUCTURAL MUTATION",
                    "CHILD FAMILY",
                    "WHY FINITE",
                )
                if key in plan
            }
            inspiration = provenance.get("structural_inspiration")
            if inspiration is not None:
                inspiration = dict(inspiration)
                inspiration["claimed_transfer"] = plan.get("STRUCTURAL MUTATION", "")
                provenance["structural_inspiration"] = inspiration
            return plan, build_generator_task(
                parents[index],
                plan,
                temperature=cfg.generator_temperature,
                top_p=cfg.generator_top_p,
                max_output_tokens=cfg.generator_max_output_tokens,
                rotate_shots=rotate,
                rng=prompt_rngs[index][1],
                provenance=provenance,
                inspiration_donor=family_tasks[index].inspiration_donor,
                emit_legacy_domain=(
                    not cfg.independent_domain_labeling
                ),
            )

        plans: list[dict | None] = [None] * len(parents)
        gen_task_by_i: dict[int, object] = {}
        gen_reply_by_i: dict[int, str | None] = {}

        # Preferred path: no barrier between the stages. A parent's stage-2 call
        # is submitted the moment ITS stage-1 reply lands, so the batch never
        # waits for the slowest plan before any generator starts. The blocking
        # two-call path below is the fallback for backends without it (tests,
        # dry runs) and is otherwise identical.
        pipelined = (
            family_tasks
            and hasattr(self.backend, "mutate_pipelined")
            and getattr(self.backend, "supports_pipelined_mutation", lambda: False)()
        )
        if pipelined:
            stage_two_tasks: dict[int, object] = {}

            def _build_stage_two(index: int, family_reply: str | None):
                plan, task = _stage_two(index, family_reply)
                plans[index] = plan
                if task is not None:
                    stage_two_tasks[index] = task
                return task

            _family_replies, gen_reply_by_i = self.backend.mutate_pipelined(
                family_tasks, _build_stage_two
            )
            gen_task_by_i = stage_two_tasks
        elif family_tasks:
            family_replies = self.backend.mutate(family_tasks)
            gen_tasks = []
            order = []
            for index, reply in enumerate(family_replies):
                plan, task = _stage_two(index, reply)
                plans[index] = plan
                if task is not None:
                    order.append(index)
                    gen_tasks.append(task)
            gen_task_by_i = dict(zip(order, gen_tasks))
            gen_replies = self.backend.mutate(gen_tasks) if gen_tasks else []
            gen_reply_by_i = dict(zip(order, gen_replies))

        self.stage_one_parsed = sum(1 for plan in plans if plan is not None)

        tasks = list(family_tasks)
        outputs: list[str | None] = [None] * len(parents)
        mutation_strategies = ["untargeted"] * len(parents)
        for i, plan in enumerate(plans):
            if plan is None:
                continue
            # Preserve stage-2 provenance even when its reply is empty or
            # malformed. The report must still identify the family plan and
            # assigned donor, and a nonempty malformed reply is ``no_code`` --
            # not an indistinguishable "empty model output".
            tasks[i] = gen_task_by_i[i]
            # Keep the raw reply intact.  Parsing and trusted assembly happen
            # exactly once in ``_make_child_from_output`` where both backend
            # paths converge.  The old path extracted here and then extracted
            # the extracted source a second time downstream, which made it
            # impossible to attach a single fail-closed Stage-2 contract.
            outputs[i] = gen_reply_by_i.get(i)

        return tasks, outputs, mutation_strategies

    def _build_child_from_output(
        self,
        task: MutationTask,
        output: str | None,
    ):
        """Extract and assemble one child without executing its verifier.

        Returns ``(child, reason, source)``. A non-None ``source`` means Stage 2
        produced an assembled program and any later failure belongs to dynamic
        verification rather than parsing/compilation.
        """
        if not output:
            rejection = str(
                (task.provenance or {}).get("stage1_rejection_reason", "")
            ).strip()
            return None, rejection or "empty model output", None
        family_plan = dict((task.provenance or {}).get("family_plan") or {})
        if task.stage == "generator":
            family = str(family_plan.get("CHILD FAMILY", "")).strip()
            if not family:
                return (
                    None,
                    "Stage-2 task has no fixed CHILD FAMILY to assemble",
                    None,
                )
            source, compile_error = compile_stage2_reply(
                output,
                family,
                require_domain=(
                    not self.evolution_config.independent_domain_labeling
                ),
            )
            if source is None:
                return None, compile_error or "Stage-2 compile failed", None
        else:
            # Compatibility for direct/internal callers that construct a
            # non-Stage-2 MutationTask.  The live two-stage pipeline never
            # reaches this branch.
            source = extract_generator_code(output)
            if source is None:
                return None, "no parseable generate() in output", None

        metadata = _child_metadata(task)
        if task.stage == "generator":
            metadata["generator_contract"] = {
                "version": TRUSTED_ASSEMBLER_VERSION,
                "family_sha256": hashlib.sha256(
                    family.encode("utf-8")
                ).hexdigest(),
                "stage2_reply_sha256": hashlib.sha256(
                    output.encode("utf-8")
                ).hexdigest(),
            }
        child = ProblemProgram(
            source_code=source,
            parent_id=task.parent.program_id,
            generation=task.parent.generation + 1,
            metadata=metadata,
        )
        return child, None, source

    def _make_child_from_output(
        self,
        task: MutationTask,
        output: str | None,
        in_flight: set[str] | None = None,
        *,
        reserve_seed_stream: bool = True,
    ):
        """Extract -> build -> verify a child from one model output.

        Returns ``(child, inst, reason, source)``. On success ``inst`` is the
        verified instance; on failure ``inst`` is None and ``source`` is the
        parsed program (None if the output had no parseable ``generate``).
        """

        child, build_reason, source = self._build_child_from_output(task, output)
        if child is None:
            return None, None, build_reason, source
        # Before execution, not after: a repeat costs 5 sandbox runs and the
        # verdict is already known. `source=None` in the return
        # also keeps it out of the self-fix retry, which would re-derive the
        # same program from the same reason.
        if in_flight is not None and child.program_id in in_flight:
            # Same batch, same source. Checked here rather than in the caller so
            # the repeat skips the 5-seed execution.
            return child, None, "duplicate of an earlier candidate in this batch", None
        memo = self.rejected_children.get(child.program_id)
        if memo is not None:
            return child, None, memo, None
        inst, reason = self.verify_program(
            child, reserve_seed_stream=reserve_seed_stream
        )
        return child, inst, reason, source

    def _make_children_from_outputs(
        self,
        tasks: list[MutationTask],
        outputs: list[str | None],
    ) -> list[
        tuple[
            ProblemProgram | None,
            ProblemInstance | None,
            str | None,
            str | None,
        ]
    ]:
        """Build in order, then verify independent candidate sources in parallel.

        Compilation and batch dedup remain deterministic in proposal order.
        Only the expensive dynamic ``verify_program`` calls fan out. Successful
        seed-stream reservations are replayed on the caller thread afterwards,
        avoiding concurrent mutation of the persisted cursor.
        """

        if len(tasks) != len(outputs):
            raise ValueError("tasks and outputs must have the same length")
        results: list[
            tuple[
                ProblemProgram | None,
                ProblemInstance | None,
                str | None,
                str | None,
            ]
            | None
        ] = [None] * len(tasks)
        # Source equality alone is not enough here. Copy safety is evaluated
        # relative to each task's inspiration donor later in proposal order;
        # the same source may fail against one donor but pass against another.
        # Deduplicate only when both the source and donor context are equal.
        unique_proposals: set[tuple[str, str | None]] = set()
        pending: list[tuple[int, ProblemProgram, str]] = []

        for index, (task, output) in enumerate(zip(tasks, outputs)):
            child, build_reason, source = self._build_child_from_output(
                task, output
            )
            if child is None:
                results[index] = (None, None, build_reason, source)
                continue
            donor = task.inspiration_donor
            donor_id = donor.program_id if donor is not None else None
            proposal_key = (child.program_id, donor_id)
            if proposal_key in unique_proposals:
                results[index] = (
                    child,
                    None,
                    "duplicate of an earlier candidate in this batch",
                    None,
                )
                continue
            unique_proposals.add(proposal_key)
            memo = self.rejected_children.get(child.program_id)
            if memo is not None:
                results[index] = (child, None, memo, None)
                continue
            pending.append((index, child, source))

        def _verify(item: tuple[int, ProblemProgram, str]):
            index, child, source = item
            inst, reason = self.verify_program(
                child, reserve_seed_stream=False
            )
            return index, child, inst, reason, source

        workers = min(
            max(1, int(self.evolution_config.program_verify_workers)),
            max(1, len(pending)),
        )
        if workers == 1:
            verified = map(_verify, pending)
            for index, child, inst, reason, source in verified:
                results[index] = (child, inst, reason, source)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="rq-program-verify",
            ) as executor:
                # map preserves proposal order even when workers finish out of
                # order, keeping logs and seed reservations reproducible.
                for index, child, inst, reason, source in executor.map(
                    _verify, pending
                ):
                    results[index] = (child, inst, reason, source)

        for result in results:
            if result is None:
                raise RuntimeError("candidate verification produced no result")
            child, inst, _reason, _source = result
            if child is not None and inst is not None:
                self.seed_stream.reserve_through(
                    child.program_id, self.evolution_config.verify_seeds - 1
                )
        return [result for result in results if result is not None]

    def _resolve_retries(self, entries: list[dict]) -> None:
        """Collapse every ``_retry`` entry into its verify_failed report.

        The self-fix round went with the single-stage prompts: ``build_fix_task``
        replayed the original mutation conversation, and that conversation no
        longer exists in the two-stage pipeline. ``fix_retry`` therefore cannot
        be honoured -- refusing loudly beats silently skipping the fix the
        config asked for.
        """
        targets = [e for e in entries if "_retry" in e]
        if not targets:
            return
        if self.evolution_config.fix_retry:
            raise RuntimeError(
                "fix_retry was removed along with the single-stage mutation "
                "prompts; set evolution.fix_retry: false"
            )
        for e in targets:
            info = e.pop("_retry")
            task = info["task"]
            e["report"] = CandidateReport(
                status="verify_failed",
                op=task.op,
                child_id=info.get("child_id"),
                reason=info["reason"],
                source_code=_logged_source(info.get("source")),
                ast_findings=list(info.get("ast_findings") or []),
                **_report_context(task),
            )

    def _score_from_rollouts(
        self,
        program: ProblemProgram,
        rollouts: list[RolloutRecord],
        *,
        instance: ProblemInstance | None = None,
    ) -> RQResult:
        """One instance's rollouts -> a one-seed R_Q. Kept for single-instance
        callers; the program-level path is :meth:`evaluate_program`."""
        stat = self._seed_stat(
            program, instance, rollouts, seed=int(getattr(instance, "seed", 0) or 0)
        )
        return self._store_result(
            program,
            compute_rq_program(
                [stat] if stat else [],
                fitness_mode=self.evolution_config.rq_fitness_mode,
                reverse_u_constant=self.evolution_config.rq_reverse_u_constant,
            ),
        )

    def _seed_stat(
        self,
        program: ProblemProgram,
        instance: ProblemInstance | None,
        rollouts: list[RolloutRecord],
    ) -> "SeedStat | None":
        """Collapse one instance's m rollouts into its (s, L, U) triple.

        Rejected rollouts (timeout / stale / worker error) leave the estimate
        entirely -- they were never drawn from the policy, so counting them as
        failures would understate s. A seed whose rollouts were ALL rejected
        returns None and is dropped from the seed set rather than scored as 0:
        one transient infra failure must not look like an unsolvable instance.
        """
        accepted = [
            r for r in rollouts if getattr(r, "status", "accepted") == "accepted"
        ]
        if not accepted:
            return None
        flags = [
            (
                clean_and_grade_solver_rollout(record, instance)[2]
                if instance is not None
                else bool(record.correct)
            )
            for record in accepted
        ]
        return score_seed(
            seed=int(getattr(instance, "seed", 0) or 0),
            correct_flags=flags,
            rollout_entropies=[r.entropy for r in accepted],
        )

    @staticmethod
    def _store_result(program: ProblemProgram, result: RQResult) -> RQResult:
        program.s_hat = result.s_hat
        program.u_score = result.u_score
        program.rq_score = result.rq_score
        program.metadata["dispersion"] = result.dispersion
        program.metadata["num_seeds"] = result.num_seeds
        program.metadata["rq_fitness_mode"] = result.fitness_mode
        program.metadata["rq_reverse_u_constant"] = result.reverse_u_constant
        return result

    def draw_instances(
        self,
        program: ProblemProgram,
        n_seeds: int | None = None,
    ) -> list[ProblemInstance]:
        """Execute ``program`` on n FRESH seeds from its never-reused stream.

        Seeds it cannot execute are skipped, not retried at a lower seed: the
        stream only moves forward, so a program that fails on some seeds is
        simply graded on fewer instances rather than on a re-drawn set.
        """
        n = int(n_seeds or 1)
        instances: list[ProblemInstance] = []
        for seed in self.seed_stream.take(program.program_id, n):
            inst = program.execute(seed=seed)
            if inst is not None:
                instances.append(inst)
        return instances

    def evaluate_programs(
        self,
        programs: list[ProblemProgram],
        *,
        store_replay: bool = False,
        instance_counts: list[int] | None = None,
    ) -> list[RQResult | None]:
        """Score each program on ONE fresh instance x G rollouts, one session.

        ``instance_counts`` may ask for more than one instance per program. The
        extras exist only to fill the training batch when the frontier is
        smaller than it (see :meth:`_allocate_instances`); **R_Q is always the
        first instance alone**, so the fitness of a champion that filled three
        slots means the same thing as one that filled one. All the instances go
        to replay, because every rollout that was paid for should produce a
        gradient.

        Every program's instances go into ONE flat rollout batch and are
        regrouped afterwards, so the evaluation costs the same number of
        wake/sleep cycles regardless of how the instances are distributed.

        With ``store_replay`` the rollouts are kept in :attr:`replay` instead of
        being discarded, and become the solver's training batch. That is only
        correct for the re-scoring pass: those rollouts come from the same
        theta_t that the following update starts from, so training on them is
        on-policy. Candidate scoring passes ``False`` -- a child inserted this
        iteration has no previous score yet and does not train until the next one.
        """
        if not programs:
            return []
        counts = instance_counts or [1] * len(programs)
        per_program: list[list[ProblemInstance]] = [
            self.draw_instances(p, n_seeds=k) for p, k in zip(programs, counts)
        ]
        flat = [inst for group in per_program for inst in group]
        if not flat:
            return [None] * len(programs)

        self.backend.begin_session()
        try:
            pending = self.backend.generate_rollouts(
                flat, n_rollouts=self.evolution_config.group_size
            )
        finally:
            self.backend.end_session()
        grouped = self.backend.finalize_rollouts(pending)
        payloads = getattr(pending, "payloads", None) or []

        results: list[RQResult | None] = []
        cursor = 0
        for program, instances in zip(programs, per_program):
            stats = []
            for instance in instances:
                rollouts = grouped[cursor] if cursor < len(grouped) else []
                cursor += 1
                stat = self._seed_stat(program, instance, rollouts)
                if stat is not None:
                    stats.append(stat)
                    if store_replay:
                        self.replay.store(
                            program.program_id,
                            instance,
                            rollouts,
                            payload=(
                                payloads[cursor - 1]
                                if cursor - 1 < len(payloads)
                                else None
                            ),
                        )
            if not stats:
                # Every seed lost its rollouts. Scoring this as 0 would zero the
                # program's s_hat/u_score in place and let one transient infra
                # failure evict a champion from its true niche.
                self.events.append(
                    {
                        "event": "eval_rollout_failed",
                        "program_id": program.program_id,
                        "seeds_drawn": len(instances),
                    }
                )
                results.append(None)
                continue
            # The FIRST instance is the measurement; any others were drawn to
            # fill the batch. Scoring on all of them would make R_Q depend on
            # how many slots a champion happened to be given.
            results.append(
                self._store_result(
                    program,
                    compute_rq_program(
                        stats[:1],
                        fitness_mode=self.evolution_config.rq_fitness_mode,
                        reverse_u_constant=(
                            self.evolution_config.rq_reverse_u_constant
                        ),
                    ),
                )
            )
        return results

    def _allocate_instances(self, champions: list[ProblemProgram]) -> list[int]:
        """One fresh instance each, then extras by the previous raw R_Q.

        The trainer needs ``train_batch_target`` prompts and the frontier is
        routinely smaller than that (median 18 against a target of 16-32 over
        the 8B run), so the shortfall is covered by drawing further FRESH seeds
        from the highest-R_Q champions rather than by shrinking the batch.
        Extra seeds are extra problems, not extra rollouts on the same problem
        -- LILO found scaling the number of levels more effective than scaling
        rollouts per level.

        Ranked by the same previous raw R_Q used to build the training batch.
        ``program.rq_score`` is deliberately not the ranking key: it is only
        overwritten by the current re-score before the batch is built, whereas
        ``PreviousRQScoreboard`` preserves exactly the t-1 value without an EWMA.

        THE SHORTFALL IS COUNTED OVER FRONTIER CHAMPIONS, NOT ALL OF THEM, and
        that distinction is the whole point of this function. Only a champion
        with 0 < s_hat < 1 contributes a training row -- ``is_frontier`` in
        dataset.py drains the rest -- so comparing the target against
        ``len(champions)`` asks the wrong question. Measured on the 4B run: the
        archive reached 48 champions while only 18-22 were on the frontier, so
        ``target(32) <= n(48)`` held, every champion got exactly one instance,
        and the training set came out at 19 rows against a 32-prompt batch.
        VerlDynamicDataset is built with ``min_size=train_batch_size``
        (verl_adapter.py), so the missing 13 rows were filled by wrapping onto
        rows already in the batch -- the same instance, and under replay the
        same stored rollouts, counted twice in the update. Over iterations 100+
        that was 41% of every batch, peaking at 75% when the frontier fell to 8.
        Allocating against the frontier count closes it: extras go to frontier
        champions until the batch is full of distinct instances.
        """
        n = len(champions)
        counts = [1] * n
        target = int(self.evolution_config.train_batch_target)
        if n == 0:
            return counts
        low, high = self.evolution_config.frontier_s_hat_range
        frontier = [
            i
            for i in range(n)
            if is_frontier(float(getattr(champions[i], "s_hat", 0.0) or 0.0), low, high)
        ]
        # Nothing on the frontier yet (bootstrap, or every champion degenerate):
        # fall back to the old whole-population behaviour rather than refusing
        # to allocate, so the batch is still filled with something.
        pool = frontier or list(range(n))
        if target <= len(pool):
            return counts
        past_scores = {
            i: self.previous_rq.selection_score(
                champions[i].program_id, self.current_iteration
            )
            for i in pool
        }
        # Bootstrap / a pre-scoreboard checkpoint has no past-only value at
        # all.  Only in that all-missing state may the already-stored score seed
        # the first batch.  Once any history exists, a missing program is a new
        # child and must not receive priority before its next iteration.
        if all(score is None for score in past_scores.values()):
            rank_score = {
                i: float(getattr(champions[i], "rq_score", 0.0) or 0.0) for i in pool
            }
        else:
            rank_score = {
                i: (float(score) if score is not None else float("-inf"))
                for i, score in past_scores.items()
            }
        order = sorted(pool, key=lambda i: rank_score[i], reverse=True)
        for j in range(target - len(pool)):
            counts[order[j % len(order)]] += 1
        return counts

    def reevaluate_champions(self) -> None:
        """Refresh champion scores under the current backend.

        One vLLM wake/sleep for the whole champion set (was one per champion).
        """
        champions = list(self.archive.champions())
        if not champions:
            return
        # Degenerate champions (s_hat in {0, 1}) are held in the archive but not
        # necessarily re-measured. They cannot enter the training batch -- the
        # frontier band drains them in dataset.py -- so under
        # replay_training_batch their rollouts are generated and discarded.
        # See EvolutionConfig.reevaluate_degenerate_every for the budget this
        # recovers and, importantly, for what skipping them costs at n=1.
        every = int(self.evolution_config.reevaluate_degenerate_every)
        if every != 1:
            low, high = self.evolution_config.frontier_s_hat_range
            due = every > 0 and (self.current_iteration % every == 0)
            if not due:
                skipped = [
                    c
                    for c in champions
                    if not is_frontier(
                        float(getattr(c, "s_hat", 0.0) or 0.0), low, high
                    )
                ]
                if skipped:
                    champions = [c for c in champions if c not in skipped]
                    self.events.append(
                        {
                            "event": "degenerate_reevaluation_skipped",
                            "iteration": self.current_iteration,
                            "skipped": len(skipped),
                            "rescored": len(champions),
                        }
                    )
        if not champions:
            return
        # Fresh seeds every re-scoring. A program that degenerates on only an
        # eps-fraction of its seed space is caught with probability
        # 1-(1-eps)^n per re-evaluation, so tail-overfitted generators leave the
        # archive within a few iterations instead of surviving on a lucky seed.
        results = self.evaluate_programs(
            champions,
            store_replay=True,
            instance_counts=self._allocate_instances(champions),
        )

        for champion, result in zip(champions, results):
            if result is None:
                # all rollouts rejected (transient timeout/worker error) --
                # keep the champion's previous scores and niche untouched.
                continue
            # Every champion is rescored against the same weights before any of
            # them moves, so re-binning cannot depend on the order of this loop.
            # Recorded BEFORE the archive moves, so the score is attributed to
            # the iteration that measured it regardless of what happens to the
            # champion's cell afterwards.
            self.previous_rq.record(
                champion.program_id, self.current_iteration, result.rq_score
            )
            removed_cells = self.archive.remove_program(champion.program_id)
            # A champion whose rescore comes back at R_Q = 0 is reinserted, not
            # dropped. Removing it emptied the cell and left the grid with no
            # record that the region had ever been reached; keeping it means the
            # cell reports what is actually the best generator found for it.
            inserted, _archive_decision = self._try_insert_with_telemetry(
                program=champion,
                u_value=result.u_score,
                rq_score=result.rq_score,
                source="champion_reevaluation",
            )
            if not inserted:
                rejection_reason = (champion.metadata or {}).get("archive_status")
                if rejection_reason == "similarity_timeout_rejected":
                    restored_cells = (
                        self.archive.restore_program_after_similarity_timeout(
                            champion,
                            removed_cells,
                            rq_score=result.rq_score,
                        )
                    )
                    self.events.append(
                        {
                            "event": "champion_preserved_after_similarity_timeout",
                            "program_id": champion.program_id,
                            "cells": [list(cell) for cell in restored_cells],
                            "s_hat": result.s_hat,
                            "rq_score": result.rq_score,
                            "timeout": (champion.metadata or {}).get(
                                "similarity_timeout"
                            ),
                        }
                    )
                    continue
                self.events.append(
                    {
                        "event": "champion_removed_after_reevaluation",
                        "program_id": champion.program_id,
                        "reason": "lost_target_bin_competition",
                        "s_hat": result.s_hat,
                        "rq_score": result.rq_score,
                    }
                )

    def _archive_preflight_with_telemetry(
        self, program: ProblemProgram
    ) -> tuple[bool, dict]:
        """Reject existing archive duplicates before solver rollouts are paid."""

        placement_cell = self.archive.placement_cell(program)
        incumbent = (
            self.archive.grid[placement_cell].champion
            if placement_cell is not None
            else None
        )
        passed = self.archive.passes_admission_preflight(program)
        reason = None if passed else (program.metadata or {}).get(
            "archive_status", "archive_preflight_rejected"
        )
        decision = {
            "phase": "preflight",
            "accepted": bool(passed),
            "reason": reason,
            "placement_cell": (
                list(placement_cell) if placement_cell is not None else None
            ),
            "placement_labels": (
                list(self.archive.cell_labels(placement_cell))
                if placement_cell is not None
                else None
            ),
            "incumbent_program_id": (
                incumbent.program_id if incumbent is not None else None
            ),
            "incumbent_rq": (
                float(incumbent.rq_score) if incumbent is not None else None
            ),
        }
        if not passed:
            self.events.append(
                {
                    "event": "archive_preflight_rejected",
                    "program_id": program.program_id,
                    **decision,
                }
            )
        return passed, decision

    def _try_insert_with_telemetry(
        self,
        *,
        program: ProblemProgram,
        u_value: float,
        rq_score: float,
        source: str = "mutation",
    ) -> tuple[bool, dict]:
        placement_cell = self.archive.placement_cell(program)
        incumbent = (
            self.archive.grid[placement_cell].champion
            if placement_cell is not None
            else None
        )
        admission_started = time.monotonic()
        print(
            "[RQ-Evolve] archive admission begin: "
            f"outer_iteration={self.current_iteration} source={source} "
            f"program_id={program.program_id} placement_cell={placement_cell}",
            flush=True,
        )
        try:
            inserted = self.archive.try_insert(
                program=program,
                u_value=u_value,
                rq_score=rq_score,
            )
        except BaseException as exc:
            print(
                "[RQ-Evolve] archive admission error: "
                f"outer_iteration={self.current_iteration} source={source} "
                f"program_id={program.program_id} duration_s="
                f"{time.monotonic() - admission_started:.3f} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        rejection_reason = None if inserted else (program.metadata or {}).get(
            "archive_status", "lost_cell_contest"
        )
        print(
            "[RQ-Evolve] archive admission end: "
            f"outer_iteration={self.current_iteration} source={source} "
            f"program_id={program.program_id} duration_s="
            f"{time.monotonic() - admission_started:.3f} "
            f"accepted={bool(inserted)} reason={rejection_reason}",
            flush=True,
        )
        decision = {
            "phase": "final_admission",
            "accepted": bool(inserted),
            "reason": rejection_reason,
            "placement_cell": (
                list(placement_cell) if placement_cell is not None else None
            ),
            "placement_labels": (
                list(self.archive.cell_labels(placement_cell))
                if placement_cell is not None
                else None
            ),
            "incumbent_program_id": (
                incumbent.program_id if incumbent is not None else None
            ),
            "incumbent_rq": (
                float(incumbent.rq_score) if incumbent is not None else None
            ),
            "incoming_rq": float(rq_score),
        }
        if not inserted:
            # WHY it was refused. try_insert distinguishes six rejections --
            # unlabelled, seed_variation, duplicate_behavior, duplicate_template,
            # near_duplicate_template, structural_duplicate -- and losing a cell
            # to a higher-R_Q incumbent is a seventh. All seven were reported as
            # one bare "rejected_non_elite" with no reason, so the novelty gates
            # could not be told apart from ordinary cell competition: 176 such
            # rejections in the 65-iteration run, none attributable.
            #
            # This is the most expensive rejection in the pipeline. try_insert
            # takes rq_score, so the child has already paid for verification,
            # classification and its full rollout budget by the time it gets here.
            self.events.append(
                {
                    "event": "insert_rejected",
                    "source": source,
                    "reason": rejection_reason,
                    "program_id": program.program_id,
                    "rq_score": float(rq_score),
                    "placement_cell": (
                        list(placement_cell) if placement_cell is not None else None
                    ),
                }
            )
        if (
            inserted
            and incumbent is not None
            and incumbent.program_id != program.program_id
        ):
            self.events.append(
                {
                    "event": "champion_replaced",
                    "source": source,
                    "placement_cell": list(placement_cell),
                    "placement_labels": list(
                        self.archive.cell_labels(placement_cell)
                    ),
                    "incoming_program_id": program.program_id,
                    "incoming_rq": float(rq_score),
                    "evicted_program_id": incumbent.program_id,
                    "evicted_rq": float(incumbent.rq_score),
                }
            )
        return inserted, decision

    def refresh_dataset(
        self,
        *,
        warmup: bool = False,
        training_champions: list[ProblemProgram] | None = None,
    ) -> None:
        # ``training_champions`` is captured immediately after incumbent
        # re-scoring and before mutation.  Keeping it separate from the live
        # archive makes the temporal contract explicit: children created in
        # iteration t may train from t+1, while every row used now has a replay
        # group sampled under the current theta_t.
        champions = (
            list(training_champions)
            if training_champions is not None
            else list(self.archive.champions())
        )
        # Record each champion's frontier decision (the learnability filter that
        # decides which problems feed training) with the SAME predicate
        # build_training_examples uses -- observability only, no behavior change.
        low, high = self.evolution_config.frontier_s_hat_range
        self.last_frontier = [
            {
                "program_id": champion.program_id,
                "s_hat": round(float(champion.s_hat), 4),
                "rq_score": round(float(champion.rq_score), 6),
                "decision": (
                    "in_frontier"
                    if is_frontier(champion.s_hat, low, high)
                    else "s_hat_out_of_range"
                ),
            }
            for champion in champions
        ]
        if self.training_config.replay_training_batch:
            replay_budget = (
                self.training_config.training_budget
                if self.training_config.training_budget is not None
                else self.evolution_config.train_batch_target
            )
            examples = build_replay_training_examples(
                champions,
                replay=self.replay,
                previous_rq=self.previous_rq,
                iteration=self.current_iteration,
                frontier_s_hat_range=self.evolution_config.frontier_s_hat_range,
                training_budget=replay_budget,
                warmup=warmup,
            )
            if not examples and not warmup:
                # Nothing has a prior measurement to be selected on. That is the
                # same situation as bootstrap -- a cold resume, or an archive
                # that turned over completely -- and the honest response is to
                # fall back to the current scores rather than hand the trainer
                # an empty dataloader. The lag re-engages the moment any
                # champion carries history again.
                if champions and all(
                    self.previous_rq.selection_score(
                        c.program_id, self.current_iteration
                    )
                    is None
                    for c in champions
                ):
                    self.events.append(
                        {
                            "event": "replay_warmup_fallback",
                            "iteration": self.current_iteration,
                            "champions": len(champions),
                        }
                    )
                    examples = build_replay_training_examples(
                        champions,
                        replay=self.replay,
                        previous_rq=self.previous_rq,
                        iteration=self.current_iteration,
                        frontier_s_hat_range=(
                            self.evolution_config.frontier_s_hat_range
                        ),
                        training_budget=replay_budget,
                        warmup=True,
                    )
            if not examples and not warmup and champions:
                # The frontier admitted nobody: every champion currently reads
                # degenerate (s_hat exactly 0 or 1). Handing the trainer an
                # empty dataloader ends the run -- verl raises IndexError out of
                # VerlDynamicDataset and there is no recovery path. A run died
                # this way at iteration 65 with 29 of 30 champions degenerate.
                #
                # Training on them produces no gradient (RLOO advantages are
                # identically 0 when a group scores alike), so this costs one
                # step and changes nothing. It keeps the loop alive so the next
                # re-evaluation can pull champions back onto the band, which is
                # what actually recovers: 20.3% of degenerate readings are
                # followed by a live one at the very next measurement.
                self.events.append(
                    {
                        "event": "frontier_empty_fallback",
                        "iteration": self.current_iteration,
                        "champions": len(champions),
                    }
                )
                examples = build_replay_training_examples(
                    champions,
                    replay=self.replay,
                    previous_rq=self.previous_rq,
                    iteration=self.current_iteration,
                    frontier_s_hat_range=self.evolution_config.frontier_s_hat_range,
                    training_budget=replay_budget,
                    warmup=True,
                    allow_degenerate=True,
                )
            # The dataset is modulo-padded up to train_batch_size. Under replay
            # that repeats IDENTICAL responses rather than resampling them, so a
            # short batch double-counts the same rollouts. Harmless but worth
            # seeing: it shrinks as the archive fills.
            budget = replay_budget
            if examples and budget and len(examples) < budget:
                self.events.append(
                    {
                        "event": "replay_batch_short",
                        "iteration": self.current_iteration,
                        "examples": len(examples),
                        "requested": budget,
                    }
                )
            self.dataset.update(examples)
            self.dataset_refresh_count += 1
            return
        examples = build_training_examples(
            champions,
            instances_per_program=self.training_config.instances_per_program,
            training_budget=self.training_config.training_budget,
            frontier_s_hat_range=self.evolution_config.frontier_s_hat_range,
            used_seeds=self.used_seeds,
            strict_anti_reuse=self.training_config.strict_anti_reuse,
            select_lowest_rq_first=self.training_config.select_lowest_rq_first,
            select_random_order=self.training_config.select_random_order,
            # Vary the shuffle per refresh so the random ablation isn't frozen to
            # one ordering across outer iterations, while staying reproducible.
            select_random_seed=(
                self.training_config.select_random_seed + self.dataset_refresh_count
            ),
            select_ignores_uncertainty=self.evolution_config.select_ignores_uncertainty,
            select_ignores_variance=self.evolution_config.select_ignores_variance,
        )
        self.dataset.update(examples)
        self.dataset_refresh_count += 1

    _USED_SEEDS_FILE = "rq_used_seeds.json"
    _ITERATION_FILE = "rq_iteration.json"

    def save_state(self, directory: str | Path, iteration: int | None = None) -> None:
        """Persist the MAP-Elites archive + used_seeds for restart.

        The verl weight checkpoint does NOT include the archive, so without this
        a resumed run restarts from a seed-only grid and loses every evolved
        champion. Called once per outer iteration (after evolution) so the
        latest archive is always on disk.

        ``archive.json`` is the latest snapshot (overwritten each call, used for
        resume). When ``iteration`` is given a versioned copy
        ``archive_iter{iteration}.json`` is also written so the per-step evolution
        trajectory is recoverable.
        """
        import shutil

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.archive.save(directory)
        if iteration is not None:
            shutil.copyfile(
                directory / "archive.json",
                directory / f"archive_iter{int(iteration)}.json",
            )
        used = {pid: sorted(seeds) for pid, seeds in self.used_seeds.items()}
        (directory / self._USED_SEEDS_FILE).write_text(
            json.dumps(
                {
                    "strict_anti_reuse": self.training_config.strict_anti_reuse,
                    "instances_per_program": self.training_config.instances_per_program,
                    "used_seeds": used,
                    "seed_cursor": self.seed_stream.to_dict(),
                    "previous_rq_scores": self.previous_rq.to_dict(),
                    "rejected_children": self.rejected_children,
                    "inspiration_draw_count": self.inspiration_draw_count,
                    "mutation_prompt_draw_count": self.mutation_prompt_draw_count,
                    "search_draw_count": self.search_draw_count,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (directory / self._ITERATION_FILE).write_text(
            json.dumps(
                {"current_iteration": self.current_iteration},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def append_evolution_log(
        self,
        directory: str | Path,
        iteration: int,
        metrics: dict,
        reports: list[CandidateReport] | None = None,
    ) -> None:
        """Append one JSON line per outer iteration to ``evolution_log.jsonl``.

        Unlike the archive (latest snapshot only), this is append-only, so the
        full evolution trajectory is preserved: per-iteration metrics plus every
        candidate report. ``CandidateReport.status`` is one of: no_parent,
        mutation_failed, no_code, verify_failed, inspiration_copy_rejected, rollout_failed,
        s_hat_zero, rq_zero, inserted, rejected_non_elite (each with op, rq_score, s_hat,
        u_score; see docs/PIPELINE.md "Evolution Candidate State").

        Each report also carries ``source_code`` (truncated at
        ``_MAX_LOGGED_SOURCE_CHARS``) and ``ast_findings``. Only champions land
        in the archive snapshots, so without those two a rejected child's
        program is unrecoverable and a gate's false-positive rate cannot be
        measured against the population it rejects. ``reports`` defaults to
        ``self.last_reports``.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        reports = self.last_reports if reports is None else reports
        record = {
            "iteration": int(iteration),
            "rq_fitness_mode": self.evolution_config.rq_fitness_mode,
            "rq_reverse_u_constant": (
                self.evolution_config.rq_reverse_u_constant
            ),
            "metrics": metrics,
            "reports": [asdict(r) for r in reports],
            "frontier": self.last_frontier,
        }
        with (directory / "evolution_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_state(self, directory: str | Path) -> bool:
        """Restore archive + used_seeds written by :meth:`save_state`.

        Returns True if a snapshot was found and loaded (the caller should then
        skip seed bootstrapping), False otherwise. The dataset is refreshed from
        the restored archive so training can resume immediately.
        """
        directory = Path(directory)
        if not (directory / "archive.json").exists():
            return False
        n_champions = self.archive.load(directory)
        if n_champions == 0:
            # An incompatible legacy snapshot can leave no canonical
            # DOMAIN x PROBLEM_TYPE champions after ``archive.load``. Reporting
            # that as a successful resume made the
            # caller skip seed bootstrapping, and the trainer then died on its
            # first batch with "VerlDynamicDataset is empty". Nothing to resume
            # is the same as no snapshot. Returning before ``used_seeds`` is
            # restored matters too: those seeds belong to the run that produced
            # the unusable archive, and carrying them over would retire seeds
            # the bootstrap is about to need.
            return False
        missing_prompt_draw_count = True
        missing_search_draw_count = True
        seeds_file = directory / self._USED_SEEDS_FILE
        if seeds_file.exists():
            payload = json.loads(seeds_file.read_text(encoding="utf-8"))
            self.used_seeds = {
                pid: set(seeds) for pid, seeds in payload.get("used_seeds", {}).items()
            }
            self.previous_rq = PreviousRQScoreboard.from_dict(
                # Pre-migration archives used the key ``lagged_scores``.
                # Accept it for offline inspection; changed training semantics
                # still require a fresh run rather than checkpoint resume.
                payload.get("previous_rq_scores")
                or payload.get("lagged_scores"),
            )
            self.rejected_children = dict(payload.get("rejected_children") or {})
            self.inspiration_draw_count = int(
                payload.get("inspiration_draw_count", self.inspiration_draw_count)
            )
            if "mutation_prompt_draw_count" in payload:
                self.mutation_prompt_draw_count = int(
                    payload["mutation_prompt_draw_count"]
                )
                missing_prompt_draw_count = False
            if "search_draw_count" in payload:
                self.search_draw_count = int(payload["search_draw_count"])
                missing_search_draw_count = False
            cursor = payload.get("seed_cursor")
            if cursor:
                self.seed_stream = SeedStream.from_dict(cursor)
            else:
                # A snapshot from before the stream existed records WHICH seeds
                # were emitted, not how far the stream ran. Resume one past the
                # largest so nothing is issued twice.
                self.seed_stream = SeedStream.from_used_seeds(self.used_seeds)
        iteration_file = directory / self._ITERATION_FILE
        if iteration_file.exists():
            payload = json.loads(iteration_file.read_text(encoding="utf-8"))
            self.current_iteration = int(
                payload.get("current_iteration", self.current_iteration)
            )
        if missing_prompt_draw_count:
            # Every successful primary-parent draw increments total_selections
            # exactly once, so it is the exact legacy proposal counter.
            self.mutation_prompt_draw_count = self.archive.total_selections
        if missing_search_draw_count:
            self.search_draw_count = self.archive.total_selections
        self.refresh_dataset()
        self.events.append({"event": "archive_restored", "champions": n_champions})
        return True
