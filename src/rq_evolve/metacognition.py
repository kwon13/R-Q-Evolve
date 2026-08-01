"""Metacognitive evidence, delayed progress, and operator-control helpers.

The MAP-Elites archive remains the only problem archive.  This module stores
only telemetry derived from rollouts that R_Q already paid for:

* a compact correct/wrong reasoning contrast for later mutation planning;
* pre/post pass-rate progress measured before re-binning the live MAP;
* an optional EMA controller for depth/breadth selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from .concepts import CONCEPT_GROUPS, concept_group_for_type
from .program import ProblemInstance, ProblemProgram
from .scoring import RQResult

if TYPE_CHECKING:
    from .backends import RolloutRecord


@dataclass(slots=True)
class ReasoningEvidence:
    source: str
    role: str
    problem: str
    response: str
    predicted_answer: str | None
    correct: bool
    entropy: float | None
    origin_program_id: str | None
    origin_concept_group: str
    origin_concept_type: str
    seed: int
    policy_version: int
    iteration: int
    response_tokens: int
    evidence_quality_version: str


@dataclass(slots=True)
class ProgressSummary:
    count: int
    pre_mean_p: float
    post_mean_p: float
    delta_p: float


@dataclass(slots=True)
class MetaProgress:
    iteration: int
    global_progress: ProgressSummary
    by_concept_group: dict[str, ProgressSummary]
    by_concept_type: dict[str, ProgressSummary]
    by_operator: dict[str, ProgressSummary]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PLAN_FIELDS_V4: tuple[str, ...] = (
    "schema_version",
    "operator",
    "failure_summary",
    "target_reasoning_move",
    "correct_wrong_contrast",
    "target_concept_group",
    "target_concept_type",
    "why_target_reasoning_move_is_necessary",
    "preserve",
    "change",
    "forbidden_changes",
    "parameters",
    "guards",
    "max_sampling_attempts",
    "answer_route",
    "problem_output_contract",
    "answer_output_contract",
    "predicted_pre_behavior",
    "predicted_post_behavior",
    "heldout_evaluation",
)

PLAN_FIELDS_V5: tuple[str, ...] = (
    *PLAN_FIELDS_V4,
    "generator_family",
    "family_config",
)

# Allowed but not required. ``family_variant`` names the registry-owned
# construction variant the plan selects; when it is absent the compiler falls
# back to the family's default variant and records that it was not plan-selected,
# so a planner that ignores the choice is visible in the manifest instead of
# failing the whole plan.
OPTIONAL_PLAN_FIELDS_V5: tuple[str, ...] = ("family_variant",)

# Compatibility alias for callers that only need the historical common fields.
REQUIRED_PLAN_FIELDS = PLAN_FIELDS_V4

REMOVED_PLAN_FIELDS: tuple[str, ...] = (
    "insight_route",
    "brute_route",
    "equivalence_assertion",
    "route_independence_reason",
    "decoy_route",
    "decoy_assertion",
)

EVIDENCE_QUALITY_VERSION = "clean_live_parent_v1"


def truncate_text_tokens(
    text: str,
    max_tokens: int,
    *,
    tokenizer=None,
    head_ratio: float = 0.6,
) -> str:
    """Keep the beginning and conclusion of a reasoning trace within a budget."""
    text = str(text or "")
    if max_tokens <= 0:
        return ""
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= max_tokens:
                return text
            marker = "\n...[trace truncated]...\n"
            marker_tokens = len(
                tokenizer.encode(marker, add_special_tokens=False)
            )
            keep_budget = max(0, max_tokens - marker_tokens)
            if keep_budget <= 1:
                return tokenizer.decode(
                    ids[:max_tokens],
                    skip_special_tokens=True,
                )
            head = max(1, int(keep_budget * head_ratio))
            tail = max(0, keep_budget - head)
            kept = ids[:head] + (ids[-tail:] if tail else [])
            return (
                tokenizer.decode(kept[:head], skip_special_tokens=True)
                + marker
                + (
                    tokenizer.decode(kept[head:], skip_special_tokens=True)
                    if tail
                    else ""
                )
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            tokenizer = None

    # Framework-free fallback for tests and non-verl backends.
    words = text.split()
    if len(words) <= max_tokens:
        return text
    marker_words = ["...[trace", "truncated]..."]
    keep_budget = max(0, max_tokens - len(marker_words))
    if keep_budget <= 1:
        return " ".join(words[:max_tokens])
    head = max(1, int(keep_budget * head_ratio))
    tail = max(0, keep_budget - head)
    return " ".join(
        words[:head] + marker_words + (words[-tail:] if tail else [])
    )


_TRACE_BOUNDARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\n\s*(?:user|assistant)\s*:", re.IGNORECASE),
    re.compile(
        r"\n\s*please\s+reason\s+step\s+by\s+step\b",
        re.IGNORECASE,
    ),
    re.compile(r"<\|im_start\|>\s*(?:user|assistant)\b", re.IGNORECASE),
)

# Standalone vLLM can stop before emitting these common second-chat markers.
# ``sanitize_solver_trace`` remains the backend-independent fallback for
# completions produced by training/rollout stacks that cannot accept stop
# strings per request.
SOLVER_CHAT_BOUNDARY_STOPS: tuple[str, ...] = (
    "\nPlease reason step by step",
    "\nUser:",
    "\nuser:",
    "\nAssistant: user",
    "\nassistant: user",
    "<|im_start|>user",
)


def sanitize_solver_trace(response: str) -> str:
    """Drop chat-template spillover or a second conversation from a rollout.

    A rollout used as metacognitive evidence must describe exactly one supplied
    problem. Some base-model generations continue after their first answer by
    emitting a fresh User:/Assistant: exchange. Without this boundary check, a
    correct first solution can be mislabeled as a confident failure about an
    unrelated second problem.
    """
    text = str(response or "").strip()
    if not text:
        return ""
    boundaries = [
        match.start()
        for pattern in _TRACE_BOUNDARY_PATTERNS
        for match in pattern.finditer(text)
        if match.start() > 0
    ]
    if boundaries:
        text = text[: min(boundaries)].rstrip()
    return text


def _trace_token_count(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return len(str(text).split())


def _evidence_provenance_issues(
    item: dict[str, Any],
    *,
    program: ProblemProgram | None = None,
) -> list[str]:
    """Require typed provenance for a trace used by the live parent."""

    issues: list[str] = []
    for name in (
        "source",
        "problem",
        "origin_program_id",
        "origin_concept_group",
        "origin_concept_type",
        "evidence_quality_version",
    ):
        if not isinstance(item.get(name), str) or not item[name].strip():
            issues.append(f"missing_or_invalid_provenance:{name}")
    for name in ("seed", "policy_version", "iteration", "response_tokens"):
        value = item.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            issues.append(f"missing_or_invalid_provenance:{name}")
    entropy = item.get("entropy")
    if isinstance(entropy, bool) or not isinstance(entropy, (int, float)):
        issues.append("missing_or_invalid_provenance:entropy")
    if item.get("source") != "observed":
        issues.append("provenance_source_must_be_observed")
    if item.get("evidence_quality_version") != EVIDENCE_QUALITY_VERSION:
        issues.append("unsupported_evidence_quality_version")
    if isinstance(item.get("response_tokens"), int) and item["response_tokens"] <= 0:
        issues.append("invalid_provenance:response_tokens")
    if program is not None:
        if item.get("origin_program_id") != program.program_id:
            issues.append("provenance_origin_program_mismatch")
        if item.get("origin_concept_group") != program.get_concept_group():
            issues.append("provenance_origin_concept_group_mismatch")
        if item.get("origin_concept_type") != program.get_concept_type():
            issues.append("provenance_origin_concept_type_mismatch")
    return issues


def reasoning_trace_quality_issues(
    response: str,
    predicted_answer: str | None,
    *,
    response_tokens: int = 0,
    max_tokens: int | None = None,
) -> list[str]:
    """Return deterministic reasons a rollout is unsafe as planning evidence.

    Metacognitive planning needs a mathematical success/failure contrast, not a
    formatting miss, a max-token cutoff, or a low-entropy decoding loop.  This
    deliberately conservative gate affects evidence eligibility only; every
    rollout still contributes to the ordinary Solver score.
    """

    text = sanitize_solver_trace(response)
    issues: list[str] = []
    if not text:
        return ["empty_trace"]
    if predicted_answer is None or not str(predicted_answer).strip():
        issues.append("missing_parsed_answer")
    if "...[trace truncated]..." in text:
        issues.append("stored_trace_truncated")
    if (
        max_tokens is not None
        and max_tokens > 0
        and int(response_tokens or 0) >= int(max_tokens)
    ):
        issues.append("reached_trace_token_limit")

    normalized_lines = [
        re.sub(r"\s+", " ", line.strip().lower())
        for line in text.splitlines()
        if line.strip()
    ]
    if len(normalized_lines) >= 8:
        most_common_count = Counter(normalized_lines).most_common(1)[0][1]
        if (
            most_common_count >= 4
            and most_common_count / len(normalized_lines) >= 0.30
        ):
            issues.append("repetitive_no_progress_loop")
    return issues


def validate_reasoning_contrast(
    evidence: Iterable[dict[str, Any]],
) -> list[str]:
    """Validate one same-instance, same-policy success/failure evidence pair."""

    items = [dict(item) for item in evidence if isinstance(item, dict)]
    issues: list[str] = []
    if len(items) != 2:
        return [f"contrast_requires_exactly_two_traces: found={len(items)}"]
    roles = {str(item.get("role", "")).lower() for item in items}
    if roles != {"success", "failure"}:
        issues.append("contrast_requires_one_success_and_one_failure")
    for index, item in enumerate(items):
        issues.extend(
            f"trace_{index}:{reason}"
            for reason in _evidence_provenance_issues(item)
        )
        role = str(item.get("role", "")).lower()
        correct = item.get("correct")
        if not isinstance(correct, bool):
            issues.append(f"trace_{index}:missing_or_invalid_correct")
        elif (role == "success") != correct:
            issues.append(f"trace_{index}:role_correct_mismatch")
        trace_issues = reasoning_trace_quality_issues(
            str(item.get("response", "")),
            item.get("predicted_answer"),
            response_tokens=(
                item["response_tokens"]
                if isinstance(item.get("response_tokens"), int)
                and not isinstance(item.get("response_tokens"), bool)
                else 0
            ),
        )
        issues.extend(f"trace_{index}:{reason}" for reason in trace_issues)

    def comparable_values(name: str) -> set[str]:
        values = {
            " ".join(str(item.get(name, "")).split())
            for item in items
            if item.get(name) is not None and str(item.get(name, "")).strip()
        }
        return values

    if len(comparable_values("problem")) != 1:
        issues.append("contrast_problem_mismatch")
    if len(comparable_values("seed")) != 1:
        issues.append("contrast_seed_mismatch")
    if len(comparable_values("origin_program_id")) != 1:
        issues.append("contrast_origin_program_mismatch")
    if len(comparable_values("origin_concept_group")) != 1:
        issues.append("contrast_origin_concept_group_mismatch")
    if len(comparable_values("origin_concept_type")) != 1:
        issues.append("contrast_origin_concept_type_mismatch")
    if len(comparable_values("policy_version")) != 1:
        issues.append("contrast_policy_version_mismatch")
    if len(comparable_values("iteration")) != 1:
        issues.append("contrast_iteration_mismatch")
    return issues


def clean_and_grade_solver_rollout(
    record: RolloutRecord,
    instance: ProblemInstance,
) -> tuple[str, str | None, bool]:
    """Return one-chat response text and a grade derived from that text.

    The raw rollout stays untouched for diagnostics. When a model spills into a
    second conversation, scoring and metacognitive evidence must nevertheless
    agree on the first supplied problem instead of grading the last boxed answer
    from an unrelated suffix.
    """
    original = str(record.response or "").strip()
    cleaned = sanitize_solver_trace(original)
    predicted = record.predicted_answer
    is_correct = bool(record.correct)
    if cleaned != original:
        from .reward import answers_match, extract_boxed

        predicted = extract_boxed(cleaned)
        is_correct = bool(
            predicted is not None
            and answers_match(predicted, instance.answer)
        )
    return cleaned, predicted, is_correct


def _clean_stored_evidence(
    item: dict[str, Any],
    program: ProblemProgram,
) -> dict[str, Any] | None:
    """Sanitize persisted evidence and re-grade it if spillover was removed."""
    copied = dict(item)
    if _evidence_provenance_issues(copied, program=program):
        return None
    original = str(copied.get("response", ""))
    cleaned = sanitize_solver_trace(original)
    if not cleaned:
        return None
    copied["response"] = cleaned

    instance = program.execute(copied["seed"])
    if instance is None:
        return None
    if " ".join(copied["problem"].split()) != " ".join(instance.problem.split()):
        return None

    # Persisted labels are never trusted: extract and grade against the current
    # live parent's answer on every use, even when sanitation changed nothing.
    from .reward import answers_match, extract_boxed

    predicted = extract_boxed(cleaned)
    correct = bool(
        predicted is not None
        and answers_match(predicted, instance.answer)
    )
    copied["predicted_answer"] = predicted
    copied["correct"] = correct
    copied["role"] = "success" if correct else "failure"
    if cleaned != original.strip():
        copied["response_tokens"] = _trace_token_count(cleaned)
    if reasoning_trace_quality_issues(
        cleaned,
        copied.get("predicted_answer"),
        response_tokens=int(copied.get("response_tokens", 0) or 0),
    ):
        return None
    return copied


def select_reasoning_evidence(
    rollouts: Iterable[RolloutRecord],
    *,
    program: ProblemProgram,
    instance: ProblemInstance,
    iteration: int,
    max_tokens: int,
    tokenizer=None,
) -> list[ReasoningEvidence]:
    """Select one concise success and one confident failure at zero rollout cost."""
    cleaned_records: list[dict[str, Any]] = []
    for record in rollouts:
        if getattr(record, "status", "accepted") != "accepted":
            continue
        cleaned, predicted, is_correct = clean_and_grade_solver_rollout(
            record,
            instance,
        )
        if not cleaned:
            continue
        # Evidence is independently extracted/graded against the live answer;
        # backend-supplied predicted/correct labels are only scoring telemetry.
        from .reward import answers_match, extract_boxed

        predicted = extract_boxed(cleaned)
        is_correct = bool(
            predicted is not None
            and answers_match(predicted, instance.answer)
        )
        trace_tokens = _trace_token_count(cleaned, tokenizer)
        if trace_tokens > max_tokens:
            continue
        quality_issues = reasoning_trace_quality_issues(
            cleaned,
            predicted,
            response_tokens=trace_tokens,
        )
        if quality_issues:
            continue
        cleaned_records.append(
            {
                "record": record,
                "response": cleaned,
                "predicted_answer": predicted,
                "correct": is_correct,
                "response_tokens": trace_tokens,
            }
        )

    correct = [item for item in cleaned_records if item["correct"]]
    wrong = [item for item in cleaned_records if not item["correct"]]
    if not correct or not wrong:
        return []

    def response_len(item: dict[str, Any]) -> int:
        record = item["record"]
        original = str(record.response or "").strip()
        if (
            item["response"] == original
            and int(getattr(record, "response_tokens", 0) or 0) > 0
        ):
            return int(record.response_tokens)
        if tokenizer is not None:
            try:
                return len(
                    tokenizer.encode(
                        item["response"],
                        add_special_tokens=False,
                    )
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return len(str(item["response"]).split())
        return len(str(item["response"]).split())

    success = min(correct, key=response_len)
    # Low-entropy wrong answers are the strongest "confident wrong" signal.
    def entropy_value(item: dict[str, Any]) -> float:
        try:
            return float(item["record"].entropy)
        except (TypeError, ValueError):
            return math.inf

    failure = min(wrong, key=lambda r: (entropy_value(r), response_len(r)))

    def make(item: dict[str, Any], role: str) -> ReasoningEvidence:
        record = item["record"]
        return ReasoningEvidence(
            source="observed",
            role=role,
            problem=instance.problem,
            response=item["response"],
            predicted_answer=item["predicted_answer"],
            correct=bool(item["correct"]),
            entropy=entropy_value(item),
            origin_program_id=program.program_id,
            origin_concept_group=str(program.get_concept_group() or ""),
            origin_concept_type=str(program.get_concept_type() or ""),
            seed=int(instance.seed),
            policy_version=int(getattr(record, "policy_version", -1)),
            iteration=int(iteration),
            response_tokens=int(item["response_tokens"]),
            evidence_quality_version=EVIDENCE_QUALITY_VERSION,
        )

    return [make(success, "success"), make(failure, "failure")]


def fit_evidence_to_total_budget(
    evidence: list[dict[str, Any]],
    total_tokens: int,
    *,
    tokenizer=None,
) -> list[dict[str, Any]]:
    """Return complete traces only; never truncate evidence after validation."""
    if not evidence or total_tokens <= 0:
        return []
    fitted = [dict(item) for item in evidence]
    token_total = sum(
        _trace_token_count(str(item.get("response", "")), tokenizer)
        for item in fitted
    )
    if token_total > total_tokens:
        return []
    return fitted


def collect_planning_evidence(
    parent: ProblemProgram,
    op: str,
    champions: Iterable[ProblemProgram],
    *,
    total_tokens: int,
    tokenizer=None,
) -> list[dict[str, Any]]:
    """Return one clean same-problem contrast for either mutation operator.

    Breadth transfers the *move* diagnosed on the live parent; mixing a parent
    success with a different champion's failure would not identify one observed
    divergence and would confound the planned transfer.
    """
    del champions  # Kept in the public signature for call-site compatibility.
    own = [
        cleaned
        for item in (parent.metadata or {}).get("reasoning_evidence") or []
        if (cleaned := _clean_stored_evidence(dict(item), parent)) is not None
    ]
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for item in own:
        try:
            seed = int(item.get("seed", 0))
        except (TypeError, ValueError):
            continue
        problem = " ".join(str(item.get("problem", "")).split())
        if not problem:
            continue
        grouped.setdefault((seed, problem), []).append(item)

    candidates: list[tuple[int, list[dict[str, Any]]]] = []
    for (seed, _problem), items in grouped.items():
        successes = [
            item
            for item in items
            if item.get("role") == "success" and item.get("correct") is not False
        ]
        failures = [
            item
            for item in items
            if item.get("role") == "failure" and item.get("correct") is not True
        ]
        if not successes or not failures:
            continue
        success = min(successes, key=lambda item: len(str(item["response"])))
        failure = min(
            failures,
            key=lambda item: (
                float(item.get("entropy", math.inf)),
                len(str(item["response"])),
            ),
        )
        pair = [success, failure]
        if not validate_reasoning_contrast(pair):
            candidates.append((seed, pair))

    candidates.sort(key=lambda item: item[0])
    selected = candidates[0][1] if candidates else []

    return fit_evidence_to_total_budget(
        selected,
        total_tokens,
        tokenizer=tokenizer,
    )


def compute_meta_progress(
    pre_scores: dict[str, dict[str, Any]],
    programs: list[ProblemProgram],
    results: list[RQResult | None],
    *,
    iteration: int,
) -> MetaProgress:
    """Compute fixed-cohort pass-rate change before any MAP move/replacement."""
    rows: list[dict[str, Any]] = []
    for program, result in zip(programs, results):
        before = pre_scores.get(program.program_id)
        if before is None or result is None:
            continue
        rows.append(
            {
                "pre": float(before["p_hat"]),
                "post": float(result.p_hat),
                "concept_group": str(before.get("concept_group") or ""),
                "concept_type": str(before.get("concept_type") or ""),
                "operator": str(before.get("operator") or "seed"),
            }
        )

    def summarize(items: list[dict[str, Any]]) -> ProgressSummary:
        if not items:
            return ProgressSummary(0, 0.0, 0.0, 0.0)
        pre = sum(item["pre"] for item in items) / len(items)
        post = sum(item["post"] for item in items) / len(items)
        return ProgressSummary(
            count=len(items),
            pre_mean_p=pre,
            post_mean_p=post,
            delta_p=post - pre,
        )

    def group_by(key: str) -> dict[str, ProgressSummary]:
        names = sorted({str(item[key]) for item in rows if str(item[key])})
        return {
            name: summarize([item for item in rows if str(item[key]) == name])
            for name in names
        }

    return MetaProgress(
        iteration=int(iteration),
        global_progress=summarize(rows),
        by_concept_group=group_by("concept_group"),
        by_concept_type=group_by("concept_type"),
        by_operator=group_by("operator"),
    )


def progress_context(
    progress: dict[str, Any] | None,
    parent: ProblemProgram,
) -> dict[str, Any]:
    """Extract only the feedback relevant to one planning request."""
    payload = progress or {}
    group = str(parent.get_concept_group() or "")
    concept_type = str(parent.get_concept_type() or "")
    return {
        "global": payload.get("global_progress", {}),
        "concept_group": (payload.get("by_concept_group") or {}).get(group, {}),
        "concept_type": (payload.get("by_concept_type") or {}).get(concept_type, {}),
        "by_operator": payload.get("by_operator", {}),
    }


def validate_mutation_plan(
    plan: dict[str, Any],
    op: str,
    *,
    reasoning_informed: bool = True,
    parent_concept_group: str | None = None,
    parent_concept_type: str | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        schema_version = int(plan.get("schema_version", 0))
    except (TypeError, ValueError):
        schema_version = 0
    if schema_version == 4:
        required_fields = PLAN_FIELDS_V4
    elif schema_version == 5:
        required_fields = PLAN_FIELDS_V5
    else:
        required_fields = PLAN_FIELDS_V5
        errors.append("schema_version must be 4 (legacy) or 5")

    for field_name in required_fields:
        if field_name not in plan:
            errors.append(f"missing plan field: {field_name}")
    if any(error.startswith("missing plan field:") for error in errors):
        return errors
    for field_name in REMOVED_PLAN_FIELDS:
        if field_name in plan:
            errors.append(
                f"removed schema_version {schema_version} plan field: "
                f"{field_name}"
            )
    optional_fields = (
        OPTIONAL_PLAN_FIELDS_V5 if schema_version == 5 else ()
    )
    unexpected_fields = sorted(
        set(plan)
        - set(required_fields)
        - set(optional_fields)
        - set(REMOVED_PLAN_FIELDS)
    )
    for field_name in unexpected_fields:
        errors.append(
            f"unexpected schema_version {schema_version} plan field: "
            f"{field_name}"
        )
    if schema_version == 5:
        generator_family = plan.get("generator_family")
        if (
            not isinstance(generator_family, str)
            or re.fullmatch(
                r"[a-z][a-z0-9_.-]*",
                generator_family.strip(),
            )
            is None
        ):
            errors.append(
                "generator_family must be a non-empty stable identifier"
            )
        if not isinstance(plan.get("family_config"), dict):
            errors.append("family_config must be an object")
    if str(plan.get("operator")) != op:
        errors.append(f"plan operator must be {op}")
    try:
        max_attempts = int(plan.get("max_sampling_attempts", 0))
    except (TypeError, ValueError):
        max_attempts = 0
    if max_attempts != 200:
        errors.append("max_sampling_attempts must be exactly 200")
    for name in (
        "target_reasoning_move",
        "target_concept_group",
        "target_concept_type",
        "why_target_reasoning_move_is_necessary",
        "answer_route",
        "heldout_evaluation",
    ):
        if not isinstance(plan.get(name), str) or not plan[name].strip():
            errors.append(f"plan field must be a non-empty string: {name}")
    evidence_fields = (
        "failure_summary",
        "correct_wrong_contrast",
        "predicted_pre_behavior",
        "predicted_post_behavior",
    )
    if reasoning_informed:
        for name in evidence_fields:
            if not isinstance(plan.get(name), str) or not plan[name].strip():
                errors.append(
                    "reasoning-informed plan field must be a non-empty string: "
                    f"{name}"
                )
    else:
        for name in evidence_fields:
            if plan.get(name) is not None:
                errors.append(
                    f"plain plan field must be null without Solver evidence: {name}"
                )
    for name in (
        "preserve",
        "change",
        "forbidden_changes",
        "parameters",
        "guards",
    ):
        if not isinstance(plan.get(name), list) or not plan[name]:
            errors.append(f"plan field must be a non-empty list: {name}")
    for name in ("preserve", "change", "forbidden_changes", "guards"):
        values = plan.get(name)
        if isinstance(values, list) and values and any(
            not isinstance(value, str) or not value.strip()
            for value in values
        ):
            errors.append(
                f"plan field must contain only non-empty strings: {name}"
            )
    parameters = plan.get("parameters")
    if isinstance(parameters, list) and parameters:
        for index, parameter in enumerate(parameters):
            if not isinstance(parameter, dict):
                errors.append(f"parameters[{index}] must be an object")
                continue
            name_value = parameter.get("name")
            if not isinstance(name_value, str) or not name_value.strip():
                errors.append(
                    f"parameters[{index}].name must be a non-empty string"
                )
            # A registered family publishes its bounded domains as JSON Schema
            # `enum` lists, and the planner prompt shows that schema verbatim, so
            # a literal enumeration is the registry's own vocabulary for a
            # domain -- and a stricter, machine-checkable one than prose. Accept
            # either form; reject empty or non-scalar enumerations.
            domain_value = parameter.get("domain")
            if isinstance(domain_value, str):
                if not domain_value.strip():
                    errors.append(
                        f"parameters[{index}].domain must be a non-empty string "
                        "or a non-empty list of scalar values"
                    )
            elif isinstance(domain_value, list):
                if not domain_value or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float, str))
                    or (isinstance(item, str) and not item.strip())
                    for item in domain_value
                ):
                    errors.append(
                        f"parameters[{index}].domain list must enumerate at "
                        "least one non-empty scalar value"
                    )
            else:
                errors.append(
                    f"parameters[{index}].domain must be a non-empty string "
                    "or a non-empty list of scalar values"
                )
    variant_value = plan.get("family_variant")
    if variant_value is not None and (
        not isinstance(variant_value, str) or not variant_value.strip()
    ):
        errors.append(
            "family_variant must be a non-empty string naming a registered "
            "variant, or be omitted"
        )
    guards = plan.get("guards")
    if (
        isinstance(guards, list)
        and guards
        and isinstance(guards[0], str)
        and not guards[0].strip().lower().startswith("necessity:")
    ):
        errors.append(
            "guards[0] must start with `necessity:` and state a checkable "
            "structural witness"
        )
    for name in ("problem_output_contract", "answer_output_contract"):
        if not isinstance(plan.get(name), dict):
            errors.append(f"{name} must be an object")
    problem_contract = plan.get("problem_output_contract")
    if isinstance(problem_contract, dict):
        for field_name in ("states", "withholds"):
            values = problem_contract.get(field_name)
            if (
                not isinstance(values, list)
                or not values
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in values
                )
            ):
                errors.append(
                    f"problem_output_contract.{field_name} must be a "
                    "non-empty list of non-empty strings"
                )
        closing_value = problem_contract.get("closing_instruction")
        if not isinstance(closing_value, str) or not closing_value.strip():
            errors.append(
                "problem_output_contract.closing_instruction must be a "
                "non-empty string"
            )
            closing = ""
        else:
            closing = closing_value.lower()
        if closing and "integer" not in closing:
            errors.append(
                "problem_output_contract.closing_instruction must request one integer"
            )
    answer_contract = plan.get("answer_output_contract")
    if isinstance(answer_contract, dict):
        answer_type = answer_contract.get("type")
        if not isinstance(answer_type, str) or answer_type.strip().lower() != "integer":
            errors.append("answer_output_contract.type must be integer")
        serialization = answer_contract.get("serialization")
        if (
            not isinstance(serialization, str)
            or not serialization.strip()
            or "sympy.Integer" not in serialization
        ):
            errors.append("answer serialization must use str(sympy.Integer(...))")
    target_group = str(plan.get("target_concept_group", "")).strip()
    target_type = str(plan.get("target_concept_type", "")).strip()
    if target_group and target_group not in CONCEPT_GROUPS:
        errors.append(f"unknown target_concept_group: {target_group}")
    if target_type and concept_group_for_type(target_type) != target_group:
        errors.append(
            "target_concept_type prefix must match target_concept_group"
        )
    if parent_concept_group and parent_concept_type:
        if op == "in_depth" and (
            target_group != parent_concept_group
            or target_type != parent_concept_type
        ):
            errors.append(
                "in-depth plan target must preserve parent concept group and type"
            )
        if op == "in_breadth" and target_group == parent_concept_group:
            errors.append(
                "in-breadth plan target must change parent concept group"
            )
    heldout = str(plan.get("heldout_evaluation", "")).lower()
    if "pass" not in heldout and "accuracy" not in heldout:
        errors.append("heldout_evaluation phase 1 must be pass-rate based")
    return errors


def mutation_plan_id(plan: dict[str, Any]) -> str:
    canonical = json.dumps(plan, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def update_operator_ema(
    ema: dict[str, float],
    progress: dict[str, Any],
    *,
    alpha: float,
) -> dict[str, float]:
    updated = dict(ema)
    for op in ("in_depth", "in_breadth"):
        summary = (progress.get("by_operator") or {}).get(op)
        if not summary or int(summary.get("count", 0)) <= 0:
            continue
        value = float(summary.get("delta_p", 0.0))
        old = float(updated.get(op, value))
        updated[op] = alpha * value + (1.0 - alpha) * old
    return updated


def adaptive_depth_probability(
    ema: dict[str, float],
    *,
    fallback: float,
    min_probability: float,
) -> float:
    if "in_depth" not in ema or "in_breadth" not in ema:
        return float(fallback)
    # A bounded softmax keeps small delta-p differences from collapsing
    # exploration while still preferring the operator with realized progress.
    depth = math.exp(max(-5.0, min(5.0, 5.0 * float(ema["in_depth"]))))
    breadth = math.exp(max(-5.0, min(5.0, 5.0 * float(ema["in_breadth"]))))
    raw = depth / (depth + breadth)
    floor = max(0.0, min(0.49, float(min_probability)))
    return min(1.0 - floor, max(floor, raw))
