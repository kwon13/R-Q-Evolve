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
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

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


REQUIRED_PLAN_FIELDS: tuple[str, ...] = (
    "schema_version",
    "operator",
    "failure_summary",
    "target_reasoning_move",
    "correct_wrong_contrast",
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

REMOVED_PLAN_FIELDS: tuple[str, ...] = (
    "insight_route",
    "brute_route",
    "equivalence_assertion",
    "route_independence_reason",
    "decoy_route",
    "decoy_assertion",
)


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
    original = str(copied.get("response", ""))
    cleaned = sanitize_solver_trace(original)
    if not cleaned:
        return None
    copied["response"] = cleaned

    try:
        instance = program.execute(int(copied.get("seed", 0)))
    except (TypeError, ValueError):
        instance = None
    if instance is None:
        return None
    if str(copied.get("problem", "")).strip() != instance.problem.strip():
        return None

    if cleaned != original.strip():
        from .reward import answers_match, extract_boxed

        predicted = extract_boxed(cleaned)
        correct = bool(
            predicted is not None
            and answers_match(predicted, instance.answer)
        )
        copied["predicted_answer"] = predicted
        copied["correct"] = correct
        copied["role"] = "success" if correct else "failure"
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
        cleaned_records.append(
            {
                "record": record,
                "response": cleaned,
                "predicted_answer": predicted,
                "correct": is_correct,
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
            response=truncate_text_tokens(
                item["response"],
                max_tokens,
                tokenizer=tokenizer,
            ),
            predicted_answer=item["predicted_answer"],
            correct=bool(item["correct"]),
            entropy=entropy_value(item),
            origin_program_id=program.program_id,
            origin_concept_group=str(program.get_concept_group() or ""),
            origin_concept_type=str(program.get_concept_type() or ""),
            seed=int(instance.seed),
            policy_version=int(getattr(record, "policy_version", -1)),
            iteration=int(iteration),
        )

    return [make(success, "success"), make(failure, "failure")]


def fit_evidence_to_total_budget(
    evidence: list[dict[str, Any]],
    total_tokens: int,
    *,
    tokenizer=None,
) -> list[dict[str, Any]]:
    """Re-truncate stored traces so their combined planning input stays bounded."""
    if not evidence or total_tokens <= 0:
        return []
    per_item = max(1, total_tokens // len(evidence))
    fitted: list[dict[str, Any]] = []
    for item in evidence:
        copied = dict(item)
        copied["response"] = truncate_text_tokens(
            str(copied.get("response", "")),
            per_item,
            tokenizer=tokenizer,
        )
        fitted.append(copied)
    return fitted


def collect_planning_evidence(
    parent: ProblemProgram,
    op: str,
    champions: Iterable[ProblemProgram],
    *,
    total_tokens: int,
    tokenizer=None,
) -> list[dict[str, Any]]:
    """Use same-program contrast for depth; allow cross-domain failure for breadth."""
    own = [
        cleaned
        for item in (parent.metadata or {}).get("reasoning_evidence") or []
        if (cleaned := _clean_stored_evidence(dict(item), parent)) is not None
    ]
    success = next((e for e in own if e.get("role") == "success"), None)
    failure = next((e for e in own if e.get("role") == "failure"), None)

    selected: list[dict[str, Any]] = []
    if op == "in_depth":
        if success and failure:
            selected = [success, failure]
    else:
        if success:
            selected.append(success)
        # Prefer a failure already observed in a different mathematical group.
        parent_group = parent.get_concept_group()
        cross_failures: list[dict[str, Any]] = []
        for champion in champions:
            if champion.program_id == parent.program_id:
                continue
            if champion.get_concept_group() == parent_group:
                continue
            for item in (champion.metadata or {}).get("reasoning_evidence") or []:
                cleaned = _clean_stored_evidence(dict(item), champion)
                if cleaned is not None and cleaned.get("role") == "failure":
                    cross_failures.append(cleaned)
        if cross_failures:
            cross_failures.sort(
                key=lambda e: (
                    float(e.get("entropy", math.inf)),
                    str(e.get("origin_program_id", "")),
                )
            )
            selected.append(cross_failures[0])
        elif failure:
            selected.append(failure)

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


def validate_mutation_plan(plan: dict[str, Any], op: str) -> list[str]:
    errors: list[str] = []
    for field_name in REQUIRED_PLAN_FIELDS:
        if field_name not in plan:
            errors.append(f"missing plan field: {field_name}")
    if errors:
        return errors
    for field_name in REMOVED_PLAN_FIELDS:
        if field_name in plan:
            errors.append(
                f"removed schema_version 3 plan field: {field_name}"
            )
    unexpected_fields = sorted(
        set(plan) - set(REQUIRED_PLAN_FIELDS) - set(REMOVED_PLAN_FIELDS)
    )
    for field_name in unexpected_fields:
        errors.append(
            f"unexpected schema_version 3 plan field: {field_name}"
        )
    try:
        schema_version = int(plan.get("schema_version", 0))
    except (TypeError, ValueError):
        schema_version = 0
    if schema_version != 3:
        errors.append("schema_version must be 3")
    if str(plan.get("operator")) != op:
        errors.append(f"plan operator must be {op}")
    try:
        max_attempts = int(plan.get("max_sampling_attempts", 0))
    except (TypeError, ValueError):
        max_attempts = 0
    if max_attempts != 200:
        errors.append("max_sampling_attempts must be exactly 200")
    for name in (
        "failure_summary",
        "target_reasoning_move",
        "correct_wrong_contrast",
        "answer_route",
        "predicted_pre_behavior",
        "predicted_post_behavior",
        "heldout_evaluation",
    ):
        if not str(plan.get(name, "")).strip():
            errors.append(f"empty plan field: {name}")
    for name in (
        "preserve",
        "change",
        "forbidden_changes",
        "parameters",
        "guards",
    ):
        if not isinstance(plan.get(name), list) or not plan[name]:
            errors.append(f"plan field must be a non-empty list: {name}")
    for name in ("problem_output_contract", "answer_output_contract"):
        if not isinstance(plan.get(name), dict):
            errors.append(f"{name} must be an object")
    problem_contract = plan.get("problem_output_contract")
    if isinstance(problem_contract, dict):
        closing = str(problem_contract.get("closing_instruction", "")).lower()
        if "integer" not in closing:
            errors.append(
                "problem_output_contract.closing_instruction must request one integer"
            )
    answer_contract = plan.get("answer_output_contract")
    if isinstance(answer_contract, dict):
        if str(answer_contract.get("type", "")).lower() != "integer":
            errors.append("answer_output_contract.type must be integer")
        if "sympy.Integer" not in str(answer_contract.get("serialization", "")):
            errors.append("answer serialization must use str(sympy.Integer(...))")
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
