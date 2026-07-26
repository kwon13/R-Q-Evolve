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
    "insight_route",
    "brute_route",
    "equivalence_assertion",
    "route_independence_reason",
    "decoy_route",
    "decoy_assertion",
    "problem_output_contract",
    "answer_output_contract",
    "predicted_pre_behavior",
    "predicted_post_behavior",
    "heldout_evaluation",
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
    accepted = [
        r
        for r in rollouts
        if getattr(r, "status", "accepted") == "accepted" and str(r.response or "").strip()
    ]
    correct = [r for r in accepted if bool(r.correct)]
    wrong = [r for r in accepted if not bool(r.correct)]
    if not correct or not wrong:
        return []

    def response_len(record: RolloutRecord) -> int:
        if int(getattr(record, "response_tokens", 0) or 0) > 0:
            return int(record.response_tokens)
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(record.response, add_special_tokens=False))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return len(str(record.response).split())
        return len(str(record.response).split())

    success = min(correct, key=response_len)
    # Low-entropy wrong answers are the strongest "confident wrong" signal.
    def entropy_value(record: RolloutRecord) -> float:
        try:
            return float(record.entropy)
        except (TypeError, ValueError):
            return math.inf

    failure = min(wrong, key=lambda r: (entropy_value(r), response_len(r)))

    def make(record: RolloutRecord, role: str) -> ReasoningEvidence:
        return ReasoningEvidence(
            source="observed",
            role=role,
            problem=instance.problem,
            response=truncate_text_tokens(
                record.response,
                max_tokens,
                tokenizer=tokenizer,
            ),
            predicted_answer=record.predicted_answer,
            correct=bool(record.correct),
            entropy=entropy_value(record),
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
    own = list((parent.metadata or {}).get("reasoning_evidence") or [])
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
                if item.get("role") == "failure":
                    cross_failures.append(dict(item))
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
    try:
        schema_version = int(plan.get("schema_version", 0))
    except (TypeError, ValueError):
        schema_version = 0
    if schema_version != 2:
        errors.append("schema_version must be 2")
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
        "insight_route",
        "brute_route",
        "equivalence_assertion",
        "route_independence_reason",
        "decoy_route",
        "decoy_assertion",
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
    if str(plan.get("insight_route", "")).strip().lower() == str(
        plan.get("decoy_route", "")
    ).strip().lower():
        errors.append("decoy_route must differ from insight_route")
    equivalence = str(plan.get("equivalence_assertion", ""))
    if "assert" not in equivalence or "==" not in equivalence:
        errors.append("equivalence_assertion must contain an equality assert")
    decoy_assertion = str(plan.get("decoy_assertion", ""))
    if "assert" not in decoy_assertion or "!=" not in decoy_assertion:
        errors.append("decoy_assertion must contain a non-equality assert")
    if "continue" not in decoy_assertion:
        errors.append("decoy_assertion must describe collision resampling with continue")
    for name in ("problem_output_contract", "answer_output_contract"):
        if not isinstance(plan.get(name), dict):
            errors.append(f"{name} must be an object")
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
