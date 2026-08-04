"""One-shot reasoning-transfer diagnostics for compiled mutation families.

The generated child is used as a worked example, not as a training update.
The frozen Solver then answers a separate parent/held-out target.  This probes
in-context transfer and internal trajectory changes; it is not evidence of a
persistent capability change in model weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np

from .prompts import SOLVER_SYSTEM_PROMPT


SHOT_DIAGNOSTIC_SCHEMA_VERSION = 1
SHOT_CONDITIONS = ("no_shot", "plain_shot", "reasoning_shot")
SHOT_PRESENTATIONS = ("assistant_turn", "user_context")

_LINEAR_EQUATION_RE = re.compile(
    r"^\((-?\d+)\)x \+ \((-?\d+)\)y \+ \((-?\d+)\)z = (-?\d+)$"
)
_MODULAR_EQUATION_RE = re.compile(
    r"^(\d+)x \+ (\d+)y \+ (\d+)z is congruent to "
    r"(-?\d+) modulo (\d+)$"
)


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _linear_rows(problem: str) -> tuple[list[list[int]], list[int]]:
    parsed = [
        match
        for line in str(problem).splitlines()
        if (match := _LINEAR_EQUATION_RE.fullmatch(line.strip())) is not None
    ]
    if len(parsed) != 2:
        raise ValueError("linear shot problem must contain exactly two equations")
    rows = [[int(match.group(i)) for i in range(1, 4)] for match in parsed]
    rhs = [int(match.group(4)) for match in parsed]
    return rows, rhs


def _modular_rows(
    problem: str,
) -> tuple[list[list[int]], list[int], int]:
    parsed = [
        match
        for line in str(problem).splitlines()
        if (match := _MODULAR_EQUATION_RE.fullmatch(line.strip())) is not None
    ]
    if len(parsed) != 2:
        raise ValueError("modular shot problem must contain exactly two congruences")
    rows = [[int(match.group(i)) for i in range(1, 4)] for match in parsed]
    rhs = [int(match.group(4)) for match in parsed]
    moduli = {int(match.group(5)) for match in parsed}
    if len(moduli) != 1:
        raise ValueError("modular shot congruences disagree on the modulus")
    return rows, rhs, next(iter(moduli))


def _linear_solution(row: Mapping[str, Any]) -> str:
    problem = str(row["problem"])
    answer = int(str(row["answer"]))
    rows, rhs = _linear_rows(problem)
    facts = dict(row.get("necessity_facts") or {})
    raw_weights = facts.get("rowspace_weights")
    if not isinstance(raw_weights, Sequence) or len(raw_weights) != 2:
        raise ValueError("linear semantics are missing two rowspace weights")
    weights = [Fraction(str(value)) for value in raw_weights]
    scale = math.lcm(*(weight.denominator for weight in weights))
    integer_weights = [int(weight * scale) for weight in weights]
    combined = [
        sum(integer_weights[index] * rows[index][column] for index in range(2))
        for column in range(3)
    ]
    if combined != [scale, scale, scale]:
        raise ValueError(
            "linear semantics weights do not combine to the target functional"
        )
    combined_rhs = sum(
        integer_weights[index] * rhs[index] for index in range(2)
    )
    if combined_rhs != scale * answer:
        raise ValueError("linear canonical solution disagrees with the answer")

    left = (
        f"{integer_weights[0]}E_1 + {integer_weights[1]}E_2"
        if integer_weights[1] >= 0
        else f"{integer_weights[0]}E_1 - {abs(integer_weights[1])}E_2"
    )
    return (
        "Let E_1 and E_2 denote the two printed equations. "
        "The individual variables are not uniquely determined, so target the "
        "requested aggregate instead. Comparing coefficients gives the scaled "
        f"combination {left}, whose left side is "
        f"{scale}(x+y+z). Its right side is "
        f"{integer_weights[0]}({rhs[0]})"
        + (
            f" + {integer_weights[1]}({rhs[1]})"
            if integer_weights[1] >= 0
            else f" - {abs(integer_weights[1])}({rhs[1]})"
        )
        + f" = {combined_rhs}. Therefore x+y+z={combined_rhs}/{scale}"
        f"={answer}, so the answer is \\boxed{{{answer}}}."
    )


def _modular_solution(row: Mapping[str, Any]) -> str:
    problem = str(row["problem"])
    answer = int(str(row["answer"]))
    rows, rhs, modulus = _modular_rows(problem)
    facts = dict(row.get("necessity_facts") or {})
    multiplier = int(facts["multiplier"]) % modulus
    inverse = pow(multiplier, -1, modulus)
    if any(
        (rows[0][column] + rows[1][column]) % modulus != multiplier
        for column in range(3)
    ):
        raise ValueError("modular rows do not share the recorded multiplier")
    residue_sum = sum(rhs) % modulus
    computed = residue_sum * inverse % modulus
    if computed != answer:
        raise ValueError("modular canonical solution disagrees with the answer")
    return (
        "Add the two congruences column by column. Every variable then has "
        f"coefficient {multiplier} modulo {modulus}, so "
        f"{multiplier}(x+y+z) is congruent to "
        f"{rhs[0]}+{rhs[1]}={residue_sum} modulo {modulus}. "
        f"The inverse of {multiplier} modulo {modulus} is {inverse}. "
        f"Hence x+y+z is congruent to {inverse}({residue_sum})={answer} "
        f"modulo {modulus}, and the least nonnegative answer is "
        f"\\boxed{{{answer}}}."
    )


def canonical_shot_solution(
    generator_family: str,
    semantic_row: Mapping[str, Any],
) -> str:
    """Write a condition-blind, executable-family-specific worked solution."""

    if not bool(semantic_row.get("valid")):
        raise ValueError("cannot create a shot from an invalid semantic row")
    if not bool(semantic_row.get("answer_correct")):
        raise ValueError("cannot create a shot with an unverified answer")
    if not bool(semantic_row.get("necessity_holds")):
        raise ValueError("cannot create a shot without verified necessity")
    if generator_family == "linear_system_aggregate":
        return _linear_solution(semantic_row)
    if generator_family == "modular_linear_system_aggregate":
        return _modular_solution(semantic_row)
    raise ValueError(f"unsupported shot generator family: {generator_family}")


def build_solver_messages(
    target_problem: str,
    *,
    shot_problem: str | None = None,
    shot_solution: str | None = None,
    shot_presentation: str = "assistant_turn",
) -> list[dict[str, str]]:
    """Build a no-shot or one-shot Solver conversation.

    No condition label, mutation plan, generator source, answer metadata, or
    analysis metadata is exposed to the Solver.
    """

    target = str(target_problem).strip()
    if not target:
        raise ValueError("target_problem must not be empty")
    if shot_presentation not in SHOT_PRESENTATIONS:
        raise ValueError(
            f"unknown shot_presentation: {shot_presentation!r}"
        )
    has_problem = shot_problem is not None
    has_solution = shot_solution is not None
    if has_problem != has_solution:
        raise ValueError("shot_problem and shot_solution must be supplied together")
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT}
    ]
    if has_problem:
        example = str(shot_problem).strip()
        solution = str(shot_solution).strip()
        if not example or not solution:
            raise ValueError("shot problem and solution must not be empty")
        if shot_presentation == "assistant_turn":
            messages.extend(
                [
                    {"role": "user", "content": example},
                    {"role": "assistant", "content": solution},
                ]
            )
            messages.append({"role": "user", "content": target})
        else:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Use the following worked example as context.\n\n"
                        "Example problem:\n"
                        f"{example}\n\n"
                        "Example solution:\n"
                        f"{solution}\n\n"
                        "Now solve this separate target problem. Do not merely "
                        "repeat the example.\n\n"
                        "Target problem:\n"
                        f"{target}"
                    ),
                }
            )
    else:
        messages.append({"role": "user", "content": target})
    return messages


def conversation_hash(messages: Sequence[Mapping[str, str]]) -> str:
    canonical = json.dumps(
        list(messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return stable_text_hash(canonical)


def cosine_similarity(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("cosine inputs must be same-shaped vectors")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        raise ValueError("cosine inputs must be non-zero")
    return float(np.dot(a, b) / denominator)


def summarize_shot_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate target accuracy and internal metrics without pseudo-replication."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["operator"]), str(record["condition"]))].append(
            record
        )
    rows: list[dict[str, Any]] = []
    for (operator, condition), items in sorted(grouped.items()):
        rows.append(
            {
                "operator": operator,
                "condition": condition,
                "family_variant": (
                    items[0].get("family_variant")
                    if condition != "no_shot"
                    else None
                ),
                "n_targets": len(items),
                "accuracy": float(
                    np.mean([bool(item["correct"]) for item in items])
                ),
                "mean_stalt": float(
                    np.mean([float(item["stalt"]) for item in items])
                ),
                "mean_temporal_path_length": float(
                    np.mean(
                        [float(item["temporal_path_length"]) for item in items]
                    )
                ),
                "mean_generated_tokens": float(
                    np.mean([int(item["generated_tokens"]) for item in items])
                ),
                "mean_layer_concentration_hhi": float(
                    np.mean(
                        [
                            float(item["mean_layer_concentration_hhi"])
                            for item in items
                        ]
                    )
                ),
                "mean_prompt_last_cosine_to_no_shot": float(
                    np.mean(
                        [
                            float(item["prompt_last_cosine_to_no_shot"])
                            for item in items
                        ]
                    )
                ),
                "unique_conversation_hashes": len(
                    {str(item["conversation_hash"]) for item in items}
                ),
            }
        )

    by_key = {
        (row["operator"], row["condition"]): row
        for row in rows
    }
    contrasts: list[dict[str, Any]] = []
    for operator in sorted({row["operator"] for row in rows}):
        plain = by_key.get((operator, "plain_shot"))
        reasoning = by_key.get((operator, "reasoning_shot"))
        baseline = by_key.get((operator, "no_shot"))
        if not plain or not reasoning or not baseline:
            continue
        contrasts.append(
            {
                "operator": operator,
                "accuracy_reasoning_minus_plain": (
                    reasoning["accuracy"] - plain["accuracy"]
                ),
                "stalt_reasoning_minus_plain": (
                    reasoning["mean_stalt"] - plain["mean_stalt"]
                ),
                "path_length_reasoning_minus_plain": (
                    reasoning["mean_temporal_path_length"]
                    - plain["mean_temporal_path_length"]
                ),
                "reasoning_accuracy_gain_over_no_shot": (
                    reasoning["accuracy"] - baseline["accuracy"]
                ),
                "plain_accuracy_gain_over_no_shot": (
                    plain["accuracy"] - baseline["accuracy"]
                ),
            }
        )
    return {
        "schema_version": SHOT_DIAGNOSTIC_SCHEMA_VERSION,
        "rows": rows,
        "contrasts": contrasts,
        "interpretation": (
            "frozen-Solver in-context transfer and internal-trajectory "
            "diagnostic; not persistent post-training capability expansion"
        ),
    }


__all__ = [
    "SHOT_CONDITIONS",
    "SHOT_DIAGNOSTIC_SCHEMA_VERSION",
    "SHOT_PRESENTATIONS",
    "build_solver_messages",
    "canonical_shot_solution",
    "conversation_hash",
    "cosine_similarity",
    "stable_text_hash",
    "summarize_shot_records",
]
