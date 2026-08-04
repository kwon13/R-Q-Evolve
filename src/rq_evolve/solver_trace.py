"""Solver rollout hygiene shared by scoring, evaluation and analysis.

Base-model generations routinely continue past their first answer by opening a
fresh ``User:``/``Assistant:`` exchange. Grading the last boxed answer in such a
completion scores an unrelated second problem, so every consumer of a rollout
first trims it back to the single supplied problem.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .program import ProblemInstance

if TYPE_CHECKING:
    from .backends import RolloutRecord


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


def sanitize_solver_trace(response: str) -> str:
    """Drop chat-template spillover or a second conversation from a rollout.

    A rollout must describe exactly one supplied problem. Some base-model
    generations continue after their first answer by emitting a fresh
    User:/Assistant: exchange. Without this boundary check, a correct first
    solution can be mislabeled as a confident failure about an unrelated second
    problem.
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
    record: "RolloutRecord",
    instance: ProblemInstance,
) -> tuple[str, str | None, bool]:
    """Return one-chat response text and a grade derived from that text.

    The raw rollout stays untouched for diagnostics. When a model spills into a
    second conversation, scoring must nevertheless agree on the first supplied
    problem instead of grading the last boxed answer from an unrelated suffix.
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
