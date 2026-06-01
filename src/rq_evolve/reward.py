"""verl-compatible reward function for boxed math answers."""

from __future__ import annotations

import re

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)


def extract_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def normalize_answer(text: str) -> str:
    text = str(text).strip().lower()
    for left, right in [("{", "}"), ("[", "]"), ("(", ")")]:
        if text.startswith(left) and text.endswith(right):
            text = text[1:-1].strip()
    return text.rstrip(".").replace(",", "").replace(" ", "")


def answers_match(predicted: str, ground_truth: str) -> bool:
    if normalize_answer(predicted) == normalize_answer(ground_truth):
        return True
    try:
        from sympy import N, simplify, sympify
        from sympy.parsing.latex import parse_latex

        def parse(expr: str):
            expr = expr.strip().replace("^", "**")
            if "\\" in expr:
                try:
                    return parse_latex(expr)
                except Exception:
                    pass
            return sympify(expr)

        a = parse(predicted)
        b = parse(ground_truth)
        try:
            if simplify(a - b) == 0:
                return True
        except Exception:
            pass
        return abs(float(N(a)) - float(N(b))) < 1e-4
    except Exception:
        return False


def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    *,
    response_str_list: list[str] | None = None,
    ground_truth_list: list[str] | None = None,
    data_sources=None,
    solution_strs: list[str] | None = None,
    ground_truths: list[str] | None = None,
    extra_infos=None,
    **kwargs,
) -> dict | list[dict]:
    """Reward function compatible with recent and legacy verl calls.

    Recent verl reward managers call:
    ``compute_score(data_source, solution_str, ground_truth, extra_info)``.
    Older/batch integrations may call ``compute_score(responses, truths)`` or
    pass ``solution_strs`` / ``ground_truths`` keyword lists.
    """
    if _looks_like_batch(data_source) and _looks_like_batch(solution_str) and ground_truth is None:
        response_str_list = list(data_source)
        ground_truth_list = list(solution_str)
        data_source = None
        solution_str = None

    responses = solution_strs if solution_strs is not None else response_str_list
    truths = ground_truths if ground_truths is not None else ground_truth_list
    if responses is not None or truths is not None:
        if responses is None or truths is None:
            raise ValueError("batch compute_score requires responses and ground truths")
        return [_score_one(response, truth) for response, truth in zip(responses, truths)]

    if solution_str is None or ground_truth is None:
        raise ValueError("compute_score requires solution_str and ground_truth")
    return _score_one(solution_str, ground_truth)


def _looks_like_batch(value) -> bool:
    return isinstance(value, (list, tuple))


def _score_one(response: str, ground_truth: str) -> dict:
    predicted = extract_boxed(response)
    correct = predicted is not None and answers_match(predicted, ground_truth)
    return {
        "score": 1.0 if correct else 0.0,
        "overall": 1.0 if correct else 0.0,
        "accuracy": 1.0 if correct else 0.0,
        "format": 1.0 if predicted is not None else 0.0,
    }
