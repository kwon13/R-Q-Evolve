"""verl-compatible reward function for boxed math answers."""

from __future__ import annotations


def extract_boxed(text: str) -> str | None:
    r"""Return the content of the LAST complete ``\boxed{...}`` in ``text``.

    Brace-matched (not regex) so arbitrarily nested answers extract correctly --
    e.g. ``\boxed{\frac{\sqrt{3}}{2}}`` -> ``\frac{\sqrt{3}}{2}``. The old
    single-level regex returned None for any answer with 2+ nested brace groups,
    which silently scored common MATH-500 answers as wrong.
    """
    text = text or ""
    token = r"\boxed"
    result: str | None = None
    pos = 0
    while True:
        idx = text.find(token, pos)
        if idx == -1:
            break
        i = idx + len(token)
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text) and text[i] == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        result = text[i + 1 : j].strip()
                        break
        pos = idx + len(token)
    return result


def normalize_answer(text: str) -> str:
    text = str(text).strip().lower()
    for left, right in [("{", "}"), ("[", "]"), ("(", ")")]:
        if text.startswith(left) and text.endswith(right):
            text = text[1:-1].strip()
    return text.rstrip(".").replace(",", "").replace(" ", "")


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Answer equality via ``math_verify`` (R-Zero parity), no fallback.

    ``math_verify`` is a hard dependency: the import is intentionally outside the
    try/except so a missing install fails loud instead of silently degrading to a
    weaker grader. Parse/verify failures on a given pair count as a non-match.
    """
    from math_verify import parse, verify

    try:
        # Wrap both sides in \boxed{} so math_verify's LaTeX extraction triggers.
        # Parsing a bare fragment (e.g. "\dfrac{1}{2}" or "\frac34") otherwise
        # fails its extractor and reports a false non-match; the \boxed form makes
        # \dfrac/\frac, fraction/decimal, and spacing equivalences all resolve.
        # verify(gold, target): ground truth first, prediction second.
        gold = parse("\\boxed{" + str(ground_truth) + "}")
        pred = parse("\\boxed{" + str(predicted) + "}")
        return bool(verify(gold, pred))
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
