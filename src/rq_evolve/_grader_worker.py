"""Hermetic answer grader: one request per stdin line, one verdict per stdout line.

Runs `math_verify` in a process the parent can KILL. The in-process watchdog it
replaces could not: `reward.py` ran the comparison in a daemon thread and gave up
on it after a timeout, but Python cannot stop a thread, so the abandoned one kept
burning a core for as long as the process lived. The trigger found in the wild is
four characters:

    parse("\\boxed{51!!}") -> factorial(factorial(51))

`51!` is a 67-digit number and sympy then tries to take ITS factorial. That never
finishes, and no length guard can catch it -- the input is 4 bytes with unbounded
cost. A separate process can simply be SIGKILLed.

Grading runs on this worker's MAIN thread, so `math_verify`'s own SIGALRM budget
works natively here and stops anything that yields to the interpreter. The
parent's kill is the second layer, for work that never yields at all.
"""
import json
import sys

from verifier import canonical_boolean, normalize_verifier


def _parse_expression(value: str):
    from math_verify import parse

    return parse("\\boxed{" + str(value) + "}")


def _parsed_equal(predicted, expected) -> bool:
    from math_verify import verify

    return bool(verify(expected, predicted))


def _grade_expression(pred: str, gold: str) -> bool:
    from math_verify import parse, verify

    # \boxed-wrapped on both sides, matching reward.answers_match: a bare
    # fragment ("\dfrac{1}{2}", "\frac34") misses math_verify's extractor and
    # reports a false non-match.
    g = parse("\\boxed{" + str(gold) + "}")
    p = parse("\\boxed{" + str(pred) + "}")
    return bool(verify(g, p))


def _strip_collection_container(text: str) -> str | None:
    """Return the inside of a finite-set/list rendering.

    A bare value or comma-separated sequence is also accepted because the
    surrounding ``\\boxed{...}`` is already the output-format delimiter.  Round
    parentheses are intentionally *not* stripped: ``(1, 2)`` may be one ordered
    pair element rather than a two-element set.
    """

    value = str(text).strip().replace(r"\left", "").replace(r"\right", "")
    if value in {r"\emptyset", r"\varnothing", "∅", "{}", r"\{\}", "[]"}:
        return ""
    for left, right in ((r"\{", r"\}"), ("{", "}"), ("[", "]")):
        if value.startswith(left) and value.endswith(right):
            return value[len(left) : len(value) - len(right)].strip()
    return value


def _split_top_level(text: str) -> list[str] | None:
    """Split commas/semicolons outside LaTeX braces and coordinate brackets."""

    if not text.strip():
        return []
    opening = {"{": "}", "[": "]", "(": ")"}
    closing = set(opening.values())
    stack: list[str] = []
    start = 0
    parts: list[str] = []
    for index, char in enumerate(text):
        if char in opening:
            stack.append(opening[char])
        elif char in closing:
            if not stack or stack[-1] != char:
                return None
            stack.pop()
        elif char in {",", ";"} and not stack:
            item = text[start:index].strip()
            if not item:
                return None
            parts.append(item)
            start = index + 1
    if stack:
        return None
    item = text[start:].strip()
    if not item:
        return None
    parts.append(item)
    return parts


def _perfect_expression_matching(predicted: list[str], expected: list[str]) -> bool:
    """Whether the two finite collections are equal modulo expression equality."""

    if len(predicted) != len(expected):
        return False
    parsed_predicted = [_parse_expression(value) for value in predicted]
    parsed_expected = [_parse_expression(value) for value in expected]

    # A mathematical set cannot contain the same element twice.  Reject a
    # malformed contract or prediction even when the duplicate has a different
    # surface form (e.g. ``1`` and ``2-1``).
    for values in (parsed_predicted, parsed_expected):
        for i in range(len(values)):
            for j in range(i):
                if _parsed_equal(values[i], values[j]):
                    return False
    edges = [
        [
            j
            for j, gold in enumerate(parsed_expected)
            if _parsed_equal(pred, gold)
        ]
        for pred in parsed_predicted
    ]
    matched_pred_for_gold = [-1] * len(expected)

    def augment(pred_index: int, seen: set[int]) -> bool:
        for gold_index in edges[pred_index]:
            if gold_index in seen:
                continue
            seen.add(gold_index)
            previous = matched_pred_for_gold[gold_index]
            if previous < 0 or augment(previous, seen):
                matched_pred_for_gold[gold_index] = pred_index
                return True
        return False

    return all(augment(i, set()) for i in range(len(predicted)))


def _grade(pred: str, gold: str, verifier=None) -> bool:
    spec = normalize_verifier(verifier, answer=str(gold))
    mode = spec["mode"]
    if mode == "expression":
        return _grade_expression(pred, gold)
    if mode == "boolean":
        predicted = canonical_boolean(pred)
        expected = canonical_boolean(gold)
        return predicted is not None and expected is not None and predicted == expected
    if mode == "one_of":
        parsed = _parse_expression(pred)
        return any(
            _parsed_equal(parsed, _parse_expression(answer))
            for answer in spec["answers"]
        )
    if mode == "set":
        inside = _strip_collection_container(pred)
        predicted = None if inside is None else _split_top_level(inside)
        if predicted is None:
            return False
        return _perfect_expression_matching(predicted, spec["elements"])
    return False  # normalize_verifier makes this unreachable; fail closed.


def _warm() -> None:
    """Pay the math_verify/sympy import (~1.5 s) before the first request."""
    try:
        from math_verify import parse  # noqa: F401
        _grade_expression("1", "1")
    except Exception:
        pass


def main() -> None:
    _warm()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            out = {
                "ok": True,
                "match": _grade(req["pred"], req["gold"], req.get("verifier")),
            }
        except Exception as exc:  # parse/verify failure grades as non-match
            out = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:150]}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
