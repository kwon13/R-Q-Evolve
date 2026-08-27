"""Declarative, JSON-safe answer-verifier contracts.

Generated programs are untrusted.  They may describe *what* constitutes an
answer, but they must never supply executable grading code.  The small schemas
in this module are therefore data only:

``{"mode": "expression"}``
    One reference expression, graded with :mod:`math_verify`.

``{"mode": "boolean"}``
    A canonical truth value.  The separate reference answer must be one of
    yes/no, true/false, or 1/0 (LaTeX ``\\text{Yes}`` is also accepted).

``{"mode": "one_of", "answers": ["...", ...]}``
    Any one complete answer in the finite list is valid.  The human-readable
    reference answer remains a separate string and must occur in the list.

``{"mode": "set", "elements": ["...", ...]}``
    The prediction must contain exactly these elements, ignoring order.

The schemas deliberately exclude arbitrary predicates.  A generated
``lambda``/function would let a child certify every solver response as correct
and would also cross the generator sandbox's trust boundary.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


VERIFIER_MODES = ("expression", "boolean", "one_of", "set")
# Keep a whole finite-verifier request comfortably inside the grader worker's
# single hard-kill budget.  This is a curriculum exact-answer contract, not a
# bulk theorem-prover interchange format.
MAX_VERIFIER_ITEMS = 32
MAX_VERIFIER_ATOM_CHARS = 512
_MODE_KEYS = {
    "expression": frozenset({"mode"}),
    "boolean": frozenset({"mode"}),
    "one_of": frozenset({"mode", "answers"}),
    "set": frozenset({"mode", "elements"}),
}


def default_verifier() -> dict[str, str]:
    """Return a fresh legacy-compatible expression verifier."""

    return {"mode": "expression"}


def _answer_atom(value: Any, *, field: str) -> str:
    """Canonicalize one JSON scalar used as an expected answer."""

    if isinstance(value, bool) or value is None:
        raise ValueError(f"verifier {field} entries must be strings or numbers")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"verifier {field} entries must be finite")
    if not isinstance(value, (str, int, float)):
        raise ValueError(f"verifier {field} entries must be strings or numbers")
    text = str(value).strip()
    if not text:
        raise ValueError(f"verifier {field} entries must not be empty")
    if len(text) > MAX_VERIFIER_ATOM_CHARS:
        raise ValueError(
            f"verifier {field} entries must be <= {MAX_VERIFIER_ATOM_CHARS} characters"
        )
    return text


def _answer_list(value: Any, *, field: str, allow_empty: bool) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"verifier {field} must be a JSON array")
    result = [_answer_atom(item, field=field) for item in value]
    if len(result) > MAX_VERIFIER_ITEMS:
        raise ValueError(
            f"verifier {field} must contain <= {MAX_VERIFIER_ITEMS} entries"
        )
    if not allow_empty and not result:
        raise ValueError(f"verifier {field} must contain at least one answer")
    if len(result) != len(set(result)):
        raise ValueError(f"verifier {field} must not contain duplicate entries")
    return result


def normalize_verifier(
    verifier: Mapping[str, Any] | None,
    *,
    answer: str | None = None,
) -> dict[str, Any]:
    """Validate and copy a verifier into its canonical JSON wire form.

    ``None`` is the backward-compatible spelling of expression equality.  All
    other mappings are fail-closed: unknown modes, extra keys, nested objects,
    callables, booleans-as-numbers, and non-finite numbers are rejected.
    Supplying ``answer`` additionally validates the separate reference string.
    """

    if verifier is None:
        resolved: dict[str, Any] = default_verifier()
    else:
        if not isinstance(verifier, Mapping):
            raise ValueError("verifier must be a JSON object")
        mode_value = verifier.get("mode")
        if not isinstance(mode_value, str):
            raise ValueError("verifier.mode must be a string")
        mode = mode_value.strip().lower()
        if mode not in VERIFIER_MODES:
            raise ValueError(
                f"unknown verifier mode {mode_value!r}; expected one of "
                + ", ".join(VERIFIER_MODES)
            )
        keys = set(verifier)
        unknown = keys - _MODE_KEYS[mode]
        missing = _MODE_KEYS[mode] - keys
        if unknown:
            raise ValueError(
                "unknown verifier field(s): " + ", ".join(sorted(map(str, unknown)))
            )
        if missing:
            raise ValueError(
                "missing verifier field(s): " + ", ".join(sorted(missing))
            )
        resolved = {"mode": mode}
        if mode == "one_of":
            resolved["answers"] = _answer_list(
                verifier["answers"], field="answers", allow_empty=False
            )
        elif mode == "set":
            # The empty set is a legitimate exact answer.
            resolved["elements"] = _answer_list(
                verifier["elements"], field="elements", allow_empty=True
            )

    if answer is not None:
        reference = str(answer).strip()
        if not reference:
            raise ValueError("reference answer must not be empty")
        mode = resolved["mode"]
        if mode == "boolean" and canonical_boolean(reference) is None:
            raise ValueError(
                "boolean reference answer must be yes/no, true/false, or 1/0"
            )
        if mode == "one_of" and reference not in resolved["answers"]:
            raise ValueError(
                "one_of verifier.answers must include the reference answer string"
            )
        if mode == "set" and not _set_reference_matches_elements(
            reference, resolved["elements"]
        ):
            raise ValueError(
                "set reference answer must render exactly verifier.elements"
            )
    return resolved


def _strip_set_container(text: str) -> str | None:
    value = str(text).strip().replace(r"\left", "").replace(r"\right", "")
    if value in {r"\emptyset", r"\varnothing", "∅", "{}", r"\{\}", "[]"}:
        return ""
    for left, right in ((r"\{", r"\}"), ("{", "}"), ("[", "]")):
        if value.startswith(left) and value.endswith(right):
            return value[len(left) : len(value) - len(right)].strip()
    return None


def _split_set_elements(text: str) -> list[str] | None:
    if not text.strip():
        return []
    opening = {"{": "}", "[": "]", "(": ")"}
    closing = set(opening.values())
    stack: list[str] = []
    parts: list[str] = []
    start = 0
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
    final = text[start:].strip()
    if not final:
        return None
    parts.append(final)
    return parts


def _set_reference_matches_elements(reference: str, elements: list[str]) -> bool:
    """Cheap structural consistency check at the untrusted sandbox boundary.

    Symbolic equivalence is checked later by the grader.  Here we require the
    reference string to visibly contain the same declared element strings so a
    child cannot show one gold answer to audit code and reward against another.
    """

    inside = _strip_set_container(reference)
    rendered = None if inside is None else _split_set_elements(inside)
    if rendered is None or len(rendered) != len(elements):
        return False
    normalize = lambda value: re.sub(r"\s+", "", value)
    return sorted(map(normalize, rendered)) == sorted(map(normalize, elements))


_LATEX_TEXT = re.compile(
    r"^\\(?:text|mathrm|mathbf|operatorname)\s*\{(.*)\}$", re.IGNORECASE
)


def canonical_boolean(value: Any) -> bool | None:
    """Return the truth value represented by a conservative answer spelling."""

    text = ("" if value is None else str(value)).strip()
    # Be useful outside the reward path too, where callers may not have removed
    # the box/math delimiters yet.
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    if text.startswith(r"\boxed"):
        tail = text[len(r"\boxed") :].strip()
        if tail.startswith("{") and tail.endswith("}"):
            text = tail[1:-1].strip()
    match = _LATEX_TEXT.fullmatch(text)
    if match:
        text = match.group(1).strip()
    while len(text) >= 2 and text[0] == "{" and text[-1] == "}":
        text = text[1:-1].strip()
    # Ignore presentation around the token, not characters *inside* it: inputs
    # such as ``tr ue`` or ``y.e.s`` are not canonical truth values.
    token = text.strip().lower().rstrip(".! ")
    if token in {"yes", "true", "1"}:
        return True
    if token in {"no", "false", "0"}:
        return False
    return None


def verifier_mode(verifier: Mapping[str, Any] | None) -> str:
    """Convenience accessor that validates instead of guessing."""

    return str(normalize_verifier(verifier)["mode"])
