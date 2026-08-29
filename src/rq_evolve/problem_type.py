"""Deterministic annotation for the computational problem-type axis.

The five labels come from the standard computational-problem distinction:
decision, search, counting, optimization, and function.  This module does not
pretend that surface rules are a semantic oracle.  It deliberately abstains on
proof prompts, damaged statements, and requests without a sufficiently clear
output contract.  The live pipeline additionally cross-checks the inferred
type against the declarative verifier and requires the same result on every
verification seed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from .concepts import PROBLEM_TYPES


PROBLEM_TYPE_RULESET = "computational-output-contract-v1"


def problem_type_ruleset_sha256() -> str:
    """Hash the exact deterministic rules used to assign archive rows.

    Archive snapshots pin this value.  A rules change therefore cannot silently
    reinterpret an old MAP under new cell semantics.
    """

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ProblemTypeAnnotation:
    """One conservative, statement-only annotation result."""

    problem_type: str | None
    confidence: str
    evidence: str
    request_window: str
    review_reason: str | None = None

    @property
    def needs_review(self) -> bool:
        return self.problem_type is None


_SPACE_RE = re.compile(r"\s+")
_AUTHOR_TRAILER_RE = re.compile(r"\s*\[i\].*?\[/i\]\s*$", re.IGNORECASE | re.DOTALL)
_PROOF_RE = re.compile(
    r"(?:^|[.?!;:]\s*)(?:prove|show|demonstrate|establish)\b(?:\s+that)?",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(?:(?:determine|decide)\s+whether|prove\s+or\s+disprove|"
    r"answer\s+yes\s+or\s+no)\b|"
    r"(?:^|[.?!;:]\s*)(?:is|are|does|do|can|must)\b[^?]{0,500}\?",
    re.IGNORECASE,
)
_OPTIMIZATION_RE = re.compile(
    r"\b(?:find|determine|compute|calculate|evaluate|what\s+(?:is|are))\s+"
    r"(?:the\s+)?(?:maximum|minimum|largest|smallest|"
    r"greatest(?!\s+common\s+(?:divisor|factor))|"
    r"least(?!\s+common\s+multiple)|optimal|"
    r"best\s+possible)\b|"
    r"\b(?:maximize|minimize)\b|"
    r"\bafter\s+what\s+(?:least|fewest|minimum)\b",
    re.IGNORECASE,
)
_COUNTING_RE = re.compile(
    r"\bhow\s+many\b|"
    r"\b(?:determine|find|compute|calculate|what\s+(?:is|are))\s+"
    r"(?:the\s+)?number\s+of\b|"
    r"\b(?:determine|find|compute|calculate)\s+(?:the\s+)?cardinality\s+of\b|"
    r"(?:^|[.?!;:]\s*)count\s+the\b",
    re.IGNORECASE,
)
_SEARCH_RE = re.compile(
    r"\b(?:find|determine)\s+all\b|"
    r"\bfor\s+(?:what|which)\b|"
    r"\b(?:construct|exhibit)\b|"
    r"\bgive\s+(?:an?\s+)?(?:example|construction)\b|"
    r"\b(?:find|determine)\s+(?:an?|the)\s+[^.?!]{0,180}?\bsuch\s+that\b|"
    r"\bsolve\s+(?:the\s+)?(?:equation|system|congruence)\b",
    re.IGNORECASE,
)
_FUNCTION_RE = re.compile(
    r"\b(?:find|determine|compute|calculate)\s+(?:the\s+)?(?:greatest\s+common\s+"
    r"(?:divisor|factor)|least\s+common\s+multiple)\b|"
    r"\b(?:compute|calculate|evaluate)\b|\bwhat\s+(?:is|are)\b|"
    r"\b(?:find|determine)\s+(?:the\s+)?(?:value|values|sum|product|remainder|"
    r"residue|digit|digits|perimeter|area|volume|length|distance|ratio|"
    r"probability|coefficient|degree|order|measure)\b",
    re.IGNORECASE,
)
_GENERIC_REQUEST_RE = re.compile(r"\b(?:find|determine|give)\b", re.IGNORECASE)


def _request_window(statement: str, *, max_chars: int = 900) -> str:
    """Return a bounded normalized tail for classification and audit evidence.

    We intentionally do not cut at the *last* request-looking verb.  In
    ``Find all x such that, after you compute x^2, ...`` that would discard the
    load-bearing ``Find all`` and manufacture a function problem.
    """

    text = _AUTHOR_TRAILER_RE.sub("", statement or "")
    text = _SPACE_RE.sub(" ", text).strip()
    return text[-max_chars:]


def annotate_problem_type(statement: str) -> ProblemTypeAnnotation:
    """Conservatively infer the requested output contract from a problem.

    Precedence follows the most specific output semantics.  For example,
    ``find the smallest n`` is optimization rather than generic search, and
    ``does a maximum exist?`` is decision rather than optimization.
    """

    request = _request_window(statement)
    if not request:
        return ProblemTypeAnnotation(
            None, "none", "", request, "empty_statement"
        )

    # Proof production is not one of the five exact-answer contracts.  Do not
    # relabel it as decision merely because the proposition has a truth value.
    proof = _PROOF_RE.search(request)
    if proof:
        return ProblemTypeAnnotation(
            None, "none", proof.group(0), request, "proof_or_justification"
        )

    decision = _DECISION_RE.search(request)
    if decision:
        return ProblemTypeAnnotation(
            "decision", "high", decision.group(0), request
        )

    optimization = _OPTIMIZATION_RE.search(request)
    if optimization:
        return ProblemTypeAnnotation(
            "optimization", "high", optimization.group(0), request
        )

    counting = _COUNTING_RE.search(request)
    if counting:
        return ProblemTypeAnnotation(
            "counting", "high", counting.group(0), request
        )

    # Optimization precedes generic search. In particular, "find the smallest
    # n such that ..." requests an extremum, not an arbitrary witness.
    search = _SEARCH_RE.search(request)
    if search:
        return ProblemTypeAnnotation("search", "high", search.group(0), request)

    function = _FUNCTION_RE.search(request)
    if function:
        return ProblemTypeAnnotation(
            "function", "high", function.group(0), request
        )

    generic = _GENERIC_REQUEST_RE.search(request)
    if generic:
        return ProblemTypeAnnotation(
            None,
            "none",
            generic.group(0),
            request,
            "generic_find_or_determine",
        )

    return ProblemTypeAnnotation(
        None, "none", "", request, "no_output_contract_cue"
    )


def problem_type_contract_errors(
    annotation: ProblemTypeAnnotation,
    verifier: dict,
    answer: str,
) -> list[str]:
    """Cross-check a statement annotation against its grading contract.

    Verifier mode alone cannot separate counting, optimization, and function;
    those all return a scalar.  It is still a useful independent constraint:
    Boolean output must be a decision and an automatically graded search must
    return the complete finite set, not an untrusted witness predicate.
    """

    label = annotation.problem_type
    if label not in PROBLEM_TYPES or annotation.confidence != "high":
        return [
            "problem type is ambiguous under the deterministic rules"
            + (
                f": {annotation.review_reason}"
                if annotation.review_reason
                else ""
            )
        ]

    mode = str((verifier or {}).get("mode", ""))
    allowed_modes = {
        "decision": {"boolean"},
        "search": {"set"},
        "counting": {"expression"},
        "optimization": {"expression"},
        "function": {"expression"},
    }
    if mode not in allowed_modes[label]:
        return [
            f"problem type {label!r} is incompatible with verifier mode "
            f"{mode!r}; expected {sorted(allowed_modes[label])}"
        ]

    if label == "decision" and str(answer).strip().lower() not in {"yes", "no"}:
        return ["decision answer must be Yes or No"]
    if label == "counting":
        token = str(answer).strip()
        if re.fullmatch(r"\+?\d+", token) is None:
            return ["counting answer must be a nonnegative integer"]
    return []


def top_level_domains(domain_paths: object) -> tuple[str, ...]:
    """Extract unique Omni-MATH top-level domains from its path list."""

    if not isinstance(domain_paths, list):
        return ()
    domains: list[str] = []
    for path in domain_paths:
        if not isinstance(path, str):
            continue
        parts = [part.strip() for part in path.split("->")]
        if len(parts) < 2 or parts[0] != "Mathematics":
            continue
        domain = parts[1]
        if domain == "Other" or domain in domains:
            continue
        domains.append(domain)
    return tuple(domains)


def integer_answer(answer: object) -> bool:
    """Whether the reference answer already satisfies the current integer gate."""

    return isinstance(answer, str) and re.fullmatch(r"-?\d+", answer.strip()) is not None
