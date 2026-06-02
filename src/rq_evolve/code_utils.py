import ast
import re

from .program import ALLOWED_IMPORT_ROOTS, ProblemInstance

FORBIDDEN_SOURCE_PATTERNS = (
    "open(",
    "input(",
    "eval(",
    "exec(",
    "__import__",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "os.",
    "sys.",
)


def extract_generator_code(text: str) -> str | None:
    """Extract the best parseable Python block containing ``generate``.

    Prefers the LAST parseable fenced block: when the model thinks first and a
    draft ``generate`` appears in a ```python``` fence inside <think>, the final
    program (the last fence) wins instead of the draft. Falls back to an
    import/def scan and the whole text when no fenced block parses.
    """
    # Fenced blocks in REVERSE document order -> last fence tried first.
    candidates: list[str] = [
        match.group(1).strip()
        for match in re.finditer(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    ][::-1]

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("import ", "from ", "def generate")):
            candidates.append("\n".join(lines[i:]).strip())

    candidates.append(text.strip())

    for candidate in candidates:
        if "def generate" not in candidate:
            continue
        trimmed = _trim_to_parseable_prefix(candidate)
        if trimmed is not None:
            return trimmed
    return None


def _trim_to_parseable_prefix(code: str) -> str | None:
    lines = code.splitlines()
    for end in range(len(lines), 0, -1):
        snippet = "\n".join(lines[:end]).strip()
        try:
            tree = ast.parse(snippet)
        except SyntaxError:
            continue
        if any(
            isinstance(node, ast.FunctionDef) and node.name == "generate"
            for node in tree.body
        ):
            return snippet
    return None


def strip_module_docstring(source_code: str) -> str:
    """Drop top-level string-literal statements from a parent generator.

    Used to clean a parent's source before injecting it into a mutation prompt:
    the module docstring (and any stray top-level prose narrative) is the
    anchor the LLM tends to imitate as an output template. Everything else --
    imports, ``generate``, the CONCEPT_* constants, comments, formatting -- is
    preserved verbatim (line-based removal, not ``ast.unparse``). Returns the
    source unchanged if it does not parse. The child is still asked to write its
    own docstring; only the *parent shown in the prompt* is stripped.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return source_code

    drop_lines: set[int] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            end = node.end_lineno or node.lineno
            drop_lines.update(range(node.lineno, end + 1))
    if not drop_lines:
        return source_code

    kept = [
        line
        for i, line in enumerate(source_code.splitlines(), start=1)
        if i not in drop_lines
    ]
    return "\n".join(kept).strip()


def lint_generator_source(source_code: str) -> list[str]:
    """Cheap static checks before executing a generated program."""
    reasons: list[str] = []
    lowered = source_code.lower()
    for pattern in FORBIDDEN_SOURCE_PATTERNS:
        if pattern in lowered:
            reasons.append(f"forbidden source pattern: {pattern}")

    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]

    if not any(
        isinstance(node, ast.FunctionDef) and node.name == "generate"
        for node in tree.body
    ):
        reasons.append("missing top-level generate function")

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    reasons.append(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                reasons.append(f"disallowed import: {node.module}")
        elif isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign)):
            continue
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        else:
            reasons.append(f"top-level executable statement: {type(node).__name__}")

    return reasons


def lint_problem_instance(instance: ProblemInstance) -> list[str]:
    """Reject obviously poor training examples."""
    reasons: list[str] = []
    problem = instance.problem.strip()
    answer = instance.answer.strip()

    if len(problem) < 10:
        reasons.append("problem text too short")
    if not answer:
        reasons.append("empty answer")
    if any(token in answer.lower() for token in ("nan", "inf", "undefined")):
        reasons.append("non-finite answer")
    if "," in answer or ";" in answer:
        reasons.append("multi-part answer")

    lowered = problem.lower()

    # 1) repr / object leakage — f-string formatted a function or object
    #    e.g. "<function lcm at 0x...>", "<built-in ...>", "0x7f..."
    if re.search(r"<(?:function|built-in|class|bound method|module)\b", problem):
        reasons.append("object repr leaked into problem text")
    if re.search(r"0x[0-9a-fA-F]{6,}", problem):
        reasons.append("memory address leaked into problem text")

    # 2) the literal answer appears in the problem text (answer leakage)
    #    guard against trivial short answers to avoid false positives
    if _answer_leaks_into_problem(answer, problem):
        reasons.append("answer leaked into problem text")

    # 3) answer disguised as a variable assignment, e.g. "z = 64" at the end
    if _answer_leaks_as_assignment(answer, problem):
        reasons.append("answer leaked via variable assignment")

    # 4) multi-problem concatenation cues (soft: marker + multiple imperatives)
    concat_markers = (
        "additionally", "now consider", "now, consider",
        "find the value of x in the following",
        "compute the sum of the first", "also compute", "and then calculate",
    )
    hits = [m for m in concat_markers if m in lowered]
    if len(hits) >= 1 and _looks_multi_answer(problem):
        reasons.append(f"possible concatenation: {hits}")

    # 4b) strong concatenation markers — these phrases essentially never occur
    #     in a single-answer competition problem, so flag them on their own
    #     (catches single-verb staplings the soft rule above misses).
    if re.search(
        r"\b(also compute|and then (?:compute|calculate)|"
        r"separately(?: compute)?|total sum of all parts)\b",
        problem, re.IGNORECASE,
    ):
        reasons.append("explicit concatenation marker")

    # 5) self-contradictory numeric range like "216 < N < 8"
    for lo, var, hi in re.findall(
        r"(\d+)\s*<\s*([A-Za-z]\w*)\s*<\s*(\d+)", problem
    ):
        if int(lo) >= int(hi):
            reasons.append(f"contradictory range: {lo} < {var} < {hi}")

    # 6) intermediate computed-value leak — an appositive that hands the solver
    #    a sub-result it was supposed to compute, e.g. "..., which is 200" or
    #    "the sum of the first 9 terms ... is 2268".
    if re.search(r"\bwhich (?:is|equals|gives)\s+-?\d{2,}", problem, re.IGNORECASE) or \
       re.search(r"\bsum\b[^.]{0,40}\bis\s+-?\d{3,}", problem, re.IGNORECASE):
        reasons.append("intermediate result leaked into problem text")

    # 7) pre-computed data dump — two or more "<expr> = <bignum>" / "label: <bignum>"
    #    facts (4+ digit RHS so genuine small givens like "PA = 9" don't trip it).
    if len(re.findall(r"[=:]\s*-?\d{4,}\b", problem)) >= 2:
        reasons.append("pre-computed data dump in problem text")

    # 8) malformed / nested LaTeX delimiters, e.g. "\( PA = 9, AB = 21, and \( PC"
    if re.search(r"\\\([^)]*\\\(", problem):
        reasons.append("malformed/nested LaTeX delimiters")

    return reasons


def _answer_leaks_into_problem(answer: str, problem: str) -> bool:
    """True if the exact answer value shows up in the problem body."""
    a = answer.strip()
    # skip very short answers (0-9, single char) — too many false positives
    if len(a) <= 2:
        return False
    # match the answer as a standalone token (not a substring of a longer number)
    pattern = r"(?<![\d.])" + re.escape(a) + r"(?![\d.])"
    return re.search(pattern, problem) is not None


def _looks_multi_answer(problem: str) -> bool:
    """Heuristic: more than one imperative 'compute/find' verb suggests
    independent subproblems."""
    verbs = re.findall(
        r"\b(compute|find|calculate|determine|evaluate|how many)\b",
        problem, re.IGNORECASE,
    )
    return len(verbs) >= 2


def _answer_leaks_as_assignment(answer: str, problem: str) -> bool:
    """True if the answer appears as a bare 'var = <number>' assignment.

    Catches the pattern where the model writes a chain like
        x = 8^10
        y = x + 8^10
        z = y + 8^10
        z = 1073741824
    disguising the final answer as another equation line.
    """
    a = answer.strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", a):  # only for numeric answers
        return False
    # any line of the form  <identifier> = <pure number literal>
    # where that number equals the answer
    for m in re.finditer(
        r"(?m)^\s*[A-Za-z_]\w*\s*=\s*(-?\d+(?:\.\d+)?)\s*$", problem
    ):
        if m.group(1) == a:
            return True
    return False