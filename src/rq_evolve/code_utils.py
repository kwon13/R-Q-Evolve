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
    # ast.parse raises ValueError ("source code string cannot contain null
    # bytes"), NOT SyntaxError, on NUL bytes in a generation -- which the
    # except-SyntaxError guards below would not catch, crashing the whole run.
    # Strip NULs up front so a poisoned generation degrades to a normal
    # parse-failure (rejected candidate) instead of killing training.
    text = text.replace("\x00", "")

    # Fenced blocks in REVERSE document order -> last fence tried first.
    candidates: list[str] = [
        match.group(1).strip()
        for match in re.finditer(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    ][::-1]

    lines = text.splitlines()
    # Cap the import/def-suffix candidates: a degenerate output with hundreds of
    # "def generate"/"import" lines would otherwise spawn hundreds of candidates,
    # each running the (bounded but non-trivial) prefix scan below -> the evolution
    # phase wedges the main thread for many minutes. A real program has 1-2 entry
    # points, so the first few suffixes are enough.
    _MAX_SUFFIX_CANDIDATES = 16
    suffix_added = 0
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("import ", "from ", "def generate")):
            candidates.append("\n".join(lines[i:]).strip())
            suffix_added += 1
            if suffix_added >= _MAX_SUFFIX_CANDIDATES:
                break

    candidates.append(text.strip())

    for candidate in candidates:
        if "def generate" not in candidate:
            continue
        trimmed = _trim_to_parseable_prefix(candidate)
        if trimmed is not None:
            return trimmed
    return None


# Max ast.parse attempts per candidate. The prefix scan below is worst-case
# O(n^2) over lines; on a multi-thousand-line degenerate generation that wedges
# the (single-threaded) evolution phase for minutes. A genuine generate() program
# is small, so capping the attempts bounds the work without losing real code.
_TRIM_MAX_ATTEMPTS = 400


def _trim_to_parseable_prefix(code: str) -> str | None:
    """Longest line-prefix of ``code`` that parses AND defines ``generate``.

    Bounded: instead of trying every prefix length (O(n) parses), a SyntaxError
    jumps straight to just before the offending line, and the total attempts are
    capped at ``_TRIM_MAX_ATTEMPTS`` so a huge unparseable blob can never hang.
    """
    lines = code.splitlines()
    end = len(lines)
    attempts = 0
    while end > 0:
        attempts += 1
        if attempts > _TRIM_MAX_ATTEMPTS:
            return None
        snippet = "\n".join(lines[:end]).strip()
        try:
            tree = ast.parse(snippet)
        except (SyntaxError, ValueError) as exc:
            # Jump to just before the failing line (O(#error-points), not O(n));
            # ValueError (NUL bytes) has no lineno -> fall back to a single step.
            lineno = getattr(exc, "lineno", None)
            if isinstance(lineno, int) and 0 < lineno - 1 < end:
                end = lineno - 1
            else:
                end -= 1
            continue
        if any(
            isinstance(node, ast.FunctionDef) and node.name == "generate"
            for node in tree.body
        ):
            return snippet
        end -= 1
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
    except (SyntaxError, ValueError):
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
    except (SyntaxError, ValueError) as exc:
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


def lint_metacognitive_generator_source(
    source_code: str,
    *,
    require_assert: bool = True,
    reject_trivial_assert: bool = True,
    reject_unbounded_sampling: bool = True,
    require_answer_routes: bool = True,
) -> list[str]:
    """Extra static contract for generated mutation children.

    Schema-version 3 planned children use one executable ``answer`` route, so
    callers disable ``require_answer_routes`` and ``require_assert`` for them.
    Legacy mutation prompts can retain the historical independent
    ``answer_insight``/``answer_brute`` cross-check.
    """
    try:
        tree = ast.parse(source_code)
    except (SyntaxError, ValueError) as exc:
        return [f"syntax error: {exc}"]

    generate = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "generate"
        ),
        None,
    )
    if generate is None:
        return ["missing top-level generate function"]

    reasons: list[str] = []
    max_attempts = next(
        (
            node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "MAX_ATTEMPTS"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ),
        None,
    )
    if max_attempts != 200:
        reasons.append("generated mutation requires top-level MAX_ATTEMPTS = 200")

    assertions = [node for node in ast.walk(generate) if isinstance(node, ast.Assert)]
    if require_assert and not assertions:
        reasons.append("generated mutation requires an assert inside generate()")

    if reject_trivial_assert:
        for assertion in assertions:
            test = assertion.test
            if isinstance(test, ast.Compare) and len(test.comparators) == 1:
                left = ast.dump(test.left, include_attributes=False)
                right = ast.dump(test.comparators[0], include_attributes=False)
                if left == right:
                    reasons.append("trivial self-comparison assert")
                    break

    if reject_unbounded_sampling:
        for node in ast.walk(generate):
            if (
                isinstance(node, ast.While)
                and isinstance(node.test, ast.Constant)
                and node.test.value is True
            ):
                reasons.append("generated mutation may not use while True")
                break
        has_bounded_sampler = any(
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and any(
                isinstance(arg, ast.Name) and arg.id == "MAX_ATTEMPTS"
                for arg in node.iter.args
            )
            and any(
                isinstance(child, ast.Raise)
                and isinstance(child.exc, ast.Call)
                and isinstance(child.exc.func, ast.Name)
                and child.exc.func.id == "RuntimeError"
                for branch in node.orelse
                for child in ast.walk(branch)
            )
            for node in ast.walk(generate)
        )
        if not has_bounded_sampler:
            reasons.append(
                "generated mutation requires `for ... in range(MAX_ATTEMPTS)` "
                "with an exhaustion else clause that raises RuntimeError"
            )

    if require_answer_routes:
        assigned_names = {
            target.id
            for node in ast.walk(generate)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                list(node.targets)
                if isinstance(node, ast.Assign)
                else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        for route_name in ("answer_insight", "answer_brute"):
            if route_name not in assigned_names:
                reasons.append(f"generated mutation must assign `{route_name}`")

        route_assignments: dict[str, list[ast.AST]] = {
            "answer_insight": [],
            "answer_brute": [],
        }
        for node in ast.walk(generate):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                list(node.targets)
                if isinstance(node, ast.Assign)
                else [node.target]
            )
            for target in targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id in route_assignments
                ):
                    route_assignments[target.id].append(node.value)

        # Catch the common false cross-check where both route variables are
        # aliases for one precomputed value. Restrict identical-RHS rejection
        # to the single-assignment case so independent accumulators that both
        # initialize at zero remain valid.
        insight_values = route_assignments["answer_insight"]
        brute_values = route_assignments["answer_brute"]
        if len(insight_values) == 1 and len(brute_values) == 1:
            if ast.dump(
                insight_values[0],
                include_attributes=False,
            ) == ast.dump(
                brute_values[0],
                include_attributes=False,
            ):
                reasons.append(
                    "answer_insight and answer_brute have identical assignments"
                )
        for route_name, other_name in (
            ("answer_insight", "answer_brute"),
            ("answer_brute", "answer_insight"),
        ):
            if any(
                isinstance(child, ast.Name) and child.id == other_name
                for value in route_assignments[route_name]
                for child in ast.walk(value)
            ):
                reasons.append(
                    f"`{route_name}` may not be computed from `{other_name}`"
                )
                break

        has_route_equivalence_assert = any(
            isinstance(assertion.test, ast.Compare)
            and any(isinstance(op, ast.Eq) for op in assertion.test.ops)
            and {
                part.id
                for part in [
                    assertion.test.left,
                    *assertion.test.comparators,
                ]
                if isinstance(part, ast.Name)
            }
            >= {"answer_insight", "answer_brute"}
            for assertion in assertions
        )
        if not has_route_equivalence_assert:
            reasons.append(
                "generated mutation must assert "
                "`answer_insight == answer_brute`"
            )

    # Randomness must flow through the seed-local rng. Creating Random(seed) is
    # required, and direct module-level random calls inside generate are rejected.
    has_local_rng = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "rng" for t in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "random"
        and node.value.func.attr == "Random"
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "seed"
        for node in ast.walk(generate)
    )
    if not has_local_rng:
        reasons.append("generated mutation requires `rng = random.Random(seed)`")
    for node in ast.walk(generate):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "random"
            and node.func.attr != "Random"
        ):
            reasons.append(f"direct random.{node.func.attr} call; use rng instead")
            break

    for node in ast.walk(generate):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Attribute)
            and node.iter.func.attr in {"items", "keys", "values"}
        ):
            reasons.append(
                "dict/set-derived iteration must be wrapped in sorted() "
                "for cross-process determinism"
            )
            break

    returns = [node for node in ast.walk(generate) if isinstance(node, ast.Return)]
    has_integer_serialization = any(
        isinstance(node.value, ast.Tuple)
        and len(node.value.elts) == 2
        and isinstance(node.value.elts[1], ast.Call)
        and isinstance(node.value.elts[1].func, ast.Name)
        and node.value.elts[1].func.id == "str"
        and len(node.value.elts[1].args) == 1
        and isinstance(node.value.elts[1].args[0], ast.Call)
        and isinstance(node.value.elts[1].args[0].func, ast.Attribute)
        and isinstance(node.value.elts[1].args[0].func.value, ast.Name)
        and node.value.elts[1].args[0].func.value.id == "sympy"
        and node.value.elts[1].args[0].func.attr == "Integer"
        for node in returns
    )
    if not has_integer_serialization:
        reasons.append(
            "generated mutation must return "
            "`problem, str(sympy.Integer(answer))`"
        )

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

    # 1) repr / object leakage
    if re.search(r"<(?:function|built-in|class|bound method|module)\b", problem):
        reasons.append("object repr leaked into problem text")
    if re.search(r"0x[0-9a-fA-F]{6,}", problem):
        reasons.append("memory address leaked into problem text")

    # 2) literal answer appears in the problem text
    if _answer_leaks_into_problem(answer, problem):
        reasons.append("answer leaked into problem text")

    # 3) answer disguised as a variable assignment
    if _answer_leaks_as_assignment(answer, problem):
        reasons.append("answer leaked via variable assignment")

    # 4) soft concatenation cue: marker + multiple imperatives
    concat_markers = (
        "additionally", "now consider", "now, consider",
        "find the value of x in the following",
        "compute the sum of the first", "also compute", "and then calculate",
    )
    hits = [m for m in concat_markers if m in lowered]
    if len(hits) >= 1 and _looks_multi_answer(problem):
        reasons.append(f"possible concatenation: {hits}")

    # 4b) strong concatenation markers
    if re.search(
        r"\b(also compute|and then (?:compute|calculate)|"
        r"separately(?: compute)?|total sum of all parts)\b",
        problem, re.IGNORECASE,
    ):
        reasons.append("explicit concatenation marker")

    # 5) self-contradictory numeric range
    for lo, var, hi in re.findall(
        r"(\d+)\s*<\s*([A-Za-z]\w*)\s*<\s*(\d+)", problem
    ):
        if int(lo) >= int(hi):
            reasons.append(f"contradictory range: {lo} < {var} < {hi}")

    # 6) intermediate computed-value leak
    if re.search(r"\bwhich (?:is|equals|gives)\s+-?\d{2,}", problem, re.IGNORECASE) or \
       re.search(r"\bsum\b[^.]{0,40}\bis\s+-?\d{3,}", problem, re.IGNORECASE):
        reasons.append("intermediate result leaked into problem text")

    # 7) pre-computed data dump
    if len(re.findall(r"[=:]\s*-?\d{4,}\b", problem)) >= 2:
        reasons.append("pre-computed data dump in problem text")

    # 8) malformed / nested LaTeX delimiters
    if re.search(r"\\\([^)]*\\\(", problem):
        reasons.append("malformed/nested LaTeX delimiters")

    # 9) structural multi-problem: two or more independent questions, even with
    #    NO concatenation marker (catches the marker-free stapling that 4/4b miss,
    #    e.g. a number-theory paragraph followed by a committee paragraph).
    if _counts_independent_questions(problem) >= 2:
        reasons.append("multiple independent questions in problem text")

    return reasons


def _counts_independent_questions(problem: str) -> int:
    """Count distinct answer-demanding questions.

    A single multi-step problem normally issues ONE final demand ("find X").
    Two or more separate demands — each phrased as its own question — is the
    signature of a stapled multi-problem, regardless of any linking word.
    """
    # explicit question marks
    q_marks = problem.count("?")
    # imperative answer-demands ("how many", "find", "compute", "solve for",
    # "determine", "calculate"), counted as occurrences
    demands = re.findall(
        r"\b(how many|find|compute|calculate|determine|solve for|"
        r"evaluate)\b",
        problem, re.IGNORECASE,
    )
    # Question marks are a clean signal (a single problem normally has one "?").
    # Demand verbs are NOISY: a legitimate multi-step problem routinely chains
    # two of them on one object -- "compute ... find ... mod 1000", "find all x
    # ... and sum them" -- which are exactly the high-RQ AIME-style problems we
    # must not reject. So count demands as multiple questions only at 3+ (i.e.
    # demands - 1), while 2+ question marks alone still flags a stapled pair.
    return max(q_marks, len(demands) - 1)


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
