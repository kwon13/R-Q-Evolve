import ast
import textwrap
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
        except (MemoryError, RecursionError):
            # A nesting bomb. Pure paren runs hit CPython's cheap "too many
            # nested parentheses" SyntaxError, but mixed nesting -- a base
            # model looping "[1," for thousands of tokens -- explodes the PEG
            # parser's arena and raises MemoryError with host RAM to spare.
            # It killed a training run at 2026-08-23 18:29: one degenerate
            # reply must cost one candidate, never the trainer. Shaving
            # trailing lines cannot defuse it (the nesting is within a line),
            # so bail out of the whole candidate.
            return None
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


def strip_label_declarations(source_code: str) -> str:
    """Delete the top-level GROUP / SKILL assignments from a parent generator.

    The parent shown in a mutation prompt must not carry its own cell. With the
    real labels visible, 97% of 118 distinct children declared the cell their
    parent already occupied, across only 12 distinct cells; hiding them cut that
    to 25%.

    Whatever sits in that tail is what the child copies, so there is no
    redaction that is safe on its own: `GROUP = "..."` produced children whose
    declared cell was literally `...`, and the skeleton placeholder produced
    `<one of the allowed GROUPS>`. Deleting the lines is the only form that
    cannot be copied -- but it removes the ending the contract requires, and on
    its own it drove "missing GROUP; missing SKILL" from 0 to 11 of 24. It is
    safe only because the system prompt now makes PART 1 commit to GROUP and
    SKILL before any code is written, so the two lines in the block are a
    transcription rather than a step the reply can reach the end without taking.

    Line-based, like :func:`strip_module_docstring`. Returns the source
    unchanged if it does not parse.
    """
    try:
        tree = ast.parse(source_code)
    except (SyntaxError, ValueError):
        return source_code

    drop: set[int] = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name) and target.id in ("GROUP", "SKILL")
            for target in targets
        ):
            drop.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    if not drop:
        return source_code

    kept = [
        line
        for i, line in enumerate(source_code.splitlines(), start=1)
        if i not in drop
    ]
    return "\n".join(kept).rstrip()


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


def lint_mutation_generator_source(
    source_code: str,
    *,
    require_assert: bool = False,
    reject_trivial_assert: bool = True,
    reject_unbounded_sampling: bool = True,
    require_answer_routes: bool = False,
    require_canonical_instance_data: bool = False,
    require_mechanical_shape: bool = True,
) -> list[str]:
    """Extra static contract for model-generated mutation children.

    The standard contract uses one executable ``answer`` route. The historical
    independent ``answer_insight``/``answer_brute`` check remains available only
    when a caller explicitly enables both route-related flags.
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
    if require_mechanical_shape and max_attempts != 200:
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

    if require_canonical_instance_data:
        assignments: dict[str, list[ast.AST]] = {}
        assignment_nodes: dict[
            str,
            list[ast.Assign | ast.AnnAssign],
        ] = {}
        for node in ast.walk(generate):
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            else:
                continue
            if value is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(value)
                    assignment_nodes.setdefault(target.id, []).append(node)

        def expression_depends_on(
            expression: ast.AST,
            dependency: str,
            *,
            seen: frozenset[str] = frozenset(),
        ) -> bool:
            names = {
                child.id
                for child in ast.walk(expression)
                if isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
            }
            if dependency in names:
                return True
            for name in names - set(seen):
                if name == dependency:
                    return True
                for value in assignments.get(name, ()):
                    if expression_depends_on(
                        value,
                        dependency,
                        seen=seen | {name},
                    ):
                        return True
            return False

        canonical_nodes = assignment_nodes.get("instance_data", [])
        if len(canonical_nodes) != 1:
            reasons.append(
                "generated mutation must assign canonical `instance_data` "
                "exactly once after all sampled-object transformations"
            )
        else:
            canonical_line = canonical_nodes[0].lineno
            for name in ("answer", "problem"):
                if any(
                    node.lineno <= canonical_line
                    for node in assignment_nodes.get(name, ())
                ):
                    reasons.append(
                        f"generated mutation must assign `{name}` only after "
                        "canonical `instance_data`"
                    )

            if not any(
                expression_depends_on(value, "instance_data")
                for value in assignments.get("answer", ())
            ):
                reasons.append(
                    "generated mutation must compute `answer` from "
                    "`instance_data`"
                )
            if not any(
                expression_depends_on(value, "instance_data")
                for value in assignments.get("problem", ())
            ):
                reasons.append(
                    "generated mutation must render `problem` from "
                    "`instance_data`"
                )

            helper_names = {
                node.name
                for node in ast.walk(generate)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            helper_names.update(
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            harmless_names = {
                "abs",
                "enumerate",
                "format",
                "int",
                "len",
                "list",
                "map",
                "max",
                "min",
                "range",
                "repr",
                "sorted",
                "str",
                "sum",
                "tuple",
                "zip",
                "math",
                "sympy",
            }
            stale_problem_names: set[str] = set()
            for value in assignments.get("problem", ()):
                direct_names = {
                    child.id
                    for child in ast.walk(value)
                    if isinstance(child, ast.Name)
                    and isinstance(child.ctx, ast.Load)
                }
                for name in direct_names:
                    if (
                        name == "instance_data"
                        or name in helper_names
                        or name in harmless_names
                        or any(
                            expression_depends_on(
                                assigned_value,
                                "instance_data",
                            )
                            for assigned_value in assignments.get(name, ())
                        )
                    ):
                        continue
                    stale_problem_names.add(name)
            if stale_problem_names:
                reasons.append(
                    "generated mutation renders `problem` from names outside "
                    "canonical `instance_data`: "
                    + ", ".join(sorted(stale_problem_names))
                )

            has_semantic_consistency_assert = False
            answer_value_dumps = {
                ast.dump(value, include_attributes=False)
                for value in assignments.get("answer", ())
            }
            for assertion in assertions:
                if assertion.lineno <= canonical_line:
                    continue
                if not isinstance(assertion.test, ast.Compare):
                    continue
                direct_names = {
                    child.id
                    for child in ast.walk(assertion.test)
                    if isinstance(child, ast.Name)
                    and isinstance(child.ctx, ast.Load)
                }
                if "answer" not in direct_names:
                    continue
                other_names = direct_names - {"answer"}
                links_canonical_data = (
                    "instance_data" in other_names
                    or any(
                        any(
                            expression_depends_on(value, "instance_data")
                            for value in assignments.get(name, ())
                        )
                        for name in other_names
                    )
                )
                comparison_parts = [
                    assertion.test.left,
                    *assertion.test.comparators,
                ]
                non_answer_parts = [
                    part
                    for part in comparison_parts
                    if not (
                        isinstance(part, ast.Name)
                        and part.id == "answer"
                    )
                ]
                merely_repeats_answer_rhs = bool(non_answer_parts) and all(
                    ast.dump(part, include_attributes=False)
                    in answer_value_dumps
                    for part in non_answer_parts
                )
                if links_canonical_data and not merely_repeats_answer_rhs:
                    has_semantic_consistency_assert = True
                    break
            if not has_semantic_consistency_assert:
                reasons.append(
                    "generated mutation must assert a non-trivial semantic "
                    "consistency comparison linking `instance_data` and "
                    "`answer` without repeating the answer assignment"
                )

            def root_name(node: ast.AST) -> str | None:
                current = node
                while isinstance(current, (ast.Attribute, ast.Subscript)):
                    current = current.value
                return current.id if isinstance(current, ast.Name) else None

            mutating_methods = {
                "add",
                "append",
                "clear",
                "discard",
                "extend",
                "insert",
                "pop",
                "remove",
                "reverse",
                "setdefault",
                "sort",
                "update",
            }
            canonical_source_names = {
                child.id
                for value in assignments.get("instance_data", ())
                for child in ast.walk(value)
                if isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
                and child.id in assignments
            }
            protected_roots = {"instance_data", *canonical_source_names}
            mutated_after_canonical = any(
                (
                    isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                    and any(
                        root_name(target) in protected_roots
                        and (
                            not isinstance(target, ast.Name)
                            or target.id != "instance_data"
                        )
                        for target in (
                            node.targets
                            if isinstance(node, ast.Assign)
                            else [node.target]
                        )
                    )
                )
                or (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and root_name(node.func.value) in protected_roots
                    and node.func.attr in mutating_methods
                )
                for node in ast.walk(generate)
                if getattr(node, "lineno", 0) > canonical_line
            )
            if mutated_after_canonical:
                reasons.append(
                    "generated mutation may not mutate `instance_data` after "
                    "canonicalization"
                )

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
    if require_mechanical_shape and not has_integer_serialization:
        reasons.append(
            "generated mutation must return "
            "`problem, str(sympy.Integer(answer))`"
        )

    return reasons


# A competition-math statement that runs past this is not a hard problem, it is
# a generator dumping its sampled object into the prose. The bound matters
# operationally, not just aesthetically: problem_text is concatenated into the
# evaluator prompt, the solver prompt, and every training row, so one runaway
# statement can exceed the rollout context and take the whole run down. Real
# instances (14 seeds + 8 verified fixtures, 110 samples) top out at 397 chars,
# so this leaves an order of magnitude of headroom.
MAX_PROBLEM_TEXT_CHARS = 4000


def lint_problem_instance(instance: ProblemInstance) -> list[str]:
    """Reject obviously poor training examples."""
    reasons: list[str] = []
    problem = instance.problem.strip()
    answer = instance.answer.strip()

    if len(problem) < 10:
        reasons.append("problem text too short")
    if len(problem) > MAX_PROBLEM_TEXT_CHARS:
        reasons.append(
            f"problem text too long: {len(problem)} chars "
            f"(limit {MAX_PROBLEM_TEXT_CHARS})"
        )
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


def set_label_declarations(source_code: str, group: str, skill: str) -> str:
    """Return ``source_code`` with exactly one GROUP / SKILL pair, at the end.

    The two-stage mutation decides the labels in stage 1, from the problem and
    its solution, before any code exists. Stage 2 is shown a parent with its
    label lines removed and is told not to write any, so appending them here is
    what puts them back -- which means they cannot be omitted (the failure that
    cost 15 of 24 children when the model was asked for them) and cannot
    disagree with stage 1.

    Any label lines the patch added anyway are dropped first, so this is
    idempotent and there is never a second pair.
    """
    body = strip_label_declarations(source_code).rstrip()
    return f'{body}\n\n\nGROUP = "{group}"\nSKILL = "{skill}"\n'


def extract_problem_template(source_code: str) -> str | None:
    """The parent's ``problem = ...`` assignment, verbatim, as it is written.

    Stage 1 of the two-stage mutation used to be shown one rendered instance --
    seed 0, with every parameter already a number. Shown that, a model changes
    the numbers: measured, 85-89% of children were near-copies of the statement
    they were given, and one child's own account of its mutation was "a
    different prime p, a different g, and a different range".

    The template makes that move visibly empty. Where the instance says
    ``p = 13``, the template says ``p = {p}``, so substituting a different value
    is not a change to anything the model can see. What is left to alter is the
    structure around the holes.

    Returns None when the source does not parse or has no such assignment; the
    caller falls back to the rendered instance rather than dropping the parent.
    """
    try:
        tree = ast.parse(source_code)
    except (SyntaxError, ValueError):
        return None

    lines = source_code.splitlines()
    best: str | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == "problem" for t in targets):
            continue
        end = node.end_lineno or node.lineno
        # Last assignment wins: a generator that builds the text in pieces ends
        # with the one that is actually returned.
        best = "\n".join(lines[node.lineno - 1 : end])
    return textwrap.dedent(best).strip() if best else None
