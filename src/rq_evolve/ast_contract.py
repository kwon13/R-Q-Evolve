"""Structural contract for a generator's answer cross-check.

A generator returns ``(problem_text, answer[, verifier])``. Nothing in the pipeline proves
those two describe the same mathematics -- ``verify_program`` proves the answer
is an integer and that the statement varies, the evaluator proves the statement
reads coherently, and neither closes the gap. The only mechanical link is the
generator's own ``assert answer == check``, where ``check`` recomputes the answer
by an independent route (see the design note at ``prompts.py:39-45``).

In one 105-iteration run that link was broken in 53 of the 151 programs that
entered the archive: the statements computing ``check`` were a character-for-
character copy of the statements computing ``answer``, so the assert held by
construction and carried no information. Five of the top ten champions by R_Q
were of this shape. R_Q cannot see it (median 22.5 against 20.7 for sound
programs) and the evaluator passed them with the full source in its prompt.

This module decides that question structurally.

WHAT IT DOES NOT DO
    It never asserts that two routes are independent. Whether two
    non-isomorphic programs compute the same function is undecidable (Rice), so
    a positive proof of independence is not available at any price. Instead a
    cross-check is *accepted* unless one of three sound refutations applies:

        identity   -- the two routes are the same program modulo renaming
        derivation -- the check consumed the answer instead of recomputing it
        degeneracy -- the check side is a constant

    Each refutation is a syntactic property of the parse tree. None of them
    consults the mathematics.

THE CHECK IS EXISTENTIAL, NOT UNIVERSAL
    A program passes when *at least one* assert certifies as a cross-check. The
    natural framing -- scan the asserts and flag the bad ones -- rejects sound
    seeds immediately: ``seed_programs/11_kth_root_count.py`` has three asserts
    of which two are parameter guards, and
    ``seed_programs/12_derangement_fixed_points.py`` pins a shared helper with a
    combinatorial identity. Neither guard is a cross-check and neither is a
    defect.
"""

from __future__ import annotations

import ast
from .safe_parse import safe_ast_parse
import copy
import re
from dataclasses import dataclass

__all__ = ["Finding", "check_generator_contract", "check_problem_text"]


@dataclass(frozen=True, slots=True)
class Finding:
    """One contract violation. ``code`` is the stable rule id."""

    code: str
    message: str
    lineno: int = 0

    def __str__(self) -> str:  # what the caller folds into ``source_errors``
        return f"{self.code}: {self.message}"


# Callables whose argument, when compared to zero, is really a difference:
# ``assert sympy.simplify(a - b) == 0`` and ``assert abs(a - b) < 1e-6`` are
# both equality claims about a and b. Missing these misreads a genuine check as
# rule A5c (constant comparison).
_ZERO_WRAPPERS = frozenset(
    {"abs", "Abs", "simplify", "nsimplify", "expand", "factor", "radsimp", "N"}
)
_PAIR_CALLS = frozenset({"isclose", "allclose", "Eq"})

# A statement that hands the solver the intended technique makes the declared
# SKILL untrue by construction: the reasoning is no longer forced, it is quoted.
_TECHNIQUE_HANDED_OVER = re.compile(
    r"\b(using|apply|applying|by|via|through|with)\s+(the\s+)?"
    r"(transformation|substitution|induction|contradiction|invariant|"
    r"casework|case\s+analysis|extremal|pigeonhole|construction|"
    r"generating\s+function|inclusion[-\s]exclusion|telescop\w*)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# frontend
# --------------------------------------------------------------------------
#
# Six shapes in the real corpus each cost a false positive during derivation.
# They are handled here rather than in the rules, so a rule cannot forget one:
#
#   1. The answer variable is often not called ``answer`` (133 of 456 archived
#      programs, and one seed, use ``max_product`` / ``area`` / ``count``).
#      It is resolved from the return tuple.
#   2. The check side is frequently an unnamed inline expression -- four of the
#      eight verified fixtures write ``assert _helper(...) == answer``. Routes
#      are therefore expressions, never variable names.
#   3. Comprehension targets are a separate scope; ``[answer for answer in ...]``
#      shadows the answer name.
#   4. Nested ``def``s are opaque: descending into one pollutes the name graph.
#   5. The assert usually sits inside a sampling loop. Scanning only the
#      function's top level finds no assert in 8 of 8 fixtures.
#   6. Chained comparisons and tolerance forms are equality claims too.

_BODY_FIELDS = frozenset({"body", "orelse", "finalbody", "handlers"})
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _shadowed_names(node: ast.AST) -> set[str]:
    """Names bound by a comprehension or lambda inside ``node`` (gotcha 3)."""
    bound: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for generator in child.generators:
                bound |= _stored_names(generator.target)
        elif isinstance(child, ast.Lambda):
            args = child.args
            for group in (args.posonlyargs, args.args, args.kwonlyargs):
                bound |= {a.arg for a in group}
            for solo in (args.vararg, args.kwarg):
                if solo is not None:
                    bound.add(solo.arg)
    return bound


def _stored_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }


def _loaded_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    } - _shadowed_names(node)


def _targets(stmt: ast.AST) -> set[str]:
    """Names ``stmt`` assigns. A nested def contributes only its own name."""
    if isinstance(stmt, _SCOPES):
        return {stmt.name}
    return _stored_names(stmt) - _shadowed_names(stmt)


def _reads(stmt: ast.AST) -> set[str]:
    """Names ``stmt`` reads, including through an augmented or indexed target."""
    reads = _loaded_names(stmt)
    for node in ast.walk(stmt):
        if isinstance(node, ast.AugAssign):
            reads |= _stored_names(node.target)
        elif isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(
            node.ctx, ast.Store
        ):
            reads |= {
                n.id for n in ast.walk(node) if isinstance(n, ast.Name)
            }
    return reads


def _find_generate(tree: ast.Module) -> ast.FunctionDef | None:
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "generate"
        ),
        None,
    )


def _parent_map(root: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _own_nodes(generate: ast.FunctionDef, kind) -> list[ast.AST]:
    """Nodes of ``kind`` belonging to ``generate`` itself (gotchas 4 and 5).

    Both the asserts and the return live at arbitrary depth -- the assert is
    usually inside a sampling loop -- but a nested helper's ``return`` is not
    the generator's answer, and a nested helper's ``assert`` is not the
    generator's cross-check. ``ast.walk`` cannot tell the difference.
    """
    found: list[ast.AST] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (*_SCOPES, ast.Lambda)):
                continue
            if isinstance(child, kind):
                found.append(child)
            walk(child)

    walk(generate)
    return found


def _assert_nodes(generate: ast.FunctionDef) -> list[ast.Assert]:
    return [n for n in _own_nodes(generate, ast.Assert)]


def _visible_statements(
    target: ast.AST,
    generate: ast.FunctionDef,
    parents: dict[ast.AST, ast.AST],
) -> list[ast.stmt]:
    """Statements that may run before ``target``, innermost block outward.

    A ``for``/``while`` ancestor contributes its whole body, not just the part
    textually above the target: the assert observes the previous iteration's
    values too, so loop-carried definitions are visible.
    """
    out: list[ast.stmt] = []
    node: ast.AST = target
    while node is not generate:
        parent = parents.get(node)
        if parent is None:
            break
        for field, value in ast.iter_fields(parent):
            if not isinstance(value, list) or field not in _BODY_FIELDS:
                continue
            if not any(item is node for item in value):
                continue
            index = next(i for i, item in enumerate(value) if item is node)
            loop = isinstance(parent, (ast.For, ast.AsyncFor, ast.While))
            preceding = list(value) if (loop and field == "body") else value[:index]
            out = [s for s in preceding if isinstance(s, ast.stmt)] + out
            break
        node = parent
    return out


def _backward_slice(statements: list[ast.stmt], names: set[str]) -> list[ast.stmt]:
    """Statements contributing to ``names``, in source order.

    Deliberately flow-insensitive: every assignment to a wanted name is kept,
    including both arms of a branch. Both sides of a comparison are sliced by
    the same algorithm, so the over-approximation is symmetric and cannot
    manufacture an equality that is not really there.
    """
    wanted = set(names)
    keep: list[ast.stmt] = []
    for stmt in reversed(statements):
        if _targets(stmt) & wanted:
            keep.append(stmt)
            wanted |= _reads(stmt)
    keep.reverse()
    return keep


def _rename(node: ast.AST, mapping: dict[str, str]) -> ast.AST:
    clone = copy.deepcopy(node)
    for child in ast.walk(clone):
        if isinstance(child, ast.Name) and child.id in mapping:
            child.id = mapping[child.id]
    return _IdempotentMinMax().visit(clone)


class _IdempotentMinMax(ast.NodeTransformer):
    """Fold ``min(x, x)``/``max(x, x)`` before route comparison.

    Repeating one expression inside an idempotent built-in is not an
    independent verification route.  Without this normalization a candidate
    can turn ``check = answer_expression`` into ``max(expr, expr)`` and evade
    the alpha-equivalent-route gate without performing a second computation.
    """

    def visit_Call(self, node: ast.Call):  # noqa: N802 - ast API spelling
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"min", "max"}
            and len(node.args) >= 2
            and not any(isinstance(argument, ast.Starred) for argument in node.args)
            and not node.keywords
        ):
            fingerprints = {
                ast.dump(argument, include_attributes=False)
                for argument in node.args
            }
            if len(fingerprints) == 1:
                return node.args[0]
        return node


def _resolve_alias(operand: ast.expr, visible: list[ast.stmt]) -> ast.expr:
    """Follow ``check = relay`` back to ``relay`` before normalising.

    Without this, routing a duplicated computation through one extra variable
    buys a pass: the slice gains a statement the other side lacks and the two
    normal forms stop matching. A static gate sits inside a selection loop, so
    a dodge this cheap would simply become the shape the archive evolves.
    """
    seen: set[str] = set()
    while isinstance(operand, ast.Name) and operand.id not in seen:
        seen.add(operand.id)
        sources = [
            stmt.value
            for stmt in visible
            if isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == operand.id
        ]
        if len(sources) != 1 or not isinstance(sources[0], ast.Name):
            break
        operand = sources[0]
    return operand


def _route_normal_form(
    operand: ast.expr, visible: list[ast.stmt]
) -> tuple[str, ...]:
    """Alpha-canonical program computing ``operand``.

    Names the route *defines* are renamed to positional placeholders; names it
    merely *reads* -- shared sampled parameters, module globals, helpers --
    keep their identity. That is alpha-equivalence modulo the shared
    environment, which is exactly the equivalence that makes two routes "the
    same computation under different variable names".
    """
    operand = _resolve_alias(operand, visible)
    sliced = _backward_slice(visible, _loaded_names(operand))
    defined: list[str] = []
    for stmt in sliced:
        for name in sorted(_targets(stmt)):
            if name not in defined:
                defined.append(name)
    mapping = {name: f"_r{i}" for i, name in enumerate(defined)}
    parts = [
        ast.dump(_rename(stmt, mapping), include_attributes=False) for stmt in sliced
    ]
    parts.append(ast.dump(_rename(operand, mapping), include_attributes=False))
    return tuple(parts)


def _latest_value(name: str, visible: list[ast.stmt]) -> ast.expr | None:
    """Last simple assignment to ``name`` visible at a cross-check."""

    for stmt in reversed(visible):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name
        ):
            return stmt.value
        if (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == name
            and stmt.value is not None
        ):
            return stmt.value
    return None


def _resolved_value(
    operand: ast.expr, visible: list[ast.stmt], *, limit: int = 8
) -> ast.expr:
    """Resolve ordinary value assignments, not only pure aliases."""

    value = operand
    seen: set[str] = set()
    for _ in range(limit):
        if not isinstance(value, ast.Name) or value.id in seen:
            break
        seen.add(value.id)
        source = _latest_value(value.id, visible)
        if source is None:
            break
        value = source
    return value


def _canonical_loop_parts(
    target: ast.expr, iterator: ast.expr, conditions: list[ast.expr]
) -> tuple[str, tuple[str, ...]]:
    bound = sorted(_stored_names(target))
    mapping = {name: f"_item{index}" for index, name in enumerate(bound)}
    iterator_dump = ast.dump(_rename(iterator, mapping), include_attributes=False)
    condition_dump = tuple(
        ast.dump(_rename(condition, mapping), include_attributes=False)
        for condition in conditions
    )
    return iterator_dump, condition_dump


def _imperative_count_signature(
    collection: str, visible: list[ast.stmt]
) -> tuple | None:
    """Canonicalize ``for/if append`` and ``for/if += 1`` counts."""

    for stmt in reversed(visible):
        if not isinstance(stmt, ast.For):
            continue
        conditions: list[ast.expr] = []
        body = list(stmt.body)
        while len(body) == 1 and isinstance(body[0], ast.If) and not body[0].orelse:
            conditions.append(body[0].test)
            body = list(body[0].body)
        if len(body) != 1:
            continue
        update = body[0]
        matched = False
        if (
            isinstance(update, ast.Expr)
            and isinstance(update.value, ast.Call)
            and isinstance(update.value.func, ast.Attribute)
            and update.value.func.attr == "append"
            and isinstance(update.value.func.value, ast.Name)
            and update.value.func.value.id == collection
            and len(update.value.args) == 1
        ):
            matched = True
        elif (
            isinstance(update, ast.AugAssign)
            and isinstance(update.target, ast.Name)
            and update.target.id == collection
            and isinstance(update.op, ast.Add)
            and isinstance(update.value, ast.Constant)
            and update.value.value == 1
        ):
            matched = True
        if matched:
            iterator, predicates = _canonical_loop_parts(
                stmt.target, stmt.iter, conditions
            )
            return ("bounded_count", iterator, predicates)
    return None


def _count_route_signature(
    operand: ast.expr, visible: list[ast.stmt]
) -> tuple | None:
    """Recognize semantically identical bounded counting idioms.

    Alpha-equivalent ASTs already catch loop-vs-loop copies.  This extra narrow
    normalizer closes the common 4B dodge where one side is a ``for`` loop and
    the other is the same range/predicate written as a comprehension.
    """

    value = _resolved_value(operand, visible)
    while (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in {"str", "int"}
        and len(value.args) == 1
        and not value.keywords
    ):
        value = _resolved_value(value.args[0], visible)

    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "len"
        and len(value.args) == 1
        and not value.keywords
    ):
        counted = value.args[0]
        if isinstance(counted, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            if len(counted.generators) != 1 or counted.generators[0].is_async:
                return None
            generator = counted.generators[0]
            iterator, predicates = _canonical_loop_parts(
                generator.target, generator.iter, list(generator.ifs)
            )
            return ("bounded_count", iterator, predicates)
        if isinstance(counted, ast.Name):
            return _imperative_count_signature(counted.id, visible)

    if isinstance(value, ast.Name):
        return _imperative_count_signature(value.id, visible)
    return None


def _is_zero(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value in (0, 0.0)


def _difference(node: ast.expr) -> tuple[ast.expr, ast.expr] | None:
    """Unwrap ``abs(a - b)`` / ``simplify(a - b)`` / ``a - b`` into ``(a, b)``."""
    while isinstance(node, ast.Call) and len(node.args) == 1:
        name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else getattr(node.func, "id", None)
        )
        if name not in _ZERO_WRAPPERS:
            return None
        node = node.args[0]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return node.left, node.right
    return None


def _equality_pairs(test: ast.expr) -> list[tuple[ast.expr, ast.expr]]:
    """Every ``(left, right)`` the assert claims equal (gotcha 6)."""
    if isinstance(test, ast.BoolOp):
        return [pair for value in test.values for pair in _equality_pairs(value)]
    if isinstance(test, ast.Call):
        name = (
            test.func.attr
            if isinstance(test.func, ast.Attribute)
            else getattr(test.func, "id", None)
        )
        if name in _PAIR_CALLS and len(test.args) >= 2:
            return [(test.args[0], test.args[1])]
        return []
    if not isinstance(test, ast.Compare):
        return []
    pairs: list[tuple[ast.expr, ast.expr]] = []
    operands = [test.left, *test.comparators]
    for index, op in enumerate(test.ops):
        left, right = operands[index], operands[index + 1]
        if isinstance(op, ast.Eq):
            if _is_zero(right):
                diff = _difference(left)
            elif _is_zero(left):
                diff = _difference(right)
            else:
                diff = None
            pairs.append(diff or (left, right))
        elif isinstance(op, (ast.Lt, ast.LtE)):
            diff = _difference(left)          # abs(a - b) < eps
            if diff is not None:
                pairs.append(diff)
    return pairs


def _returned(generate: ast.FunctionDef) -> tuple[ast.expr | None, ast.expr | None]:
    """``(problem_expr, answer_expr)`` from the last 2/3-element return.

    Scoped: a nested helper's ``return transformed[0]`` is that helper's value,
    not the generator's answer. Reading it instead left ``answer_names`` empty
    and made every assert look like an invariant.
    """
    problem = answer = None
    for node in _own_nodes(generate, ast.Return):
        value = node.value
        if isinstance(value, ast.Tuple) and len(value.elts) in (2, 3):
            problem, answer = value.elts[:2]
    return problem, answer


def _alias_closure(generate: ast.FunctionDef, names: set[str]) -> set[str]:
    """Extend ``names`` through pure renames, following ``answer = other``.

    The historical dual-route shape computes ``answer_insight`` and
    ``answer_brute`` independently, asserts them equal, and only then binds
    ``answer``. Keyed on the returned name alone that assert reads as an
    invariant between two unrelated values -- it accounted for 56 of 75 A2
    firings on the archive, every one of them a real cross-check.
    """
    # Only a name assigned exactly once can be a rename. ``val = c`` followed by
    # ``val = p * val + q`` initialises an accumulator; reading it as an alias
    # drags the input parameter ``c`` into the answer's identity and makes every
    # route that samples ``c`` look derived from the answer.
    assigned: dict[str, int] = {}
    for stmt in _own_nodes(generate, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        for target in targets:
            for name in _stored_names(target):
                assigned[name] = assigned.get(name, 0) + 1
    for stmt in _own_nodes(generate, ast.For):
        for name in _stored_names(stmt.target):
            assigned[name] = assigned.get(name, 0) + 2  # loop-carried, never an alias

    renames: dict[str, set[str]] = {}
    for stmt in _own_nodes(generate, ast.Assign):
        if (
            len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Name)
            and assigned.get(stmt.targets[0].id) == 1
        ):
            renames.setdefault(stmt.targets[0].id, set()).add(stmt.value.id)
    out = set(names)
    changed = True
    while changed:
        changed = False
        for target, sources in renames.items():
            if target in out and not sources <= out:
                out |= sources
                changed = True
    return out


def _name_graph(body: list[ast.stmt]) -> list[tuple[set[str], set[str]]]:
    """``(targets, reads)`` data/control dependencies, excluding nested defs.

    A value updated in ``for _ in range(n)`` depends on ``n`` through the
    number of updates even when the loop body never reads ``n`` directly.
    Likewise, a branch assignment depends on its condition.  Omitting these
    control edges falsely labels ordinary recurrence and enumeration programs
    as having statement parameters disconnected from their answers.
    """
    edges: list[tuple[set[str], set[str]]] = []

    def assignment_edges(
        target: ast.expr, value: ast.expr
    ) -> list[tuple[set[str], set[str]]]:
        """Keep pairwise tuple unpacking from coupling unrelated values."""

        if (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
        ):
            split: list[tuple[set[str], set[str]]] = []
            for child_target, child_value in zip(target.elts, value.elts):
                split.extend(assignment_edges(child_target, child_value))
            return split
        return [(_stored_names(target), _loaded_names(value))]

    def loop_controlled(
        nested: list[tuple[set[str], set[str]]], control_reads: set[str]
    ) -> None:
        edges.extend(nested)
        # Only an accumulator/recurrence that re-reads a value it assigns is
        # coupled to the iteration count solely through control flow.  A plain
        # ``value = 0`` repeated n times is still semantically constant and
        # must not acquire a spurious dependency on n.
        carried = {
            name
            for assigned, reads in nested
            for name in assigned
            if name in reads
        }
        if carried and control_reads:
            edges.append((carried, control_reads))

    for stmt in body:
        if isinstance(stmt, _SCOPES):
            edges.append(({stmt.name}, _loaded_names(stmt)))
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            edges.append((_stored_names(stmt.target), _loaded_names(stmt.iter)))
            loop_controlled(_name_graph(stmt.body), _loaded_names(stmt.iter))
            edges.extend(_name_graph(stmt.orelse))
        elif isinstance(stmt, ast.While):
            loop_controlled(_name_graph(stmt.body), _loaded_names(stmt.test))
            edges.extend(_name_graph(stmt.orelse))
        elif isinstance(stmt, ast.If):
            # General semantic control dependence is deliberately not claimed:
            # ``if n: answer = 1; else: answer = 1`` does not make answer
            # depend on n.  Direct data reads inside either branch remain.
            edges.extend(_name_graph(stmt.body))
            edges.extend(_name_graph(stmt.orelse))
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            edges.extend(_name_graph(stmt.body))
        elif isinstance(stmt, ast.Try):
            edges.extend(_name_graph(stmt.body))
            for handler in stmt.handlers:
                edges.extend(_name_graph(handler.body))
            edges.extend(_name_graph(stmt.orelse))
            edges.extend(_name_graph(stmt.finalbody))
        elif (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], (ast.Tuple, ast.List))
            and isinstance(stmt.value, (ast.Tuple, ast.List))
        ):
            edges.extend(assignment_edges(stmt.targets[0], stmt.value))
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            edges.append((_targets(stmt), _reads(stmt)))
    return edges


def _reachable(edges: list[tuple[set[str], set[str]]], seed: set[str]) -> set[str]:
    """Names ``seed`` transitively depends on. Cycle-safe by the fixpoint."""
    seen = set(seed)
    changed = True
    while changed:
        changed = False
        for targets, reads in edges:
            if targets & seen and not reads <= seen:
                seen |= reads
                changed = True
    return seen


def _component(edges: list[tuple[set[str], set[str]]], seed: set[str]) -> set[str]:
    """Weakly-connected component of ``seed`` -- ancestors *and* descendants.

    The directed version ("the answer must depend on it") fires on four of the
    six seeds: ``10_inequalities`` states ``s`` while the answer uses ``m`` with
    ``s = 3 * m`` (siblings), and ``11_kth_root_count`` states parameters that
    are *descendants* of the answer. Coupling through a common ancestor is
    still coupling.
    """
    seen = set(seed)
    changed = True
    while changed:
        changed = False
        for targets, reads in edges:
            group = targets | reads
            if group & seen and not group <= seen:
                seen |= group
                changed = True
    return seen


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------


def check_generator_contract(source_code: str) -> list[Finding]:
    """Structural findings for one generator. Empty list means it passed.

    Syntax errors are not this checker's business -- ``lint_generator_source``
    owns them, and reporting them twice would double every message.
    """
    try:
        tree = safe_ast_parse(source_code)
    except (SyntaxError, ValueError):
        return []

    generate = _find_generate(tree)
    if generate is None:
        return []

    findings: list[Finding] = []
    parents = _parent_map(generate)
    asserts = _assert_nodes(generate)
    _, answer_expr = _returned(generate)
    edges = _name_graph(generate.body)
    locals_defined = {name for targets, _ in edges for name in targets}
    # Locals only: ``str(sympy.Integer(answer))`` also loads ``sympy``, and a
    # module name says nothing about which side of an assert holds the answer.
    answer_names = (
        _loaded_names(answer_expr) & locals_defined if answer_expr is not None else set()
    )
    answer_names = _alias_closure(generate, answer_names) & locals_defined

    # ---- A1: no assert at all -------------------------------------------
    if not asserts:
        findings.append(
            Finding(
                "A1",
                "generate() has no assert; nothing links the problem statement "
                "to the returned answer",
                generate.lineno,
            )
        )
        return findings

    # ---- certification pass ---------------------------------------------
    # Collect a diagnosis per assert; the program passes if any one certifies.
    diagnoses: list[Finding] = []
    certified = False

    for node in asserts:
        pairs = _equality_pairs(node.test)
        if not pairs:
            continue
        visible = _visible_statements(node, generate, parents)
        for left, right in pairs:
            # A cross-check is identified by DIRECT reference to the answer on
            # exactly one side. Using the answer's dependency closure instead
            # rejects two of the eight verified fixtures: every genuine second
            # route reads the same sampled parameters the answer was built
            # from, so both sides "touch" the closure and the assert reads as
            # an invariant.
            left_has = bool(_loaded_names(left) & answer_names)
            right_has = bool(_loaded_names(right) & answer_names)
            if left_has == right_has:
                # Both sides hold the answer, or neither does: a guard or an
                # invariant, not a cross-check. Not a defect on its own --
                # seeds carry several alongside a real one.
                continue

            answer_side, check_side = (left, right) if left_has else (right, left)

            # A5c -- the check side is a constant
            local_check = _loaded_names(check_side) & locals_defined
            if not local_check:
                diagnoses.append(
                    Finding(
                        "A5c",
                        "the cross-check compares the answer to a constant "
                        "expression, so it holds by construction",
                        node.lineno,
                    )
                )
                continue

            # A4d -- the check consumed the answer instead of recomputing it.
            #
            # "Consumed" means the answer's value flows directly into the
            # check's definition, as in ``pick_interior = interior_points``.
            # Transitive dependence is NOT consumption: in
            # ``11_kth_root_count.py`` the answer is the sampled parameter ``t``
            # and the prime ``p`` is chosen so that ``t | p - 1``, so every
            # route reaches ``t`` through the construction. A closure-based test
            # rejects that seed; a direct-read test does not.
            consumed = any(
                _targets(stmt) & local_check and _reads(stmt) & answer_names
                for stmt in visible
            )
            if consumed:
                diagnoses.append(
                    Finding(
                        "A4d",
                        "the cross-check is derived from the answer it is "
                        "supposed to verify",
                        node.lineno,
                    )
                )
                continue

            answer_count = _count_route_signature(answer_side, visible)
            check_count = _count_route_signature(check_side, visible)
            if (
                answer_count is not None
                and check_count is not None
                and answer_count == check_count
            ):
                diagnoses.append(
                    Finding(
                        "A3v",
                        "the two sides count the same bounded range with the "
                        "same predicate; rewriting a loop as a comprehension "
                        "does not make an independent check",
                        node.lineno,
                    )
                )
                continue

            # A3v -- the two routes are the same program modulo renaming
            if _route_normal_form(answer_side, visible) == _route_normal_form(
                check_side, visible
            ):
                diagnoses.append(
                    Finding(
                        "A3v",
                        "the two sides of the cross-check are the same "
                        "computation under different variable names; the "
                        "assert holds by construction and proves nothing",
                        node.lineno,
                    )
                )
                continue

            certified = True
            break
        if certified:
            break

    if not certified:
        if diagnoses:
            findings.append(diagnoses[0])
        else:
            # A2 -- asserts exist but none straddles the answer's dependencies.
            findings.append(
                Finding(
                    "A2",
                    "no assert compares the answer against an independently "
                    "computed value; the existing asserts are guards or "
                    "invariants, not a cross-check",
                    asserts[0].lineno,
                )
            )

    # ---- P1: a stated parameter disconnected from the answer -------------
    findings.extend(_check_statement_parameters(tree, generate, edges, answer_names))
    return findings


def _rng_names(generate: ast.FunctionDef) -> set[str]:
    """Names bound to a ``random.Random(...)`` instance."""
    found: set[str] = set()
    for stmt in _own_nodes(generate, ast.Assign):
        value = stmt.value
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == "Random":
            for target in stmt.targets:
                found |= _stored_names(target)
    return found


def _declared_distractors(tree: ast.Module) -> set[str]:
    """Opt-out for a deliberate red herring: ``DISTRACTORS = ("noise",)``."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "DISTRACTORS" for t in node.targets
        ):
            continue
        if isinstance(node.value, (ast.Tuple, ast.List, ast.Set)):
            return {
                e.value
                for e in node.value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
    return set()


def _check_statement_parameters(
    tree: ast.Module,
    generate: ast.FunctionDef,
    edges: list[tuple[set[str], set[str]]],
    answer_names: set[str],
) -> list[Finding]:
    """A quantity printed in the problem must be coupled to the answer.

    Coupling is the *undirected* component, not ancestry -- see ``_component``.
    A program may declare ``DISTRACTORS`` to opt a name out; the seed corpus
    contains no deliberate red herrings and the mutation contract does not
    permit them, so the default is that a stated parameter can matter.
    """
    if not answer_names:
        return []
    problem_expr, _ = _returned(generate)
    if problem_expr is None:
        return []

    locals_defined = {name for targets, _ in edges for name in targets}
    # ``rng`` is read by every sampled parameter, so leaving it in the graph
    # merges the whole function into one component and the rule never fires.
    # Sharing a random source is plumbing, not mathematical coupling.
    plumbing = {"seed"} | {
        name
        for targets, _ in edges
        for name in targets
        if name in _rng_names(generate)
    }
    # Names bound to an f-string: the rendered statement and any fragment of
    # it. Their assignments are excluded from the graph below.
    rendering: set[str] = set()
    for stmt in _own_nodes(generate, (ast.Assign, ast.AnnAssign)):
        if any(isinstance(n, ast.JoinedStr) for n in ast.walk(stmt)):
            rendering |= _targets(stmt)

    interpolated: set[str] = set()
    for stmt in _own_nodes(generate, (ast.Assign, ast.AnnAssign)):
        if not (_targets(stmt) & (rendering | _loaded_names(problem_expr))):
            continue
        for node in ast.walk(stmt):
            if isinstance(node, ast.FormattedValue):
                interpolated |= _loaded_names(node.value)
    for node in ast.walk(problem_expr):
        if isinstance(node, ast.FormattedValue):
            interpolated |= _loaded_names(node.value)

    # The rendering statement itself must not supply the coupling. It reads
    # every interpolated name at once, so leaving it in makes each of them
    # trivially connected to all the others and the rule can never fire.
    coupled = [
        (targets - plumbing, reads - plumbing)
        for targets, reads in edges
        if (targets - plumbing) and not (targets & rendering)
    ]

    # A helper is not a stated quantity, and neither is the random source.
    helpers = {n.name for n in generate.body if isinstance(n, _SCOPES)}
    interpolated &= locals_defined
    interpolated -= _declared_distractors(tree) | plumbing | helpers | rendering
    orphans = sorted(interpolated - _component(coupled, answer_names - plumbing))
    if not orphans:
        return []
    return [
        Finding(
            "P1",
            "the problem statement names "
            + ", ".join(repr(o) for o in orphans)
            + ", which share no dependency path with the answer",
            generate.lineno,
        )
    ]


def check_problem_text(problem: str) -> list[Finding]:
    """Findings about one rendered problem statement.

    Separate from the source check because it needs an executed instance.
    """
    match = _TECHNIQUE_HANDED_OVER.search(problem or "")
    if match is None:
        return []
    return [
        Finding(
            "P2",
            "the statement instructs the solver to use "
            f"{match.group(3).lower()!r}; a technique that is named is not a "
            "technique the problem forces",
        )
    ]
