"""Algorithmic-SHAPE description of a generator, for the stage-2 ``skeleton_forbidden`` operator.

Why not the AST node-type skeleton. The probe's ``skel()`` flattens the tree to
a node-type sequence and compares with SequenceMatcher. That is the right METRIC
(it is what measured parent->child median 0.996) and the wrong PROHIBITION: over
the 48-champion snapshot ``archive_iter157.json`` there are 31 distinct node-type
skeletons but only 13 distinct (answer-shape, check-shape) pairs, and the top 4
pairs cover 67% of the archive. Telling a model "do not emit ListComp" buys a
GeneratorExp -- a different skeleton string, the same algorithm. Telling it "do
not count by summing a predicate over a range" forbids the thing that repeats.

Calibration for the metric, measured on this repo:
  - 8 hand-written seed_programs, pairwise node-type skeleton sim: median 0.397.
    That is what real structural diversity reads as. The floor is not 0.
  - champions with DIFFERENT coarse shape pairs: median 0.648
  - champions with the SAME coarse shape pair:   median 0.756
  - parent -> accepted child:                    median 0.996
So the achievable headroom for a stage-2 operator is roughly 0.99 -> 0.40, and
0.65 is already "different algorithm, same contract".

Route extraction. A route is not one assignment: ``check = 0`` followed by a loop
that does ``check += 1`` is one route with three statements. ``route_stmts``
therefore takes every TOP-LEVEL statement of ``generate`` whose subtree writes
the name, in order, which keeps the loop that does the work attached to the
accumulator it initialises. Taking only the first ``Assign`` (the obvious
implementation) mislabels 11 of 48 champions as "check is a literal constant".
"""
from __future__ import annotations

import ast

_RNG_DRAWS = ("randint", "randrange", "choice", "choices", "sample", "shuffle", "uniform")
_LOOPISH = (ast.For, ast.While, ast.comprehension)
_COMPS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)


def _generate_fn(tree: ast.AST) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate":
            return node
    return None


def _writes(node: ast.AST, name: str) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in sub.targets
        ):
            return True
        if isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Name) and sub.target.id == name:
            return True
        if isinstance(sub, ast.For) and isinstance(sub.target, ast.Name) and sub.target.id == name:
            return True
    return False


def route_stmts(fn: ast.FunctionDef, name: str) -> list[ast.stmt]:
    """Every top-level statement of ``generate`` whose subtree writes ``name``."""
    return [s for s in fn.body if _writes(s, name)]


def _calls(nodes) -> set[str]:
    out: set[str] = set()
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                if isinstance(fn, ast.Name):
                    out.add(fn.id)
                elif isinstance(fn, ast.Attribute):
                    out.add(fn.attr)
    return out


def _depth(nodes) -> int:
    best = 0

    def walk(node, d):
        nonlocal best
        best = max(best, d)
        for child in ast.iter_child_nodes(node):
            walk(child, d + 1 if isinstance(child, _LOOPISH) else d)

    for node in nodes:
        walk(node, 0)
    return best


def _binop_ops(nodes) -> set[type]:
    return {
        type(sub.op)
        for node in nodes
        for sub in ast.walk(node)
        if isinstance(sub, ast.BinOp)
    }


def route_shape(stmts: list[ast.stmt], drawn: dict[str, str]) -> str:
    """Coarse algorithmic shape of one route, in words a model can act on.

    The degenerate shapes are tested first and returned alone: naming
    "BARE COPY of the sampled parameter 'n'" next to three other clauses lets a
    model satisfy the prohibition by changing one of the other three.
    """
    if not stmts:
        return "absent -- the name is never assigned"
    last = stmts[-1]
    if len(stmts) == 1 and isinstance(last, ast.Assign) and isinstance(last.value, ast.Name):
        if last.value.id in drawn:
            return (
                f"a BARE COPY of the sampled parameter {last.value.id!r} -- the value is "
                "handed back exactly as it was drawn, so nothing in the statement is computed"
            )
        return f"a bare copy of the local name {last.value.id!r} -- no arithmetic of its own"
    if len(stmts) == 1 and isinstance(last, ast.Assign) and isinstance(last.value, ast.Constant):
        return "a literal constant, the same for every seed and independent of every parameter"

    calls = _calls(stmts)
    ops = _binop_ops(stmts)
    depth = _depth(stmts)
    has_loop = any(isinstance(s, (ast.For, ast.While)) for st in stmts for s in ast.walk(st))
    has_comp = any(isinstance(s, _COMPS) for st in stmts for s in ast.walk(st))

    bits: list[str] = []
    if ast.LShift in ops:
        bits.append("enumeration of every subset via an integer bitmask")
    if (has_loop or has_comp) and "range" in calls:
        bits.append(f"enumeration over an explicit range, loop/comprehension nesting depth {depth}")
    elif has_loop or has_comp:
        bits.append(f"iteration over a built collection, nesting depth {depth}")
    if {"comb", "perm", "factorial", "binomial"} & calls:
        bits.append("a closed-form binomial/factorial count")
    if ast.Mod in ops:
        bits.append("divisibility / modular arithmetic")
    if {"max", "min"} & calls:
        bits.append("extremal selection over generated candidates")
    if {"gcd", "lcm", "isqrt", "sqrt"} & calls:
        bits.append("a number-theoretic primitive (gcd/lcm/isqrt)")
    if any(isinstance(s, ast.While) for st in stmts for s in ast.walk(st)) and not has_comp:
        bits.append("a while-loop search that steps until a condition holds")
    if not bits:
        bits.append("a single arithmetic/closed-form expression with no enumeration")
    return "; ".join(bits)


def structural_fingerprint(source_code: str) -> dict | None:
    """Structured shape record for one generator, or None if it does not parse."""
    try:
        tree = ast.parse(source_code)
    except (SyntaxError, ValueError):
        return None
    fn = _generate_fn(tree)
    if fn is None:
        return None

    drawn: dict[str, str] = {}
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr in _RNG_DRAWS
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    drawn[target.id] = node.value.func.attr

    answer = route_stmts(fn, "answer")
    check = route_stmts(fn, "check")
    helpers = [n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n is not fn]
    recursive = [
        h.name
        for h in helpers
        if any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == h.name
               for c in ast.walk(h))
    ]
    check_names = {n.id for st in check for n in ast.walk(st) if isinstance(n, ast.Name)}

    return {
        "n_params": len(drawn),
        "draw_kinds": sorted(set(drawn.values())),
        "answer_shape": route_shape(answer, drawn),
        "check_shape": route_shape(check, drawn),
        "answer_stmts": len(answer),
        "check_stmts": len(check),
        "helpers": [h.name for h in helpers],
        "recursive_helpers": recursive,
        "check_reads_answer": "answer" in check_names,
        "control": sorted({
            type(n).__name__ for n in ast.walk(fn)
            if isinstance(n, (ast.For, ast.While, ast.If) + _COMPS)
        }),
    }


def render_fingerprint(fp: dict) -> str:
    """The exact block that goes into the ``skeleton_forbidden`` prompt."""
    def _route(label, shape, count):
        # The statement count is informative for a real route and noise after a
        # degenerate one ("a BARE COPY ..., written as 1 statement(s)").
        degenerate = shape.startswith(("a BARE COPY", "a bare copy", "a literal constant", "absent"))
        tail = "" if degenerate or not count else f", written as {count} statement(s)"
        return f"- its {label} route is {shape}{tail}"

    lines = [
        f"- it draws {fp['n_params']} parameter(s), using {', '.join(fp['draw_kinds']) or 'no rng call at all'}",
        _route("ANSWER", fp["answer_shape"], fp["answer_stmts"]),
        _route("CHECK", fp["check_shape"], fp["check_stmts"]),
        "- helper functions: "
        + (", ".join(fp["helpers"]) + (f" ({len(fp['recursive_helpers'])} recursive)" if fp["recursive_helpers"] else "")
           if fp["helpers"] else "none -- every step is inline in generate()"),
        f"- control constructs it uses: {', '.join(fp['control']) or 'straight-line code only'}",
    ]
    if fp["check_reads_answer"]:
        lines.append("- its CHECK route reads `answer`, so it is not an independent route at all")
    return "\n".join(lines)


def shape_key(fp: dict) -> tuple[str, str]:
    """Coarse bucket for reporting: (first answer clause, first check clause)."""
    return (fp["answer_shape"].split(";")[0].strip(), fp["check_shape"].split(";")[0].strip())
