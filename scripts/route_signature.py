"""Route signature: a novelty measure that reads the OPERATIONS, not the shape.

Companion to `skel()` / `sim()` in scripts/probe_operators_report.py.

The AST-skeleton SequenceMatcher answers "is this the same code shape?".
This answers "does this program compute with the same mathematical
operations?" -- which is what "structurally new problem family" was meant to
mean. Measured on the three controls that decide it:

    for-loop vs comprehension, SAME mathematics   skeleton 0.920  route 0.733
    divisor count -> Euler phi, DIFFERENT maths   skeleton 0.934  route 0.412
    only the numeric literals changed             skeleton 1.000  route 1.000

The skeleton metric orders the first two BACKWARDS: it calls the pure
respelling more novel than the change of mathematics.

Three properties the skeleton metric does not have:
  order-invariant   sibling statements and helper defs may be reordered
  length-invariant  a SET of labels, not a sequence -- a longer child that
                    uses the same operations is not scored as novel
                    (skeleton similarity correlates -0.83 with |length diff|)
  content-bearing   the called function name and the operator class ARE the
                    label; identifiers and literals are dropped, because
                    renaming and re-ranging are the non-mutations the project
                    already rejects
"""
from __future__ import annotations

import ast

__all__ = ["route_signature", "route_sim", "novel_form"]

# Sampling and formatting are plumbing every generator must contain; counting
# them as shared mathematics floors every pair at the contract's own overlap.
_PLUMBING = frozenset({
    "Random", "randint", "randrange", "choice", "choices", "sample",
    "shuffle", "uniform", "seed", "str", "format", "print", "join", "strip",
})

# An accumulator threaded through a loop IS the aggregate. Spelling it as a
# comprehension is not mathematics, so both reduce to one label.
_AGG = {"Add": "sum", "Mult": "prod", "BitOr": "union", "BitAnd": "inter"}

_LOOP = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp,
         ast.DictComp, ast.GeneratorExp)


def _fname(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _ops(node: ast.AST) -> list[str]:
    """The mathematical operations this node performs; [] when it performs none."""
    if isinstance(node, ast.Call):
        name = _fname(node.func).split(".")[-1]
        return [] if name in _PLUMBING or not name else ["call:" + name]
    if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp)):
        return ["op:" + type(node.op).__name__]
    if isinstance(node, ast.Compare):
        return ["cmp:" + type(o).__name__ for o in node.ops]
    if isinstance(node, ast.AugAssign):
        kind = type(node.op).__name__
        return ["call:" + _AGG.get(kind, kind.lower())]
    if isinstance(node, _LOOP):
        return ["loop"]
    if isinstance(node, (ast.If, ast.IfExp, ast.Match)):
        return ["branch"]
    if isinstance(node, ast.Subscript):
        return ["index"]
    return []


def route_signature(source: str, depth: int = 1) -> frozenset[str] | None:
    """Set of operation labels plus one level of operation-in-operation context.

    Returns None when the source does not parse, matching `skel()`.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    nodes = list(ast.walk(tree))
    lab = {id(n): _ops(n) for n in nodes}
    sig: set[str] = set()
    for n in nodes:
        sig |= set(lab[id(n)])
    for _ in range(depth):
        for n in nodes:
            mine = lab[id(n)]
            if not mine:
                continue
            inner = sorted({o for c in ast.walk(n) if c is not n for o in lab[id(c)]})
            if inner:
                ctx = "<" + ",".join(inner[:6]) + ">"
                sig |= {m + ctx for m in mine}
    return frozenset(sig)


def route_sim(a: frozenset[str] | None, b: frozenset[str] | None) -> float:
    """Jaccard over route signatures. 1.0 = same operations, 0.0 = disjoint."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def novel_form(child: frozenset[str] | None,
               seen: list[frozenset[str]],
               tau: float = 0.60) -> bool:
    """True when the child's route matches nothing already visited.

    `seen` must be the archive PLUS every child accepted so far in this run.
    The statistic that went to zero by generation 10 is "forms not previously
    visited", and a child that only duplicates an EARLIER CHILD is not a new
    form -- scoring against the archive alone cannot see that.
    """
    return bool(child) and all(route_sim(child, s) < tau for s in seen)
