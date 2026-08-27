"""Structural gates against a generator that ignores its seed.

Corollary 2.2 of the design says the fitness already charges a program for
difficulty dispersion, so no explicit consistency reward is needed -- and none
may be added, because rewarding consistency hands a free maximum to a generator
that returns the same instance for every seed. The defence has to be a hard
pass/fail gate instead: a gate has no gradient for evolution to learn to exploit,
where a fitness term does.

Three checks, all computed from instances the verifier already executed:

  A. canonical template count -- mask every number and count the distinct
     skeletons, so a family that only permutes surface labels is visible.
  B. mathematical content -- the multiset of numbers appearing in the problem,
     across the seed family. This is the discriminating check. A CONSTANT
     answer is perfectly legitimate: an invariant or feasibility family
     ("find k - k") is a real problem family whose answer happens not to move.
     What is not legitimate is a family where NEITHER the numbers in the
     statement NOR the answer move -- then the seed changed nothing
     mathematical and only decorated a fixed computation.
  C. z-sensitivity of the solution trace -- what fraction of the values
     computed on the way to the answer actually depend on the seed. A
     source-level cross-check on B, since a generator can vary the printed
     numbers while computing a constant.

A and B read the rendered instances. C reads the program's own execution, so it
lives here rather than in the text linter.
"""

from __future__ import annotations

import ast
from .safe_parse import safe_ast_parse
import re
from dataclasses import dataclass

from .structural_fingerprint import exact_top_level_build_instance

# Masked in the canonical template: digits, and the quoted/bracketed entities a
# generator most often permutes while leaving the mathematics fixed.
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ConstancyVerdict:
    passed: bool
    reason: str = ""
    templates: int = 0
    answers: int = 0
    z_sensitive_fraction: float = 1.0


def canonical_template(problem: str) -> str:
    """The problem with every number replaced by N and whitespace collapsed."""
    return _WHITESPACE.sub(" ", _NUMBER.sub("N", str(problem or ""))).strip().lower()


def count_templates(problems: list[str]) -> int:
    return len({canonical_template(p) for p in problems})


def count_answers(answers: list[str]) -> int:
    return len({str(a).strip() for a in answers})


def _generate_function(tree: ast.Module) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "generate":
            return node
    return None


def _rng_names(generate: ast.FunctionDef, *, builder_entrypoint: bool = False) -> set[str]:
    """Names bound to a seeded RNG -- reading one IS reading the seed."""
    # The trusted wrapper constructs ``rng = random.Random(seed)`` and passes it
    # into ``build_instance(rng)``.  Inside that model-owned entrypoint the
    # argument is therefore the seed-tainted root even though ``seed`` itself is
    # deliberately absent.
    names = {"rng"} if builder_entrypoint else {"seed"}
    for node in ast.walk(generate):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        attr = getattr(func, "attr", None) or getattr(func, "id", None)
        if attr in {"Random", "seed"}:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def z_sensitive_fraction(source: str) -> float:
    """Share of ``generate``'s assigned names that depend on the seed.

    Computed as a transitive closure over the assignment graph starting from
    the seed and any RNG bound to it: a name is seed-dependent if it reads one.
    Returns 1.0 when the function cannot be parsed or assigns nothing, because
    a gate must not reject on its own inability to look.
    """
    try:
        tree = safe_ast_parse(source)
    except (SyntaxError, ValueError):
        return 1.0
    builder, builder_claimed = exact_top_level_build_instance(tree)
    if builder_claimed and builder is None:
        # A duplicate, rebound, or malformed builder must not inherit the
        # canonical wrapper's perfect-looking sensitivity.  Returning zero is
        # fail-closed at the archive's min_z_sensitive gate.
        return 0.0
    generate = builder or _generate_function(tree)
    if generate is None:
        return 1.0

    tainted = _rng_names(generate, builder_entrypoint=builder is not None)
    edges: list[tuple[set[str], set[str]]] = []
    for node in ast.walk(generate):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = {
                n.id
                for t in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for n in ast.walk(t)
                if isinstance(n, ast.Name)
            }
            reads = {
                n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)
            } if node.value is not None else set()
            if targets:
                edges.append((targets, reads))
        elif isinstance(node, ast.For):
            targets = {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
            reads = {n.id for n in ast.walk(node.iter) if isinstance(n, ast.Name)}
            if targets:
                edges.append((targets, reads))

    assigned = {name for targets, _ in edges for name in targets} - tainted
    if not assigned:
        return 1.0

    changed = True
    while changed:
        changed = False
        for targets, reads in edges:
            if reads & tainted and not targets <= tainted:
                tainted |= targets
                changed = True
    return len(assigned & tainted) / len(assigned)


def problem_numbers(problem: str) -> tuple[str, ...]:
    """The multiset of numbers printed in one problem, in order."""
    return tuple(_NUMBER.findall(str(problem or "")))


def numeric_content_varies(problems: list[str]) -> bool:
    """True if the numbers in the statement are not the same for every seed."""
    return len({problem_numbers(p) for p in problems}) > 1


def check_constancy(
    source: str,
    problems: list[str],
    answers: list[str],
    *,
    min_z_sensitive: float = 0.25,
) -> ConstancyVerdict:
    """Run the gates over one seed family.

    The rejection rule is deliberately narrow: a family fails only when neither
    the numbers in its statements nor its answers move across seeds. Requiring
    the ANSWERS to vary would reject a legitimate invariant family, and
    requiring more than one canonical template would reject the normal shape of
    a healthy generator -- one skeleton, different numbers.
    """
    templates = count_templates(problems)
    distinct_answers = count_answers(answers)
    sensitivity = z_sensitive_fraction(source)
    numbers_move = numeric_content_varies(problems)

    if not numbers_move and distinct_answers <= 1:
        return ConstancyVerdict(
            False,
            "neither the numbers in the problem nor the answer change across "
            "the seed family: the seed only decorates a fixed computation",
            templates, distinct_answers, sensitivity,
        )
    if sensitivity < min_z_sensitive:
        return ConstancyVerdict(
            False,
            f"only {sensitivity:.0%} of the values computed toward the answer "
            "depend on the seed",
            templates, distinct_answers, sensitivity,
        )
    return ConstancyVerdict(True, "", templates, distinct_answers, sensitivity)
