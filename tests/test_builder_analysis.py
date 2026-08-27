"""Compatibility checks for trusted build_instance source assembly."""

import ast

import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.code_utils import compile_stage2_reply
from rq_evolve.constancy import check_constancy, z_sensitive_fraction
from rq_evolve.program import ProblemProgram
from rq_evolve.structural_fingerprint import (
    render_fingerprint,
    structural_fingerprint,
)


STAGE2_REPLY = '''MODE: expression
CORE:
```python
def build_instance(rng):
    n = rng.randint(2, 20)
    answer = n * 2
    check = n + n
    constant_bookkeeping = 17
    parameters = {"n": n}
    return parameters, answer, check
```
'''
ASSEMBLED, COMPILE_ERROR = compile_stage2_reply(
    STAGE2_REPLY,
    "Let n = [[n]]. Compute twice n.",
)
assert ASSEMBLED is not None, COMPILE_ERROR


def _flatten(node: ast.AST) -> tuple[str, ...]:
    result: list[str] = []

    def walk(current: ast.AST) -> None:
        result.append(type(current).__name__)
        for child in ast.iter_child_nodes(current):
            walk(child)

    walk(node)
    return tuple(result)


def test_builder_rng_argument_is_the_constancy_taint_root():
    # Five assigned values live in build_instance: n/answer/check/parameters are
    # tainted, constant_bookkeeping is not.  The much larger canonical wrapper
    # must contribute no nodes to this calculation.
    assert z_sensitive_fraction(ASSEMBLED) == pytest.approx(4 / 5)


def test_duplicate_or_malformed_builder_fails_closed_for_constancy():
    duplicate = ASSEMBLED.replace(
        "def build_instance(rng):",
        "def build_instance(rng):\n    return {}, 1, 1, {}\n\ndef build_instance(rng):",
        1,
    )
    malformed = ASSEMBLED.replace("def build_instance(rng):", "def build_instance(seed):")
    annotated = ASSEMBLED.replace(
        "def build_instance(rng):", "def build_instance(rng) -> tuple:"
    )

    assert z_sensitive_fraction(duplicate) == 0.0
    assert z_sensitive_fraction(malformed) == 0.0
    assert z_sensitive_fraction(annotated) == 0.0
    verdict = check_constancy(
        duplicate,
        ["Compute twice 2.", "Compute twice 3."],
        ["4", "6"],
    )
    assert verdict.passed is False
    assert verdict.z_sensitive_fraction == 0.0


def test_legacy_constancy_analysis_still_targets_generate():
    legacy = '''
import random
def generate(seed):
    rng = random.Random(seed)
    a = rng.randint(1, 9)
    b = a * 2
    c = 17
    return f"{a} {b} {c}", str(b)
'''
    assert z_sensitive_fraction(legacy) == pytest.approx(2 / 3)


def test_structural_fingerprint_reads_builder_routes_not_wrapper_routes():
    fingerprint = structural_fingerprint(ASSEMBLED)

    assert fingerprint is not None
    assert fingerprint["entrypoint"] == "build_instance"
    assert fingerprint["n_params"] == 1
    assert "single arithmetic" in fingerprint["answer_shape"]
    assert fingerprint["check_reads_answer"] is False
    assert "inline in build_instance()" in render_fingerprint(fingerprint)


def test_structural_fingerprint_rejects_ambiguous_builder_name():
    rebound = ASSEMBLED + "\nbuild_instance = generate\n"
    duplicate = ASSEMBLED.replace(
        "def build_instance(rng):",
        "def build_instance(rng):\n    return {}, 1, 1, {}\n\ndef build_instance(rng):",
        1,
    )

    assert structural_fingerprint(rebound) is None
    assert structural_fingerprint(duplicate) is None


def test_archive_skeleton_excludes_the_canonical_wrapper_and_uses_v2_cache():
    program = ProblemProgram(
        source_code=ASSEMBLED,
        metadata={"_ast_skeleton": ["stale-whole-module-cache"]},
    )
    tree = ast.parse(ASSEMBLED)
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_instance"
    )

    skeleton = MAPElitesArchive().program_skeleton(program)

    assert skeleton == _flatten(builder)
    assert skeleton != ("stale-whole-module-cache",)
    assert program.metadata["_ast_skeleton_v2_builder_aware"] == list(skeleton)


def test_archive_skeleton_preserves_legacy_whole_module_shape():
    legacy = "def generate(seed):\n    answer = seed + 1\n    return str(seed), str(answer)\n"
    program = ProblemProgram(source_code=legacy)

    assert MAPElitesArchive().program_skeleton(program) == _flatten(ast.parse(legacy))


def test_archive_skeleton_does_not_fingerprint_a_duplicate_builder_wrapper():
    duplicate = ASSEMBLED.replace(
        "def build_instance(rng):",
        "def build_instance(rng):\n    return {}, 1, 1, {}\n\ndef build_instance(rng):",
        1,
    )

    assert MAPElitesArchive().program_skeleton(
        ProblemProgram(source_code=duplicate)
    ) is None
