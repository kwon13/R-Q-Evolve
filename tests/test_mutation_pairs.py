"""The four verified parent->child pairs, used as a contract fixture.

These are what a correct operator-A / operator-B mutation looks like. They are
NOT few-shot examples: injecting them costs +8k tokens against a 12k rollout
window, and greedy decoding copies examples instead of mutating the live parent.
They live here so the operator contract, the axis labels and the generator
contract are pinned by something that actually runs.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from rq_evolve.code_utils import (
    lint_generator_source,
    lint_mutation_generator_source,
    lint_problem_instance,
)
from rq_evolve.concepts import GROUPS, SKILLS
from rq_evolve.evolution import RQEvolver
from rq_evolve.program import ProblemProgram

FIXTURES = Path(__file__).parent / "fixtures" / "mutation_pairs"
FILES = {
    "in_depth": FIXTURES / "operator_A_skill_within_group.txt",
    "in_breadth": FIXTURES / "operator_B_group_within_skill.txt",
}

# Every flag the mutation prompt's skeleton promises. require_answer_routes is
# excluded: the templates use one `answer` plus a semantic assert, not the
# retired answer_insight/answer_brute pair.
SKELETON_CONTRACT = dict(
    require_assert=True,
    reject_trivial_assert=True,
    reject_unbounded_sampling=True,
    require_answer_routes=False,
    require_canonical_instance_data=True,
    require_mechanical_shape=True,
)


def _pairs(op: str) -> list[tuple[ProblemProgram, ProblemProgram]]:
    blocks = re.findall(
        r"```python\n(.*?)```", FILES[op].read_text(encoding="utf-8"), re.S
    )
    assert len(blocks) % 2 == 0 and blocks, f"{op}: expected parent/child pairs"
    return [
        (ProblemProgram(source_code=blocks[i]), ProblemProgram(source_code=blocks[i + 1]))
        for i in range(0, len(blocks), 2)
    ]


ALL_PROGRAMS = [
    pytest.param(program, id=f"{op}-{i // 2}-{'parent' if i % 2 == 0 else 'child'}")
    for op in FILES
    for i, program in enumerate(p for pair in _pairs(op) for p in pair)
]
ALL_PAIRS = [
    pytest.param(op, parent, child, id=f"{op}-{i}")
    for op in FILES
    for i, (parent, child) in enumerate(_pairs(op))
]


@pytest.mark.parametrize("program", ALL_PROGRAMS)
def test_fixture_satisfies_the_full_generator_contract(program):
    assert lint_generator_source(program.source_code) == []
    assert lint_mutation_generator_source(program.source_code, **SKELETON_CONTRACT) == []
    assert program.declared_group() in GROUPS
    assert program.declared_skill() in SKILLS


@pytest.mark.parametrize("program", ALL_PROGRAMS)
def test_fixture_executes_and_varies_across_seeds(program):
    problems = set()
    for seed in range(5):
        instance = program.execute(seed=seed)
        assert instance is not None, f"seed {seed} crashed or hit the 5s sandbox kill"
        assert lint_problem_instance(instance) == []
        assert re.fullmatch(r"-?\d+", instance.answer.strip()), instance.answer
        problems.add(" ".join(instance.problem.split()))
    assert len(problems) >= 2, "the visible problem must change across seeds"


@pytest.mark.parametrize("op,parent,child", ALL_PAIRS)
def test_fixture_pair_satisfies_its_operator_contract(op, parent, child):
    task = SimpleNamespace(op=op, parent=parent)
    assert RQEvolver._validate_mutation_contract(task, child) is None


@pytest.mark.parametrize("op,parent,child", ALL_PAIRS)
def test_operator_moves_exactly_one_axis(op, parent, child):
    """The point of the pair: one axis is held, the other genuinely moves."""
    if op == "in_depth":
        assert child.declared_group() == parent.declared_group()
        assert child.declared_skill() != parent.declared_skill()
    else:
        assert child.declared_skill() == parent.declared_skill()
        assert child.declared_group() != parent.declared_group()


@pytest.mark.parametrize("op,parent,child", ALL_PAIRS)
def test_child_records_its_evidence_header(op, parent, child):
    header = "\n".join(
        line for line in child.source_code.splitlines() if line.startswith("#")
    )
    assert "CRUX:" in header
    assert "BYPASS_BLOCKED:" in header
    # Operator A must argue the parent's move cannot reach the child's answer;
    # operator B must name the relation of the new domain it actually uses.
    assert ("PARENT_CRUX_FAILS:" if op == "in_depth" else "NATIVE_STRUCTURE:") in header


def test_fixtures_are_not_on_the_few_shot_injection_path():
    """Injecting them costs +8k tokens against a 12k window -- keep them out."""
    from rq_evolve.prompts import SHOT_TEMPLATE_DIR, _load_shot_examples

    for op in FILES:
        assert _load_shot_examples(op) == ""
        assert not (SHOT_TEMPLATE_DIR / FILES[op].name).exists()


def test_the_retired_concept_vocabulary_is_gone_everywhere():
    for path in FILES.values():
        assert not re.search(r"CONCEPT_(GROUP|TYPE|REASON)", path.read_text("utf-8"))
