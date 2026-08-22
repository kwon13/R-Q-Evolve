import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.program import ProblemProgram


def _program(
    group: str, value: int, skill: str = "transformation"
) -> ProblemProgram:
    return ProblemProgram(
        source_code=f'''
import random


def generate(seed):
    return f"What is {value} + {{seed}}?", str({value} + seed)


GROUP = "{group}"
SKILL = "{skill}"
'''
    )


def test_random_selection_strategy_does_not_call_ucb():
    archive = MAPElitesArchive(selection_strategy="random")
    first = _program("algebra", 1)
    second = _program("geometry", 2)
    archive.try_insert(first, u_value=1.0, rq_score=0.1)
    archive.try_insert(second, u_value=2.0, rq_score=0.2)

    def fail_ucb(_occupied):
        raise AssertionError("random selection should not call _sample_ucb")

    archive._sample_ucb = fail_ucb
    assert archive.sample_parent() is not None


def test_unknown_selection_strategy_is_rejected():
    with pytest.raises(ValueError):
        MAPElitesArchive(selection_strategy="roulette")


def test_seed_variation_allows_varied_problems_with_constant_answer():
    program = ProblemProgram(
        source_code='''
def generate(seed):
    return f"Find the value of {seed} - {seed}.", "0"

GROUP = "algebra"
SKILL = "invariant"
'''
    )
    archive = MAPElitesArchive()

    assert archive._passes_seed_variation(program) is True
    assert program.metadata["validity_check"]["n_distinct_problems"] == 5
    assert program.metadata["validity_check"]["n_distinct_answers"] == 1


def test_seed_variation_rejects_unchanged_visible_problem_with_hidden_answers():
    program = ProblemProgram(
        source_code='''
def generate(seed):
    return "Find the hidden number.", str(seed)

GROUP = "algebra"
SKILL = "construction"
'''
    )
    archive = MAPElitesArchive()

    assert archive._passes_seed_variation(program) is False
