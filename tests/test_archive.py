import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.program import ProblemProgram


def _program(group: str, value: int) -> ProblemProgram:
    return ProblemProgram(
        source_code=f'''
import random


def generate(seed):
    return "What is {value} + {{seed}}?", str({value} + seed)


CONCEPT_GROUP = "{group}"
CONCEPT_TYPE = "{group}.toy"
'''
    )


def test_random_selection_strategy_does_not_call_ucb():
    archive = MAPElitesArchive(selection_strategy="random")
    first = _program("algebra", 1)
    second = _program("geometry", 2)
    archive.try_insert(first, h_value=1.0, problem_text="algebra", rq_score=0.1)
    archive.try_insert(second, h_value=2.0, problem_text="geometry", rq_score=0.2)

    def fail_ucb(_occupied):
        raise AssertionError("random selection should not call _sample_ucb")

    archive._sample_ucb = fail_ucb
    assert archive.sample_parent() is not None


def test_unknown_selection_strategy_is_rejected():
    with pytest.raises(ValueError):
        MAPElitesArchive(selection_strategy="roulette")
