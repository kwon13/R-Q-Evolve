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


def test_every_occupied_niche_can_be_a_mutation_parent():
    """R_Q = 0 champions stay on the map AND reproduce.

    Filtering them out of the parent pool made them sterile: they held a cell
    and never mutated, so the cells most in need of new material -- the ones
    the policy cannot solve yet -- were exactly the ones evolution could not
    start from. On a 4B probe that left 5 of 10 champions eligible, two of them
    in the same skill column, and both evolved children landed there.

    R_Q = 0 covers two opposite cases and the test pins both: s_hat = 0
    (unsolvable today) and s_hat = 1 (solved outright).
    """
    def _distinct(group: str, skill: str, template: str) -> ProblemProgram:
        # Distinct problem TEMPLATES: the archive rejects a second champion whose
        # statement skeleton matches an existing one, which would make this test
        # about template dedup rather than about the parent pool.
        return ProblemProgram(
            source_code=(
                "import random\n\n\n"
                "def generate(seed):\n"
                f'    return f"{template}", str(seed)\n\n\n'
                f'GROUP = "{group}"\n'
                f'SKILL = "{skill}"\n'
            )
        )

    archive = MAPElitesArchive(selection_strategy="random")
    unsolvable = _distinct(
        "geometry", "extremal_principle", "Largest triangle with perimeter {seed}?"
    )
    solved_outright = _distinct(
        "algebra", "invariant", "Constant term after {seed} shifts?"
    )
    unsolvable.s_hat, unsolvable.u_score = 0.0, 0.4
    solved_outright.s_hat, solved_outright.u_score = 1.0, 0.4
    for program in (unsolvable, solved_outright):
        assert archive.try_insert(program, u_value=0.4, rq_score=0.0)
        # Neither is "learnable" -- that is precisely the case under test.
        assert archive._is_learnable(program) is False

    drawn = {archive.sample_parent().program_id for _ in range(60)}
    assert drawn == {unsolvable.program_id, solved_outright.program_id}, (
        "an archive of only R_Q=0 champions must still yield parents, and both "
        "cells must be reachable"
    )
