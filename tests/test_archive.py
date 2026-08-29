import hashlib

import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.problem_type import PROBLEM_TYPE_RULESET, problem_type_ruleset_sha256
from rq_evolve.program import ProblemProgram


def _certify(
    program: ProblemProgram, domain: str, problem_type: str
) -> ProblemProgram:
    """Emulate the metadata written by ``RQEvolver.verify_program``."""
    program.metadata["problem_type"] = problem_type
    program.metadata["descriptor_contract"] = {
        "domain_authority": "source_exact_one_literal",
        "problem_type_authority": "deterministic_statement_and_verifier",
        "problem_type_ruleset": PROBLEM_TYPE_RULESET,
        "problem_type_ruleset_sha256": problem_type_ruleset_sha256(),
        "verified_seeds": 5,
        "domain": domain,
        "problem_type": problem_type,
        "source_sha256": hashlib.sha256(
            program.source_code.encode("utf-8")
        ).hexdigest(),
    }
    return program


def _program(domain: str, value: int, problem_type: str = "function") -> ProblemProgram:
    program = ProblemProgram(
        source_code=f"""
import random


def generate(seed):
    return f"What is {value} + {{seed}}?", str({value} + seed)


DOMAIN = "{domain}"
"""
    )
    return _certify(program, domain, problem_type)


def _counting_program(domain: str, value: int) -> ProblemProgram:
    program = ProblemProgram(
        source_code=f'''
import random


def generate(seed):
    return f"How many integers lie between {{seed}} and {{seed + {value}}}?", str({value} + 1)


DOMAIN = "{domain}"
'''
    )
    return _certify(program, domain, "counting")


def test_random_selection_strategy_does_not_call_ucb():
    archive = MAPElitesArchive(selection_strategy="random")
    first = _program("algebra", 1)
    second = _counting_program("geometry", 2)
    archive.try_insert(first, u_value=1.0, rq_score=0.1)
    archive.try_insert(second, u_value=2.0, rq_score=0.2)

    def fail_ucb(_occupied):
        raise AssertionError("random selection should not call _sample_ucb")

    archive._sample_ucb = fail_ucb
    assert archive.sample_parent() is not None


def test_unknown_selection_strategy_is_rejected():
    with pytest.raises(ValueError):
        MAPElitesArchive(selection_strategy="roulette")


def test_sliding_window_mce_requires_positive_window():
    with pytest.raises(ValueError, match="mce_window_iterations"):
        MAPElitesArchive(
            selection_strategy="sliding_window_mce",
            mce_window_iterations=0,
        )


def test_sliding_window_mce_uses_virtual_pulls_and_survival_rate():
    archive = MAPElitesArchive(
        selection_strategy="sliding_window_mce",
        ucb_c=1 / (2**0.5),
        mce_window_iterations=3,
    )
    first = _program("algebra", 1)
    second = _counting_program("geometry", 2)
    assert archive.try_insert(first, u_value=1.0, rq_score=0.1)
    assert archive.try_insert(second, u_value=1.0, rq_score=0.1)

    archive.begin_selection_iteration(0)
    selected_first = archive.sample_parent()
    selected_second = archive.sample_parent()
    assert selected_first is not None
    assert selected_second is not None
    # The first unresolved pull acts as a virtual loss, so the other unvisited
    # cell receives the second slot instead of one arm monopolising the batch.
    assert selected_first.program_id != selected_second.program_id

    archive.record_parent_outcome(selected_first.program_id, True, iteration=0)
    archive.record_parent_outcome(selected_second.program_id, False, iteration=0)

    selected_third = archive.sample_parent()
    assert selected_third is not None
    # Equal pull counts give equal exploration bonuses; MCE exploitation is the
    # offspring survival rate, not either champion's fitness.
    assert selected_third.program_id == selected_first.program_id
    archive.record_parent_outcome(selected_third.program_id, False, iteration=0)


def test_sliding_window_mce_forgets_expired_outcomes():
    archive = MAPElitesArchive(
        selection_strategy="sliding_window_mce",
        mce_window_iterations=3,
    )
    first = _program("algebra", 1)
    second = _counting_program("geometry", 2)
    assert archive.try_insert(first, u_value=1.0, rq_score=0.1)
    assert archive.try_insert(second, u_value=1.0, rq_score=0.1)

    archive.begin_selection_iteration(0)
    selected = archive.sample_parent()
    assert selected is not None
    archive.record_parent_outcome(selected.program_id, True, iteration=0)
    assert sum(len(n.mce_history) for n in archive.grid.values()) == 1

    archive.begin_selection_iteration(3)
    assert sum(len(n.mce_history) for n in archive.grid.values()) == 0


def test_sliding_window_mce_history_round_trips_in_snapshot():
    archive = MAPElitesArchive(
        selection_strategy="sliding_window_mce",
        mce_window_iterations=7,
    )
    program = _program("algebra", 1)
    assert archive.try_insert(program, u_value=1.0, rq_score=0.1)
    archive.begin_selection_iteration(4)
    parent = archive.sample_parent()
    assert parent is not None
    archive.record_parent_outcome(parent.program_id, True, iteration=4)

    restored = MAPElitesArchive(
        selection_strategy="sliding_window_mce",
        mce_window_iterations=7,
    )
    assert restored.load_payload(archive.to_payload()) == 1
    cell = restored.program_to_cell(program)
    assert cell is not None
    assert restored.grid[cell].mce_history == [{"iteration": 4, "reward": 1}]


def test_seed_variation_allows_varied_problems_with_constant_answer():
    program = ProblemProgram(
        source_code="""
def generate(seed):
    return f"Find the value of {seed} - {seed}.", "0"

DOMAIN = "algebra"
"""
    )
    archive = MAPElitesArchive()

    assert archive._passes_seed_variation(program) is True
    assert program.metadata["validity_check"]["n_distinct_problems"] == 5
    assert program.metadata["validity_check"]["n_distinct_answers"] == 1


def test_seed_variation_rejects_unchanged_visible_problem_with_hidden_answers():
    program = ProblemProgram(
        source_code="""
def generate(seed):
    return "Find the hidden number.", str(seed)

DOMAIN = "algebra"
"""
    )
    archive = MAPElitesArchive()

    assert archive._passes_seed_variation(program) is False


def test_every_occupied_niche_can_be_a_mutation_parent():
    """R_Q = 0 champions stay on the map AND reproduce.

    Filtering them out of the parent pool made them sterile: they held a cell
    and never mutated, so the cells most in need of new material -- the ones
    the policy cannot solve yet -- were exactly the ones evolution could not
    start from. On a 4B probe that left 5 of 10 champions eligible, two of them
    in the same problem-type column, and both evolved children landed there.

    R_Q = 0 covers two opposite cases and the test pins both: s_hat = 0
    (unsolvable today) and s_hat = 1 (solved outright).
    """

    def _distinct(domain: str, problem_type: str, template: str) -> ProblemProgram:
        # Distinct problem TEMPLATES: the archive rejects a second champion whose
        # statement skeleton matches an existing one, which would make this test
        # about template dedup rather than about the parent pool.
        program = ProblemProgram(
            source_code=(
                "import random\n\n\n"
                "def generate(seed):\n"
                f'    return f"{template}", str(seed)\n\n\n'
                f'DOMAIN = "{domain}"\n'
            )
        )
        return _certify(program, domain, problem_type)

    archive = MAPElitesArchive(selection_strategy="random")
    unsolvable = _distinct(
        "geometry", "optimization", "Largest triangle with perimeter {seed}?"
    )
    solved_outright = _distinct(
        "algebra", "function", "Constant term after {seed} shifts?"
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


def _family(domain: str, problem_type: str, statement: str) -> ProblemProgram:
    program = ProblemProgram(
        source_code=(
            "import random\n\n\n"
            "def generate(seed):\n"
            "    rng = random.Random(seed)\n"
            "    n = rng.randint(10, 99)\n"
            # The answer must not be a number the statement prints, or the
            # instance lint rejects it before the duplicate gate is reached.
            "    answer = n * 7 + 3\n"
            f'    return f"{statement}", str(answer)\n\n\n'
            f'DOMAIN = "{domain}"\n'
        )
    )
    return _certify(program, domain, problem_type)


def test_a_restatement_of_another_cell_is_not_new_coverage():
    """Exact template hashes only catch an exact match. Live, that let

        Let n = N. How many distinct prime factors does n have?
        Let n = N be a positive integer. How many distinct prime factors does n have?

    hold two different cells: five words apart, two different hashes. With
    noisy problem-type labels can make the restatement land somewhere else and
    report coverage the curriculum does not have.
    """
    archive = MAPElitesArchive(selection_strategy="random")
    first = _family(
        "number_theory",
        "counting",
        "Let n = {n}. How many distinct prime factors does n have? State only the integer.",
    )
    restated = _family(
        "number_theory",
        "function",
        "Let n = {n} be a positive integer. How many distinct prime factors does n have? State only the integer.",
    )
    assert archive.try_insert(first, u_value=0.4, rq_score=0.2)
    assert not archive.try_insert(
        restated, u_value=0.4, rq_score=0.9
    ), "a higher score must not buy a second cell for the same question"
    assert restated.metadata["archive_status"] == "near_duplicate_template_rejected"
    assert restated.metadata["duplicate_of"] == first.program_id
    assert restated.metadata["duplicate_ratio"] >= 0.9
    assert len(archive.champions()) == 1


def test_duplicate_preflight_rejects_without_mutating_archive():
    """Score-independent novelty gates can run before solver rollouts."""

    archive = MAPElitesArchive(selection_strategy="random")
    first = _family(
        "number_theory",
        "counting",
        "Let n = {n}. How many positive divisors does n have? State only the integer.",
    )
    restated = _family(
        "number_theory",
        "function",
        "Let n = {n} be positive. How many positive divisors does n have? State only the integer.",
    )
    assert archive.try_insert(first, u_value=0.4, rq_score=0.2)
    before = archive.to_payload()

    assert archive.passes_admission_preflight(restated) is False
    assert restated.metadata["archive_status"] in {
        "near_duplicate_template_rejected",
        "structural_duplicate_rejected",
    }
    assert archive.to_payload() == before


def test_a_genuinely_different_question_still_gets_its_cell():
    """The gate is text similarity, which is a weak proxy for mathematical
    sameness -- so it sits high enough that neighbouring questions survive."""
    archive = MAPElitesArchive(selection_strategy="random")
    divisors = _family(
        "number_theory",
        "counting",
        "Let n = {n}. How many positive divisors does n have? State only the integer.",
    )
    triangle = _family(
        "geometry",
        "optimization",
        "A triangle has integer sides and perimeter {n}. Find its greatest possible area doubled. State only the integer.",
    )
    assert archive.try_insert(divisors, u_value=0.4, rq_score=0.2)
    assert archive.try_insert(triangle, u_value=0.4, rq_score=0.2)
    assert len(archive.champions()) == 2
