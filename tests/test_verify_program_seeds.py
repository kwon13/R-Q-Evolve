"""verify_program's contract with the seed stream.

The candidate's FIRST scoring draw has to be a seed nothing has already looked
at. verify_program renders seeds 0..n-1 for its structural checks and derives
one descriptor contract across those instances, so unless the stream is moved
past them, `take` hands back seed 0 and the candidate is admitted on the very
instance every gate already ran against. Nothing else advances the cursor for a
new program_id.

This file exists because the call that does it was written against a method
name SeedStream does not have. Every unit test passed -- none of them ran
verify_program -- and the run died on its first seed program.
"""

from pathlib import Path

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.config import EvolutionConfig
from rq_evolve.evolution import RQEvolver
from rq_evolve.program import ProblemProgram

SOURCE = """
import random


def generate(seed):
    rng = random.Random(seed)
    n = rng.randint(3, 40)

    answer = n * (n + 1) // 2

    check = 0
    for k in range(1, n + 1):
        check += k

    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Let n = {n}. Compute the sum of the first n positive integers. "
        f"State only the integer."
    )
    return problem, str(answer), {"mode": "expression"}


DOMAIN = "algebra"
"""


class _Backend:
    max_model_len = None

    def mutate(self, tasks):
        return [None] * len(tasks)


def _evolver(**kw) -> RQEvolver:
    return RQEvolver(
        archive=MAPElitesArchive(),
        backend=_Backend(),
        evolution_config=EvolutionConfig(**kw),
    )


def _program() -> ProblemProgram:
    return ProblemProgram(source_code=SOURCE, generation=0)


def test_a_clean_program_verifies():
    evolver = _evolver(verify_seeds=5)
    inst, reason = evolver.verify_program(_program())
    assert inst is not None, reason


def test_the_stream_is_moved_past_the_seeds_verification_rendered():
    evolver = _evolver(verify_seeds=5)
    program = _program()
    assert evolver.seed_stream.peek(program.program_id) == 0

    inst, reason = evolver.verify_program(program)
    assert inst is not None, reason
    assert evolver.seed_stream.peek(program.program_id) == 5


def test_the_first_scoring_draw_is_not_an_instance_a_gate_already_saw():
    """The whole point: `take` must not hand back a verified seed."""
    evolver = _evolver(verify_seeds=5)
    program = _program()
    evolver.verify_program(program)

    drawn = evolver.seed_stream.take(program.program_id, 3)
    assert drawn == [5, 6, 7]
    assert not set(drawn) & set(range(5))


def test_verifying_twice_does_not_rewind_the_cursor():
    evolver = _evolver(verify_seeds=5)
    program = _program()
    evolver.verify_program(program)
    evolver.seed_stream.take(program.program_id, 4)  # cursor -> 9
    evolver.verify_program(program)
    assert evolver.seed_stream.peek(program.program_id) == 9


def test_a_single_seed_verification_still_advances_by_one():
    evolver = _evolver(verify_seeds=1)
    program = _program()
    inst, reason = evolver.verify_program(program)
    assert inst is not None, reason
    assert evolver.seed_stream.peek(program.program_id) == 1


def test_the_shipped_seed_programs_all_verify():
    """load_seed_programs runs this on every file; a miss aborts the run."""
    evolver = _evolver()
    seed_dir = Path(__file__).resolve().parent.parent / "seed_programs_domain_type"
    programs = evolver.load_seed_programs(seed_dir)
    rejected = [e for e in evolver.events if e.get("event") == "seed_rejected"]
    assert not rejected, rejected
    assert len(programs) == len(list(seed_dir.glob("*.py")))


def test_the_seven_shipped_seeds_cover_the_approved_distinct_cells():
    """The first five form the literal diagonal; the two extra rows stay unique."""
    expected = {
        "00_algebra_decision.py": ("algebra", "decision"),
        "01_geometry_search.py": ("geometry", "search"),
        "02_number_theory_counting.py": ("number_theory", "counting"),
        "03_discrete_mathematics_optimization.py": (
            "discrete_mathematics",
            "optimization",
        ),
        "04_applied_mathematics_function.py": ("applied_mathematics", "function"),
        "05_calculus_decision.py": ("calculus", "decision"),
        "06_precalculus_search.py": ("precalculus", "search"),
    }
    evolver = _evolver()
    seed_dir = Path(__file__).resolve().parent.parent / "seed_programs_domain_type"
    programs = evolver.load_seed_programs(seed_dir)
    actual = {
        program.metadata["source_file"]: (
            program.get_domain(),
            program.get_problem_type(),
        )
        for program in programs
    }

    assert actual == expected
    assert len(set(actual.values())) == 7
    assert {domain for domain, _ in actual.values()} == {
        "algebra",
        "geometry",
        "number_theory",
        "discrete_mathematics",
        "applied_mathematics",
        "calculus",
        "precalculus",
    }
