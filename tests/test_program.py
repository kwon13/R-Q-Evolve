from pathlib import Path

from rq_evolve.program import ProblemProgram


def test_seed_program_executes():
    root = Path(__file__).resolve().parents[1]
    program = ProblemProgram.from_file(root / "seed_programs" / "01_gcd_lcm.py")
    inst = program.execute(seed=0)
    assert inst is not None
    assert inst.answer
    assert inst.problem

