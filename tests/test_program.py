from pathlib import Path

from rq_evolve.program import ProblemProgram


def test_every_seed_program_executes():
    """Name no single seed: the set is curated and its members come and go."""
    root = Path(__file__).resolve().parents[1]
    seeds = sorted((root / "seed_programs").glob("*.py"))
    assert seeds, "the seed directory must not be empty"
    for path in seeds:
        program = ProblemProgram.from_file(path)
        inst = program.execute(seed=0)
        assert inst is not None, path.name
        assert inst.answer, path.name
        assert inst.problem, path.name

