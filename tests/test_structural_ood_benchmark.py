from pathlib import Path

from rq_evolve.ast_contract import check_generator_contract, check_problem_text
from rq_evolve.code_utils import lint_generator_source, lint_problem_instance
from rq_evolve.concepts import validate_label_decl
from rq_evolve.evolved_performance import build_seed_id_rows
from rq_evolve.program import ProblemProgram


ROOT = Path(__file__).resolve().parent.parent
CHALLENGE_DIR = ROOT / "challenge_seed_programs" / "structural_ood_v2"
EXPECTED_LABELS = {
    ("sequence", "induction"),
    ("algebra", "transformation"),
    ("inequality", "transformation"),
    ("number_theory", "transformation"),
    ("combinatorics", "counting"),
    ("algebra", "casework"),
}


def test_structural_ood_generators_are_valid_and_vary():
    paths = sorted(CHALLENGE_DIR.glob("*.py"))
    assert len(paths) == 6
    observed_labels = set()
    for path in paths:
        program = ProblemProgram.from_file(path)
        assert lint_generator_source(program.source_code) == []
        assert check_generator_contract(program.source_code) == []
        assert validate_label_decl(
            program.declared_group(), program.declared_skill()
        ) == []
        observed_labels.add((program.declared_group(), program.declared_skill()))

        instances = [program.execute(seed) for seed in range(5)]
        assert all(instance is not None for instance in instances)
        assert len({instance.problem for instance in instances}) > 1
        for instance in instances:
            assert lint_problem_instance(instance) == []
            assert check_problem_text(instance.problem) == []

    assert observed_labels == EXPECTED_LABELS


def test_structural_ood_builder_records_distinct_benchmark_name():
    rows, programs = build_seed_id_rows(
        CHALLENGE_DIR,
        examples_per_program=3,
        seed_start=3_000_000,
        benchmark_name="evolved_performance_structural_ood_v2",
    )
    assert len(rows) == 18
    assert len(programs) == 6
    assert {row["benchmark"] for row in rows} == {
        "evolved_performance_structural_ood_v2"
    }
    assert {(row["group"], row["skill"]) for row in rows} == EXPECTED_LABELS
    assert len({row["instance_sha256"] for row in rows}) == len(rows)
