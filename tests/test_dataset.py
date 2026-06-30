from rq_evolve.dataset import build_training_examples
from rq_evolve.program import ProblemProgram


def _program(
    label: str,
    *,
    rq_score: float,
    p_hat: float = 0.5,
    h_score: float = 0.1,
    h_bin: int = 0,
    d_bin: int = 0,
) -> ProblemProgram:
    return ProblemProgram(
        program_id=label,
        p_hat=p_hat,
        h_score=h_score,
        rq_score=rq_score,
        niche_h=h_bin,
        niche_div=d_bin,
        source_code=f'''
def generate(seed):
    value = {len(label) * 100} + seed
    return "{label} seed " + str(seed), str(value)
''',
    )


def test_training_examples_prioritize_global_rq_when_budget_binds():
    low_rq_high_h = _program("low", rq_score=0.1, h_bin=9, d_bin=0)
    high_rq_low_h = _program("high", rq_score=0.9, h_bin=0, d_bin=0)
    mid_rq = _program("mid", rq_score=0.5, h_bin=5, d_bin=0)

    examples = build_training_examples(
        [low_rq_high_h, high_rq_low_h, mid_rq],
        instances_per_program=2,
        training_budget=3,
        frontier_p_hat_range=(0.0, 1.0),
        n_h_bins=10,
        n_div_bins=1,
        used_seeds={},
        strict_anti_reuse=True,
    )

    assert [row["program_id"] for row in examples] == ["high", "high", "mid"]
    assert [row["rq_score"] for row in examples] == [0.9, 0.9, 0.5]
