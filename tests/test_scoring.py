from rq_evolve.scoring import compute_rq_full, estimate_pass_rate


def test_estimate_pass_rate():
    assert estimate_pass_rate([True, False, True, True]) == 0.75


def test_compute_rq_full():
    result = compute_rq_full([True, False, True, False], 2.0)
    assert result.p_hat == 0.5
    assert result.rq_score == 0.5
    assert result.num_correct == 2

