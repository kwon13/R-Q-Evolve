import pytest

from rq_evolve.scoring import compute_rq_full, estimate_pass_rate


def test_estimate_pass_rate():
    assert estimate_pass_rate([True, False, True, True]) == 0.75


def test_compute_rq_full():
    result = compute_rq_full([True, False, True, False], 2.0)
    assert result.p_hat == 0.5
    assert result.rq_score == 0.5
    assert result.num_correct == 2



def test_uncertainty_is_exploration_cost_not_a_per_token_rate():
    """H is summed over the trajectory and averaged over rollouts only.

    Dividing by the response length T would price a problem that keeps the
    solver uncertain for 800 tokens the same as one that does so for 80. The
    length normalisation is dropped on purpose; the /N average over rollouts
    stays. This pins the consequence: R_Q now scales with how long the solver
    stayed uncertain.
    """
    from rq_evolve.scoring import compute_rq_full

    short = compute_rq_full([True, False], uncertainty=80.0)
    long = compute_rq_full([True, False], uncertainty=800.0)
    assert long.rq_score == pytest.approx(10 * short.rq_score)


def test_nothing_clips_or_bounds_the_uncertainty_term():
    """H left the grid, so its magnitude only has to survive ranking.

    While H was a binning axis, h_range=[0.1, 0.6] would have collapsed every
    summed-entropy program into the top bin. It is fitness now, and fitness is
    compared, never bucketed.
    """
    from rq_evolve.scoring import compute_rq_value, selection_priority

    assert compute_rq_value(0.5, 1234.0) == pytest.approx(0.25 * 1234.0)
    assert selection_priority(0.5, 308.5) == pytest.approx(308.5)
    # Ranking is what matters, and it is preserved under any positive scale.
    a, b = compute_rq_value(0.5, 100.0), compute_rq_value(0.5, 200.0)
    assert (a < b) is (a * 7 < b * 7)
