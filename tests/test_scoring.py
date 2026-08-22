"""The n x m estimator: what R_Q measures and what it refuses to measure."""

import random
import statistics

import pytest

from rq_evolve.scoring import (
    compute_rq_program,
    estimate_dispersion,
    estimate_success_rate,
    score_seed,
    selection_priority,
    unbiased_learnability,
)


def _seed(z, correct, entropies=None, m=None):
    flags = list(correct)
    return score_seed(z, flags, entropies if entropies is not None else [1.0] * len(flags))


def test_estimate_success_rate():
    assert estimate_success_rate([True, False, True, True]) == 0.75


# --- Lemma 1: the m/(m-1) correction ---------------------------------------


@pytest.mark.parametrize("m,s_true", [(2, 0.5), (2, 0.3), (4, 0.5), (10, 0.3)])
def test_learnability_is_unbiased(m, s_true):
    """The plug-in estimates ((m-1)/m)L, which at m=2 is half the true value."""
    random.seed(7)
    n = 200_000
    mean = sum(
        unbiased_learnability(
            estimate_success_rate([random.random() < s_true for _ in range(m)]), m
        )
        for _ in range(n)
    ) / n
    assert mean == pytest.approx(s_true * (1 - s_true), abs=0.005)


def test_a_single_rollout_cannot_identify_learnability():
    """m=1 makes s_hat 0 or 1, so the plug-in is identically 0 for everything.

    Reporting 0 is the honest answer, and it is why m >= 2 is a config
    requirement rather than a tuning preference.
    """
    assert unbiased_learnability(0.0, 1) == 0.0
    assert unbiased_learnability(1.0, 1) == 0.0


# --- Corollary 2.1: pooling measures the wrong thing ------------------------


def test_a_trivial_impossible_mix_scores_zero_not_maximum():
    """Half trivial, half impossible: pooled learnability is at its MAXIMUM.

    Every instance is degenerate, so the program supplies no learning signal
    anywhere. A pooled fitness would rank it top; the per-seed average gives 0.
    Such a program needs no calibration against the solver to write, which is
    why the pooled form would be a cheaper attack surface than single-instance
    scoring was.
    """
    program = [_seed(z, [True] * 4, [1.0] * 4) for z in range(3)]
    program += [_seed(z, [False] * 4, [5.0] * 4) for z in range(3, 6)]
    result = compute_rq_program(program)

    assert result.s_hat == pytest.approx(0.5)
    assert result.s_hat * (1 - result.s_hat) == pytest.approx(0.25)  # pooled: max
    assert result.learnability == 0.0
    assert result.rq_score == 0.0
    assert result.dispersion > 0.0


# --- the per-seed product ---------------------------------------------------


def test_a_degenerate_seeds_entropy_cannot_reach_fitness():
    """Impossible instances provoke long, confused, high-entropy rollouts.

    Multiplying the AVERAGES lets that entropy inflate fitness through the
    second factor. Multiplying per seed zeroes the seed's whole term.
    """
    result = compute_rq_program([
        _seed(0, [True, False], [1.0, 1.0]),    # L > 0, U = 1
        _seed(1, [False, False], [9.0, 9.0]),   # L = 0, U = 9
    ])
    product_of_means = result.learnability * result.u_score

    assert result.rq_score == pytest.approx(0.25)
    assert product_of_means == pytest.approx(1.25)
    assert result.rq_score < product_of_means


# --- Lemma 3: dispersion ----------------------------------------------------


def test_dispersion_is_unbiased():
    """Raw variance of s_hat overstates D by W/m; the correction removes it."""
    truths = (0.2, 0.4, 0.5, 0.6, 0.8)
    random.seed(3)
    n = 6000
    mean = sum(
        estimate_dispersion([
            _seed(z, [random.random() < s for _ in range(4)], [1.0] * 4)
            for z, s in enumerate(truths)
        ])
        for _ in range(n)
    ) / n
    assert mean == pytest.approx(statistics.variance(truths), abs=0.01)


def test_dispersion_of_identical_instances_is_zero():
    random.seed(11)
    n = 6000
    mean = sum(
        estimate_dispersion([
            _seed(z, [random.random() < 0.5 for _ in range(4)], [1.0] * 4)
            for z in range(5)
        ])
        for _ in range(n)
    ) / n
    assert mean == pytest.approx(0.0, abs=0.01)


def test_dispersion_needs_two_seeds():
    assert estimate_dispersion([_seed(0, [True, False])]) == 0.0
    assert estimate_dispersion([]) == 0.0


def test_dispersion_is_reported_but_never_priced():
    """Rewarding consistency directly hands a free maximum to a generator that
    ignores its seed; the defence against those is a validity gate."""
    consistent = compute_rq_program([_seed(z, [True, False]) for z in range(4)])
    spread = compute_rq_program(
        [_seed(0, [True, False]), _seed(1, [True, True]),
         _seed(2, [False, False]), _seed(3, [True, False])]
    )
    assert consistent.dispersion < spread.dispersion
    # ...and fitness is the per-seed average either way, with no D term in it.
    assert consistent.rq_score == pytest.approx(
        sum(st.contribution for st in consistent.per_seed) / len(consistent.per_seed)
    )


# --- aggregate shape --------------------------------------------------------


def test_the_result_carries_the_per_seed_breakdown():
    result = compute_rq_program([_seed(z, [True, False]) for z in range(5)])
    assert result.num_seeds == 5
    assert result.num_rollouts == 10
    assert result.num_correct == 5
    assert [st.seed for st in result.per_seed] == [0, 1, 2, 3, 4]


def test_an_empty_seed_set_scores_zero_rather_than_raising():
    result = compute_rq_program([])
    assert result.rq_score == 0.0 and result.num_seeds == 0


def test_selection_priority_ablations():
    assert selection_priority(0.5, 308.5) == pytest.approx(308.5)
    assert selection_priority(0.5, 308.5, ignore_uncertainty=True) == pytest.approx(0.25)
    assert selection_priority(0.5, 308.5, 4.0, ignore_variance=True) == pytest.approx(4.0)
    assert selection_priority(0.5, 308.5, 4.0,
                              ignore_uncertainty=True, ignore_variance=True) == 1.0
