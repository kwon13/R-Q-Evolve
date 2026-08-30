"""R_Q and the quantities it is built from, estimated over n seeds x m rollouts.

Notation follows the paper. A program p is graded on n FRESH seeds; each
instance x_z gets m independent solver rollouts with verifier outcomes r_{z,j}
and length-normalized rollout entropies h_{z,j}:

    s_hat_z      = (1/m) sum_j r_{z,j}                  per-instance success
    L_hat_z      = m/(m-1) * s_hat_z (1 - s_hat_z)      UNBIASED learnability
    U_hat_z      = (1/m) sum_j h_{z,j}                  policy uncertainty
    R_Q_hat      = (1/n) sum_z L_hat_z * w(U_hat_z)     program fitness
    D_hat        = ddof-1 var of s_hat_z  -  L_bar/m    difficulty dispersion

Three properties this shape has and the obvious alternatives do not:

**The m/(m-1) correction is not cosmetic.** The plug-in s(1-s) estimates
((m-1)/m)*L, and at the small m this budget allows (m=2) that is a 50% shortfall.
At m=1 it is identically 0 for every program, so m>=2 is an identifiability
requirement, not a preference.

**Per-seed product, then average -- not the product of the averages.** A seed
the solver cannot touch (s in {0,1}) has L_hat_z = 0, and impossible instances
provoke long, confused, high-entropy rollouts. Multiplying the averages lets
that entropy leak into fitness through the second factor; multiplying per seed
zeroes the seed's contribution outright.

**Pooled learnability is a different quantity.** By the total-variance
identity, s_bar(1 - s_bar) = W + D: pooling seeds measures the average learning
signal PLUS the spread of difficulty across instances. A program emitting
trivial instances half the time and impossible ones the other half has W = 0 --
no learning signal anywhere -- and a pooled score at the maximum 1/4. Fitness
must therefore be the per-instance average, and D is reported separately as a
diagnostic rather than folded in.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


RQ_FITNESS_MODES = ("standard", "reverse_u", "no_u")


def uncertainty_weight(
    u_score: float,
    fitness_mode: str = "standard",
    *,
    reverse_u_constant: float = 2.0,
) -> float:
    """Return the configured uncertainty factor ``w(U)``.

    ``standard`` is the production score ``U``; ``reverse_u`` is the requested
    ablation ``C-U``; and ``no_u`` fixes the factor to one. Reverse-U is kept
    exact rather than clipped: if ``U > C``, a negative contribution is part
    of that ablation's stated definition.
    """
    mode = str(fitness_mode)
    if mode == "standard":
        return float(u_score)
    if mode == "reverse_u":
        constant = float(reverse_u_constant)
        if not math.isfinite(constant) or constant <= 0.0:
            raise ValueError("reverse_u_constant must be a positive finite value")
        return constant - float(u_score)
    if mode == "no_u":
        return 1.0
    raise ValueError(
        f"unknown R_Q fitness mode {mode!r}; expected one of {RQ_FITNESS_MODES}"
    )


@dataclass(slots=True)
class SeedStat:
    """One instance's contribution: what seed z produced and what it scored."""

    seed: int
    s_hat: float
    learnability: float
    u_score: float
    num_rollouts: int
    num_correct: int

    @property
    def contribution(self) -> float:
        """L_hat_z * U_hat_z, this seed's term in R_Q."""
        return self.learnability * self.u_score

    def fitness_contribution(
        self,
        fitness_mode: str = "standard",
        *,
        reverse_u_constant: float = 2.0,
    ) -> float:
        """This seed's term under the selected uncertainty ablation."""
        return self.learnability * uncertainty_weight(
            self.u_score,
            fitness_mode,
            reverse_u_constant=reverse_u_constant,
        )

    @property
    def degenerate(self) -> bool:
        """s in {0,1}: the solver never varies here, so the seed teaches nothing."""
        return self.learnability <= 0.0


@dataclass(slots=True)
class RQResult:
    rq_score: float
    s_hat: float
    learnability: float
    u_score: float
    dispersion: float
    num_rollouts: int
    num_correct: int
    num_seeds: int = 0
    per_seed: list[SeedStat] = field(default_factory=list)
    fitness_mode: str = "standard"
    reverse_u_constant: float = 2.0


def estimate_success_rate(correct_flags: list[bool]) -> float:
    """s_hat_z = (1/m) sum_j r_{z,j}."""
    if not correct_flags:
        return 0.0
    return sum(bool(x) for x in correct_flags) / len(correct_flags)


def unbiased_learnability(s_hat: float, num_rollouts: int) -> float:
    """L_hat_z = m/(m-1) * s(1-s), unbiased for s(1-s) when m >= 2.

    Returns 0.0 at m < 2: with one rollout s_hat is 0 or 1, the plug-in is
    identically zero, and no estimator can recover L from a single Bernoulli
    draw. Reporting 0 is honest -- the seed carries no measurable signal.
    """
    m = int(num_rollouts)
    if m < 2:
        return 0.0
    s = float(s_hat)
    return (m / (m - 1)) * s * (1.0 - s)


def score_seed(
    seed: int,
    correct_flags: list[bool],
    rollout_entropies: list[float],
) -> SeedStat:
    """Collapse one instance's m rollouts into its (s, L, U) triple."""
    s_hat = estimate_success_rate(correct_flags)
    m = len(correct_flags)
    return SeedStat(
        seed=int(seed),
        s_hat=s_hat,
        learnability=unbiased_learnability(s_hat, m),
        u_score=(
            sum(rollout_entropies) / len(rollout_entropies)
            if rollout_entropies
            else 0.0
        ),
        num_rollouts=m,
        num_correct=sum(bool(x) for x in correct_flags),
    )


def estimate_dispersion(per_seed: list[SeedStat]) -> float:
    """D_hat: difficulty spread ACROSS instances, unbiased (paper Lemma 3).

    The raw variance of s_hat_z overstates D by W/m, because each s_hat_z is
    itself a noisy m-sample estimate; subtracting L_bar/m removes exactly that.
    Needs n >= 2 seeds; returns 0.0 below that rather than raising, since a
    one-seed program has no cross-instance spread to speak of.

    Diagnostic only. It is deliberately NOT part of fitness: rewarding
    consistency directly would hand a free maximum to a generator that ignores
    its seed, and the defence against those is a validity gate, not a score.
    """
    if len(per_seed) < 2:
        return 0.0
    raw = statistics.variance([st.s_hat for st in per_seed])
    mean_l = sum(st.learnability for st in per_seed) / len(per_seed)
    mean_m = sum(st.num_rollouts for st in per_seed) / len(per_seed)
    correction = mean_l / mean_m if mean_m else 0.0
    return raw - correction


def compute_rq_program(
    per_seed: list[SeedStat],
    *,
    fitness_mode: str = "standard",
    reverse_u_constant: float = 2.0,
) -> RQResult:
    """Compute the configured per-seed fitness, plus reported raw aggregates.

    ``s_hat`` and ``u_score`` on the result are seed averages, carried for the
    frontier band and for logging; the fitness itself never goes through them.
    ``fitness_mode`` selects ``L*U`` (standard), ``L*(C-U)`` (reverse_u), or
    ``L`` (no_u).  The raw U is always retained for comparable diagnostics.
    """
    # Validate even an empty batch so a misspelled experiment arm cannot run
    # silently until the first successful rollout arrives.
    uncertainty_weight(
        0.0,
        fitness_mode,
        reverse_u_constant=reverse_u_constant,
    )
    if not per_seed:
        return RQResult(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0,
            0,
            [],
            fitness_mode=fitness_mode,
            reverse_u_constant=reverse_u_constant,
        )
    n = len(per_seed)
    return RQResult(
        rq_score=sum(
            st.fitness_contribution(
                fitness_mode,
                reverse_u_constant=reverse_u_constant,
            )
            for st in per_seed
        )
        / n,
        s_hat=sum(st.s_hat for st in per_seed) / n,
        learnability=sum(st.learnability for st in per_seed) / n,
        u_score=sum(st.u_score for st in per_seed) / n,
        dispersion=estimate_dispersion(per_seed),
        num_rollouts=sum(st.num_rollouts for st in per_seed),
        num_correct=sum(st.num_correct for st in per_seed),
        num_seeds=n,
        per_seed=list(per_seed),
        fitness_mode=fitness_mode,
        reverse_u_constant=reverse_u_constant,
    )


def selection_priority(
    s_hat: float,
    rq_score: float,
    u_score: float = 0.0,
    *,
    ignore_uncertainty: bool = False,
    ignore_variance: bool = False,
) -> float:
    """Priority used to rank champions for mutation / training-data selection.

    Production ranks by the full R_Q. Two ablations drop one factor from the
    *driving signal only* (the archive keeps storing / logging the real R_Q):

    * ``ignore_uncertainty`` -- U term := 1, rank by learnability alone.
    * ``ignore_variance``    -- learnability := 1, rank by U (``u_score``).

    Enabling both leaves a constant 1.0 (degenerate: no signal). Callers should
    not set both at once.
    """
    if ignore_uncertainty and ignore_variance:
        return 1.0
    if ignore_uncertainty:
        return float(s_hat) * (1.0 - float(s_hat))
    if ignore_variance:
        return float(u_score)
    return float(rq_score)


def is_frontier(s_hat: float, low: float, high: float) -> bool:
    """Training data uses frontier problems; archive can keep easier material."""
    return low < s_hat < high
