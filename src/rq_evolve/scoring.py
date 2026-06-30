from dataclasses import dataclass


@dataclass(slots=True)
class RQResult:
    rq_score: float
    p_hat: float
    p_variance: float
    uncertainty: float
    num_rollouts: int
    num_correct: int


def estimate_pass_rate(correct_flags: list[bool]) -> float:
    if not correct_flags:
        return 0.0
    return sum(bool(x) for x in correct_flags) / len(correct_flags)


def compute_rq_value(p_hat: float, uncertainty: float) -> float:
    return float(p_hat) * (1.0 - float(p_hat)) * float(uncertainty)


def selection_priority(
    p_hat: float,
    rq_score: float,
    h: float = 0.0,
    *,
    ignore_uncertainty: bool = False,
    ignore_variance: bool = False,
) -> float:
    """Priority used to rank champions for mutation / training-data selection.

    Production ranks by the full R_Q (``rq_score`` = s(1-s)*H). Two ablations
    drop one factor from the *driving signal only* (the archive keeps storing /
    logging the real R_Q, so the MAP still records true scores):

    * ``ignore_uncertainty`` -- H term := 1, rank by s(1-s) (pass-rate variance).
    * ``ignore_variance``    -- s(1-s) term := 1, rank by H (uncertainty/``h``).

    Enabling both drops every factor, leaving a constant 1.0 (degenerate: the
    priority carries no signal). Callers should not set both at once.
    """
    if ignore_uncertainty and ignore_variance:
        return 1.0
    if ignore_uncertainty:
        return float(p_hat) * (1.0 - float(p_hat))
    if ignore_variance:
        return float(h)
    return float(rq_score)


def compute_rq(p_hat: float, uncertainty: float) -> RQResult:
    p_var = float(p_hat) * (1.0 - float(p_hat))
    return RQResult(
        rq_score=compute_rq_value(p_hat, uncertainty),
        p_hat=float(p_hat),
        p_variance=p_var,
        uncertainty=float(uncertainty),
        num_rollouts=0,
        num_correct=0,
    )


def compute_rq_full(correct_flags: list[bool], uncertainty: float) -> RQResult:
    p_hat = estimate_pass_rate(correct_flags)
    p_var = p_hat * (1.0 - p_hat)
    return RQResult(
        rq_score=compute_rq_value(p_hat, uncertainty),
        p_hat=p_hat,
        p_variance=p_var,
        uncertainty=float(uncertainty),
        num_rollouts=len(correct_flags),
        num_correct=sum(bool(x) for x in correct_flags),
    )


def is_frontier(p_hat: float, low: float, high: float) -> bool:
    """Training data uses frontier problems; archive can keep easier material."""
    return low < p_hat < high

