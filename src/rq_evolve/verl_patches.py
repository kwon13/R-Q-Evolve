"""Verification helpers for the R_Q uncertainty (entropy vs. Gini) measure.

The capacity / uncertainty term U in R_Q is, exactly,

    sum_t ||h_t||^2 * (1 - ||pi_t||^2),

where ``1 - ||pi_t||^2 = 1 - sum_v pi_t(v)^2`` is the Gini impurity of the token
distribution pi_t. The shipped pipeline approximates the per-token factor
``(1 - ||pi_t||^2)`` with the Shannon entropy ``-sum_v pi_t(v) log pi_t(v)`` that
verl returns from ``compute_log_prob(calculate_entropy=True)``.

To measure U with the exact Gini factor instead, verl's per-token entropy
function (``verl.utils.torch_functional.entropy_from_logits`` and its chunked
variant) is env-gated: when ``RQ_UNCERTAINTY_MEASURE=gini`` it returns
``1 - sum p^2``. That gate lives inside verl itself so the swap happens in the
actor *worker* process (where the logits live) at the right time -- during the
forward pass, long after each worker's GPU is assigned.

Why not a Ray ``worker_process_setup_hook``? A job-level setup hook perturbs
Ray's per-actor GPU assignment and triggers "Duplicate GPU detected" / NCCL
invalid-usage at FSDP init. The env var + verl-side gate avoids touching worker
startup entirely.

This module only *verifies* the gate is present (the patch is applied directly
to the installed verl source). If verl is reinstalled the gate is lost, so the
driver calls :func:`assert_verl_uncertainty_gate` before launching workers to
fail loudly instead of silently running entropy in a "gini" run.
"""

from __future__ import annotations

ENV_VAR = "RQ_UNCERTAINTY_MEASURE"

# Marker the verl edit exposes once patched; see
# verl/utils/torch_functional.py (RQ-Evolve uncertainty patch).
_GATE_ATTR = "RQ_USE_GINI_UNCERTAINTY"


def gini_from_logits(logits):
    """Reference impl of the per-distribution Gini impurity ``1 - sum_v p(v)^2``.

    Kept for tests / parity checks; the live computation is the env-gated branch
    inside ``verl.utils.torch_functional.entropy_from_logits``.
    """
    import torch

    pd = torch.softmax(logits.float(), dim=-1)
    return 1.0 - torch.sum(pd * pd, dim=-1)


def assert_verl_uncertainty_gate(measure: str) -> None:
    """Verify the installed verl carries the Gini env-gate before launch.

    No-op in entropy mode. In gini mode, raise a clear error if the verl edit is
    missing (e.g. verl was reinstalled), rather than silently measuring entropy.
    """
    measure = (measure or "entropy").strip().lower()
    if measure != "gini":
        return

    import verl.utils.torch_functional as verl_F

    if not hasattr(verl_F, _GATE_ATTR):
        raise RuntimeError(
            "uncertainty_measure=gini but the installed verl is missing the "
            "Gini uncertainty gate. Re-apply the patch in "
            f"{verl_F.__file__} (search 'RQ-Evolve uncertainty patch'): the "
            "entropy_from_logits functions must return 1 - sum(p^2) when "
            f"{ENV_VAR}=gini. See src/rq_evolve/verl_patches.py."
        )
    if not getattr(verl_F, _GATE_ATTR):
        raise RuntimeError(
            f"uncertainty_measure=gini but verl read {ENV_VAR}="
            f"{getattr(verl_F, 'RQ_UNCERTAINTY_MEASURE', None)!r} at import. "
            f"Ensure {ENV_VAR}=gini is set in the environment before verl is "
            "imported (the VerlTrainerAdapter sets it and propagates it to "
            "workers via the Ray runtime_env)."
        )
