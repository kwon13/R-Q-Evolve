"""Small runtime controls shared by the repository's vLLM entry points.

The sampler choice must be configured before vLLM creates its engine workers.
Keeping that operation here prevents the comparison and held-out evaluation
paths from silently using different backends.
"""

from __future__ import annotations

import os


VLLM_SAMPLER_BACKENDS: tuple[str, ...] = (
    "pytorch",
    "auto",
    "flashinfer",
)


def configure_vllm_sampler_backend(backend: str) -> str:
    """Configure vLLM's top-k/top-p sampler and return the effective env value.

    ``pytorch`` avoids FlashInfer's runtime JIT dependency, ``flashinfer``
    explicitly enables it, and ``auto`` preserves any caller-provided
    ``VLLM_USE_FLASHINFER_SAMPLER`` setting (or vLLM's own automatic choice).
    """

    normalized = str(backend).strip().lower()
    if normalized not in VLLM_SAMPLER_BACKENDS:
        raise ValueError(
            "vLLM sampler backend must be one of "
            + ", ".join(VLLM_SAMPLER_BACKENDS)
        )
    if normalized == "pytorch":
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    elif normalized == "flashinfer":
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
    return os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "auto")


__all__ = [
    "VLLM_SAMPLER_BACKENDS",
    "configure_vllm_sampler_backend",
]
