"""Make verl's agent loop honour per-call sampling overrides.

THE BUG. ``verl_backend._generate_with_batch`` sets temperature / top_p /
max_tokens on ``DataProto.meta_info``, which is how every other verl rollout
backend takes a per-call override (see ``verl/workers/rollout/hf_rollout.py``:
``prompts.meta_info.get("temperature", self.config.temperature)``).
``AgentLoopWorker.generate_sequences`` -- the async vLLM rollout path in the
supported verl 0.7/0.9 layouts -- builds its sampling_params from the rollout
config alone and reads meta_info only for ``validate``. Every override is
therefore dropped in silence without this patch.

WHAT IT COST. With ``rollout.temperature: 1.0`` in the config:

* ``judge_temperature: 0.0`` ran at 1.0, so the taxonomy judge -- a gate that is
  supposed to be deterministic -- returned four different verdicts on four
  byte-identical children, including one that echoed a prompt placeholder.
* ``code_temperature: 0.4`` ran at 1.0.
* the vLLM engines carry a fixed per-replica seed
  (``vllm_async_server.py``: ``"seed": self.replica_rank + config.seed``), so
  identical prompts landing on the same replica at the same point in the RNG
  stream still produced byte-identical mutations -- which is why the failure
  looked like determinism and non-determinism at once.

WHY A PATCH. ``AgentLoopWorker`` is a Ray actor: it holds its own copy of the
rollout config in its own process, so mutating the manager's copy in the driver
does not reach it. The override has to be read where sampling_params is built.

Idempotent. Keeps a .orig backup. Run it again after any verl reinstall --
``scripts/train_with_verl.py`` asserts the patch is present at startup rather
than letting the run go quiet again.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# v2 adds `logprobs` and `allowed_token_ids`, which the authoritative DOMAIN
# labeler needs. A versioned marker makes an environment carrying the earlier
# temperature-only patch fail preflight and receive the additive v2 block.
MARKER = "# [rq-evolve] per-call sampling overrides from meta_info v2"

ANCHOR = """        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature
"""
ANCHOR_ALT = """        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature
"""

INSERT = f"""
        {MARKER}
        # Applied by patches/verl_agent_loop_sampling.py. Mirrors hf_rollout,
        # which already lets a caller override sampling per call. Validation
        # keeps precedence: its overrides are set above and a validation batch
        # carries none of these keys.
        # logprobs / allowed_token_ids carry the DOMAIN labeler: one binary
        # question per candidate domain at max_tokens=1, restricted to the
        # YES and NO token ids so exp(logprob) is the exact two-way probability
        # rather than a renormalised slice of the top-k.
        for _rq_key in (
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "logprobs",
            "allowed_token_ids",
        ):
            if _rq_key in batch.meta_info:
                sampling_params[_rq_key] = batch.meta_info[_rq_key]
"""


def target_file() -> Path:
    import verl  # noqa: F401  -- resolve whichever verl is importable

    return (
        Path(verl.__file__).parent
        / "experimental"
        / "agent_loop"
        / "agent_loop.py"
    )


def is_applied(path: Path | None = None) -> bool:
    path = path or target_file()
    return MARKER in path.read_text(encoding="utf-8")


def apply() -> str:
    path = target_file()
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return f"already applied: {path}"
    target_anchor = None
    if ANCHOR in text:
        target_anchor = ANCHOR
    elif ANCHOR_ALT in text:
        target_anchor = ANCHOR_ALT
    else:
        raise SystemExit(
            f"anchor not found in {path}; verl's agent loop changed shape and "
            "this patch must be re-derived before any run is trusted"
        )
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(target_anchor, target_anchor + INSERT, 1), encoding="utf-8")
    return f"patched {path} (backup at {backup})"


if __name__ == "__main__":
    if "--check" in sys.argv:
        print("applied" if is_applied() else "NOT APPLIED")
        sys.exit(0 if is_applied() else 1)
    print(apply())
