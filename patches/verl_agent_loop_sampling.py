"""Make verl's agent loop honour per-call sampling overrides.

THE BUG. ``verl_backend._generate_with_batch`` sets temperature / top_p /
max_tokens on ``DataProto.meta_info``, which is how every other verl rollout
backend takes a per-call override (see ``verl/workers/rollout/hf_rollout.py``:
``prompts.meta_info.get("temperature", self.config.temperature)``).
``AgentLoopWorker.generate_sequences`` -- the path verl 0.7.x actually uses for
an async vLLM rollout -- builds its sampling_params from the rollout config
alone and reads meta_info only for ``validate``. Every override was therefore
dropped in silence.

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

MARKER = "# [rq-evolve] per-call sampling overrides from meta_info"

ANCHOR = """        # override sampling params for validation
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
        for _rq_key in ("temperature", "top_p", "top_k", "max_tokens"):
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
    if ANCHOR not in text:
        raise SystemExit(
            f"anchor not found in {path}; verl's agent loop changed shape and "
            "this patch must be re-derived before any run is trusted"
        )
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(ANCHOR, ANCHOR + INSERT, 1), encoding="utf-8")
    return f"patched {path} (backup at {backup})"


if __name__ == "__main__":
    if "--check" in sys.argv:
        print("applied" if is_applied() else "NOT APPLIED")
        sys.exit(0 if is_applied() else 1)
    print(apply())
