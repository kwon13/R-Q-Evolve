"""Append-only JSONL logging of rollout samples and phase metrics.

Two files under the run's ``rq_archive/`` directory (next to the existing
``evolution_log.jsonl``, which stays problem-level):

  rollout_samples.jsonl  -- one line per rollout SAMPLE, accepted AND rejected,
                            with reject_reason + policy/adapter version +
                            timestamps. Nothing is dropped silently.
  rollout_metrics.jsonl  -- one line per outer iteration: the RolloutMetrics
                            snapshot (counts, latencies, tokens/s, idle times).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

SAMPLES_FILE = "rollout_samples.jsonl"
METRICS_FILE = "rollout_metrics.jsonl"

# Rejection reasons a sample line may carry. Kept as a tuple (not an enum) so
# ad-hoc reasons like "worker_error:<repr>" can extend the taxonomy without a
# schema change; these are the stable, documented ones.
REJECT_REASONS = (
    "invalid_answer",   # no parseable \boxed{...} in the response
    "timeout",          # chunk exceeded request_timeout_s after retries
    "overlong",         # response hit max_response_length without EOS
    "duplicate",        # identical normalized response within the group
    "stale_policy",     # policy lag beyond staleness_mode/max_policy_lag
    "worker_error",     # rollout worker raised (prefix; suffixed with detail)
)


class JsonlSampleLogger:
    """Thread-safe line-buffered JSONL appender.

    ``enabled=False`` turns every call into a no-op so call sites don't need
    their own guards. One instance per file.
    """

    def __init__(self, path: str | Path | None, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled) and path is not None
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self.lines_written = 0
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        if not self.enabled:
            return
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self.lines_written += 1


def make_sample_record(
    *,
    iteration: int,
    program_id: str | None,
    instance_seed: int | None,
    rollout_idx: int,
    chunk_id: int,
    status: str,
    reject_reason: str | None,
    correct: bool | None,
    predicted_answer: str | None,
    prompt_tokens: int,
    response_tokens: int,
    latency_s: float,
    ts_start: float,
    ts_end: float,
    policy_version: int,
    adapter_version: int,
    global_step: int,
    source_checkpoint: str,
    entropy: float | None = None,
    verifier_mode: str | None = None,
) -> dict:
    """Schema-in-one-place constructor for a rollout_samples.jsonl line."""
    return {
        "iteration": int(iteration),
        "program_id": program_id,
        "instance_seed": instance_seed,
        "rollout_idx": int(rollout_idx),
        "chunk_id": int(chunk_id),
        "status": status,                      # accepted | rejected
        "reject_reason": reject_reason,        # null when accepted
        "correct": correct,
        "predicted_answer": predicted_answer,
        "verifier_mode": verifier_mode,
        "prompt_tokens": int(prompt_tokens),
        "response_tokens": int(response_tokens),
        "latency_s": round(float(latency_s), 3),
        "ts_start": float(ts_start),
        "ts_end": float(ts_end),
        "policy_version": int(policy_version),
        "adapter_version": int(adapter_version),
        "global_step": int(global_step),
        "source_checkpoint": source_checkpoint,
        "entropy": entropy,
    }
