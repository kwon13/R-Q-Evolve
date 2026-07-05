"""Rollout/pipeline instrumentation: counters, timers, policy-version tracking.

Everything here is pure bookkeeping (no verl/ray imports at module level) so it
is unit-testable on CPU. The scheduler (async_rollout.py) and the sampler
(verl_adapter.py) feed these objects; ``to_wandb()`` payloads ride the existing
``commit=False`` bridge so they merge into verl's per-step wandb commits.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty list (metric, not math lib)."""
    if not values:
        return 0.0
    import math

    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q / 100.0 * len(ordered)) - 1))
    return float(ordered[idx])


class RolloutMetrics:
    """Thread-safe counters/gauges for one evolve phase's streamed rollouts.

    One instance per outer iteration (reset by the caller); the scheduler
    thread, verify workers, and the entropy/weight-sync sites all write here.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            # chunk-level
            self.chunks_submitted = 0
            self.chunks_completed = 0
            self.chunks_failed = 0
            self.chunks_timeout = 0
            self.chunks_retried = 0
            # request(sample)-level
            self.requests_submitted = 0
            self.requests_completed = 0
            self.samples_accepted = 0
            self.samples_rejected = 0
            self.reject_reasons: dict[str, int] = {}
            # detection-only flags (overlong/duplicate/invalid_answer seen even
            # when the corresponding reject_* knob keeps the sample accepted)
            self.flags: dict[str, int] = {}
            # latency / throughput
            self.chunk_latencies_s: list[float] = []
            self.sample_latencies_s: list[float] = []
            self.prompt_tokens = 0
            self.response_tokens = 0
            self.phase_started_at: float | None = None
            self.phase_ended_at: float | None = None
            # gauges
            self.pending_chunks = 0
            self.max_pending_chunks = 0
            self.queue_depth = 0
            self.max_queue_depth = 0
            self.rollout_idle_s = 0.0
            # stage durations (verify/entropy/weight_sync/... via add_duration)
            self.durations_s: dict[str, float] = {}

    # -- scheduler-side ------------------------------------------------------

    def phase_start(self) -> None:
        # First call wins: one evolve phase runs the scheduler several times
        # (reevaluate + each inner batch); the span covers all of them until
        # the metrics object is reset for the next outer iteration.
        with self._lock:
            if self.phase_started_at is None:
                self.phase_started_at = time.monotonic()

    def phase_end(self) -> None:
        with self._lock:
            self.phase_ended_at = time.monotonic()

    def on_submit(self, n_requests: int) -> None:
        with self._lock:
            self.chunks_submitted += 1
            self.requests_submitted += n_requests

    def on_retry(self) -> None:
        with self._lock:
            self.chunks_retried += 1

    def on_chunk_complete(
        self,
        n_requests: int,
        latency_s: float,
        sample_latencies_s: list[float] | None = None,
        prompt_tokens: int = 0,
        response_tokens: int = 0,
    ) -> None:
        with self._lock:
            self.chunks_completed += 1
            self.requests_completed += n_requests
            self.chunk_latencies_s.append(float(latency_s))
            if sample_latencies_s:
                self.sample_latencies_s.extend(float(x) for x in sample_latencies_s)
            self.prompt_tokens += int(prompt_tokens)
            self.response_tokens += int(response_tokens)

    def on_chunk_failed(self, reason: str) -> None:
        with self._lock:
            self.chunks_failed += 1
            if reason == "timeout":
                self.chunks_timeout += 1

    def gauge(self, pending_chunks: int, queue_depth: int) -> None:
        with self._lock:
            self.pending_chunks = int(pending_chunks)
            self.max_pending_chunks = max(self.max_pending_chunks, self.pending_chunks)
            self.queue_depth = int(queue_depth)
            self.max_queue_depth = max(self.max_queue_depth, self.queue_depth)

    def add_rollout_idle(self, seconds: float) -> None:
        with self._lock:
            self.rollout_idle_s += float(seconds)

    # -- consumer/driver-side ------------------------------------------------

    def on_sample(self, accepted: bool, reject_reason: str | None = None) -> None:
        with self._lock:
            if accepted:
                self.samples_accepted += 1
            else:
                self.samples_rejected += 1
                key = reject_reason or "unknown"
                self.reject_reasons[key] = self.reject_reasons.get(key, 0) + 1

    def on_flag(self, name: str) -> None:
        with self._lock:
            self.flags[name] = self.flags.get(name, 0) + 1

    def add_duration(self, name: str, seconds: float) -> None:
        with self._lock:
            self.durations_s[name] = self.durations_s.get(name, 0.0) + float(seconds)

    @contextmanager
    def timed(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.add_duration(name, time.monotonic() - t0)

    # -- export ---------------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            span = None
            if self.phase_started_at is not None:
                end = self.phase_ended_at or time.monotonic()
                span = max(1e-9, end - self.phase_started_at)
            lat = self.chunk_latencies_s
            slat = self.sample_latencies_s
            out = {
                "chunks_submitted": self.chunks_submitted,
                "chunks_completed": self.chunks_completed,
                "chunks_failed": self.chunks_failed,
                "chunks_timeout": self.chunks_timeout,
                "chunks_retried": self.chunks_retried,
                "requests_submitted": self.requests_submitted,
                "requests_completed": self.requests_completed,
                "samples_accepted": self.samples_accepted,
                "samples_rejected": self.samples_rejected,
                "prompt_tokens": self.prompt_tokens,
                "response_tokens": self.response_tokens,
                "tokens_per_s": (
                    (self.prompt_tokens + self.response_tokens) / span if span else 0.0
                ),
                "chunk_latency_avg_s": (sum(lat) / len(lat)) if lat else 0.0,
                "chunk_latency_p50_s": _percentile(lat, 50),
                "chunk_latency_p95_s": _percentile(lat, 95),
                "sample_latency_avg_s": (sum(slat) / len(slat)) if slat else 0.0,
                "sample_latency_p95_s": _percentile(slat, 95),
                "pending_chunks": self.pending_chunks,
                "max_pending_chunks": self.max_pending_chunks,
                "max_queue_depth": self.max_queue_depth,
                "rollout_idle_s": self.rollout_idle_s,
                "phase_span_s": span or 0.0,
            }
            for name, seconds in self.durations_s.items():
                out[f"time_{name}_s"] = seconds
            for reason, count in self.reject_reasons.items():
                out[f"rejected_{reason}"] = count
            for name, count in self.flags.items():
                out[f"flag_{name}"] = count
            return out

    def to_wandb(self, prefix: str = "rollout/") -> dict:
        return {f"{prefix}{k}": v for k, v in self.snapshot().items()}


@dataclass
class PolicyVersionTracker:
    """Monotonic policy/adapter version, bumped on every trainer->vLLM sync.

    ``install()`` wraps ``trainer.checkpoint_manager.update_weights`` on the
    INSTANCE (no library edit), which covers every sync site: verl's per-step
    push, the initial sync, and the backend's evolve-phase ``_wake()``.
    Versions stamp each rollout sample so the staleness gate can compare the
    sample's generation-time version against the current one.
    """

    policy_version: int = 0
    adapter_version: int = 0
    last_sync_global_step: int = 0
    last_sync_ts: float = 0.0
    last_sync_duration_s: float = 0.0
    source_checkpoint: str = ""
    lora_enabled: bool = False
    _model_path: str = ""
    _installed: bool = field(default=False, repr=False)

    def install(self, trainer, *, lora_enabled: bool, model_path: str) -> None:
        if self._installed:
            return
        manager = getattr(trainer, "checkpoint_manager", None)
        if manager is None or not hasattr(manager, "update_weights"):
            print(
                "[RQ-Evolve] PolicyVersionTracker: trainer has no "
                "checkpoint_manager.update_weights -- versions stay at 0"
            )
            return
        self.lora_enabled = bool(lora_enabled)
        self._model_path = str(model_path)
        original = manager.update_weights
        tracker = self

        def wrapped(global_steps=None, *args, **kwargs):
            t0 = time.monotonic()
            result = original(global_steps, *args, **kwargs)
            tracker.record_sync(int(global_steps or 0), time.monotonic() - t0)
            return result

        manager.update_weights = wrapped
        self._installed = True

    def record_sync(self, global_step: int, duration_s: float) -> None:
        self.policy_version += 1
        if self.lora_enabled:
            self.adapter_version += 1
        self.last_sync_global_step = int(global_step)
        self.last_sync_ts = time.time()
        self.last_sync_duration_s = float(duration_s)
        self.source_checkpoint = f"{self._model_path}@global_step_{global_step}"

    def stamp(self) -> dict:
        """Metadata frozen into a chunk/sample at submission time."""
        return {
            "policy_version": self.policy_version,
            "adapter_version": self.adapter_version,
            "global_step": self.last_sync_global_step,
            "source_checkpoint": self.source_checkpoint,
        }

    def to_wandb(self, prefix: str = "rollout/") -> dict:
        return {
            f"{prefix}policy_version": self.policy_version,
            f"{prefix}adapter_version": self.adapter_version,
            f"{prefix}weight_sync_s": self.last_sync_duration_s,
        }


def check_staleness(
    sample_version: int,
    current_version: int,
    *,
    mode: str,
    max_policy_lag: int,
) -> tuple[bool, int]:
    """Return (is_fresh, lag). ``mode`` is 'strict' or 'bounded'."""
    lag = int(current_version) - int(sample_version)
    if mode == "strict":
        return lag == 0, lag
    return lag <= int(max_policy_lag), lag
