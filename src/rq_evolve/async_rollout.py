"""Chunked streaming rollout scheduler (producer-consumer evolve phase).

Legacy path: ONE blocking ``AgentLoopManager.generate_sequences`` call for the
whole inner batch -- verification starts only after the slowest sample of the
whole batch finishes. This module splits the batch into chunks (default: one
problem's G rollouts), submits them round-robin DIRECTLY to verl's
``AgentLoopWorker`` Ray actors (asyncio actors: any-size batch, per-request
load balancing via the shared GlobalRequestLoadBalancer, no divisor padding),
and consumes each chunk the moment IT completes:

    scheduler (caller thread)                 verify pool (verify_workers threads)
    ------------------------                  ------------------------------------
    submit up to max_in_flight chunks   -->   decode + extract_boxed + math_verify
    ray.wait(num_returns=1, poll)       -->   overlong/duplicate/staleness gates
    timeout scan + retry/fail           -->   per-sample JSONL (accepted+rejected)

Bounds: ``max_in_flight_chunks`` (engine backpressure), ``queue_maxsize``
(verify backpressure -- submission pauses when consumers fall behind),
``request_timeout_s`` + ``max_retries`` (no chunk waits forever; failures are
explicit records, never silent drops).

The entropy (u_score) pass is NOT streamed: it is an FSDP actor forward on
the same GPUs vLLM occupies, so it runs once, batched, after the last chunk
drains -- identical memory profile to the legacy path (see
``VerlPolicyBackend.finalize_rollouts``).

Ray is injected via a small transport object so the scheduler itself is
unit-testable on CPU (tests/dry-run use a thread-pool transport).
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .backends import RolloutRecord
from .config import AsyncRolloutConfig
from .metrics import RolloutMetrics, check_staleness
from .program import ProblemInstance
from .reward import answers_match, extract_boxed
from .rollout_log import JsonlSampleLogger, make_sample_record


@dataclass(slots=True)
class ChunkJob:
    """One schedulable unit: ``len(instances) * n_rollouts`` generations."""

    chunk_id: int
    instances: list[ProblemInstance]
    gen_batch: Any        # DataProto, already repeated x n_rollouts
    batch: Any            # popped remainder (uid, ...) for full_batch/entropy
    n_rollouts: int
    # PolicyVersionTracker.stamp() at build time: policy_version,
    # adapter_version, global_step, source_checkpoint.
    meta: dict = field(default_factory=dict)
    attempts: int = 0
    ts_submit: float = 0.0        # monotonic, for latency/timeout
    ts_submit_wall: float = 0.0   # wall clock, for JSONL


@dataclass(slots=True)
class ChunkFailure:
    reason: str  # "timeout" | "worker_error:<detail>"


@dataclass(slots=True)
class ChunkResult:
    job: ChunkJob
    # Per-instance groups of RolloutRecord, aligned with job.instances.
    grouped: list[list[RolloutRecord]]
    # batch.repeat(n).union(output) for the deferred entropy pass; None for
    # failed chunks (their records are rejected and get entropy 0.0).
    full_batch: Any = None
    # The RAW generation output, kept separate from full_batch. Replay hands
    # this back to the trainer in place of a second sampling pass, and it must
    # be the generation side only: full_batch already carries the prompt-side
    # keys, which the trainer unions in itself.
    output: Any = None


class RayAgentLoopTransport:
    """Round-robin submission to verl AgentLoopWorker actors via plain Ray.

    Each worker is an asyncio Ray actor whose ``generate_sequences`` accepts an
    arbitrary-size DataProto (it fans samples into per-request asyncio tasks
    and routes them through the shared request load balancer), so no
    divisor padding is needed and calls to the same worker interleave.
    """

    def __init__(self, agent_loop_workers: list) -> None:
        if not agent_loop_workers:
            raise ValueError("no agent_loop_workers to submit rollouts to")
        self.workers = list(agent_loop_workers)
        self._rr = 0

    def submit(self, gen_batch):
        worker = self.workers[self._rr % len(self.workers)]
        self._rr += 1
        return worker.generate_sequences.remote(gen_batch)

    def wait(self, handles: list, timeout_s: float):
        import ray

        return ray.wait(handles, num_returns=1, timeout=timeout_s)

    def get(self, handle):
        import ray

        return ray.get(handle)

    def cancel(self, handle) -> None:
        import ray

        try:
            # Best-effort: reaches the worker's asyncio task; the underlying
            # vLLM request may still run to completion (result is discarded).
            ray.cancel(handle, force=False)
        except Exception:
            pass


class LocalThreadTransport:
    """Ray-free transport for tests/dry-run: workers are plain callables.

    ``workers`` are ``fn(gen_batch) -> payload`` executed on a thread pool;
    handles are concurrent.futures.Future objects.
    """

    def __init__(self, workers: list[Callable[[Any], Any]], max_workers: int | None = None) -> None:
        if not workers:
            raise ValueError("no workers")
        self.workers = list(workers)
        self._rr = 0
        self._pool = ThreadPoolExecutor(max_workers=max_workers or len(workers))

    def submit(self, gen_batch):
        worker = self.workers[self._rr % len(self.workers)]
        self._rr += 1
        return self._pool.submit(worker, gen_batch)

    def wait(self, handles: list, timeout_s: float):
        import concurrent.futures as cf

        done, not_done = cf.wait(handles, timeout=timeout_s, return_when=cf.FIRST_COMPLETED)
        return list(done), list(not_done)

    def get(self, handle):
        return handle.result()

    def cancel(self, handle) -> None:
        handle.cancel()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def _normalize_response(text: str) -> str:
    return " ".join(text.split())


class ChunkedRolloutScheduler:
    """Bounded in-flight submission + streamed verification of chunk results.

    The submission/wait loop runs in the CALLING thread (the evolve phase is a
    synchronous call site); verification fans out to ``verify_workers`` threads.
    ``run()`` returns ChunkResults sorted by chunk_id -- the caller reassembles
    per-instance groups in original order.
    """

    def __init__(
        self,
        *,
        transport,
        cfg: AsyncRolloutConfig,
        tokenizer,
        metrics: RolloutMetrics,
        sample_logger: JsonlSampleLogger,
        current_version_fn: Callable[[], int],
        iteration: int = -1,
        poll_s: float = 1.0,
    ) -> None:
        self.transport = transport
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.metrics = metrics
        self.sample_logger = sample_logger
        self.current_version_fn = current_version_fn
        self.iteration = int(iteration)
        self.poll_s = float(poll_s)

    def run(self, jobs: list[ChunkJob]) -> list[ChunkResult]:
        cfg = self.cfg
        metrics = self.metrics
        metrics.phase_start()
        todo: deque[ChunkJob] = deque(jobs)
        pending: dict[Any, ChunkJob] = {}
        consumer_futures: list = []

        try:
            with ThreadPoolExecutor(
                max_workers=max(1, cfg.verify_workers), thread_name_prefix="rq-verify"
            ) as pool:
                results = self._run_loop(todo, pending, consumer_futures, pool)
        finally:
            metrics.phase_end()
        results.sort(key=lambda r: r.job.chunk_id)
        return results

    def _run_loop(self, todo, pending, consumer_futures, pool) -> list[ChunkResult]:
        cfg = self.cfg
        metrics = self.metrics

        def outstanding_consumes() -> int:
            return sum(1 for f in consumer_futures if not f.done())

        while todo or pending:
            # 1) fill in-flight up to the bound; pause when verifiers lag
            #    (queue_maxsize) -- backpressure instead of unbounded buffering.
            while (
                todo
                and len(pending) < cfg.max_in_flight_chunks
                and outstanding_consumes() < cfg.queue_maxsize
            ):
                job = todo.popleft()
                job.attempts += 1
                job.ts_submit = time.monotonic()
                job.ts_submit_wall = time.time()
                handle = self.transport.submit(job.gen_batch)
                pending[handle] = job
                metrics.on_submit(len(job.instances) * job.n_rollouts)

            if not pending:
                # work remains but nothing in flight: verifiers are the
                # bottleneck (backpressure) -> the rollout engine is idle.
                time.sleep(min(0.05, self.poll_s))
                metrics.add_rollout_idle(min(0.05, self.poll_s))
                metrics.gauge(0, outstanding_consumes())
                continue

            # 2) consume ANY one completion (streaming: no full-batch barrier)
            ready, _ = self.transport.wait(list(pending), timeout_s=self.poll_s)
            now = time.monotonic()
            for handle in ready:
                job = pending.pop(handle)
                latency_s = now - job.ts_submit
                try:
                    payload = self.transport.get(handle)
                except Exception as exc:
                    self._fail_or_retry(
                        job,
                        todo,
                        consumer_futures,
                        pool,
                        f"worker_error:{type(exc).__name__}: {exc}",
                    )
                    continue
                consumer_futures.append(
                    pool.submit(self._consume_safe, job, payload, latency_s)
                )

            # 3) timeout scan: no chunk waits forever
            for handle, job in list(pending.items()):
                if now - job.ts_submit > cfg.request_timeout_s:
                    del pending[handle]
                    self.transport.cancel(handle)
                    self._fail_or_retry(job, todo, consumer_futures, pool, "timeout")

            metrics.gauge(len(pending), outstanding_consumes())

        # _consume_safe never raises, so one bad chunk can't discard the
        # phase's other results here.
        return [f.result() for f in consumer_futures]

    def _fail_or_retry(self, job, todo, consumer_futures, pool, reason: str) -> None:
        if job.attempts <= self.cfg.max_retries:
            self.metrics.on_retry()
            todo.append(job)
            return
        self.metrics.on_chunk_failed("timeout" if reason == "timeout" else "worker_error")
        consumer_futures.append(
            pool.submit(self._consume_safe, job, ChunkFailure(reason), 0.0)
        )

    # ------------------------------------------------------------------
    # consumer side (verify pool threads)
    # ------------------------------------------------------------------

    def _consume_safe(self, job: ChunkJob, payload, latency_s: float) -> ChunkResult:
        """_consume, but an exception in verification can never torch the phase.

        A raising consumer (disk-full JSONL append, DataProto union mismatch,
        OOM while building full_batch) degrades to an all-rejected failure
        record for ITS chunk only; every other chunk's result survives.
        """
        try:
            return self._consume(job, payload, latency_s)
        except Exception as exc:
            self.metrics.on_flag("verify_error")
            reason = ChunkFailure(f"worker_error:verify_error:{type(exc).__name__}: {exc}")
            try:
                return self._consume_failure(job, reason, latency_s, time.time())
            except Exception:
                # even the failure path failed (e.g. the logger IS the problem):
                # build the rejected records without logging.
                grouped = [
                    [
                        RolloutRecord(
                            response="",
                            predicted_answer=None,
                            correct=False,
                            entropy=0.0,
                            status="rejected",
                            reject_reason="worker_error",
                        )
                        for _ in range(job.n_rollouts)
                    ]
                    for _ in job.instances
                ]
                return ChunkResult(job=job, grouped=grouped, full_batch=None)

    def _consume(self, job: ChunkJob, payload, latency_s: float) -> ChunkResult:
        ts_end_wall = time.time()
        if isinstance(payload, ChunkFailure):
            return self._consume_failure(job, payload, latency_s, ts_end_wall)

        cfg = self.cfg
        n = job.n_rollouts
        fresh, lag = check_staleness(
            int(job.meta.get("policy_version", -1)),
            self.current_version_fn(),
            mode=cfg.staleness_mode,
            max_policy_lag=cfg.max_policy_lag,
        )

        with self.metrics.timed("verify"):
            output = payload
            responses = output.batch.get("responses")
            n_rows = len(job.instances) * n
            if responses is None:
                decoded = [""] * n_rows
                resp_width = 0
                resp_token_counts = [0] * n_rows
                prompt_tokens_total = 0
            else:
                # decode per chunk in the consumer thread (fast-tokenizer decode
                # holds no shared mutable state, unlike padded encode)
                decoded = [
                    self.tokenizer.decode(row.tolist(), skip_special_tokens=True)
                    for row in responses
                ]
                resp_width = int(responses.shape[1])
                resp_token_counts, prompt_tokens_total = _token_counts(output, resp_width)

            grouped: list[list[RolloutRecord]] = []
            for ci, inst in enumerate(job.instances):
                rows: list[RolloutRecord] = []
                seen_norm: set[str] = set()
                for ri in range(n):
                    idx = ci * n + ri
                    text = decoded[idx] if idx < len(decoded) else ""
                    n_resp_tokens = resp_token_counts[idx] if idx < len(resp_token_counts) else 0
                    reject_reason: str | None = None

                    if not fresh:
                        reject_reason = "stale_policy"
                        self.metrics.on_flag("stale_policy_lag_%d" % lag)
                    if reject_reason is None and resp_width and n_resp_tokens >= resp_width:
                        self.metrics.on_flag("overlong")
                        if cfg.reject_overlong:
                            reject_reason = "overlong"

                    pred = extract_boxed(text)
                    if pred is None:
                        self.metrics.on_flag("invalid_answer")
                        if reject_reason is None and cfg.reject_invalid_answer:
                            reject_reason = "invalid_answer"

                    norm = _normalize_response(text)
                    if norm and norm in seen_norm:
                        self.metrics.on_flag("duplicate")
                        if reject_reason is None and cfg.reject_duplicates:
                            reject_reason = "duplicate"
                    seen_norm.add(norm)

                    accepted = reject_reason is None
                    # spend sympy/math_verify only on samples that will count
                    correct = bool(
                        accepted and pred is not None and answers_match(pred, inst.answer)
                    )
                    record = RolloutRecord(
                        response=text,
                        predicted_answer=pred,
                        correct=correct,
                        entropy=0.0,  # backfilled by the deferred entropy pass
                        status="accepted" if accepted else "rejected",
                        reject_reason=reject_reason,
                        policy_version=int(job.meta.get("policy_version", -1)),
                        adapter_version=int(job.meta.get("adapter_version", -1)),
                        global_step=int(job.meta.get("global_step", -1)),
                        source_checkpoint=str(job.meta.get("source_checkpoint", "")),
                        ts_start=job.ts_submit_wall,
                        ts_end=ts_end_wall,
                        response_tokens=n_resp_tokens,
                    )
                    rows.append(record)
                    self.metrics.on_sample(accepted, reject_reason)
                    self._log_sample(job, inst, ri, record, latency_s)
                grouped.append(rows)

        sample_latencies = _sample_latencies(output)
        self.metrics.on_chunk_complete(
            n_requests=n_rows,
            latency_s=latency_s,
            sample_latencies_s=sample_latencies,
            prompt_tokens=prompt_tokens_total,
            response_tokens=sum(resp_token_counts),
        )
        full_batch = None
        if job.batch is not None and responses is not None:
            full_batch = job.batch.repeat(repeat_times=n, interleave=True).union(output)
        return ChunkResult(
            job=job, grouped=grouped, full_batch=full_batch, output=output
        )

    def _consume_failure(
        self, job: ChunkJob, failure: ChunkFailure, latency_s: float, ts_end_wall: float
    ) -> ChunkResult:
        base = "timeout" if failure.reason == "timeout" else "worker_error"
        grouped: list[list[RolloutRecord]] = []
        for inst in job.instances:
            rows: list[RolloutRecord] = []
            for ri in range(job.n_rollouts):
                record = RolloutRecord(
                    response="",
                    predicted_answer=None,
                    correct=False,
                    entropy=0.0,
                    status="rejected",
                    reject_reason=base,
                    policy_version=int(job.meta.get("policy_version", -1)),
                    adapter_version=int(job.meta.get("adapter_version", -1)),
                    global_step=int(job.meta.get("global_step", -1)),
                    source_checkpoint=str(job.meta.get("source_checkpoint", "")),
                    ts_start=job.ts_submit_wall,
                    ts_end=ts_end_wall,
                )
                rows.append(record)
                self.metrics.on_sample(False, base)
                self._log_sample(job, inst, ri, record, latency_s, detail=failure.reason)
            grouped.append(rows)
        return ChunkResult(job=job, grouped=grouped, full_batch=None)

    def _log_sample(
        self,
        job: ChunkJob,
        inst: ProblemInstance,
        rollout_idx: int,
        record: RolloutRecord,
        latency_s: float,
        detail: str | None = None,
    ) -> None:
        reason = record.reject_reason
        if detail and reason == "worker_error":
            reason = detail  # keep the exception detail in the log line
        self.sample_logger.log(
            make_sample_record(
                iteration=self.iteration,
                program_id=getattr(inst, "program_id", None),
                instance_seed=getattr(inst, "seed", None),
                rollout_idx=rollout_idx,
                chunk_id=job.chunk_id,
                status=record.status,
                reject_reason=reason,
                # rejected samples skip answers_match (their correctness was
                # never evaluated): null, not false -- false means graded-wrong.
                correct=record.correct if record.status == "accepted" else None,
                predicted_answer=record.predicted_answer,
                prompt_tokens=0,  # per-sample prompt tokens not tracked (chunk-level in metrics)
                response_tokens=record.response_tokens,
                latency_s=latency_s,
                ts_start=record.ts_start,
                ts_end=record.ts_end,
                policy_version=record.policy_version,
                adapter_version=record.adapter_version,
                global_step=record.global_step,
                source_checkpoint=record.source_checkpoint,
                entropy=None,  # deferred pass runs after logging; see docs
            )
        )


def _token_counts(output, resp_width: int) -> tuple[list[int], int]:
    """Per-row response token counts + total prompt tokens from the masks."""
    try:
        attention_mask = output.batch["attention_mask"]
        resp_mask = attention_mask[:, -resp_width:]
        prompt_mask = attention_mask[:, :-resp_width]
        return (
            [int(x) for x in resp_mask.sum(dim=-1).tolist()],
            int(prompt_mask.sum().item()),
        )
    except Exception:
        n_rows = output.batch["responses"].shape[0] if "responses" in output.batch else 0
        return [0] * int(n_rows), 0


def _sample_latencies(output) -> list[float]:
    """Per-sample generate seconds from the agent loop's AgentLoopMetrics."""
    try:
        metrics = output.meta_info.get("metrics") or []
        return [float(m["generate_sequences"]) for m in metrics if "generate_sequences" in m]
    except Exception:
        return []
