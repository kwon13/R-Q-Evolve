#!/usr/bin/env python
"""Async-rollout dry run: prove the streaming pipeline before real training.

Two modes:

  --mode mock  (CPU-only, runs even with busy GPUs)
      Drives the REAL ChunkedRolloutScheduler + verification/filtering/JSONL
      path with fake workers (no ray, no vLLM): mixed short/long generations,
      one hanging chunk (timeout -> retry -> explicit failure), one stale-policy
      chunk, invalid-answer and duplicate responses. Asserts:
        * short chunks complete before long ones (no whole-batch barrier)
        * in-flight never exceeds max_in_flight_chunks (backpressure)
        * timeout -> retry -> failure records (nothing silently dropped)
        * stale chunk rejected with reason stale_policy
        * one JSONL line per sample (accepted AND rejected)

  --mode live  (needs free GPUs)
      Boots the full trainer stack (ray + FSDP workers + vLLM agent loop)
      exactly like training, pushes weights once, streams engineered short/long
      prompts, prints rollout metrics + the short-vs-long completion timeline.

Run inside `conda activate azr-bw-blackwell` from the repo root:
    python scripts/dry_run_async.py --mode mock
    python scripts/dry_run_async.py --mode live --config configs/rq_evolve_smoke_lora.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# mock mode
# ---------------------------------------------------------------------------

N_PROBLEMS = 32
G = 4                      # rollouts per problem
RESP_WIDTH = 64            # mock "max_response_length"
HANG_CHUNK = 5             # this chunk hangs every attempt -> timeout path
STALE_CHUNK = 7            # stamped 2 policy versions behind -> stale_policy
NOBOX_CHUNK = 9            # responses without \boxed{} -> invalid_answer flag
DUP_CHUNK = 11             # identical responses -> duplicate flag
HANG_S = 6.0
CURRENT_POLICY_VERSION = 3


class FakeTokenizer:
    """decode() maps a row's first token id to a pre-registered text."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return self.texts[int(ids[0])] if ids else ""


class FakeOutput:
    """Duck-typed DataProto surface used by the consumer: .batch / .meta_info."""

    def __init__(self, responses, attention_mask, gen_seconds: list[float]) -> None:
        self.batch = {"responses": responses, "attention_mask": attention_mask}
        self.meta_info = {
            "metrics": [{"generate_sequences": s} for s in gen_seconds]
        }


class MockGenBatch:
    """Opaque payload the fake worker turns into a FakeOutput after a delay."""

    def __init__(self, text_indices, delay_s, overlong_rows=(), hang=False):
        self.text_indices = list(text_indices)
        self.delay_s = float(delay_s)
        self.overlong_rows = set(overlong_rows)
        self.hang = bool(hang)


def _fake_worker(gen_batch: MockGenBatch):
    import torch

    time.sleep(HANG_S if gen_batch.hang else gen_batch.delay_s)
    n = len(gen_batch.text_indices)
    responses = torch.zeros((n, RESP_WIDTH), dtype=torch.long)
    attention = torch.zeros((n, 2 * RESP_WIDTH), dtype=torch.long)
    for row, text_idx in enumerate(gen_batch.text_indices):
        responses[row, 0] = text_idx
        attention[row, :RESP_WIDTH] = 1  # prompt tokens
        # response tokens: full width == overlong (hit the cap), else half
        n_resp = RESP_WIDTH if row in gen_batch.overlong_rows else RESP_WIDTH // 2
        attention[row, RESP_WIDTH : RESP_WIDTH + n_resp] = 1
    gen_seconds = [gen_batch.delay_s] * n
    return FakeOutput(responses, attention, gen_seconds)


def run_mock() -> int:
    from rq_evolve.async_rollout import (
        ChunkedRolloutScheduler,
        ChunkJob,
        LocalThreadTransport,
    )
    from rq_evolve.config import AsyncRolloutConfig
    from rq_evolve.metrics import RolloutMetrics
    from rq_evolve.program import ProblemInstance
    from rq_evolve.rollout_log import JsonlSampleLogger

    cfg = AsyncRolloutConfig(
        streaming_enabled=True,
        chunk_size=1,
        max_in_flight_chunks=8,
        request_timeout_s=1.5,
        max_retries=1,
        verify_workers=4,
        queue_maxsize=16,
        staleness_mode="bounded",
        max_policy_lag=1,
    )

    texts: list[str] = []
    jobs: list[ChunkJob] = []
    for i in range(N_PROBLEMS):
        inst = ProblemInstance(
            problem=f"What is {i}+{i}?", answer=str(2 * i),
            program_id=f"mock_{'short' if i % 2 == 0 else 'long'}_{i}", seed=0,
        )
        row_texts = []
        for g in range(G):
            if i == NOBOX_CHUNK:
                text = f"I think the answer might be {2 * i} but I won't box it."
            elif i == DUP_CHUNK:
                text = f"The answer is \\boxed{{{2 * i}}}."  # identical G times
            elif g == 0:
                text = f"Wrong attempt: \\boxed{{{2 * i + 1}}}."
            else:
                text = f"Compute {i}+{i}. The answer is \\boxed{{{2 * i}}} (try {g})."
            row_texts.append(text)
        text_indices = []
        for text in row_texts:
            text_indices.append(len(texts))
            texts.append(text)
        delay = 0.1 if i % 2 == 0 else 1.0
        jobs.append(
            ChunkJob(
                chunk_id=i,
                instances=[inst],
                gen_batch=MockGenBatch(
                    text_indices,
                    delay,
                    overlong_rows=(0,) if i % 2 == 1 else (),
                    hang=(i == HANG_CHUNK),
                ),
                batch=None,  # no entropy pass in the mock
                n_rollouts=G,
                meta={
                    "policy_version": (
                        CURRENT_POLICY_VERSION - 2 if i == STALE_CHUNK else CURRENT_POLICY_VERSION
                    ),
                    "adapter_version": 0,
                    "global_step": 42,
                    "source_checkpoint": "mock@global_step_42",
                },
            )
        )

    tmp = Path(tempfile.mkdtemp(prefix="rq_dryrun_")) / "rollout_samples.jsonl"
    logger = JsonlSampleLogger(tmp)
    metrics = RolloutMetrics()
    # Pool capacity >= max_in_flight: mirrors vLLM continuous batching, where
    # every in-flight request progresses concurrently (the chunk timeout covers
    # engine queueing too, but the mock's thread pool must not fake-starve it).
    transport = LocalThreadTransport(
        [_fake_worker] * 4, max_workers=cfg.max_in_flight_chunks + 2
    )
    scheduler = ChunkedRolloutScheduler(
        transport=transport,
        cfg=cfg,
        tokenizer=FakeTokenizer(texts),
        metrics=metrics,
        sample_logger=logger,
        current_version_fn=lambda: CURRENT_POLICY_VERSION,
        iteration=0,
        poll_s=0.1,
    )
    t0 = time.time()
    results = scheduler.run(jobs)
    transport.shutdown()

    snapshot = metrics.snapshot()
    print("== mock dry-run metrics ==")
    for key, value in sorted(snapshot.items()):
        print(f"  {key}: {value}")

    # ---- assertions: the executable proof of the streaming design ----------
    failures: list[str] = []

    short_done, long_done = [], []
    completion_at: dict[int, float] = {}
    for result in results:
        rows = result.grouped[0]
        done = max(r.ts_end for r in rows) - t0
        completion_at[result.job.chunk_id] = done
        if result.job.chunk_id == HANG_CHUNK:
            continue  # failure record, not a generation completion
        (short_done if result.job.chunk_id % 2 == 0 else long_done).append(done)
    mean_short = sum(short_done) / len(short_done)
    mean_long = sum(long_done) / len(long_done)
    if not mean_short < mean_long:
        failures.append(
            f"short chunks NOT faster: mean_short={mean_short:.2f}s "
            f"mean_long={mean_long:.2f}s (whole-batch barrier?)"
        )
    # a later-submitted short chunk must beat an earlier-submitted long chunk
    if not completion_at[2] < completion_at[1]:
        failures.append("chunk 2 (short, submitted after) did not finish before chunk 1 (long)")
    if snapshot["max_pending_chunks"] > cfg.max_in_flight_chunks:
        failures.append(
            f"in-flight exceeded bound: {snapshot['max_pending_chunks']} > "
            f"{cfg.max_in_flight_chunks}"
        )
    if snapshot["chunks_timeout"] < 1 or snapshot["chunks_retried"] < 1:
        failures.append("hanging chunk did not produce timeout + retry")
    if snapshot.get("rejected_timeout", 0) != G:
        failures.append(
            f"expected {G} timeout-rejected samples, got {snapshot.get('rejected_timeout', 0)}"
        )
    if snapshot.get("rejected_stale_policy", 0) != G:
        failures.append(
            f"expected {G} stale_policy-rejected samples, got "
            f"{snapshot.get('rejected_stale_policy', 0)}"
        )
    if snapshot.get("flag_invalid_answer", 0) < G:
        failures.append("invalid_answer responses were not flagged")
    if snapshot.get("flag_duplicate", 0) < G - 1:
        failures.append("duplicate responses were not flagged")
    expected_lines = N_PROBLEMS * G
    if logger.lines_written != expected_lines:
        failures.append(
            f"JSONL lines {logger.lines_written} != samples {expected_lines} "
            f"(silent drop!)"
        )
    # every line parses and rejected lines carry a reason
    for line in tmp.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["status"] == "rejected" and not record["reject_reason"]:
            failures.append(f"rejected sample without a reason: {record}")
            break

    print(f"\nsamples JSONL: {tmp} ({logger.lines_written} lines)")
    print(f"mean completion: short={mean_short:.2f}s long={mean_long:.2f}s")
    if failures:
        print("\nDRY RUN FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nDRY RUN OK: streaming consumption, bounded pending, timeout/retry, "
          "staleness gate, and lossless JSONL all verified.")
    return 0


# ---------------------------------------------------------------------------
# live mode
# ---------------------------------------------------------------------------

def run_live(config_path: str, n_problems: int, n_rollouts: int) -> int:
    from rq_evolve.config import load_config
    from rq_evolve.verl_adapter import VerlAdapterConfig, VerlTrainerAdapter

    sys.path.insert(0, str(ROOT / "scripts"))
    from train_with_verl import _read_inline_verl_config  # reuse the entry logic

    config = load_config(config_path)
    inline = _read_inline_verl_config(config_path)
    adapter = VerlTrainerAdapter(
        config=VerlAdapterConfig(
            config_path=config.verl.config_path,
            reward_function=config.verl.reward_function,
            inline_config=inline,
        ),
        rq_config=config,
        project_root=ROOT,
    )
    try:
        adapter.dry_run_rollout(n_problems=n_problems, n_rollouts=n_rollouts)
        return 0
    finally:
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--config", default=str(ROOT / "configs" / "rq_evolve_smoke_lora.yaml"))
    parser.add_argument("--n-problems", type=int, default=24)
    parser.add_argument("--n-rollouts", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "mock":
        return run_mock()
    return run_live(args.config, args.n_problems, args.n_rollouts)


if __name__ == "__main__":
    sys.exit(main())
