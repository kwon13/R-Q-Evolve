"""CPU tests for the chunked streaming scheduler (no ray, no vLLM).

Fake workers run on LocalThreadTransport; outputs duck-type the DataProto
surface the consumer uses (.batch dict + .meta_info). These tests are the
contract for requirement "streaming producer-consumer with bounded pending,
timeout/retry, staleness, and lossless logging".
"""

import time

import torch

from rq_evolve.async_rollout import (
    ChunkedRolloutScheduler,
    ChunkJob,
    LocalThreadTransport,
)
from rq_evolve.config import AsyncRolloutConfig
from rq_evolve.metrics import RolloutMetrics
from rq_evolve.program import ProblemInstance
from rq_evolve.rollout_log import JsonlSampleLogger

RESP_WIDTH = 16
G = 2


class FakeTokenizer:
    def __init__(self, texts):
        self.texts = texts

    def decode(self, ids, skip_special_tokens=True):
        return self.texts[int(ids[0])] if ids else ""


class FakeOutput:
    def __init__(self, text_indices, overlong_rows=()):
        n = len(text_indices)
        responses = torch.zeros((n, RESP_WIDTH), dtype=torch.long)
        attention = torch.zeros((n, 2 * RESP_WIDTH), dtype=torch.long)
        for row, text_idx in enumerate(text_indices):
            responses[row, 0] = text_idx
            attention[row, :RESP_WIDTH] = 1
            n_resp = RESP_WIDTH if row in overlong_rows else RESP_WIDTH // 2
            attention[row, RESP_WIDTH : RESP_WIDTH + n_resp] = 1
        self.batch = {"responses": responses, "attention_mask": attention}
        self.meta_info = {"metrics": [{"generate_sequences": 0.1}] * n}


class FakeGenBatch:
    def __init__(self, text_indices, delay_s, overlong_rows=(), fail=False):
        self.text_indices = text_indices
        self.delay_s = delay_s
        self.overlong_rows = overlong_rows
        self.fail = fail


def fake_worker(gen_batch):
    time.sleep(gen_batch.delay_s)
    if gen_batch.fail:
        raise RuntimeError("boom")
    return FakeOutput(gen_batch.text_indices, gen_batch.overlong_rows)


def make_jobs(specs, texts):
    """specs: list of dicts {answer_texts, delay, meta?, overlong?, fail?}."""
    jobs = []
    for i, spec in enumerate(specs):
        inst = ProblemInstance(
            problem=f"p{i}", answer=spec.get("answer", "4"),
            program_id=f"prog_{i}", seed=0,
        )
        text_indices = []
        for text in spec["texts"]:
            text_indices.append(len(texts))
            texts.append(text)
        jobs.append(
            ChunkJob(
                chunk_id=i,
                instances=[inst],
                gen_batch=FakeGenBatch(
                    text_indices,
                    spec.get("delay", 0.05),
                    spec.get("overlong", ()),
                    spec.get("fail", False),
                ),
                batch=None,
                n_rollouts=G,
                meta=spec.get(
                    "meta",
                    {"policy_version": 1, "adapter_version": 0, "global_step": 1,
                     "source_checkpoint": "m@1"},
                ),
            )
        )
    return jobs


def make_scheduler(cfg, texts, tmp_path, current_version=1, n_workers=3):
    metrics = RolloutMetrics()
    logger = JsonlSampleLogger(tmp_path / "samples.jsonl")
    transport = LocalThreadTransport([fake_worker] * n_workers)
    scheduler = ChunkedRolloutScheduler(
        transport=transport,
        cfg=cfg,
        tokenizer=FakeTokenizer(texts),
        metrics=metrics,
        sample_logger=logger,
        current_version_fn=lambda: current_version,
        iteration=0,
        poll_s=0.02,
    )
    return scheduler, metrics, logger, transport


BOXED = ["ok \\boxed{4}.", "also \\boxed{5}."]


def test_streaming_completion_order_not_submission_order(tmp_path):
    texts = []
    # chunk 0: slow; chunks 1..3: fast -- fast ones must finish first
    specs = [{"texts": BOXED, "delay": 0.8}] + [
        {"texts": BOXED, "delay": 0.05} for _ in range(3)
    ]
    jobs = make_jobs(specs, texts)
    cfg = AsyncRolloutConfig(max_in_flight_chunks=4, request_timeout_s=10, verify_workers=2)
    scheduler, metrics, logger, transport = make_scheduler(cfg, texts, tmp_path)
    results = scheduler.run(jobs)
    transport.shutdown()
    assert [r.job.chunk_id for r in results] == [0, 1, 2, 3]  # sorted for caller
    done_at = {r.job.chunk_id: max(rec.ts_end for rec in r.grouped[0]) for r in results}
    assert done_at[1] < done_at[0] and done_at[2] < done_at[0]
    assert metrics.snapshot()["samples_accepted"] == 4 * G
    assert logger.lines_written == 4 * G


def test_in_flight_never_exceeds_bound(tmp_path):
    texts = []
    jobs = make_jobs([{"texts": BOXED, "delay": 0.05} for _ in range(12)], texts)
    cfg = AsyncRolloutConfig(max_in_flight_chunks=2, request_timeout_s=10, verify_workers=2)
    scheduler, metrics, _, transport = make_scheduler(cfg, texts, tmp_path, n_workers=6)
    scheduler.run(jobs)
    transport.shutdown()
    assert metrics.snapshot()["max_pending_chunks"] <= 2


def test_timeout_then_retry_then_explicit_failure(tmp_path):
    texts = []
    specs = [
        {"texts": BOXED, "delay": 5.0},   # hangs past timeout on every attempt
        {"texts": BOXED, "delay": 0.05},
    ]
    jobs = make_jobs(specs, texts)
    cfg = AsyncRolloutConfig(
        max_in_flight_chunks=4, request_timeout_s=0.3, max_retries=1, verify_workers=2
    )
    scheduler, metrics, logger, transport = make_scheduler(cfg, texts, tmp_path)
    results = scheduler.run(jobs)
    transport.shutdown()
    snap = metrics.snapshot()
    assert snap["chunks_retried"] == 1
    assert snap["chunks_timeout"] == 1
    assert snap["rejected_timeout"] == G
    failed = next(r for r in results if r.job.chunk_id == 0)
    assert all(rec.status == "rejected" and rec.reject_reason == "timeout"
               for rec in failed.grouped[0])
    assert failed.full_batch is None
    # nothing dropped: every sample (incl. the failed chunk's) has a JSONL line
    assert logger.lines_written == 2 * G


def test_worker_error_becomes_failure_record(tmp_path):
    texts = []
    jobs = make_jobs(
        [{"texts": BOXED, "fail": True}, {"texts": BOXED}], texts
    )
    cfg = AsyncRolloutConfig(max_in_flight_chunks=4, request_timeout_s=5,
                             max_retries=0, verify_workers=2)
    scheduler, metrics, _, transport = make_scheduler(cfg, texts, tmp_path)
    results = scheduler.run(jobs)
    transport.shutdown()
    failed = next(r for r in results if r.job.chunk_id == 0)
    assert all(rec.reject_reason == "worker_error" for rec in failed.grouped[0])
    assert metrics.snapshot()["rejected_worker_error"] == G


def test_staleness_strict_and_bounded(tmp_path):
    fresh_meta = {"policy_version": 5, "adapter_version": 0, "global_step": 1,
                  "source_checkpoint": "m@1"}
    lag1_meta = dict(fresh_meta, policy_version=4)
    lag2_meta = dict(fresh_meta, policy_version=3)

    # bounded, max_lag=1: lag 0 and 1 accepted, lag 2 rejected
    texts = []
    jobs = make_jobs(
        [
            {"texts": BOXED, "meta": fresh_meta},
            {"texts": BOXED, "meta": lag1_meta},
            {"texts": BOXED, "meta": lag2_meta},
        ],
        texts,
    )
    cfg = AsyncRolloutConfig(staleness_mode="bounded", max_policy_lag=1,
                             request_timeout_s=5, verify_workers=2)
    scheduler, metrics, _, transport = make_scheduler(cfg, texts, tmp_path, current_version=5)
    results = scheduler.run(jobs)
    transport.shutdown()
    by_id = {r.job.chunk_id: r for r in results}
    assert all(rec.status == "accepted" for rec in by_id[0].grouped[0])
    assert all(rec.status == "accepted" for rec in by_id[1].grouped[0])
    assert all(rec.reject_reason == "stale_policy" for rec in by_id[2].grouped[0])

    # strict: any lag rejected
    texts2 = []
    jobs2 = make_jobs(
        [{"texts": BOXED, "meta": fresh_meta}, {"texts": BOXED, "meta": lag1_meta}],
        texts2,
    )
    cfg2 = AsyncRolloutConfig(staleness_mode="strict", request_timeout_s=5, verify_workers=2)
    scheduler2, _, _, transport2 = make_scheduler(cfg2, texts2, tmp_path / "s", current_version=5)
    (tmp_path / "s").mkdir(exist_ok=True)
    results2 = scheduler2.run(jobs2)
    transport2.shutdown()
    by_id2 = {r.job.chunk_id: r for r in results2}
    assert all(rec.status == "accepted" for rec in by_id2[0].grouped[0])
    assert all(rec.reject_reason == "stale_policy" for rec in by_id2[1].grouped[0])


def test_filter_flags_and_optional_rejection(tmp_path):
    nobox = ["no box here at all.", "still no box."]
    dup = ["same \\boxed{4}.", "same \\boxed{4}."]
    overlong_spec = {"texts": BOXED, "overlong": (0, 1)}

    # defaults: overlong rejected; invalid_answer / duplicate detected only
    texts = []
    jobs = make_jobs([{"texts": nobox}, {"texts": dup}, dict(overlong_spec)], texts)
    cfg = AsyncRolloutConfig(request_timeout_s=5, verify_workers=2)
    scheduler, metrics, _, transport = make_scheduler(cfg, texts, tmp_path)
    results = scheduler.run(jobs)
    transport.shutdown()
    by_id = {r.job.chunk_id: r for r in results}
    snap = metrics.snapshot()
    assert snap["flag_invalid_answer"] == G
    assert snap["flag_duplicate"] == G - 1
    assert snap["rejected_overlong"] == G
    # invalid answers stay ACCEPTED with correct=False (s_hat semantics)
    assert all(rec.status == "accepted" and rec.correct is False
               for rec in by_id[0].grouped[0])
    assert all(rec.status == "rejected" for rec in by_id[2].grouped[0])

    # opt-in strict filtering rejects them
    texts2 = []
    jobs2 = make_jobs([{"texts": nobox}, {"texts": dup}], texts2)
    cfg2 = AsyncRolloutConfig(
        request_timeout_s=5, verify_workers=2,
        reject_invalid_answer=True, reject_duplicates=True,
    )
    scheduler2, metrics2, _, transport2 = make_scheduler(cfg2, texts2, tmp_path / "opt")
    (tmp_path / "opt").mkdir(exist_ok=True)
    results2 = scheduler2.run(jobs2)
    transport2.shutdown()
    by_id2 = {r.job.chunk_id: r for r in results2}
    assert all(rec.reject_reason == "invalid_answer" for rec in by_id2[0].grouped[0])
    dup_rows = by_id2[1].grouped[0]
    assert dup_rows[0].status == "accepted"
    assert dup_rows[1].reject_reason == "duplicate"


def test_verify_error_degrades_to_failure_record(tmp_path):
    """An exception in the verify path must reject ITS chunk only -- never
    discard the phase's other results (the end-of-phase f.result() trap)."""

    class RaisingTokenizer(FakeTokenizer):
        def decode(self, ids, skip_special_tokens=True):
            text = super().decode(ids, skip_special_tokens)
            if text == "RAISE":
                raise ValueError("synthetic verify failure")
            return text

    texts = []
    jobs = make_jobs([{"texts": ["RAISE", "RAISE"]}, {"texts": BOXED}], texts)
    cfg = AsyncRolloutConfig(request_timeout_s=5, verify_workers=2)
    metrics = RolloutMetrics()
    logger = JsonlSampleLogger(tmp_path / "samples.jsonl")
    transport = LocalThreadTransport([fake_worker] * 2)
    scheduler = ChunkedRolloutScheduler(
        transport=transport,
        cfg=cfg,
        tokenizer=RaisingTokenizer(texts),
        metrics=metrics,
        sample_logger=logger,
        current_version_fn=lambda: 1,
        iteration=0,
        poll_s=0.02,
    )
    results = scheduler.run(jobs)  # must NOT raise
    transport.shutdown()
    by_id = {r.job.chunk_id: r for r in results}
    assert all(rec.reject_reason == "worker_error" for rec in by_id[0].grouped[0])
    assert all(rec.status == "accepted" for rec in by_id[1].grouped[0])
    snap = metrics.snapshot()
    assert snap["flag_verify_error"] == 1
    assert metrics.phase_ended_at is not None  # phase_end ran despite the error


def test_record_metadata_stamped(tmp_path):
    texts = []
    meta = {"policy_version": 9, "adapter_version": 2, "global_step": 77,
            "source_checkpoint": "/m@global_step_77"}
    jobs = make_jobs([{"texts": BOXED, "meta": meta}], texts)
    cfg = AsyncRolloutConfig(request_timeout_s=5, verify_workers=1)
    scheduler, _, _, transport = make_scheduler(cfg, texts, tmp_path, current_version=9)
    results = scheduler.run(jobs)
    transport.shutdown()
    rec = results[0].grouped[0][0]
    assert rec.policy_version == 9
    assert rec.adapter_version == 2
    assert rec.global_step == 77
    assert rec.source_checkpoint == "/m@global_step_77"
    assert rec.ts_end >= rec.ts_start > 0
    assert rec.response_tokens == RESP_WIDTH // 2
