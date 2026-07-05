import json
import threading

from rq_evolve.rollout_log import JsonlSampleLogger, make_sample_record


def _record(i: int = 0, status: str = "accepted", reason=None) -> dict:
    return make_sample_record(
        iteration=1,
        program_id=f"prog_{i}",
        instance_seed=0,
        rollout_idx=i,
        chunk_id=i,
        status=status,
        reject_reason=reason,
        correct=True,
        predicted_answer="4",
        prompt_tokens=10,
        response_tokens=20,
        latency_s=1.234,
        ts_start=100.0,
        ts_end=101.0,
        policy_version=3,
        adapter_version=1,
        global_step=42,
        source_checkpoint="/m@global_step_42",
        entropy=None,
    )


def test_logger_writes_parseable_lines(tmp_path):
    path = tmp_path / "samples.jsonl"
    logger = JsonlSampleLogger(path)
    logger.log(_record(0))
    logger.log(_record(1, status="rejected", reason="timeout"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and logger.lines_written == 2
    first, second = (json.loads(line) for line in lines)
    assert first["status"] == "accepted" and first["reject_reason"] is None
    assert second["status"] == "rejected" and second["reject_reason"] == "timeout"
    # metadata the async-correctness contract requires on EVERY sample
    for key in (
        "policy_version",
        "adapter_version",
        "global_step",
        "source_checkpoint",
        "ts_start",
        "ts_end",
    ):
        assert key in first


def test_logger_disabled_is_noop(tmp_path):
    path = tmp_path / "off.jsonl"
    logger = JsonlSampleLogger(path, enabled=False)
    logger.log(_record())
    assert not path.exists() and logger.lines_written == 0
    logger_none = JsonlSampleLogger(None)
    logger_none.log(_record())  # must not raise
    assert logger_none.lines_written == 0


def test_logger_thread_safe(tmp_path):
    path = tmp_path / "concurrent.jsonl"
    logger = JsonlSampleLogger(path)
    n_threads, per_thread = 8, 100

    def worker(tid):
        for i in range(per_thread):
            logger.log(_record(tid * per_thread + i))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_threads * per_thread
    for line in lines:
        json.loads(line)  # no interleaved/corrupt lines
