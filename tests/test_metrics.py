import threading

from rq_evolve.metrics import (
    PolicyVersionTracker,
    RolloutMetrics,
    _percentile,
    check_staleness,
)


def test_percentile():
    assert _percentile([], 95) == 0.0
    assert _percentile([1.0], 95) == 1.0
    values = [float(x) for x in range(1, 101)]
    assert _percentile(values, 50) == 50.0
    assert _percentile(values, 95) == 95.0


def test_counters_and_snapshot():
    m = RolloutMetrics()
    m.phase_start()
    m.on_submit(10)
    m.on_chunk_complete(10, 2.0, [1.0, 3.0], prompt_tokens=100, response_tokens=200)
    m.on_chunk_failed("timeout")
    m.on_retry()
    for _ in range(7):
        m.on_sample(True)
    for _ in range(3):
        m.on_sample(False, "timeout")
    m.on_flag("duplicate")
    m.gauge(5, 2)
    m.gauge(3, 1)
    m.add_duration("verify", 1.5)
    m.phase_end()
    snap = m.snapshot()
    assert snap["chunks_submitted"] == 1
    assert snap["chunks_completed"] == 1
    assert snap["chunks_timeout"] == 1
    assert snap["chunks_retried"] == 1
    assert snap["requests_submitted"] == 10
    assert snap["samples_accepted"] == 7
    assert snap["samples_rejected"] == 3
    assert snap["rejected_timeout"] == 3
    assert snap["flag_duplicate"] == 1
    assert snap["chunk_latency_avg_s"] == 2.0
    assert snap["max_pending_chunks"] == 5
    assert snap["max_queue_depth"] == 2
    assert snap["pending_chunks"] == 3
    assert snap["time_verify_s"] == 1.5
    assert snap["prompt_tokens"] == 100 and snap["response_tokens"] == 200
    assert snap["tokens_per_s"] > 0


def test_phase_start_first_call_wins():
    m = RolloutMetrics()
    m.phase_start()
    first = m.phase_started_at
    m.phase_start()
    assert m.phase_started_at == first
    m.reset()
    assert m.phase_started_at is None


def test_thread_safety_smoke():
    m = RolloutMetrics()
    n_threads, per_thread = 8, 500

    def worker():
        for _ in range(per_thread):
            m.on_sample(True)
            m.on_sample(False, "duplicate")
            m.on_flag("overlong")

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = m.snapshot()
    assert snap["samples_accepted"] == n_threads * per_thread
    assert snap["rejected_duplicate"] == n_threads * per_thread
    assert snap["flag_overlong"] == n_threads * per_thread


def test_version_tracker_stamp_and_sync():
    tracker = PolicyVersionTracker()
    tracker.lora_enabled = True
    tracker._model_path = "/models/foo"
    assert tracker.stamp()["policy_version"] == 0
    tracker.record_sync(global_step=7, duration_s=0.5)
    stamp = tracker.stamp()
    assert stamp["policy_version"] == 1
    assert stamp["adapter_version"] == 1
    assert stamp["global_step"] == 7
    assert stamp["source_checkpoint"] == "/models/foo@global_step_7"
    tracker.lora_enabled = False
    tracker.record_sync(global_step=8, duration_s=0.1)
    assert tracker.policy_version == 2
    assert tracker.adapter_version == 1  # only bumps when lora is on


def test_version_tracker_install_wraps_update_weights():
    class Manager:
        def __init__(self):
            self.calls = []

        def update_weights(self, global_steps=None):
            self.calls.append(global_steps)
            return "ok"

    class Trainer:
        pass

    trainer = Trainer()
    trainer.checkpoint_manager = Manager()
    tracker = PolicyVersionTracker()
    tracker.install(trainer, lora_enabled=False, model_path="/m")
    assert trainer.checkpoint_manager.update_weights(3) == "ok"
    assert trainer.checkpoint_manager.calls == [3]
    assert tracker.policy_version == 1
    assert tracker.last_sync_global_step == 3
    # idempotent: second install must not double-wrap
    tracker.install(trainer, lora_enabled=False, model_path="/m")
    trainer.checkpoint_manager.update_weights(4)
    assert tracker.policy_version == 2


def test_check_staleness():
    assert check_staleness(5, 5, mode="strict", max_policy_lag=1) == (True, 0)
    assert check_staleness(4, 5, mode="strict", max_policy_lag=1) == (False, 1)
    assert check_staleness(4, 5, mode="bounded", max_policy_lag=1) == (True, 1)
    assert check_staleness(3, 5, mode="bounded", max_policy_lag=1) == (False, 2)
    assert check_staleness(3, 5, mode="bounded", max_policy_lag=2) == (True, 2)
