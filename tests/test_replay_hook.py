"""The hook that serves the training batch from the re-scoring rollouts.

Its one non-negotiable property: it must never hand back a row belonging to a
different prompt. A wrong batch trains the policy on someone else's response and
nothing downstream can detect it, so every path that cannot prove alignment has
to fall through to real generation instead.
"""

import types

import pytest

from rq_evolve.backends import RolloutRecord
from rq_evolve.program import ProblemInstance
from rq_evolve.replay import RolloutReplayBuffer
from rq_evolve.replay_hook import ReplayRolloutHook

M = 2


class _Tensors:
    """Stands in for a TensorDict: only ``batch_size`` is read."""

    def __init__(self, rows, keys=()):
        self.batch_size = (rows,)
        self._data = {key: object() for key in keys}

    def keys(self):
        return self._data.keys()

    def __contains__(self, key):
        return key in self._data

    def pop(self, key):
        return self._data.pop(key)


class _FakeBatch:
    """The parts of DataProto the hook actually reads."""

    def __init__(self, rows, extras=None, prompts=None, tag=""):
        self.batch = _Tensors(rows)
        self.non_tensor_batch = {}
        if extras is not None:
            self.non_tensor_batch["extra_info"] = extras
        if prompts is not None:
            self.non_tensor_batch["raw_prompt"] = prompts
        self.meta_info = {}
        self.tag = tag


def _payload(n=M, tag=""):
    return _FakeBatch(n, tag=tag)


def _instance(pid, seed, problem):
    return ProblemInstance(problem=problem, answer="1", program_id=pid, seed=seed)


def _records(n=M):
    return [
        RolloutRecord(response="x", predicted_answer="1", correct=True, entropy=1.0)
        for _ in range(n)
    ]


def _buffer(entries):
    buf = RolloutReplayBuffer()
    buf.begin_iteration(0)
    for pid, seed, problem in entries:
        buf.store(
            pid, _instance(pid, seed, problem), _records(),
            payload=_payload(tag=f"{pid}/{seed}"),
        )
    return buf


def _request(entries):
    """A gen_batch the way the trainer builds it: each prompt repeated m times."""
    extras, prompts = [], []
    for pid, seed, problem in entries:
        for _ in range(M):
            extras.append({"program_id": pid, "seed": seed})
            prompts.append([
                {"role": "system", "content": "rules"},
                {"role": "user", "content": problem},
            ])
    return _FakeBatch(len(extras), extras=extras, prompts=prompts)


def _hook(buf):
    return ReplayRolloutHook(buf, group_size=M)


def test_a_fully_cached_batch_is_planned_from_the_buffer():
    entries = [("a", 0, "问 A"), ("b", 3, "问 B")]
    hook = _hook(_buffer(entries))
    plan = hook._plan(_request(entries))
    assert plan is not None
    assert [p.tag for p in plan] == ["a/0", "b/3"]


def test_a_prompt_that_does_not_match_is_never_served():
    """The check that makes a silent swap impossible."""
    hook = _hook(_buffer([("a", 0, "the stored problem")]))
    request = _request([("a", 0, "a DIFFERENT problem")])
    assert hook._plan(request) is None
    assert hook.stats.misses["prompt_mismatch"] == 1


def test_an_uncached_program_falls_through():
    hook = _hook(_buffer([("a", 0, "q")]))
    assert hook._plan(_request([("zzz", 0, "q")])) is None
    assert hook.stats.misses["not_in_buffer"] == 1


def test_an_uncached_seed_of_a_cached_program_falls_through():
    """Seeds are never reused, so a stale seed must not resolve to a sibling."""
    hook = _hook(_buffer([("a", 0, "q")]))
    assert hook._plan(_request([("a", 99, "q")])) is None
    assert hook.stats.misses["not_in_buffer"] == 1


def test_a_batch_without_extra_info_falls_through():
    hook = _hook(_buffer([("a", 0, "q")]))
    assert hook._plan(_FakeBatch(2)) is None
    assert hook.stats.misses["no_extra_info"] == 1


def test_a_rollout_n_that_does_not_divide_the_batch_falls_through():
    entries = [("a", 0, "q")]
    hook = ReplayRolloutHook(_buffer(entries), group_size=3)
    assert hook._plan(_request(entries)) is None
    assert hook.stats.misses["rollout_n_mismatch"] == 1


def test_a_cached_group_of_the_wrong_size_falls_through():
    buf = RolloutReplayBuffer()
    buf.begin_iteration(0)
    buf.store("a", _instance("a", 0, "q"), _records(), payload=_payload(n=5))
    hook = _hook(buf)
    assert hook._plan(_request([("a", 0, "q")])) is None
    assert hook.stats.misses["payload_size_mismatch"] == 1


def test_a_group_whose_rows_are_not_one_instance_falls_through():
    """If the trainer stops repeating the way this assumes, do not guess."""
    hook = _hook(_buffer([("a", 0, "q"), ("b", 1, "r")]))
    request = _request([("a", 0, "q")])
    request.non_tensor_batch["extra_info"][1] = {"program_id": "b", "seed": 1}
    assert hook._plan(request) is None
    assert hook.stats.misses["group_not_contiguous"] == 1


# --- the wrapper itself -----------------------------------------------------


def test_install_falls_through_to_the_original_on_a_miss():
    calls = []

    def original(batch, *a, **k):
        calls.append(batch)
        return "GENERATED"

    manager = types.SimpleNamespace(generate_sequences=original)
    hook = _hook(_buffer([("a", 0, "q")]))
    assert hook.install(manager) is True

    assert manager.generate_sequences(_request([("zzz", 0, "q")])) == "GENERATED"
    assert len(calls) == 1
    assert hook.stats.served_steps == 0
    assert hook.stats.steps == 1


def test_uninstall_restores_the_original():
    def original(batch, *a, **k):
        return "GENERATED"

    manager = types.SimpleNamespace(generate_sequences=original)
    hook = _hook(_buffer([]))
    hook.install(manager)
    hook.uninstall()
    assert manager.generate_sequences is original


def test_stats_report_the_hit_rate():
    hook = _hook(_buffer([]))
    hook.stats.steps, hook.stats.served_steps = 4, 3
    assert hook.stats.to_wandb()["replay/hit_rate"] == pytest.approx(0.75)


def test_the_row_count_comes_from_the_non_tensor_side_when_tensors_are_empty():
    """The training gen_batch carries no tensors at all.

    The dataset returns raw_prompt as chat messages plus one dummy tensor, and
    _get_gen_batch pops no tensor keys, so batch.batch is empty while the
    non-tensor arrays hold every row. Counting only tensors reported 0 rows and
    declined every step.
    """
    class _NoTensors:
        batch = None
        non_tensor_batch = {"extra_info": [{}] * 8, "raw_prompt": [[]] * 8}
        meta_info = {}

    assert ReplayRolloutHook._batch_size(_NoTensors()) == 8


def test_a_tensorless_batch_is_planned_normally():
    entries = [("a", 0, "q"), ("b", 1, "r")]
    hook = _hook(_buffer(entries))
    request = _request(entries)
    request.batch = None            # exactly the training-path shape
    plan = hook._plan(request)
    assert plan is not None and [p.tag for p in plan] == ["a/0", "b/1"]


def test_replayed_request_metadata_has_no_tensordict_key_collision():
    served = _FakeBatch(2)
    served.batch = _Tensors(2, keys=("temperature",))
    served.non_tensor_batch["global_steps"] = [0, 0]
    served.non_tensor_batch["responses"] = ["a", "b"]

    collisions = ReplayRolloutHook._install_request_meta(
        served,
        {"temperature": 1.0, "global_steps": 7},
    )

    assert collisions == {"temperature", "global_steps"}
    assert "temperature" not in served.batch
    assert "global_steps" not in served.non_tensor_batch
    assert served.non_tensor_batch["responses"] == ["a", "b"]
    assert served.meta_info == {"temperature": 1.0, "global_steps": 7}


# --- the reward that replay makes load-bearing ------------------------------


def test_solver_rollouts_carry_the_real_ground_truth():
    """verl's agent loop scores a generation AS IT PRODUCES IT.

    `extract_reward` then reads that stored `rm_scores` tensor rather than
    recomputing, so the ground truth handed to the generation IS the one the
    training reward comes from. An empty placeholder there scores every rollout
    0 -- zero reward, zero advantage, zero gradient -- and nothing in the loop
    reports it. This pins the plumbing so that cannot come back.
    """
    import inspect

    from rq_evolve import verl_backend

    src = inspect.getsource(verl_backend.VerlPolicyBackend._make_prompt_batch)
    assert "ground_truths" in src, "_make_prompt_batch must accept ground truths"
    assert 'reward_model_arr[i] = {"ground_truth": truth}' in src

    # both solver-rollout paths must supply them
    batched = inspect.getsource(verl_backend.VerlPolicyBackend.generate_rollouts)
    streaming = inspect.getsource(
        verl_backend.VerlPolicyBackend._generate_rollouts_streaming
    )
    assert "ground_truths=[inst.answer for inst in instances]" in batched
    assert "ground_truths=[inst.answer for inst in chunk_instances]" in streaming


def test_evolve_phase_calls_are_not_counted_as_training_steps():
    """The hook wraps ONE method that three callers share.

    Only the trainer's call is replayable. This backend's own evolve phase
    (mutation, judge, R_Q scoring) and verl's validation both go through
    generate_sequences with no extra_info, and both must generate. Counting
    them as steps reported replay/hit_rate 0.4 on a run whose two training
    steps were both served from the buffer.
    """
    calls = []

    def original(batch, *a, **k):
        calls.append(batch)
        return "GENERATED"

    manager = types.SimpleNamespace(generate_sequences=original)
    hook = _hook(_buffer([("a", 0, "q")]))
    hook.install(manager)

    # What verl_backend._make_prompt_batch builds: no extra_info anywhere.
    evolve_batch = _FakeBatch(8)
    evolve_batch.non_tensor_batch = {
        "data_source": ["rq_evolve"] * 8,
        "raw_prompt": [[{"role": "user", "content": "mutate this"}]] * 8,
        "raw_prompt_ids": [[1, 2]] * 8,
        "reward_model": [{"ground_truth": ""}] * 8,
    }
    for _ in range(3):
        assert manager.generate_sequences(evolve_batch) == "GENERATED"

    assert hook.stats.steps == 0, "evolve-phase calls are not training steps"
    assert hook.stats.non_training_calls == 3
    assert hook.stats.served_steps == 0
    assert len(calls) == 3

    # The classifier does not swallow a real training batch on the way past.
    assert hook._plan(_request([("a", 0, "q")])) is not None

    # And the three pass-throughs stay out of the ratio: one training step,
    # served, reads 1.0 -- not 1/4.
    hook.stats.steps = hook.stats.served_steps = 1
    report = hook.stats.to_wandb()
    assert report["replay/hit_rate"] == pytest.approx(1.0)
    assert report["replay/non_training_calls"] == 3
