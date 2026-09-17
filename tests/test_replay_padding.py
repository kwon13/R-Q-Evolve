"""Padded training rows must retain the exact measured rollout group."""

from collections import Counter
import random
from types import SimpleNamespace

import pytest

from rq_evolve.backends import RolloutRecord
from rq_evolve.dataset import (
    DynamicProblemDataset,
    VerlDynamicDataset,
    build_replay_training_examples,
)
from rq_evolve.program import ProblemInstance, ProblemProgram
from rq_evolve.replay import RolloutReplayBuffer
from rq_evolve.replay_hook import ReplayRolloutHook
from rq_evolve.seed_stream import SeedStream


GROUP_SIZE = 10
BATCH_SIZE = 32
PROGRAM_CODE = "def generate(seed):\n    return str(seed), str(seed)\n"


def _dataset(num_groups, *, iteration=1, seed_refresh=False, buffer=None):
    buffer = buffer if buffer is not None else RolloutReplayBuffer()
    buffer.begin_iteration(iteration)
    champion = ProblemProgram(
        source_code=PROGRAM_CODE,
        program_id="p",
        s_hat=0.5,
        rq_score=0.5,
    )
    seeds = SeedStream(seed_refresh=seed_refresh).take("p", num_groups)
    for slot, seed in enumerate(seeds):
        instance = ProblemInstance(
            problem=f"What is {seed} + 1?",
            answer=str(seed + 1),
            program_id="p",
            seed=seed,
        )
        records = [
            RolloutRecord(
                response=f"rollout {slot}/{sample}",
                predicted_answer=str(seed + 1 + sample),
                correct=sample < GROUP_SIZE // 2,
                entropy=1.0,
            )
            for sample in range(GROUP_SIZE)
        ]
        # Only shape and identity matter to planning; a real DataProto and GPU
        # generation would add no coverage of dataset-to-replay alignment.
        payload = SimpleNamespace(
            batch=SimpleNamespace(batch_size=(GROUP_SIZE,)),
            non_tensor_batch={},
            tag=slot,
        )
        buffer.store("p", instance, records, payload=payload)

    examples = build_replay_training_examples(
        [champion],
        replay=buffer,
        iteration=iteration,
        frontier_s_hat_range=(0.0, 1.0),
    )
    dataset = VerlDynamicDataset(
        DynamicProblemDataset(examples), tokenizer=None, min_size=BATCH_SIZE
    )
    return buffer, examples, dataset


def _shuffled_rows(dataset, *, seed=23):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return [dataset[index] for index in indices]


def _request(rows):
    # verl shuffles prompts, then repeats each prompt rollout.n times. The
    # training gen_batch carries its row count on the non-tensor side.
    return SimpleNamespace(
        batch=None,
        non_tensor_batch={
            "extra_info": [
                dict(row["extra_info"])
                for row in rows
                for _ in range(GROUP_SIZE)
            ],
            "raw_prompt": [
                row["raw_prompt"] for row in rows for _ in range(GROUP_SIZE)
            ],
        },
        meta_info={},
    )


def _assert_exact_plan(hook, rows):
    plan = hook._plan(_request(rows))
    assert plan is not None
    assert len(plan) == len(rows)
    by_id = {
        group.group_id: group for group in hook.buffer.get("p")
    }
    for row, payload in zip(rows, plan):
        extra = row["extra_info"]
        group = by_id[extra["replay_group_id"]]
        assert extra["seed"] == group.instance.seed
        assert payload is group.payload
    assert not hook.stats.misses
    return plan


@pytest.mark.parametrize(
    "num_groups,pad_offset,iteration,seed_refresh",
    [
        (1, 0, 0, False),
        (10, 0, 1, False),
        (11, 0, 1, False),
        (11, 7, 3, False),
        (11, 12, 4, False),
        (32, 3, 2, False),
        (11, 7, 3, True),
        (32, 3, 2, True),
    ],
)
def test_padded_shuffled_replay_keeps_each_group_and_balances_reuse(
    num_groups, pad_offset, iteration, seed_refresh
):
    buffer, examples, dataset = _dataset(
        num_groups, iteration=iteration, seed_refresh=seed_refresh
    )
    groups = buffer.get("p")
    group_ids = [group.group_id for group in groups]
    assert all(isinstance(group_id, str) and group_id for group_id in group_ids)
    assert len(set(group_ids)) == num_groups
    assert [row["replay_group_id"] for row in examples] == group_ids
    if not seed_refresh:
        assert {row["seed"] for row in examples} == {0}

    dataset.pad_offset = pad_offset
    rows = _shuffled_rows(dataset)
    assert len(rows) == BATCH_SIZE
    plan = _assert_exact_plan(ReplayRolloutHook(buffer, group_size=GROUP_SIZE), rows)

    counts = Counter(payload.tag for payload in plan)
    repeats, surplus = divmod(BATCH_SIZE, num_groups)
    assert set(counts) == set(range(num_groups))
    assert sorted(counts.values()) == sorted(
        [repeats + 1] * surplus + [repeats] * (num_groups - surplus)
    )


def test_rotating_padding_shares_the_surplus_across_all_groups():
    buffer, _, dataset = _dataset(11)
    hook = ReplayRolloutHook(buffer, group_size=GROUP_SIZE)
    total_counts = Counter()
    short_groups = set()
    for offset in range(11):
        dataset.pad_offset = offset
        plan = _assert_exact_plan(hook, _shuffled_rows(dataset, seed=offset))
        counts = Counter(payload.tag for payload in plan)
        assert sorted(counts.values()) == [2] + [3] * 10
        short_groups.update(tag for tag, count in counts.items() if count == 2)
        total_counts.update(counts)

    assert short_groups == set(range(11))
    assert total_counts == Counter({slot: BATCH_SIZE for slot in range(11)})


def test_frontier_filter_leaving_eleven_groups_preserves_balanced_replay():
    buffer, _, _ = _dataset(11)
    frontier = ProblemProgram(
        source_code=PROGRAM_CODE, program_id="p", s_hat=0.5, rq_score=0.5
    )
    degenerate = ProblemProgram(
        source_code=PROGRAM_CODE, program_id="dropped", s_hat=0.0, rq_score=0.0
    )
    for _ in range(BATCH_SIZE - 11):
        buffer.store(
            "dropped",
            ProblemInstance(
                problem="What is 0 + 1?", answer="1", program_id="dropped", seed=0
            ),
            [
                RolloutRecord(
                    response="0", predicted_answer="0", correct=False, entropy=1.0
                )
                for _ in range(GROUP_SIZE)
            ],
        )
    examples = build_replay_training_examples(
        [degenerate, frontier],
        replay=buffer,
        iteration=1,
        frontier_s_hat_range=(0.0, 1.0),
    )
    assert buffer.stats()["replay_groups"] == BATCH_SIZE
    assert len(examples) == 11
    assert {row["program_id"] for row in examples} == {"p"}

    dataset = VerlDynamicDataset(
        DynamicProblemDataset(examples), tokenizer=None, min_size=BATCH_SIZE
    )
    dataset.pad_offset = 4
    plan = _assert_exact_plan(
        ReplayRolloutHook(buffer, group_size=GROUP_SIZE), _shuffled_rows(dataset)
    )
    assert sorted(Counter(payload.tag for payload in plan).values()) == [2] + [3] * 10


def test_chunked_requests_and_repeated_calls_keep_the_same_group_identity():
    buffer, _, dataset = _dataset(11)
    dataset.pad_offset = 5
    rows = _shuffled_rows(dataset)
    hook = ReplayRolloutHook(buffer, group_size=GROUP_SIZE)
    complete = _assert_exact_plan(hook, rows)
    chunks = [rows[:6], rows[6:17], rows[17:]]
    chunked = [payload for chunk in chunks for payload in _assert_exact_plan(hook, chunk)]
    repeated = _assert_exact_plan(hook, rows)
    assert all(actual is expected for actual, expected in zip(chunked, complete))
    assert all(actual is expected for actual, expected in zip(repeated, complete))


def test_previous_iteration_ids_cannot_resolve_to_new_seed_zero_rollouts():
    buffer, examples, dataset = _dataset(11, iteration=1)
    old_ids = {row["replay_group_id"] for row in examples}
    old_request = _request(_shuffled_rows(dataset))
    buffer, new_examples, new_dataset = _dataset(11, iteration=2, buffer=buffer)
    assert old_ids.isdisjoint(row["replay_group_id"] for row in new_examples)

    hook = ReplayRolloutHook(buffer, group_size=GROUP_SIZE)
    assert hook._plan(old_request) is None
    assert hook.stats.misses["not_in_buffer"] == 1
    _assert_exact_plan(
        ReplayRolloutHook(buffer, group_size=GROUP_SIZE), _shuffled_rows(new_dataset)
    )


@pytest.mark.parametrize("num_groups", [1, 11])
def test_legacy_seed_only_rows_replay_only_when_the_group_is_unambiguous(num_groups):
    buffer, _, dataset = _dataset(num_groups)
    request = _request(_shuffled_rows(dataset))
    for extra in request.non_tensor_batch["extra_info"]:
        extra.pop("replay_group_id")
    hook = ReplayRolloutHook(buffer, group_size=GROUP_SIZE)
    plan = hook._plan(request)
    if num_groups == 1:
        assert plan is not None
        assert len(plan) == BATCH_SIZE
        assert all(payload is buffer.get("p")[0].payload for payload in plan)
    else:
        assert plan is None
        assert hook.stats.misses["not_in_buffer"] == 1
