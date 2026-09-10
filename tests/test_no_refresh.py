"""Tests for No-refresh ablation (seed_refresh=False).

When seed_refresh is False:
- SeedStream always yields seed 0 and cursor never advances.
- Evaluator draws seed 0 across all iterations.
- ReplayRolloutHook can serve multiple seed-0 groups for the same program.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.backends import PendingRollouts, RolloutRecord
from rq_evolve.config import ArchiveConfig, EvolutionConfig
from rq_evolve.dataset import build_training_examples
from rq_evolve.evolution import RQEvolver
from rq_evolve.program import ProblemInstance, ProblemProgram
from rq_evolve.replay import ReplayGroup, RolloutReplayBuffer
from rq_evolve.replay_hook import ReplayRolloutHook
from rq_evolve.seed_stream import SeedStream

SAMPLE_CODE = (
    "def generate(seed):\n"
    '    return f"problem with seed {seed}", str(seed * 2)\n\n\n'
    'DOMAIN = "algebra"\n'
)


def test_seed_stream_no_refresh_always_zeros():
    stream = SeedStream(seed_refresh=False)
    assert stream.take("p1", 1) == [0]
    assert stream.take("p1", 1) == [0]
    assert stream.take("p1", 3) == [0, 0, 0]
    assert stream.peek("p1") == 0
    stream.reserve_through("p1", 10)
    assert stream.peek("p1") == 0
    assert stream.take("p1", 2) == [0, 0]


def test_seed_stream_serialization_preserves_no_refresh():
    stream = SeedStream(seed_refresh=False)
    stream.take("p1", 5)
    payload = stream.to_dict()
    assert payload.get("__seed_refresh__") is False

    restored = SeedStream.from_dict(payload)
    assert restored.seed_refresh is False
    assert restored.take("p1", 1) == [0]


class _DummyBackend:
    def __init__(self):
        self.seen_problems = []
        self.seen_instances = []

    def begin_session(self):
        pass

    def end_session(self):
        pass

    def sync_weights(self):
        pass

    def mutate(self, tasks):
        return [None] * len(tasks)

    def generate_rollouts(self, instances, n_rollouts):
        self.seen_instances.extend(instances)
        self.seen_problems.extend(i.problem for i in instances)
        grouped = [
            [
                RolloutRecord(
                    response=f"ans_{k}",
                    predicted_answer="0",
                    correct=True,
                    entropy=0.5,
                )
                for k in range(n_rollouts)
            ]
            for _ in instances
        ]
        return PendingRollouts(
            instances=list(instances), n_rollouts=n_rollouts, grouped=grouped
        )

    def finalize_rollouts(self, pending):
        return pending.grouped


def test_evolver_draw_instances_no_refresh():
    backend = _DummyBackend()
    evolver = RQEvolver(
        archive=MAPElitesArchive(**asdict(ArchiveConfig())),
        backend=backend,
        evolution_config=EvolutionConfig(group_size=2, seed_refresh=False),
    )
    program = ProblemProgram(source_code=SAMPLE_CODE)

    # First draw
    insts_1 = evolver.draw_instances(program, n_seeds=1)
    assert len(insts_1) == 1
    assert insts_1[0].seed == 0
    assert insts_1[0].problem == "problem with seed 0"

    # Second draw: must still be seed 0 under no refresh
    insts_2 = evolver.draw_instances(program, n_seeds=3)
    assert len(insts_2) == 3
    for inst in insts_2:
        assert inst.seed == 0
        assert inst.problem == "problem with seed 0"


def test_evolver_repeated_evaluation_under_no_refresh():
    backend = _DummyBackend()
    evolver = RQEvolver(
        archive=MAPElitesArchive(**asdict(ArchiveConfig())),
        backend=backend,
        evolution_config=EvolutionConfig(group_size=2, seed_refresh=False),
    )
    program = ProblemProgram(source_code=SAMPLE_CODE)

    # Multi-iteration evaluations
    for _ in range(3):
        evolver.evaluate_programs([program], store_replay=True)

    # All instances rolled out must have been seed 0
    assert len(backend.seen_instances) == 3
    for inst in backend.seen_instances:
        assert inst.seed == 0
        assert inst.problem == "problem with seed 0"


def test_replay_hook_serves_multiple_seed_zero_groups():
    buffer = RolloutReplayBuffer()
    inst = ProblemInstance(
        problem="what is 0 * 2?",
        answer="0",
        program_id="p1",
        seed=0,
        verifier={"type": "exact"},
    )
    buffer.store(
        "p1",
        inst,
        [RolloutRecord(response="r1", predicted_answer="0", correct=True, entropy=0.1)],
        payload=["dummy_proto_1"],
    )
    buffer.store(
        "p1",
        inst,
        [RolloutRecord(response="r2", predicted_answer="0", correct=True, entropy=0.1)],
        payload=["dummy_proto_2"],
    )

    hook = ReplayRolloutHook(buffer, group_size=1)
    # Lookup sequential groups with same seed
    looked_up_1 = hook._lookup("p1", seed=0, index=0)
    looked_up_2 = hook._lookup("p1", seed=0, index=1)
    looked_up_3 = hook._lookup("p1", seed=0, index=2)  # beyond bounds falls back to last

    assert looked_up_1.payload == ["dummy_proto_1"]
    assert looked_up_2.payload == ["dummy_proto_2"]
    assert looked_up_3.payload == ["dummy_proto_2"]


def test_dataset_build_training_examples_no_refresh():
    prog = ProblemProgram(source_code=SAMPLE_CODE)
    prog.rq_score = 0.5
    prog.s_hat = 0.5
    prog.u_score = 1.0

    used_seeds: dict[str, set[int]] = {}
    examples_1 = build_training_examples(
        [prog],
        instances_per_program=1,
        training_budget=1,
        frontier_s_hat_range=(0.1, 0.9),
        used_seeds=used_seeds,
        seed_refresh=False,
    )
    assert len(examples_1) == 1
    assert examples_1[0]["seed"] == 0

    # Second call without seed refresh still yields seed 0
    examples_2 = build_training_examples(
        [prog],
        instances_per_program=1,
        training_budget=1,
        frontier_s_hat_range=(0.1, 0.9),
        used_seeds=used_seeds,
        seed_refresh=False,
    )
    assert len(examples_2) == 1
    assert examples_2[0]["seed"] == 0
