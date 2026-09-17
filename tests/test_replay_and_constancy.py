"""Measurement-as-training, current-R_Q selection, and the constancy gate."""

import hashlib
from dataclasses import asdict

import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.backends import PendingRollouts, RolloutRecord
from rq_evolve.config import ArchiveConfig, EvolutionConfig, TrainingDataConfig
from rq_evolve.constancy import check_constancy, z_sensitive_fraction
from rq_evolve.dataset import build_replay_training_examples
from rq_evolve.evolution import RQEvolver
from rq_evolve.program import ProblemInstance, ProblemProgram
from rq_evolve.replay import RolloutReplayBuffer
from rq_evolve.problem_type import (
    PROBLEM_TYPE_RULESET,
    problem_type_ruleset_sha256,
)

GEN = (
    "def generate(seed):\n"
    '    return f"what is {seed} plus one?", str(seed + 1)\n\n\n'
    'DOMAIN = "algebra"\n'
)


def _certify(
    program: ProblemProgram, domain: str = "algebra", problem_type: str = "function"
) -> ProblemProgram:
    program.metadata.update(
        {
            "domain": domain,
            "problem_type": problem_type,
            "descriptor_contract": {
                "domain_authority": "source_exact_one_literal",
                "problem_type_authority": "deterministic_statement_and_verifier",
                "problem_type_ruleset": PROBLEM_TYPE_RULESET,
                "problem_type_ruleset_sha256": problem_type_ruleset_sha256(),
                "domain": domain,
                "problem_type": problem_type,
                "source_sha256": hashlib.sha256(
                    program.source_code.encode("utf-8")
                ).hexdigest(),
            },
        }
    )
    return program


def _program(pid="p"):
    return ProblemProgram(source_code=GEN, program_id=pid)


def _inst(seed, pid="p"):
    return ProblemInstance(
        problem=f"q{seed}", answer=str(seed), program_id=pid, seed=seed
    )


def _rollouts(*correct):
    return [
        RolloutRecord(response="x", predicted_answer="1", correct=c, entropy=1.0)
        for c in correct
    ]


# --- the buffer -------------------------------------------------------------


def test_rejected_rollouts_never_enter_the_buffer():
    """A rollout that was never drawn from the policy must not be trained on."""
    buf = RolloutReplayBuffer()
    buf.begin_iteration(0)
    rejected = RolloutRecord(
        response="",
        predicted_answer=None,
        correct=False,
        entropy=0.0,
        status="rejected",
        reject_reason="timeout",
    )
    buf.store("p", _inst(0), [*_rollouts(True), rejected])
    assert buf.get("p")[0].size == 1


def test_a_group_with_no_accepted_rollouts_is_not_stored():
    buf = RolloutReplayBuffer()
    buf.begin_iteration(0)
    rejected = RolloutRecord(
        response="",
        predicted_answer=None,
        correct=False,
        entropy=0.0,
        status="rejected",
        reject_reason="timeout",
    )
    buf.store("p", _inst(0), [rejected, rejected])
    assert not buf.has("p")


def test_the_buffer_is_cleared_every_iteration():
    """Rollouts are on-policy for exactly one update; carrying them over would
    need an importance-ratio correction the design does not have."""
    buf = RolloutReplayBuffer()
    buf.begin_iteration(0)
    buf.store("p", _inst(0), _rollouts(True, False))
    buf.begin_iteration(1)
    assert not buf.has("p")


def test_degenerate_groups_are_counted_not_silently_carried():
    """All-correct / all-wrong groups produce zero advantage under LOO."""
    buf = RolloutReplayBuffer()
    buf.begin_iteration(0)
    buf.store("a", _inst(0, "a"), _rollouts(True, True))  # degenerate
    buf.store("b", _inst(0, "b"), _rollouts(True, False))  # useful
    stats = buf.stats()
    assert stats["replay_groups"] == 2
    assert stats["replay_degenerate_groups"] == 1
    assert stats["replay_degenerate_frac"] == pytest.approx(0.5)


# --- the training batch -----------------------------------------------------


def _champion(pid, s_hat, rq):
    program = _program(pid)
    program.s_hat, program.rq_score = s_hat, rq
    return program


def test_the_batch_is_the_stored_rollouts_and_nothing_else():
    buf = RolloutReplayBuffer()
    buf.begin_iteration(1)
    champ = _champion("p", 0.5, 0.5)
    for z in (11, 12, 13):
        buf.store("p", _inst(z), _rollouts(True, False))
    rows = build_replay_training_examples(
        [champ], replay=buf, iteration=1, frontier_s_hat_range=(0.0, 1.0),
    )
    assert [r["seed"] for r in rows] == [11, 12, 13]
    assert [r["replay_group_id"] for r in rows] == [g.group_id for g in buf.get("p")]
    assert all(r["replay_rollouts"] == 2 for r in rows)


def test_current_measurement_needs_no_score_history():
    buf = RolloutReplayBuffer()
    buf.begin_iteration(0)
    buf.store("fresh", _inst(0, "fresh"), _rollouts(True, False))
    rows = build_replay_training_examples(
        [_champion("fresh", 0.5, 0.5)], replay=buf, iteration=0,
        frontier_s_hat_range=(0.0, 1.0),
    )
    assert [r["program_id"] for r in rows] == ["fresh"]
    assert rows[0]["selection_score"] == 0.5
    assert rows[0]["selection_iteration"] == 0


def test_stale_replay_cannot_supply_current_training():
    buf = RolloutReplayBuffer()
    buf.begin_iteration(0)
    buf.store("p", _inst(0), _rollouts(True, False))
    rows = build_replay_training_examples(
        [_champion("p", 0.5, 1.0)], replay=buf, iteration=1,
        frontier_s_hat_range=(0.0, 1.0), allow_degenerate=True,
    )
    assert rows == []


def test_a_currently_degenerate_program_is_dropped():
    buf = RolloutReplayBuffer()
    buf.begin_iteration(1)
    buf.store("p", _inst(0), _rollouts(True, True))
    rows = build_replay_training_examples(
        [_champion("p", 1.0, 0.0)], replay=buf, iteration=1,
        frontier_s_hat_range=(0.0, 1.0),
    )
    assert rows == []


def test_a_high_score_without_current_rollouts_is_not_selected():
    buf = RolloutReplayBuffer()
    buf.begin_iteration(1)
    rows = build_replay_training_examples(
        [_champion("failed", 0.5, 100.0)], replay=buf, iteration=1,
        frontier_s_hat_range=(0.0, 1.0), allow_degenerate=True,
    )
    assert rows == []


@pytest.mark.parametrize("budget,expected", [(None, ["high", "mid", "low"]),
                                           (2, ["high", "mid"])])
def test_current_fitness_orders_and_limits_the_batch(budget, expected):
    buf = RolloutReplayBuffer()
    buf.begin_iteration(1)
    champs = []
    for pid, score in (("low", 0.1), ("high", 0.9), ("mid", 0.5)):
        buf.store(pid, _inst(0, pid), _rollouts(True, False))
        champs.append(_champion(pid, 0.5, score))
    rows = build_replay_training_examples(
        champs, replay=buf, iteration=1, frontier_s_hat_range=(0.0, 1.0),
        training_budget=budget,
    )
    assert [r["program_id"] for r in rows] == expected
    assert [r["selection_score"] for r in rows] == sorted(
        [r["rq_score"] for r in rows], reverse=True,
    )


def test_random_replay_order_is_seeded_and_independent_of_rq_values():
    buf = RolloutReplayBuffer()
    buf.begin_iteration(1)
    pids = ("a", "b", "c", "d")
    for index, pid in enumerate(pids):
        buf.store(pid, _inst(index, pid), _rollouts(True, False))

    def selected_ids(scores):
        return [row["program_id"] for row in build_replay_training_examples(
            [_champion(pid, 0.5, rq) for pid, rq in zip(pids, scores)],
            replay=buf, iteration=1, frontier_s_hat_range=(0.0, 1.0),
            training_budget=3, select_random_order=True, select_random_seed=23,
        )]

    assert selected_ids((1, 2, 3, 4)) == selected_ids((4, 3, 2, 1))
    assert selected_ids((1, 2, 3, 4)) == selected_ids((1, 2, 3, 4))


def test_random_extra_instance_allocation_is_independent_of_rq_values():
    def allocation(scores):
        champions = [_champion(pid, 0.5, rq) for pid, rq in zip(("a", "b", "c"), scores)]
        evolver = RQEvolver(
            archive=MAPElitesArchive(), backend=None,
            evolution_config=EvolutionConfig(
                train_batch_target=8, frontier_s_hat_range=(0.0, 1.0),
            ),
            training_config=TrainingDataConfig(select_random_order=True, select_random_seed=29),
        )
        evolver.current_iteration = 1
        evolver.replay.begin_iteration(1)
        for champion in champions:
            evolver.replay.store(champion.program_id, _inst(0, champion.program_id),
                                 _rollouts(True, False))
        return evolver._allocate_instances(champions)

    first = allocation((0.1, 0.2, 0.3))
    assert first == allocation((100.0, -5.0, 0.0))
    assert sum(first) == 5  # Three primary groups already occupy the other slots.


# --- the constancy gate -----------------------------------------------------


CONSTANT_DECORATED = (
    "import random\n"
    "def generate(seed):\n"
    "    rng = random.Random(seed)\n"
    "    total = 2 + 2\n"
    '    label = rng.choice(["A", "B", "C", "D", "E"])\n'
    '    return f"Set {label}: what is 2 + 2?", str(total)\n'
)
INVARIANT_FAMILY = (
    "def generate(seed):\n" '    return f"Find the value of {seed} - {seed}.", "0"\n'
)


def _family(source, n=5):
    program = ProblemProgram(
        source_code=(source + '\n\nDOMAIN = "algebra"\n')
    )
    instances = [program.execute(seed=z) for z in range(n)]
    return (
        program.source_code,
        [i.problem for i in instances],
        [i.answer for i in instances],
    )


def test_a_seed_ignoring_generator_is_rejected():
    """Only a label rotates; the mathematics is the same computation every time."""
    verdict = check_constancy(*_family(CONSTANT_DECORATED))
    assert verdict.passed is False
    assert "decorates a fixed computation" in verdict.reason


def test_an_invariant_family_with_a_constant_answer_is_accepted():
    """The answer not moving is legitimate. The NUMBERS not moving is not.

    Rejecting on the answer alone would throw out every invariant and
    feasibility family, an important class of problems in the archive.
    """
    verdict = check_constancy(*_family(INVARIANT_FAMILY))
    assert verdict.passed is True
    assert verdict.answers == 1


def test_the_gate_admits_every_seed_program_in_the_corpus():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "seed_programs"
    for path in sorted(root.glob("*.py")):
        program = ProblemProgram.from_file(path)
        instances = [program.execute(seed=z) for z in range(5)]
        verdict = check_constancy(
            program.source_code,
            [i.problem for i in instances],
            [i.answer for i in instances],
        )
        assert verdict.passed, f"{path.name}: {verdict.reason}"


def test_z_sensitivity_reads_the_seed_through_the_rng():
    """`rng = random.Random(seed)` means reading rng IS reading the seed."""
    source = (
        "import random\n"
        "def generate(seed):\n"
        "    rng = random.Random(seed)\n"
        "    a = rng.randint(1, 9)\n"
        "    b = a * 2\n"
        "    c = 17\n"
        '    return f"{a} {b} {c}", str(b)\n'
    )
    assert z_sensitive_fraction(source) == pytest.approx(2 / 3)


def test_the_archive_gate_rejects_a_seed_ignoring_generator():
    archive = MAPElitesArchive(**asdict(ArchiveConfig()))
    program = ProblemProgram(
        source_code=(
            CONSTANT_DECORATED + '\n\nDOMAIN = "algebra"\n'
        )
    )
    _certify(program)
    assert archive.try_insert(program=program, u_value=1.0, rq_score=0.5) is False
    assert program.metadata["archive_status"] == "seed_variation_rejected"
    assert program.metadata["validity_check"]["constancy_passed"] is False


# --- the whole loop ---------------------------------------------------------


class _CountingBackend:
    max_model_len = 12000

    def __init__(self):
        self.rollouts_generated = 0

    def sync_weights(self):
        pass

    def begin_session(self):
        pass

    def end_session(self):
        pass

    def mutate(self, tasks):
        return [None] * len(tasks)

    def generate_rollouts(self, instances, n_rollouts):
        self.rollouts_generated += len(instances) * n_rollouts
        grouped = [
            [
                RolloutRecord(
                    response="x",
                    predicted_answer="1",
                    correct=(k % 2 == 0),
                    entropy=1.0,
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


def _loop_evolver(backend, group_size=2, batch=6):
    """Two champions, ``batch`` prompts, G rollouts each.

    ``train_batch_target`` above the champion count is the normal case -- the
    frontier is routinely smaller than the batch -- so this also exercises the
    fill: two champions divide six slots as three fresh instances each.
    """
    return RQEvolver(
        archive=MAPElitesArchive(**asdict(ArchiveConfig())),
        backend=backend,
        evolution_config=EvolutionConfig(
            group_size=group_size,
            train_batch_target=batch,
            inner_iterations=0,
            inner_iteration_batch_size=1,
            frontier_s_hat_range=(0.0, 1.0),
        ),
        training_config=TrainingDataConfig(replay_training_batch=True),
    )


def test_extra_instances_use_current_fitness_and_available_groups():
    evolver = _loop_evolver(_CountingBackend(), batch=3)
    high = _champion("high", 0.5, 0.9)
    low = _champion("low", 0.5, 0.1)
    failed = _champion("failed", 0.5, 100.0)
    evolver.current_iteration = 2
    evolver.replay.begin_iteration(2)
    for p in [high, low]:
        evolver.replay.store(p.program_id, _inst(0, p.program_id), _rollouts(True, False))
    assert evolver._allocate_instances([high, low, failed]) == [1, 0, 0]
    evolver.replay.store("high", _inst(1, "high"), _rollouts(True, False))
    assert evolver._allocate_instances([high, low, failed]) == [0, 0, 0]


def test_training_pool_excludes_programs_no_longer_in_the_archive():
    backend = _CountingBackend()
    evolver = _loop_evolver(backend, batch=1)
    incumbent = _champion("measured-incumbent", 0.5, 0.4)
    evolver.current_iteration = 2
    evolver.replay.begin_iteration(2)
    evolver.replay.store(
        incumbent.program_id,
        _inst(7, incumbent.program_id),
        _rollouts(True, False),
    )

    # Cached responses alone do not qualify a program for training after it
    # leaves the archive.
    evolver.refresh_dataset()
    assert evolver.dataset.snapshot() == []


# One sentence per tag, and genuinely different sentences: the archive's
# near-duplicate gate compares numeric-free statements by containment, so two
# fixtures that differ only in a "[A]"/"[B]" tag are one problem to it and the
# second would be turned away -- which is exactly what these tests must not
# depend on. Answers stay seed+1 across tags; only the wording changes.
_TAG_QUESTIONS = {
    "A": "what is {seed} plus one?",
    "B": "a shelf holds {seed} books and one more arrives. how many books stand on it?",
    "C": "a choir of {seed} singers gains a soloist. how many voices sing together?",
}


def _tag_program(tag, domain):
    question = _TAG_QUESTIONS[tag]
    program = ProblemProgram(
        source_code=(
            "def generate(seed):\n"
            f'    return f"{question}", str(seed + 1)\n\n\n'
            f'DOMAIN = "{domain}"\n'
        )
    )
    _certify(program, domain=domain)
    return program


def _seeded(evolver, tag, domain):
    program = _tag_program(tag, domain)
    result = evolver.evaluate_programs([program])[0]
    evolver.archive.try_insert(
        program=program, u_value=result.u_score, rq_score=result.rq_score
    )
    return program


def test_the_loop_trains_only_on_rollouts_it_already_paid_for():
    """Primary and extra calls fill the budget; training reuses both sets."""
    backend = _CountingBackend()
    evolver = _loop_evolver(backend)
    _seeded(evolver, "A", "algebra")
    _seeded(evolver, "B", "geometry")

    per_iteration = []
    for t in range(3):
        before = backend.rollouts_generated
        evolver.run_outer_iteration(t)
        per_iteration.append(
            (backend.rollouts_generated - before, list(evolver.dataset.snapshot()))
        )

    # Every iteration spends exactly train_batch_target x G and not a rollout
    # more: six instances filled from two champions, two rollouts each.
    assert {spend for spend, _ in per_iteration} == {6 * 2}
    # Every iteration, including the first, selects by its current evaluation.
    assert [len(rows) for _, rows in per_iteration] == [6, 6, 6]


def test_a_program_present_before_reassessment_can_train_without_history():
    backend = _CountingBackend()
    evolver = _loop_evolver(backend)
    _seeded(evolver, "A", "algebra")
    evolver.run_outer_iteration(0)
    newcomer = _seeded(evolver, "B", "geometry")
    evolver.run_outer_iteration(1)
    trained = {row["program_id"] for row in evolver.dataset.snapshot()}
    assert newcomer.program_id in trained


def test_every_trained_instance_is_one_that_was_measured():
    backend = _CountingBackend()
    evolver = _loop_evolver(backend)
    _seeded(evolver, "A", "algebra")
    for t in range(2):
        evolver.run_outer_iteration(t)

    measured = {
        group.instance.seed
        for groups in evolver.replay.groups.values()
        for group in groups
    }
    trained = {row["seed"] for row in evolver.dataset.snapshot()}
    assert trained and trained <= measured


def test_instances_are_never_reused_across_iterations():
    """Seeds come from a monotone stream, so memorisation has nothing to grip."""
    backend = _CountingBackend()
    evolver = _loop_evolver(backend)
    _seeded(evolver, "A", "algebra")

    seen: list[set[int]] = []
    for t in range(3):
        evolver.run_outer_iteration(t)
        seen.append({row["seed"] for row in evolver.dataset.snapshot()})

    trained = [s for s in seen if s]
    assert len(trained) >= 2
    assert not trained[0] & trained[1]


@pytest.mark.parametrize("legacy_key", ["previous_rq_scores", "lagged_scores"])
def test_legacy_score_history_is_ignored_on_resume(tmp_path, legacy_key):
    import json

    evolver = _loop_evolver(_CountingBackend())
    program = _seeded(evolver, "A", "algebra")
    evolver.run_outer_iteration(0)
    evolver.save_state(tmp_path)
    state_path = tmp_path / evolver._USED_SEEDS_FILE
    state = json.loads(state_path.read_text())
    assert "previous_rq_scores" not in state
    state[legacy_key] = {program.program_id: [0, 999.0, 999.0]}
    state_path.write_text(json.dumps(state))

    resumed = _loop_evolver(_CountingBackend())
    assert resumed.load_state(tmp_path)
    assert not resumed.dataset.snapshot()  # No rollout cache was restored.
    resumed.run_outer_iteration(1)
    assert resumed.dataset.snapshot()
    assert all(row["selection_score"] == row["rq_score"] != 999.0
               for row in resumed.dataset.snapshot())
    assert not any(e["event"] == "replay_warmup_fallback" for e in resumed.events)


class _ScriptedBackend(_CountingBackend):
    """Control primary/extra outcomes without invoking an LLM."""

    def __init__(self, specs):
        super().__init__()
        self.specs = specs
        self.batches = []
        self.payloads = []

    def generate_rollouts(self, instances, n_rollouts):
        spec = self.specs[len(self.batches)]
        self.batches.append(list(instances))
        self.rollouts_generated += len(instances) * n_rollouts
        grouped = []
        for inst in instances:
            flags, entropy = spec[inst.program_id]
            grouped.append([
                RolloutRecord(
                    response="x", predicted_answer="1",
                    correct=bool(flags and flags[k % len(flags)]), entropy=entropy,
                    status="accepted" if flags else "rejected",
                    reject_reason=None if flags else "worker_error",
                ) for k in range(n_rollouts)
            ])
        payloads = [object() for _ in instances]
        self.payloads.extend(payloads)
        return PendingRollouts(
            instances=list(instances), n_rollouts=n_rollouts,
            grouped=grouped, payloads=payloads,
        )


@pytest.mark.parametrize("budget", [1, 3])
def test_reassessment_reverses_ranking_before_selection_and_extra_allocation(budget):
    evolver = _loop_evolver(_CountingBackend(), batch=budget)
    old_high = _seeded(evolver, "A", "algebra")
    new_high = _seeded(evolver, "B", "geometry")
    old_high.rq_score, new_high.rq_score = 10.0, 0.01
    backend = _ScriptedBackend([
        {old_high.program_id: ((True, False), 0.1),
         new_high.program_id: ((True, False), 2.0)},
        # Extra data must not overwrite the primary score or success rate.
        {new_high.program_id: ((True, True), 100.0)},
    ])
    evolver.backend = backend
    evolver.run_outer_iteration(2)

    rows = evolver.dataset.snapshot()
    assert rows[0]["program_id"] == new_high.program_id
    assert new_high.rq_score == pytest.approx(1.0)
    assert new_high.s_hat == pytest.approx(0.5)
    assert new_high.u_score == pytest.approx(2.0)
    assert new_high.metadata["num_seeds"] == 1
    assert old_high.rq_score == pytest.approx(0.05)
    assert len(rows) == budget
    if budget == 3:
        assert [inst.program_id for inst in backend.batches[1]] == [new_high.program_id]
        assert [row["program_id"] for row in rows] == [
            new_high.program_id, new_high.program_id, old_high.program_id,
        ]
        assert len({row["seed"] for row in rows if row["program_id"] == new_high.program_id}) == 2
    else:
        assert len(backend.batches) == 1  # No extra call for an already full pool.
    for row in rows:
        group = next(g for g in evolver.replay.get(row["program_id"])
                     if g.group_id == row["replay_group_id"])
        assert any(group.payload is payload for payload in backend.payloads)
        assert row["selection_iteration"] == 2
        assert row["selection_score"] == row["rq_score"]


def test_failed_primary_cannot_use_an_old_score_for_selection_or_allocation():
    evolver = _loop_evolver(_CountingBackend(), batch=3)
    failed = _seeded(evolver, "A", "algebra")
    healthy = _seeded(evolver, "B", "geometry")
    failed.rq_score = 99.0
    backend = _ScriptedBackend([
        {failed.program_id: (None, 0.0), healthy.program_id: ((True, False), 1.0)},
        {healthy.program_id: ((True, False), 2.0)},
    ])
    evolver.backend = backend
    evolver.run_outer_iteration(2)
    assert failed.rq_score == 99.0  # Archive failure handling remains intact.
    assert not evolver.replay.has(failed.program_id)
    assert [inst.program_id for inst in backend.batches[1]] == [healthy.program_id] * 2
    assert {row["program_id"] for row in evolver.dataset.snapshot()} == {healthy.program_id}
    assert healthy.u_score == 1.0  # Extras are training data, not a second score.


def test_current_frontier_membership_controls_extra_allocation():
    evolver = _loop_evolver(_CountingBackend(), batch=3)
    no_longer_useful = _seeded(evolver, "A", "algebra")
    now_useful = _seeded(evolver, "B", "geometry")
    now_useful.s_hat = 0.0
    backend = _ScriptedBackend([
        {no_longer_useful.program_id: ((True, True), 1.0),
         now_useful.program_id: ((True, False), 1.0)},
        {now_useful.program_id: ((True, False), 1.0)},
    ])
    evolver.backend = backend
    evolver.run_outer_iteration(2)
    assert [inst.program_id for inst in backend.batches[1]] == [now_useful.program_id] * 2
    assert {r["program_id"] for r in evolver.dataset.snapshot()} == {now_useful.program_id}


def test_failed_extras_preserve_the_primary_fitness_and_cached_group():
    evolver = _loop_evolver(_CountingBackend(), batch=3)
    champion = _seeded(evolver, "A", "algebra")
    backend = _ScriptedBackend([
        {champion.program_id: ((True, False), 2.0)},
        {champion.program_id: (None, 0.0)},
    ])
    evolver.backend = backend
    evolver.run_outer_iteration(2)
    assert champion.rq_score == 1.0
    assert len(evolver.dataset.snapshot()) == 1
    assert evolver.dataset.snapshot()[0]["selection_score"] == 1.0
    assert len(evolver.replay.get(champion.program_id)) == 1
    assert any(e["event"] == "replay_batch_short" for e in evolver.events)


def _script_mutation_children(evolver, child_batches):
    """Stub generation/validation; keep real scoring, caching and admission."""
    from rq_evolve.prompts import MutationTask

    pending = iter(child_batches)
    by_id = {c.program_id: c for batch in child_batches for c in batch}

    def mutate(parents, **_kwargs):
        children = next(pending)
        assert len(children) == len(parents)
        return (
            [MutationTask(op="mutate", prompt="", parent=p) for p in parents],
            [c.program_id for c in children],
            ["test"] * len(children),
        )

    def build(_tasks, outputs):
        result = []
        for pid in outputs:
            child = by_id[pid]
            evolver.seed_stream.reserve_through(pid, 4)
            result.append((child, child.execute(seed=0), None, child.source_code))
        return result

    evolver.evolution_config.inner_iterations = sum(map(len, child_batches))
    evolver.evolution_config.inner_iteration_batch_size = len(child_batches[0])
    evolver.evolution_config.adaptive_mutation_refill = False
    evolver.evolution_config.two_stage_mutation = True
    evolver._mutate_in_two_stages = mutate
    evolver._make_children_from_outputs = build
    evolver._apply_domain_labeling = lambda _entries: None


@pytest.mark.parametrize("budget", [1, 3])
@pytest.mark.parametrize("seed_refresh", [True, False])
def test_mutation_child_trains_now_and_receives_extras_by_current_fitness(budget, seed_refresh):
    from rq_evolve.replay_hook import ReplayRolloutHook

    evolver = _loop_evolver(_CountingBackend(), batch=budget)
    evolver.seed_stream.seed_refresh = seed_refresh
    incumbent = _seeded(evolver, "A", "algebra")
    child = _tag_program("B", "geometry")
    _script_mutation_children(evolver, [[child]])
    backend = _ScriptedBackend([
        {incumbent.program_id: ((True, False), 0.1)},
        {child.program_id: ((True, False), 2.0)},
        {child.program_id: ((True, True), 100.0)},
    ])
    evolver.backend = backend
    evolver.run_outer_iteration(2)

    rows = evolver.dataset.snapshot()
    assert evolver.last_reports[0].status == "inserted"
    assert rows[0]["program_id"] == child.program_id
    assert len(rows) == budget
    assert child.rq_score == 1.0
    assert child.s_hat == 0.5
    assert child.u_score == 2.0
    hook = ReplayRolloutHook(evolver.replay, group_size=2)
    primary = hook._lookup(child.program_id, rows[0]["seed"], group_id=rows[0]["replay_group_id"])
    assert primary.payload is backend.payloads[1]
    assert rows[0]["selection_iteration"] == 2
    assert rows[0]["selection_score"] == child.rq_score
    if budget == 3:
        assert [i.program_id for i in backend.batches[2]] == [child.program_id]
        assert [r["program_id"] for r in rows] == [child.program_id] * 2 + [incumbent.program_id]
        extra = hook._lookup(child.program_id, rows[1]["seed"], group_id=rows[1]["replay_group_id"])
        assert extra.payload is backend.payloads[2]
        assert primary.group_id != extra.group_id
    else:
        assert len(backend.batches) == 2  # Child evaluation is reused, not repeated.


def test_replaced_incumbent_and_child_are_excluded_from_current_training():
    evolver = _loop_evolver(_CountingBackend(), batch=1)
    incumbent = _seeded(evolver, "A", "algebra")
    first = _tag_program("B", "algebra")
    final = _tag_program("C", "algebra")
    _script_mutation_children(evolver, [[first], [final]])
    backend = _ScriptedBackend([
        {incumbent.program_id: ((True, False), 0.1)},
        {first.program_id: ((True, False), 1.0)},
        {final.program_id: ((True, False), 2.0)},
    ])
    evolver.backend = backend
    evolver.run_outer_iteration(2)

    assert [r.status for r in evolver.last_reports] == ["inserted", "inserted"]
    assert {p.program_id for p in evolver.archive.champions()} == {final.program_id}
    assert evolver.replay.has(incumbent.program_id)
    assert evolver.replay.has(first.program_id)
    assert [r["program_id"] for r in evolver.dataset.snapshot()] == [final.program_id]
    assert evolver.replay.get(final.program_id)[0].payload is backend.payloads[2]


@pytest.mark.parametrize("flags,entropy,domain,status,cached", [
    ((True, False), 0.1, "algebra", "rejected_non_elite", True),
    (None, 0.0, "geometry", "rollout_failed", False),
    ((True, True), 2.0, "geometry", "inserted", True),
])
def test_ineligible_child_does_not_enter_training(flags, entropy, domain, status, cached):
    evolver = _loop_evolver(_CountingBackend(), batch=2)
    incumbent = _seeded(evolver, "A", "algebra")
    child = _tag_program("B", domain)
    _script_mutation_children(evolver, [[child]])
    backend = _ScriptedBackend([
        {incumbent.program_id: ((True, False), 1.0)},
        {child.program_id: (flags, entropy)},
        {incumbent.program_id: ((True, False), 1.0)},
    ])
    evolver.backend = backend
    evolver.run_outer_iteration(2)

    assert evolver.last_reports[0].status == status
    assert evolver.replay.has(child.program_id) is cached
    assert {r["program_id"] for r in evolver.dataset.snapshot()} == {incumbent.program_id}
    assert [i.program_id for i in backend.batches[2]] == [incumbent.program_id]


def test_multiple_children_keep_their_own_payloads_in_the_same_batch():
    evolver = _loop_evolver(_CountingBackend(), batch=3)
    incumbent = _seeded(evolver, "A", "algebra")
    first = _tag_program("B", "geometry")
    second = _tag_program("C", "number_theory")
    _script_mutation_children(evolver, [[first, second]])
    backend = _ScriptedBackend([
        {incumbent.program_id: ((True, False), 0.1)},
        {first.program_id: ((True, False), 1.0), second.program_id: ((True, False), 2.0)},
    ])
    evolver.backend = backend
    evolver.run_outer_iteration(2)

    assert [r.status for r in evolver.last_reports] == ["inserted", "inserted"]
    assert [r["program_id"] for r in evolver.dataset.snapshot()] == [
        second.program_id, first.program_id, incumbent.program_id,
    ]
    assert evolver.replay.get(first.program_id)[0].payload is backend.payloads[1]
    assert evolver.replay.get(second.program_id)[0].payload is backend.payloads[2]


def test_evolution_log_records_the_actual_current_selection(tmp_path):
    import json

    evolver = _loop_evolver(_CountingBackend(), batch=1)
    _seeded(evolver, "A", "algebra")
    evolver.run_outer_iteration(2)
    evolver.append_evolution_log(tmp_path, iteration=2, metrics={}, reports=[])
    row = json.loads((tmp_path / "evolution_log.jsonl").read_text())
    selected = row["training_selection"]
    assert len(selected) == 1
    assert selected[0]["selection_iteration"] == 2
    assert selected[0]["selection_score"] == evolver.dataset.snapshot()[0]["rq_score"]
