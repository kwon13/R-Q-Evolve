"""Phase 2: measurement-as-training, lagged selection, and the constancy gate."""

from dataclasses import asdict

import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.backends import PendingRollouts, RolloutRecord
from rq_evolve.config import ArchiveConfig, EvolutionConfig, TrainingDataConfig
from rq_evolve.constancy import check_constancy, z_sensitive_fraction
from rq_evolve.dataset import build_replay_training_examples
from rq_evolve.evolution import RQEvolver
from rq_evolve.program import ProblemInstance, ProblemProgram
from rq_evolve.replay import LaggedScoreboard, RolloutReplayBuffer

GEN = (
    "def generate(seed):\n"
    '    return f"what is {seed} plus one?", str(seed + 1)\n\n\n'
    'GROUP = "algebra"\nSKILL = "counting"\n'
)


def _program(pid="p"):
    return ProblemProgram(source_code=GEN, program_id=pid)


def _inst(seed, pid="p"):
    return ProblemInstance(problem=f"q{seed}", answer=str(seed), program_id=pid, seed=seed)


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
    rejected = RolloutRecord(response="", predicted_answer=None, correct=False,
                             entropy=0.0, status="rejected", reject_reason="timeout")
    buf.store("p", _inst(0), [*_rollouts(True), rejected])
    assert buf.get("p")[0].size == 1


def test_a_group_with_no_accepted_rollouts_is_not_stored():
    buf = RolloutReplayBuffer()
    buf.begin_iteration(0)
    rejected = RolloutRecord(response="", predicted_answer=None, correct=False,
                             entropy=0.0, status="rejected", reject_reason="timeout")
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
    buf.store("a", _inst(0, "a"), _rollouts(True, True))     # degenerate
    buf.store("b", _inst(0, "b"), _rollouts(True, False))    # useful
    stats = buf.stats()
    assert stats["replay_groups"] == 2
    assert stats["replay_degenerate_groups"] == 1
    assert stats["replay_degenerate_frac"] == pytest.approx(0.5)


# --- lagged selection -------------------------------------------------------


def test_a_program_first_scored_this_iteration_has_no_lagged_score():
    """Newly inserted elites wait one iteration; that IS the winner's-curse break."""
    board = LaggedScoreboard()
    board.record("p", 0, 1.0)
    assert board.selection_score("p", 0) is None
    assert board.selection_score("p", 1) == pytest.approx(1.0)


def test_the_lagged_score_is_an_ewma_of_past_iterations():
    board = LaggedScoreboard(ewma_alpha=0.5)
    board.record("p", 0, 1.0)
    board.record("p", 1, 0.0)
    assert board.selection_score("p", 2) == pytest.approx(0.5)


def test_the_scoreboard_survives_a_round_trip():
    board = LaggedScoreboard(ewma_alpha=0.25)
    board.record("p", 3, 0.8)
    restored = LaggedScoreboard.from_dict(board.to_dict(), ewma_alpha=0.25)
    assert restored.selection_score("p", 4) == pytest.approx(0.8)


# --- the training batch -----------------------------------------------------


def _champion(pid, s_hat, rq):
    program = _program(pid)
    program.s_hat, program.rq_score = s_hat, rq
    return program


def test_the_batch_is_the_stored_rollouts_and_nothing_else():
    """No sampling pass: every instance trained on is one that was measured."""
    buf = RolloutReplayBuffer(); buf.begin_iteration(1)
    board = LaggedScoreboard(); board.record("p", 0, 0.5)
    champ = _champion("p", 0.5, 0.5)
    for z in (11, 12, 13):
        buf.store("p", _inst(z), _rollouts(True, False))

    rows = build_replay_training_examples(
        [champ], replay=buf, lagged=board, iteration=1,
        frontier_s_hat_range=(0.0, 1.0),
    )
    assert [r["seed"] for r in rows] == [11, 12, 13]
    assert all(r["replay_rollouts"] == 2 for r in rows)


def test_an_elite_with_no_lagged_score_does_not_train_yet():
    buf = RolloutReplayBuffer(); buf.begin_iteration(0)
    buf.store("fresh", _inst(0, "fresh"), _rollouts(True, False))
    rows = build_replay_training_examples(
        [_champion("fresh", 0.5, 0.5)],
        replay=buf, lagged=LaggedScoreboard(), iteration=0,
        frontier_s_hat_range=(0.0, 1.0),
    )
    assert rows == []


def test_a_currently_degenerate_elite_is_dropped_even_with_a_good_past_score():
    """Selected on the past, but the batch would be all-zero advantage now."""
    buf = RolloutReplayBuffer(); buf.begin_iteration(1)
    buf.store("p", _inst(0), _rollouts(True, True))
    board = LaggedScoreboard(); board.record("p", 0, 9.0)
    rows = build_replay_training_examples(
        [_champion("p", 1.0, 0.0)],   # s_hat = 1.0 -> outside the frontier band
        replay=buf, lagged=board, iteration=1,
        frontier_s_hat_range=(0.0, 1.0),
    )
    assert rows == []


def test_the_batch_is_ordered_by_the_lagged_score():
    buf = RolloutReplayBuffer(); buf.begin_iteration(1)
    board = LaggedScoreboard()
    champs = []
    for pid, past in (("low", 0.1), ("high", 0.9), ("mid", 0.5)):
        buf.store(pid, _inst(0, pid), _rollouts(True, False))
        board.record(pid, 0, past)
        champs.append(_champion(pid, 0.5, 0.5))
    rows = build_replay_training_examples(
        champs, replay=buf, lagged=board, iteration=1,
        frontier_s_hat_range=(0.0, 1.0),
    )
    assert [r["program_id"] for r in rows] == ["high", "mid", "low"]


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
    "def generate(seed):\n"
    '    return f"Find the value of {seed} - {seed}.", "0"\n'
)


def _family(source, n=5):
    program = ProblemProgram(
        source_code=source + '\n\nGROUP = "algebra"\nSKILL = "counting"\n'
    )
    instances = [program.execute(seed=z) for z in range(n)]
    return program.source_code, [i.problem for i in instances], [i.answer for i in instances]


def test_a_seed_ignoring_generator_is_rejected():
    """Only a label rotates; the mathematics is the same computation every time."""
    verdict = check_constancy(*_family(CONSTANT_DECORATED))
    assert verdict.passed is False
    assert "decorates a fixed computation" in verdict.reason


def test_an_invariant_family_with_a_constant_answer_is_accepted():
    """The answer not moving is legitimate. The NUMBERS not moving is not.

    Rejecting on the answer alone would throw out every invariant and
    feasibility family, which is a whole SKILL of the archive.
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
        source_code=CONSTANT_DECORATED + '\n\nGROUP = "algebra"\nSKILL = "counting"\n'
    )
    assert archive.try_insert(program=program, u_value=1.0, rq_score=0.5) is False
    assert program.metadata["archive_status"] == "seed_variation_rejected"
    assert program.metadata["validity_check"]["constancy_passed"] is False


# --- the whole loop ---------------------------------------------------------


class _CountingBackend:
    max_model_len = 12000

    def __init__(self):
        self.rollouts_generated = 0

    def sync_weights(self): pass
    def begin_session(self): pass
    def end_session(self): pass
    def mutate(self, tasks): return [None] * len(tasks)

    def generate_rollouts(self, instances, n_rollouts):
        self.rollouts_generated += len(instances) * n_rollouts
        grouped = [
            [
                RolloutRecord(response="x", predicted_answer="1",
                              correct=(k % 2 == 0), entropy=1.0)
                for k in range(n_rollouts)
            ]
            for _ in instances
        ]
        return PendingRollouts(instances=list(instances), n_rollouts=n_rollouts,
                               grouped=grouped)

    def finalize_rollouts(self, pending): return pending.grouped


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
            group_size=group_size, train_batch_target=batch,
            inner_iterations=0, inner_iteration_batch_size=1,
            frontier_s_hat_range=(0.0, 1.0),
        ),
        training_config=TrainingDataConfig(replay_training_batch=True),
    )


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


def _seeded(evolver, tag, group):
    question = _TAG_QUESTIONS[tag]
    program = ProblemProgram(
        source_code=(
            "def generate(seed):\n"
            f'    return f"{question}", str(seed + 1)\n\n\n'
            f'GROUP = "{group}"\nSKILL = "counting"\n'
        )
    )
    result = evolver.evaluate_programs([program])[0]
    evolver.archive.try_insert(
        program=program, u_value=result.u_score, rq_score=result.rq_score
    )
    return program


def test_the_loop_trains_only_on_rollouts_it_already_paid_for():
    """The budget claim: one re-scoring pass, no second sampling pass."""
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
    # ...and every one of them yields a batch (iteration 0 via the warm-up
    # fallback, since nothing has a prior measurement to be selected on yet).
    assert [len(rows) for _, rows in per_iteration] == [6, 6, 6]


def test_the_lag_is_in_force_once_anything_has_history():
    """The fallback is for "no earlier iteration exists", not a general bypass.

    A champion inserted at iteration t must not be selected on the rollouts it
    is about to train on -- that is the winner's-curse coupling the lag breaks.
    """
    backend = _CountingBackend()
    evolver = _loop_evolver(backend)
    _seeded(evolver, "A", "algebra")
    evolver.run_outer_iteration(0)

    newcomer = _seeded(evolver, "B", "geometry")
    evolver.run_outer_iteration(1)

    # "A" has history, so the fallback does not fire and "B" -- first scored at
    # iteration 1 -- is held back.
    trained = {row["program_id"] for row in evolver.dataset.snapshot()}
    assert newcomer.program_id not in trained
    assert trained


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


def test_a_cold_resume_falls_back_to_warmup_rather_than_an_empty_batch():
    """An archive restored without lagged history has nothing to lag against.

    Handing verl an empty dataloader kills the run; falling back to the current
    scores is the same call bootstrap makes, and the lag re-engages as soon as
    any champion carries history.
    """
    backend = _CountingBackend()
    evolver = _loop_evolver(backend)
    program = _seeded(evolver, "A", "algebra")
    evolver.lagged.history.clear()          # what a pre-scoreboard snapshot looks like

    evolver.run_outer_iteration(0)

    assert evolver.dataset.snapshot(), "cold resume produced an empty training set"
    assert any(e["event"] == "replay_warmup_fallback" for e in evolver.events)
