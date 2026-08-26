"""What happens when the frontier band admits nobody.

A run died at iteration 65 with ``IndexError: VerlDynamicDataset is empty``.
Two independent defects put it there and both are pinned here.

First, `reevaluate_degenerate_every: 0` froze every degenerate champion's score
forever, so degenerate champions accumulated monotonically: 8 champions with 5
on the frontier became 30 with 1, and mean R_Q fell 0.059 -> 0.0021.

Second, nothing caught the empty frontier. verl's dataloader raises out of
__getitem__ and the run is over -- there is no recovery path from inside the
trainer, and 65 iterations of archive went with it.
"""

import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.config import EvolutionConfig, TrainingDataConfig
from rq_evolve.dataset import build_replay_training_examples
from rq_evolve.evolution import RQEvolver
from rq_evolve.program import ProblemInstance, ProblemProgram
from rq_evolve.replay import LaggedScoreboard, RolloutReplayBuffer

BAND = (0.0, 1.0)


def _program(tag: str, s_hat: float) -> ProblemProgram:
    p = ProblemProgram(
        source_code=(
            "import random\n\n\n"
            "def generate(seed):\n"
            "    rng = random.Random(seed)\n"
            f"    n = rng.randint(2, 40) + {len(tag)}\n"
            '    answer = n * (n + 1) // 2\n'
            "    check = sum(range(1, n + 1))\n"
            '    assert answer == check, f"answer={answer} check={check}"\n'
            '    return f"Let n = {n}. Sum the first n positive integers.", str(answer)\n\n\n'
            'GROUP = "algebra"\n'
            'SKILL = "counting"\n'
        ),
        metadata={"group": "algebra", "skill": "counting"},
    )
    p.s_hat = s_hat
    p.rq_score = s_hat * (1 - s_hat)
    p.u_score = 0.5
    return p


class _Replay:
    """Stands in for RolloutReplayBuffer: every program has one stored group."""

    class _Group:
        size = 2

        def __init__(self, pid):
            self.instance = ProblemInstance(
                problem=f"q for {pid}", answer="1", program_id=pid, seed=0
            )
            self.responses = ["a", "b"]
            self.rewards = [1.0, 0.0]

    def __init__(self, pids):
        self.pids = set(pids)

    def has(self, pid):
        return pid in self.pids

    def get(self, pid):
        return [self._Group(pid)]


class _Lagged:
    def selection_score(self, pid, iteration):
        return 0.5


# --- defect 1: the score freeze ------------------------------------------


def test_degenerate_champions_are_re_measured_by_default():
    """0 turns a transient miss into a permanent one; the default must not."""
    assert EvolutionConfig().reevaluate_degenerate_every == 1


def test_the_knob_still_allows_skipping_when_asked():
    assert EvolutionConfig(reevaluate_degenerate_every=0).reevaluate_degenerate_every == 0


# --- defect 2: the empty dataloader --------------------------------------


def test_the_band_normally_excludes_degenerate_champions():
    champs = [_program("a", 0.0), _program("b", 0.5), _program("c", 1.0)]
    rows = build_replay_training_examples(
        champs,
        replay=_Replay(c.program_id for c in champs),
        lagged=_Lagged(),
        iteration=2,
        frontier_s_hat_range=BAND,
    )
    assert {r["program_id"] for r in rows} == {champs[1].program_id}


def test_an_all_degenerate_archive_yields_nothing_under_the_band():
    """The state that killed the run: the band is right, and it admits nobody."""
    champs = [_program("a", 0.0), _program("b", 1.0), _program("c", 0.0)]
    rows = build_replay_training_examples(
        champs,
        replay=_Replay(c.program_id for c in champs),
        lagged=_Lagged(),
        iteration=2,
        frontier_s_hat_range=BAND,
    )
    assert rows == []


def test_allow_degenerate_keeps_the_dataloader_non_empty():
    champs = [_program("a", 0.0), _program("b", 1.0), _program("c", 0.0)]
    rows = build_replay_training_examples(
        champs,
        replay=_Replay(c.program_id for c in champs),
        lagged=_Lagged(),
        iteration=2,
        frontier_s_hat_range=BAND,
        allow_degenerate=True,
    )
    assert len(rows) == 3, "one row per champion, band bypassed"


# --- the two together, through refresh_dataset ---------------------------


class _Backend:
    max_model_len = None

    def mutate(self, tasks):
        return [None] * len(tasks)


def _evolver_with(champs) -> RQEvolver:
    archive = MAPElitesArchive()
    ev = RQEvolver(
        archive=archive,
        backend=_Backend(),
        evolution_config=EvolutionConfig(),
        training_config=TrainingDataConfig(replay_training_batch=True),
    )
    for c in champs:
        archive.try_insert(c, c.u_score, c.rq_score)
    ev.replay = _Replay(c.program_id for c in champs)
    ev.lagged = _Lagged()
    ev.current_iteration = 5
    return ev


def test_refresh_dataset_does_not_hand_the_trainer_an_empty_batch():
    """IndexError: VerlDynamicDataset is empty -- the run's actual cause of death."""
    champs = [_program("a", 0.0), _program("b", 1.0)]
    ev = _evolver_with(champs)
    ev.refresh_dataset()

    assert len(ev.dataset) > 0, "an all-degenerate archive must not empty the loader"
    assert any(e["event"] == "frontier_empty_fallback" for e in ev.events)


def test_the_fallback_stays_out_of_the_way_when_the_band_has_someone():
    champs = [_program("a", 0.0), _program("b", 0.5)]
    ev = _evolver_with(champs)
    ev.refresh_dataset()

    assert len(ev.dataset) > 0
    assert not any(e["event"] == "frontier_empty_fallback" for e in ev.events), (
        "the band admitted someone; the escape must not fire"
    )
