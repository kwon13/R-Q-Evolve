"""Fresh seeds are an integrity device, not a convenience."""

import json
from dataclasses import asdict

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.backends import PendingRollouts, RolloutRecord
from rq_evolve.config import ArchiveConfig, EvolutionConfig
from rq_evolve.evolution import RQEvolver
from rq_evolve.program import ProblemProgram
from rq_evolve.seed_stream import SeedStream

HONEST = (
    "def generate(seed):\n"
    '    return f"what is {seed} plus one?", str(seed + 1)\n\n\n'
    'GROUP = "algebra"\nSKILL = "counting"\n'
)

# The failure fresh seeds exist to catch: calibrated on the graded instance,
# degenerate everywhere else. Under a fixed seed-0 evaluation it is invisible.
TAIL_OVERFIT = (
    "def generate(seed):\n"
    "    if seed == 0:\n"
    '        return "a genuinely open question about 0", "7"\n'
    '    return f"trivial: what is {seed}?", str(seed)\n\n\n'
    'GROUP = "algebra"\nSKILL = "counting"\n'
)


def test_seeds_are_never_reissued():
    stream = SeedStream()
    first = stream.take("p", 5)
    second = stream.take("p", 5)
    assert first == [0, 1, 2, 3, 4]
    assert second == [5, 6, 7, 8, 9]
    assert not set(first) & set(second)


def test_each_program_has_its_own_cursor():
    stream = SeedStream()
    assert stream.take("a", 3) == [0, 1, 2]
    assert stream.take("b", 3) == [0, 1, 2]
    assert stream.take("a", 1) == [3]


def test_a_legacy_used_seed_record_resumes_past_the_largest():
    """Old snapshots record WHICH seeds went out, not how far the stream ran."""
    stream = SeedStream.from_used_seeds({"p": [0, 1, 7]})
    assert stream.take("p", 2) == [8, 9]


def test_the_cursor_survives_a_round_trip():
    stream = SeedStream()
    stream.take("p", 4)
    restored = SeedStream.from_dict(json.loads(json.dumps(stream.to_dict())))
    assert restored.take("p", 1) == [4]


# --- the evolver actually draws from it -------------------------------------


class _Backend:
    def __init__(self):
        self.seen_problems = []

    def begin_session(self): pass
    def end_session(self): pass
    def sync_weights(self): pass
    def mutate(self, tasks): return [None] * len(tasks)

    def generate_rollouts(self, instances, n_rollouts):
        self.seen_problems.extend(i.problem for i in instances)
        grouped = [
            [RolloutRecord(response="x", predicted_answer="1",
                           correct=(k % 2 == 0), entropy=1.0)
             for k in range(n_rollouts)]
            for _ in instances
        ]
        return PendingRollouts(instances=list(instances), n_rollouts=n_rollouts,
                               grouped=grouped)

    def finalize_rollouts(self, pending): return pending.grouped


def _evolver(backend, n=5, m=2):
    return RQEvolver(
        archive=MAPElitesArchive(**asdict(ArchiveConfig())),
        backend=backend,
        evolution_config=EvolutionConfig(eval_seeds=n, rollouts_per_seed=m),
    )


def test_a_program_is_graded_on_n_distinct_fresh_instances():
    backend = _Backend()
    evolver = _evolver(backend, n=5, m=2)
    result = evolver.evaluate_programs([ProblemProgram(source_code=HONEST)])[0]

    assert result.num_seeds == 5
    assert result.num_rollouts == 10          # n x m
    assert len(set(backend.seen_problems)) == 5


def test_a_second_evaluation_uses_different_instances():
    """Re-scoring must not re-grade the instances the program already passed."""
    backend = _Backend()
    evolver = _evolver(backend, n=3)
    program = ProblemProgram(source_code=HONEST)

    evolver.evaluate_programs([program])
    first = list(backend.seen_problems)
    backend.seen_problems.clear()
    evolver.evaluate_programs([program])

    assert not set(first) & set(backend.seen_problems)


def test_a_seed_zero_special_case_no_longer_hides():
    """The whole point: grading on seed 0 forever makes this program look fine.

    Drawing fresh seeds exposes the other branch, which is where the program
    actually lives.
    """
    backend = _Backend()
    evolver = _evolver(backend, n=5)
    evolver.evaluate_programs([ProblemProgram(source_code=TAIL_OVERFIT)])

    assert sum("genuinely open" in p for p in backend.seen_problems) == 1
    assert sum("trivial" in p for p in backend.seen_problems) == 4


def test_the_cursor_is_persisted_and_restored(tmp_path):
    backend = _Backend()
    evolver = _evolver(backend, n=3)
    program = ProblemProgram(source_code=HONEST)
    evolver.archive.try_insert(program=program, u_value=1.0, rq_score=0.5)
    evolver.evaluate_programs([program])
    evolver.save_state(tmp_path, iteration=0)

    resumed = _evolver(_Backend(), n=3)
    resumed.load_state(tmp_path)
    assert resumed.seed_stream.peek(program.program_id) == 3
