"""The GROUP x SKILL grid: what a cell is, and what H is no longer."""

import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.concepts import GROUPS, SKILLS
from rq_evolve.program import ProblemProgram


def _program(
    group: str, skill: str, value: int = 1, tag: str = ""
) -> ProblemProgram:
    """A minimal generator carrying one (GROUP, SKILL) pair.

    ``tag`` varies the problem wording. The archive rejects two champions whose
    numeric-free problem skeletons match, so programs that must coexist need
    distinct phrasing, not just distinct numbers.
    """
    phrase = tag or f"{group}-{skill}"
    return ProblemProgram(
        source_code=f'''
def generate(seed):
    return f"Take the {phrase} route: what is {value} + {{seed}}?", str({value} + seed)


GROUP = "{group}"
SKILL = "{skill}"
'''
    )


def test_grid_shape_is_the_two_vocabularies():
    archive = MAPElitesArchive()
    assert archive.n_group_bins == len(GROUPS)
    assert archive.n_skill_bins == len(SKILLS)
    assert len(archive.grid) == len(GROUPS) * len(SKILLS)


def test_cell_is_a_pure_function_of_the_declared_labels():
    archive = MAPElitesArchive()
    cell = archive.program_to_cell(_program("geometry", "casework"))
    assert cell == (GROUPS.index("geometry"), SKILLS.index("casework"))
    assert archive.cell_labels(cell) == ("geometry", "casework")


def test_uncertainty_no_longer_moves_a_program_between_cells():
    """H is fitness, not a coordinate: the same labels land in the same cell."""
    archive = MAPElitesArchive()
    low = _program("algebra", "induction", value=1, tag="first")
    high = _program("algebra", "induction", value=2, tag="second")
    assert archive.program_to_cell(low) == archive.program_to_cell(high)

    archive.try_insert(low, h_value=0.05, problem_text="p", rq_score=0.1)
    archive.try_insert(high, h_value=5.0, problem_text="q", rq_score=0.2)
    # One cell, so the higher R_Q wins it outright rather than both surviving
    # in separate uncertainty bands.
    assert len(archive.champions()) == 1
    assert archive.champions()[0].program_id == high.program_id


def test_h_is_still_recorded_and_still_decides_the_champion():
    archive = MAPElitesArchive()
    winner = _program("algebra", "counting", value=7, tag="winner")
    archive.try_insert(winner, h_value=0.42, problem_text="p", rq_score=0.3)
    assert archive.champions()[0].h_score == pytest.approx(0.42)

    # Same cell, lower R_Q -> loses, and the incumbent is untouched.
    loser = _program("algebra", "counting", value=8, tag="loser")
    assert archive.try_insert(loser, h_value=9.9, problem_text="q", rq_score=0.01) is False
    assert archive.champions()[0].program_id == winner.program_id


def test_the_operators_each_move_exactly_one_coordinate():
    archive = MAPElitesArchive()
    parent = _program("algebra", "counting")
    skill_shifted = _program("algebra", "invariant")   # operator A
    domain_shifted = _program("geometry", "counting")  # operator B

    pg, ps = archive.program_to_cell(parent)
    ag, as_ = archive.program_to_cell(skill_shifted)
    bg, bs = archive.program_to_cell(domain_shifted)

    assert (ag, as_ != ps) == (pg, True), "A holds GROUP, moves SKILL"
    assert (bs, bg != pg) == (ps, True), "B holds SKILL, moves GROUP"


def test_an_unlabelled_program_is_rejected_not_hashed_into_a_cell():
    archive = MAPElitesArchive()
    unlabelled = ProblemProgram(
        source_code='def generate(seed):\n    return "q", str(seed)\n'
    )
    assert archive.program_to_cell(unlabelled) is None
    assert archive.target_cell(unlabelled) is None
    assert archive.try_insert(
        unlabelled, h_value=1.0, problem_text="q", rq_score=0.5
    ) is False
    assert unlabelled.metadata["archive_status"] == "unlabelled_rejected"
    assert archive.champions() == []


def test_out_of_vocabulary_labels_are_rejected_too():
    archive = MAPElitesArchive()
    for group, skill in (("topology", "counting"), ("algebra", "handwaving")):
        assert archive.program_to_cell(_program(group, skill)) is None


def test_stats_report_per_axis_coverage():
    """One number cannot tell 'two domains only' from 'two skills only'."""
    archive = MAPElitesArchive()
    for i, skill in enumerate(("counting", "invariant", "induction")):
        archive.try_insert(
            _program("algebra", skill, value=i),
            h_value=1.0,
            problem_text=f"p{i}",
            rq_score=0.1 * (i + 1),
        )
    stats = archive.stats()
    assert stats["num_champions"] == 3
    assert stats["group_coverage"] == pytest.approx(1 / len(GROUPS))
    assert stats["skill_coverage"] == pytest.approx(3 / len(SKILLS))


def test_snapshot_round_trip_preserves_cells():
    archive = MAPElitesArchive()
    for group, skill in (("algebra", "counting"), ("geometry", "invariant")):
        archive.try_insert(
            _program(group, skill, value=len(group)),
            h_value=1.0,
            problem_text=f"{group}-{skill}",
            rq_score=0.5,
        )
    payload = archive.to_payload()
    assert payload["meta"]["axes"] == ["group", "skill"]

    restored = MAPElitesArchive()
    assert restored.load_payload(payload) == 2
    before = {archive.program_to_cell(p) for p in archive.champions()}
    after = {restored.program_to_cell(p) for p in restored.champions()}
    assert before == after


def test_a_pre_migration_snapshot_is_dropped_rather_than_misplaced(capsys):
    """Old champions carry no SKILL, so there is no honest cell for them."""
    archive = MAPElitesArchive()
    legacy = {
        "meta": {"n_h_bins": 10, "n_div_bins": 6, "diversity_axis": "concept_group"},
        "champions": [
            {
                "source_code": (
                    'def generate(seed):\n    return "q", str(seed)\n\n'
                    'CONCEPT_GROUP = "algebra"\nCONCEPT_TYPE = "algebra.toy"\n'
                ),
                "program_id": "old",
                "rq_score": 0.5,
                "h_score": 0.3,
                # Coordinates from the retired H x diversity grid.
                "niche_h": 4,
                "niche_div": 3,
            }
        ],
    }
    assert archive.load_payload(legacy) == 0
    assert archive.champions() == []
    out = capsys.readouterr().out
    assert "predates the GROUP x SKILL grid" in out


def test_a_snapshot_that_loses_every_champion_is_not_a_resume(tmp_path):
    """A pre-migration archive drops all champions, and resuming into an empty
    grid killed a 4B run on its first batch with "VerlDynamicDataset is empty".

    ``load_state`` used to return True whenever archive.json existed, so the
    adapter took the resume branch and never bootstrapped from seeds.
    """
    import json

    from rq_evolve.evolution import RQEvolver

    (tmp_path / "archive.json").write_text(
        json.dumps(
            {
                "meta": {"axes": ["h", "diversity"]},
                "champions": [
                    {
                        "source_code": "def generate(seed):\n    return 'q', '1'\n",
                        "program_id": "old0",
                        "niche_h": 0,
                        "niche_div": 1,
                        "rq_score": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "rq_used_seeds.json").write_text(
        json.dumps({"used_seeds": {"old0": [0, 1, 2]}}), encoding="utf-8"
    )

    evolver = RQEvolver(archive=MAPElitesArchive(), backend=None)
    assert evolver.load_state(tmp_path) is False
    assert evolver.archive.champions() == []
    # The dead run's consumed seeds must not retire seeds the bootstrap needs.
    assert evolver.used_seeds == {}


def test_flat_binning_fills_every_slot_before_any_competition():
    """The grid arm reserves capacity per (GROUP, SKILL); the flat arm does not.

    Two programs sharing a cell knock each other out under "grid" and coexist
    under "flat" -- that difference is the whole ablation, and it is what makes
    the pair (archive_binning=flat, reevaluate_champions=False) a test of
    whether the MAP earns its place.
    """
    from rq_evolve.archive import MAPElitesArchive

    for binning, expected in (("grid", 1), ("flat", 2)):
        archive = MAPElitesArchive(binning=binning)
        for value, tag in ((3, "alpha"), (5, "beta")):
            archive.try_insert(
                program=_program("algebra", "counting", value=value, tag=tag),
                h_value=1.0,
                problem_text=f"p{value}",
                rq_score=float(value),
            )
        assert len(archive.champions()) == expected, binning


def test_flat_binning_still_refuses_an_unlabelled_program():
    """Dropping the grid must not also relax the label contract -- the arm has
    to isolate one change, not two."""
    from rq_evolve.archive import MAPElitesArchive

    archive = MAPElitesArchive(binning="flat")
    inserted = archive.try_insert(
        program=_program("not_a_group", "not_a_skill"),
        h_value=1.0,
        problem_text="p",
        rq_score=1.0,
    )
    assert inserted is False
    assert archive.champions() == []


def test_flat_binning_challenges_the_weakest_occupant_once_full():
    from rq_evolve.archive import MAPElitesArchive
    from rq_evolve.concepts import GROUPS, SKILLS

    archive = MAPElitesArchive(binning="flat")
    total = len(GROUPS) * len(SKILLS)
    # Alphabetic tags: the template-duplicate gate replaces every digit with
    # 'N', so "t0"/"t1" normalise to the same skeleton and only the first would
    # be admitted.
    names = [f"{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(total)]
    for i, name in enumerate(names):
        archive.try_insert(
            program=_program("algebra", "counting", value=i + 1, tag=name),
            h_value=1.0,
            problem_text=name,
            rq_score=float(i + 1),
        )
    assert len(archive.champions()) == total
    weakest = min(c.rq_score for c in archive.champions())

    # Two digits on purpose: at seed 0 the answer equals `value`, and the
    # answer-leak lint only inspects answers longer than two characters.
    assert archive.try_insert(
        program=_program("algebra", "counting", value=99, tag="strong"),
        h_value=1.0,
        problem_text="strong",
        rq_score=10_000.0,
    )
    assert len(archive.champions()) == total
    assert min(c.rq_score for c in archive.champions()) > weakest


def test_archive_binning_only_accepts_the_two_modes():
    import pytest

    from rq_evolve.archive import MAPElitesArchive
    from rq_evolve.config import EvolutionConfig

    with pytest.raises(ValueError, match="binning"):
        MAPElitesArchive(binning="bogus")
    with pytest.raises(ValueError, match="archive_binning"):
        EvolutionConfig(archive_binning="bogus")
