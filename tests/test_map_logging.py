"""The archive picture and the judge telemetry that scalar coverage cannot carry."""

import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.concepts import GROUPS, SKILLS
from rq_evolve.config import EvolutionConfig
from rq_evolve.evolution import RQEvolver
from rq_evolve.map_figure import occupied_cells, render_archive_figure
from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import MUTATION_OP


def _champion(group: str, skill: str, rq: float, s_hat: float) -> ProblemProgram:
    program = ProblemProgram(
        source_code=(
            "def generate(seed):\n"
            f'    return f"{group} {skill} {{seed}}?", str(seed + 1)\n\n\n'
            f'GROUP = "{group}"\nSKILL = "{skill}"\n'
        )
    )
    program.s_hat = s_hat
    return program


def _filled(cells) -> MAPElitesArchive:
    archive = MAPElitesArchive()
    for group, skill, rq, s_hat in cells:
        archive.try_insert(
            program=_champion(group, skill, rq, s_hat),
            u_value=1.0, rq_score=rq,
        )
    return archive


def test_the_figure_covers_the_whole_vocabulary_not_just_the_filled_part():
    """An empty cell has to be visible as empty; that is the point of the image."""
    archive = _filled([("algebra", "invariant", 0.3, 0.5)])
    figure = render_archive_figure(archive, iteration=1, stats=archive.stats())
    assert figure is not None
    ax = figure.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == list(GROUPS)
    assert [t.get_text() for t in ax.get_xticklabels()] == list(SKILLS)


def test_a_zero_scoring_champion_is_drawn_as_occupied():
    """R_Q=0 now holds a cell, so it must not render as an empty one."""
    archive = _filled([("geometry", "extremal_principle", 0.0, 0.0)])
    assert occupied_cells(archive) == {
        (GROUPS.index("geometry"), SKILLS.index("extremal_principle"))
    }
    assert render_archive_figure(archive, iteration=1) is not None


def test_newly_filled_cells_are_outlined():
    archive = _filled([("algebra", "counting", 0.2, 0.4)])
    before = set()
    figure = render_archive_figure(
        archive, iteration=1, new_cells=occupied_cells(archive) - before
    )
    boxes = [p for p in figure.axes[0].patches if not p.get_fill()]
    assert len(boxes) == 1


def test_an_empty_archive_still_renders():
    assert render_archive_figure(MAPElitesArchive(), iteration=0) is not None


# --- judge telemetry -------------------------------------------------------


class _Judge:
    max_model_len = 12000

    def __init__(self, replies):
        self.replies = replies

    def mutate(self, tasks):
        return [self.replies[i % len(self.replies)] for i in range(len(tasks))]


def _entry(group="algebra", skill="counting"):
    program = _champion(group, skill, 0.0, 0.0)
    program.metadata["op"] = MUTATION_OP
    from rq_evolve.program import ProblemInstance

    return {
        "task": type("T", (), {"op": MUTATION_OP})(),
        "child": program,
        "inst": ProblemInstance(
            problem="What is 2 + 2?", answer="4", program_id="p", seed=0
        ),
    }


def test_the_tally_separates_a_mismatch_from_a_refusal():
    """accept_rate alone cannot tell those apart, and they need opposite fixes."""
    evolver = RQEvolver(
        archive=MAPElitesArchive(),
        backend=_Judge([
            "GROUP: algebra\nSKILL: counting",          # agrees
            "GROUP: algebra\nSKILL: invariant",         # SKILL mismatch
            "GROUP: none\nSKILL: none\nFAILURE_REASON: underdetermined",
        ]),
        evolution_config=EvolutionConfig(use_evaluator=True),
    )
    evolver.judge_tally = {k: 0 for k in (
        "reached", "agreed", "group_agreed", "skill_agreed",
        "label_mismatch", "failed_closed", "skill_none", "group_none")}
    evolver.judge_skill_counts = {}

    evolver._apply_judge([_entry(), _entry(), _entry()])

    t = evolver.judge_tally
    assert t["reached"] == 3
    assert t["agreed"] == 1
    assert t["label_mismatch"] == 1
    assert t["failed_closed"] == 1
    assert t["group_agreed"] == 2 and t["skill_agreed"] == 1


def test_the_emitted_skill_vocabulary_is_recorded():
    """A SKILL the judge never returns is a cell no child can be archived into."""
    evolver = RQEvolver(
        archive=MAPElitesArchive(),
        backend=_Judge(["GROUP: algebra\nSKILL: counting"]),
        evolution_config=EvolutionConfig(use_evaluator=True),
    )
    evolver.judge_tally = {k: 0 for k in (
        "reached", "agreed", "group_agreed", "skill_agreed",
        "label_mismatch", "failed_closed", "skill_none", "group_none")}
    evolver.judge_skill_counts = {}

    evolver._apply_judge([_entry(), _entry()])
    assert evolver.judge_skill_counts == {"counting": 2}
