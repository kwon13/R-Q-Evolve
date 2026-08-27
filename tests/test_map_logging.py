"""The complete 7 x 5 DOMAIN x PROBLEM_TYPE archive picture."""

import hashlib

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.concepts import DOMAINS, PROBLEM_TYPES
from rq_evolve.map_figure import occupied_cells, render_archive_figure
from rq_evolve.problem_type import (
    PROBLEM_TYPE_RULESET,
    problem_type_ruleset_sha256,
)
from rq_evolve.program import ProblemProgram


def _champion(
    domain: str, problem_type: str, rq: float, s_hat: float
) -> ProblemProgram:
    program = ProblemProgram(
        source_code=(
            "def generate(seed):\n"
            '    return f"Compute {seed} plus one.", str(seed + 1), '
            '{"mode": "expression"}\n\n'
            f'DOMAIN = "{domain}"\n'
        ),
        metadata={"problem_type": problem_type},
    )
    program.metadata["descriptor_contract"] = {
        "domain_authority": "source_exact_one_literal",
        "problem_type_authority": "deterministic_statement_and_verifier",
        "problem_type_ruleset": PROBLEM_TYPE_RULESET,
        "problem_type_ruleset_sha256": problem_type_ruleset_sha256(),
        "domain": domain,
        "problem_type": problem_type,
        "source_sha256": hashlib.sha256(
            program.source_code.encode("utf-8")
        ).hexdigest(),
    }
    program.s_hat = s_hat
    return program


def _filled(cells) -> MAPElitesArchive:
    archive = MAPElitesArchive()
    for domain, problem_type, rq, s_hat in cells:
        archive.try_insert(
            program=_champion(domain, problem_type, rq, s_hat),
            u_value=1.0,
            rq_score=rq,
        )
    return archive


def test_figure_covers_the_full_grid_without_a_supported_cell_mask():
    archive = _filled([("algebra", "function", 0.3, 0.5)])
    figure = render_archive_figure(archive, iteration=1, stats=archive.stats())
    assert figure is not None
    axis = figure.axes[0]
    assert [tick.get_text() for tick in axis.get_yticklabels()] == list(DOMAINS)
    assert [tick.get_text() for tick in axis.get_xticklabels()] == list(PROBLEM_TYPES)
    assert axis.get_ylabel() == "DOMAIN"
    assert axis.get_xlabel() == "PROBLEM TYPE"


def test_zero_scoring_champion_is_drawn_as_occupied():
    archive = _filled([("geometry", "decision", 0.0, 0.0)])
    assert occupied_cells(archive) == {
        (DOMAINS.index("geometry"), PROBLEM_TYPES.index("decision"))
    }
    assert render_archive_figure(archive, iteration=1) is not None


def test_newly_filled_cells_are_outlined():
    archive = _filled([("algebra", "counting", 0.2, 0.4)])
    figure = render_archive_figure(
        archive,
        iteration=1,
        new_cells=occupied_cells(archive),
    )
    assert len([patch for patch in figure.axes[0].patches if not patch.get_fill()]) == 1


def test_empty_archive_still_renders():
    assert render_archive_figure(MAPElitesArchive(), iteration=0) is not None
