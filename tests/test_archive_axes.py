"""The complete DOMAIN x PROBLEM_TYPE archive and its resume contract."""

import copy
import hashlib
import random

import pytest

from rq_evolve.archive import ArchiveSchemaError, MAPElitesArchive
from rq_evolve.concepts import DOMAINS, PROBLEM_TYPES
from rq_evolve.problem_type import PROBLEM_TYPE_RULESET, problem_type_ruleset_sha256
from rq_evolve.program import ProblemProgram


_SUBJECTS = (
    "lattice points inside a convex hull",
    "prime gaps below a bound",
    "leaves of a rooted binary tree",
    "coins stacked in decreasing piles",
    "tilings of a strip by dominoes",
    "chords crossing inside a circle",
    "roots arranged around the unit disk",
)
_ACTIONS = (
    "counted after a single shuffle",
    "measured along the main diagonal",
    "summed over every odd index",
    "compared against their mirror images",
    "grouped by residue modulo three",
)


def _distinct_tags(count: int) -> list[str]:
    assert count <= len(_SUBJECTS) * len(_ACTIONS)
    return [
        f"{_SUBJECTS[i % len(_SUBJECTS)]} " f"{_ACTIONS[i // len(_SUBJECTS)]}"
        for i in range(count)
    ]


def _certify(
    program: ProblemProgram, domain: str, problem_type: str
) -> ProblemProgram:
    """Emulate the source-bound result of deterministic verification."""
    program.metadata["problem_type"] = problem_type
    program.metadata["descriptor_contract"] = {
        "domain_authority": "source_exact_one_literal",
        "problem_type_authority": "deterministic_statement_and_verifier",
        "problem_type_ruleset": PROBLEM_TYPE_RULESET,
        "problem_type_ruleset_sha256": problem_type_ruleset_sha256(),
        "verified_seeds": 5,
        "domain": domain,
        "problem_type": problem_type,
        "source_sha256": hashlib.sha256(
            program.source_code.encode("utf-8")
        ).hexdigest(),
    }
    return program


def _program(
    domain: str,
    problem_type: str,
    value: int = 1,
    tag: str = "",
) -> ProblemProgram:
    phrase = tag or f"{domain} studied as a {problem_type} problem"
    program = ProblemProgram(
        source_code=f"""
def generate(seed):
    problem = f"{phrase}, then add {{seed}} to {value}."
    return problem, str({value} + seed)


DOMAIN = "{domain}"
"""
    )
    return _certify(program, domain, problem_type)


def test_grid_is_the_unmasked_seven_by_five_product():
    archive = MAPElitesArchive()
    assert archive.n_domain_bins == len(DOMAINS) == 7
    assert archive.n_problem_type_bins == len(PROBLEM_TYPES) == 5
    assert set(archive.grid) == {
        (domain_bin, type_bin) for domain_bin in range(7) for type_bin in range(5)
    }
    assert len(archive.grid) == 35
    assert not hasattr(archive, "sample_target_cell")


def test_cell_is_a_pure_function_of_the_post_hoc_descriptors():
    archive = MAPElitesArchive()
    program = _program("geometry", "search")
    cell = archive.program_to_cell(program)
    assert cell == (DOMAINS.index("geometry"), PROBLEM_TYPES.index("search"))
    assert archive.cell_labels(cell) == ("geometry", "search")


def test_structural_inspiration_selection_is_descriptor_blind():
    archive = MAPElitesArchive()
    parent = _program("algebra", "function", tag="primary recurrence family")
    same_domain = _program(
        "algebra", "counting", tag="finite coefficient counting family"
    )
    other_domain = _program(
        "geometry", "search", tag="coordinate intersection witness family"
    )
    for program, root in (
        (parent, "root-a"),
        (same_domain, "root-b"),
        (other_domain, "root-c"),
    ):
        program.metadata["lineage_root_id"] = root
        cell = archive.program_to_cell(program)
        assert cell is not None
        archive.grid[cell].champion = program

    selection = archive.sample_structural_inspiration(parent, rng=random.Random(4))
    assert selection.donor in (same_domain, other_domain)
    assert selection.provenance["eligible_count"] == 2
    assert selection.provenance["selection_pool_size"] == 2
    assert selection.provenance["selection_tier"] == "cross_lineage_uniform"
    assert "domain" in selection.provenance
    assert "problem_type" in selection.provenance
    assert "group" not in selection.provenance
    assert "skill" not in selection.provenance


def test_fitness_never_changes_a_programs_cell():
    archive = MAPElitesArchive()
    low = _program("algebra", "function", value=1, tag="first relation")
    high = _program("algebra", "function", value=2, tag="second relation")
    assert archive.program_to_cell(low) == archive.program_to_cell(high)

    assert archive.try_insert(low, u_value=0.05, rq_score=0.1)
    assert archive.try_insert(high, u_value=5.0, rq_score=0.2)
    assert [p.program_id for p in archive.champions()] == [high.program_id]
    assert high.u_score == pytest.approx(5.0)


def test_missing_or_out_of_vocabulary_descriptors_are_rejected():
    archive = MAPElitesArchive()
    missing = ProblemProgram(
        source_code='def generate(seed):\n    return "q", str(seed)\n'
    )
    invalid_domain = _program("topology", "counting")
    invalid_type = _program("algebra", "proof")

    for program in (missing, invalid_domain, invalid_type):
        assert archive.program_to_cell(program) is None
        assert archive.placement_cell(program) is None
        assert not archive.try_insert(program, u_value=1.0, rq_score=0.5)
        assert program.metadata["archive_status"] == "unlabelled_rejected"
    assert archive.champions() == []


def test_missing_stale_or_ambiguous_descriptor_contract_is_rejected():
    archive = MAPElitesArchive()

    missing = _program("algebra", "function", tag="missing provenance")
    missing.metadata.pop("descriptor_contract")
    # A source declaration must not revive the retired type-authority path.
    missing.source_code += '\nPROBLEM_TYPE = "function"\n'

    stale_rules = _program("geometry", "search", tag="stale rules")
    stale_rules.metadata["descriptor_contract"][
        "problem_type_ruleset_sha256"
    ] = "0" * 64

    stale_source = _program("number_theory", "counting", tag="stale source")
    stale_source.metadata["descriptor_contract"]["source_sha256"] = "f" * 64

    base = _program("calculus", "function", tag="ambiguous domain")
    ambiguous = ProblemProgram(
        source_code=base.source_code + '\nDOMAIN = "geometry"\n'
    )
    _certify(ambiguous, "calculus", "function")

    for program in (missing, stale_rules, stale_source, ambiguous):
        assert archive.program_to_cell(program) is None
        assert not archive.try_insert(program, u_value=1.0, rq_score=0.5)
        assert program.metadata["archive_status"] == "unlabelled_rejected"
    assert archive.champions() == []


def test_stats_report_both_axis_coverages_over_all_35_cells():
    archive = MAPElitesArchive()
    rows = (
        ("algebra", "decision"),
        ("algebra", "counting"),
        ("geometry", "function"),
    )
    tags = _distinct_tags(3)
    for i, (domain, problem_type) in enumerate(rows):
        assert archive.try_insert(
            _program(domain, problem_type, value=i + 1, tag=tags[i]),
            u_value=1.0,
            rq_score=0.1 * (i + 1),
        )
    stats = archive.stats()
    assert stats["num_champions"] == 3
    assert stats["total_niches"] == 35
    assert stats["coverage"] == pytest.approx(3 / 35)
    assert stats["domain_coverage"] == pytest.approx(2 / 7)
    assert stats["problem_type_coverage"] == pytest.approx(3 / 5)
    assert "group_coverage" not in stats
    assert "skill_coverage" not in stats


def test_snapshot_round_trip_preserves_schema_and_cells():
    archive = MAPElitesArchive()
    rows = (("algebra", "counting"), ("geometry", "search"))
    tags = _distinct_tags(2)
    for i, (domain, problem_type) in enumerate(rows):
        assert archive.try_insert(
            _program(domain, problem_type, value=i + 3, tag=tags[i]),
            u_value=1.0,
            rq_score=0.5,
        )

    payload = archive.to_payload()
    assert payload["meta"]["axes"] == ["domain", "problem_type"]
    assert payload["meta"]["domain_labels"] == list(DOMAINS)
    assert payload["meta"]["problem_type_labels"] == list(PROBLEM_TYPES)
    assert payload["meta"]["schema_version"] == 2
    assert payload["meta"]["problem_type_ruleset"] == PROBLEM_TYPE_RULESET
    assert (
        payload["meta"]["problem_type_ruleset_sha256"]
        == problem_type_ruleset_sha256()
    )
    assert "supported_cells" not in payload["meta"]
    assert payload["niches"]
    assert all("domain_bin" in row for row in payload["niches"])
    assert all("problem_type_bin" in row for row in payload["niches"])
    assert all("group_bin" not in row for row in payload["niches"])
    assert all("skill_bin" not in row for row in payload["niches"])

    restored = MAPElitesArchive()
    assert restored.load_payload(payload) == 2
    assert {archive.program_to_cell(p) for p in archive.champions()} == {
        restored.program_to_cell(p) for p in restored.champions()
    }


@pytest.mark.parametrize(
    "meta_patch",
    [
        {"schema": "old-schema"},
        {"schema_version": 999},
        {"axes": ["group", "skill"]},
        {"domain_labels": ["geometry"]},
        {"problem_type_labels": ["function"]},
    ],
)
def test_schema_or_vocabulary_mismatch_fails_closed(meta_patch):
    live = MAPElitesArchive()
    incumbent = _program("algebra", "function", tag="existing live question")
    assert live.try_insert(incumbent, u_value=1.0, rq_score=0.4)

    payload = live.to_payload()
    payload["meta"].update(meta_patch)
    before = [p.program_id for p in live.champions()]
    with pytest.raises(ArchiveSchemaError, match="incompatible archive"):
        live.load_payload(payload)
    assert [p.program_id for p in live.champions()] == before


def test_the_group_skill_archive_cannot_be_resumed():
    legacy = {
        "meta": {
            "axes": ["group", "skill"],
            "group_labels": ["algebra"],
            "skill_labels": ["counting"],
        },
        "champions": [],
    }
    with pytest.raises(ArchiveSchemaError, match="incompatible archive"):
        MAPElitesArchive().load_payload(legacy)


def test_current_metadata_cannot_smuggle_an_old_labelled_champion():
    payload = MAPElitesArchive().to_payload()
    payload["champions"] = [
        {
            "source_code": (
                'def generate(seed):\n    return "q", str(seed)\n\n'
                'GROUP = "algebra"\nSKILL = "counting"\n'
            ),
            "program_id": "legacy-program",
        }
    ]
    with pytest.raises(ArchiveSchemaError, match="no valid DOMAIN/PROBLEM_TYPE"):
        MAPElitesArchive().load_payload(payload)


def test_malformed_donor_certification_fails_before_live_archive_is_cleared():
    live = MAPElitesArchive()
    incumbent = _program("algebra", "function", tag="live incumbent")
    incumbent.metadata["structural_donor_certification"] = {
        "passed": True,
        "source": "test",
    }
    assert live.try_insert(incumbent, u_value=1.0, rq_score=0.4)

    payload = copy.deepcopy(live.to_payload())
    assert payload["structural_donors"]
    payload["structural_donors"][0]["metadata"][
        "structural_donor_certification"
    ] = "not-a-mapping"
    before = [program.program_id for program in live.champions()]

    with pytest.raises(ArchiveSchemaError, match="donor certification"):
        live.load_payload(payload)
    assert [program.program_id for program in live.champions()] == before


def test_flat_binning_still_uses_exactly_35_slots():
    archive = MAPElitesArchive(binning="flat")
    tags = _distinct_tags(35)
    for i, tag in enumerate(tags):
        assert archive.try_insert(
            _program("algebra", "function", value=i + 1, tag=tag),
            u_value=1.0,
            rq_score=float(i + 1),
        )
    assert len(archive.champions()) == 35

    assert archive.try_insert(
        _program("algebra", "function", value=99, tag="strong replacement"),
        u_value=1.0,
        rq_score=10_000.0,
    )
    assert len(archive.champions()) == 35


def test_flat_snapshot_cannot_resume_as_grid_or_the_reverse():
    payload = MAPElitesArchive(binning="flat").to_payload()
    with pytest.raises(ArchiveSchemaError, match="binning"):
        MAPElitesArchive(binning="grid").load_payload(payload)


def test_archive_binning_only_accepts_the_two_modes():
    with pytest.raises(ValueError, match="binning"):
        MAPElitesArchive(binning="bogus")
