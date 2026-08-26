"""Contracts for the manually certified structural-donor v2 arm."""

import random

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.config import load_config, load_raw_config
from rq_evolve.evolution import RQEvolver
from rq_evolve.program import ProblemProgram


def _program(
    statement: str,
    group: str,
    skill: str,
    *,
    root: str,
    certified: bool,
    rq: float,
) -> ProblemProgram:
    source = f"""
import random

def generate(seed):
    rng = random.Random(seed)
    n = rng.randint(10, 40)
    answer = n + 1
    check = 1 + n
    assert answer == check
    problem = f"{statement} Let n = {{n}}. Determine n plus one."
    return problem, str(answer)

GROUP = "{group}"
SKILL = "{skill}"
"""
    metadata = {"lineage_root_id": root}
    if certified:
        metadata["structural_donor_certification"] = {
            "passed": True,
            "source": "test",
        }
    return ProblemProgram(source_code=source, rq_score=rq, metadata=metadata)


def _place(archive: MAPElitesArchive, *programs: ProblemProgram) -> None:
    for program in programs:
        cell = archive.program_to_cell(program)
        assert cell is not None
        archive.grid[cell].champion = program
        certified = program.metadata.get("structural_donor_certification", {}).get(
            "passed"
        )
        if certified:
            archive.structural_donors[program.program_id] = program


def test_v2_config_uses_manual_donors_without_any_evaluator():
    cfg = load_config(
        "configs/rq_evolve_4b_4gpu_structural_inspiration_v2.yaml"
    ).evolution
    raw = load_raw_config("configs/rq_evolve_4b_4gpu_structural_inspiration_v2.yaml")

    assert cfg.structural_inspiration is True
    assert cfg.use_evaluator is False
    assert cfg.verify_seeds == 32
    assert cfg.structural_inspiration_require_certified_donor is True
    assert cfg.structural_inspiration_require_positive_rq is True
    assert cfg.structural_inspiration_max_token_jaccard == 0.45
    assert len(cfg.manual_certified_seed_files) == 6
    assert raw.verl_config.trainer.default_local_dir.endswith(
        "rq_evolve_4b_4gpu_certified_structural_inspiration_v2"
    )


def test_uncertified_children_can_enter_map_but_not_donor_pool():
    archive = MAPElitesArchive()
    program = _program(
        "A finite algebra problem.",
        "algebra",
        "transformation",
        root="root-a",
        certified=False,
        rq=0.2,
    )

    assert archive.try_insert(program, u_value=0.3, rq_score=0.2) is True


def test_donor_pool_requires_both_certificate_and_positive_current_rq():
    archive = MAPElitesArchive()
    parent = _program(
        "Primary family.", "algebra", "counting", root="parent", certified=True, rq=0.2
    )
    good = _program(
        "Certified positive donor.",
        "sequence",
        "contradiction",
        root="good",
        certified=True,
        rq=0.1,
    )
    zero = _program(
        "Certified but zero donor.",
        "geometry",
        "invariant",
        root="zero",
        certified=True,
        rq=0.0,
    )
    uncertified = _program(
        "Positive but uncertified donor.",
        "number_theory",
        "casework",
        root="uncertified",
        certified=False,
        rq=0.3,
    )
    _place(archive, parent, good, zero, uncertified)

    picked = archive.sample_structural_inspiration(
        parent,
        rng=random.Random(0),
        require_certified=True,
        require_positive_rq=True,
    )

    assert picked.donor is good
    assert uncertified.program_id not in archive.structural_donors
    assert picked.provenance["nonpositive_rq_donor_count"] == 1
    assert picked.provenance["donor_rq_score"] == 0.1
    assert picked.provenance["donor_certification_source"] == "test"


def test_manual_donor_survives_map_eviction_and_snapshot_roundtrip():
    archive = MAPElitesArchive()
    parent = _program(
        "Primary family.", "algebra", "counting", root="parent", certified=False, rq=0.2
    )
    donor = _program(
        "Persistent reviewed seed.",
        "sequence",
        "contradiction",
        root="seed",
        certified=True,
        rq=0.1,
    )
    replacement = _program(
        "Later generated champion.",
        "sequence",
        "contradiction",
        root="child",
        certified=False,
        rq=0.9,
    )
    _place(archive, parent, donor)
    archive.grid[archive.program_to_cell(donor)].champion = replacement

    restored = MAPElitesArchive()
    restored.load_payload(archive.to_payload())
    restored_parent = next(
        p for p in restored.champions() if p.program_id == parent.program_id
    )
    picked = restored.sample_structural_inspiration(
        restored_parent,
        rng=random.Random(0),
        require_certified=True,
        require_positive_rq=True,
    )

    assert picked.donor is not None
    assert picked.donor.program_id == donor.program_id


def test_token_jaccard_rejects_a_donor_restatement():
    archive = MAPElitesArchive()
    donor = _program(
        "For each ordered pair of positive integers call the pair feasible when real numbers satisfy the sum and product constraints.",
        "inequality",
        "counting",
        root="donor",
        certified=True,
        rq=0.1,
    )
    child = _program(
        "For every ordered pair of positive integers call it feasible if real numbers satisfy the product and sum constraints.",
        "algebra",
        "transformation",
        root="child",
        certified=True,
        rq=0.1,
    )

    verdict = archive.compare_with_structural_inspiration(
        child, donor, max_token_jaccard=0.45
    )

    assert verdict["token_jaccard"] >= 0.45
    assert verdict["rejected"] is True
    assert verdict["reason"] == "donor_token_jaccard"


def test_only_allowlisted_seeds_receive_structural_donor_certification():
    config = load_config(
        "configs/rq_evolve_4b_4gpu_structural_inspiration_v2.yaml"
    ).evolution
    evolver = RQEvolver(
        archive=MAPElitesArchive(), backend=None, evolution_config=config
    )

    seeds = evolver.load_seed_programs("seed_programs")

    certified = [
        p
        for p in seeds
        if p.metadata.get("structural_donor_certification", {}).get("passed")
    ]
    assert len(certified) == 6
    assert all(
        p.metadata["structural_donor_certification"]["source"]
        == "manual_seed_allowlist"
        for p in certified
    )
