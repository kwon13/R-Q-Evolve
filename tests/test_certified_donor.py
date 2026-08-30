"""Contracts for the manually certified structural-donor v2 arm."""

import hashlib
import random

import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.config import EvolutionConfig, load_config, load_raw_config
from rq_evolve.evolution import RQEvolver
from rq_evolve.problem_type import PROBLEM_TYPE_RULESET, problem_type_ruleset_sha256
from rq_evolve.program import ProblemProgram


def _program(
    statement: str,
    domain: str,
    problem_type: str,
    *,
    root: str,
    certified: bool,
    rq: float,
) -> ProblemProgram:
    request = {
        "decision": (
            'answer = "Yes" if n + 1 > n else "No"\n'
            '    check = "Yes"\n'
            '    problem = f"{prefix} Let n = {n}. Is n + 1 greater than n? '
            'Answer Yes or No."\n'
            '    verifier = {"mode": "boolean"}'
        ),
        "search": (
            'values = [n, n + 1]\n'
            '    answer = r"\\{" + ",".join(str(x) for x in values) + r"\\}"\n'
            '    checked_values = [x for x in range(n - 1, n + 3) if n <= x <= n + 1]\n'
            '    check = r"\\{" + ",".join(str(x) for x in checked_values) + r"\\}"\n'
            '    problem = f"{prefix} Let n = {n}. Find all integers x such that '
            'n <= x <= n + 1."\n'
            '    verifier = {"mode": "set", "elements": [str(x) for x in values]}'
        ),
        "counting": (
            'answer = n + 1\n'
            '    check = sum(1 for x in range(n + 1) if 0 <= x <= n)\n'
            '    problem = f"{prefix} Let n = {n}. How many integers x satisfy '
            '0 <= x <= n?"\n'
            '    verifier = {"mode": "expression"}'
        ),
        "optimization": (
            'answer = n + 1\n'
            '    check = max(range(n + 2))\n'
            '    problem = f"{prefix} Let n = {n}. Find the maximum value of an '
            'integer x satisfying 0 <= x <= n + 1."\n'
            '    verifier = {"mode": "expression"}'
        ),
        "function": (
            'answer = n + 1\n'
            '    check = sum((n, 1))\n'
            '    problem = f"{prefix} Let n = {n}. Compute n + 1."\n'
            '    verifier = {"mode": "expression"}'
        ),
    }[problem_type]
    source = f"""
import random

def generate(seed):
    rng = random.Random(seed)
    n = rng.randint(10, 40)
    prefix = {statement!r}
    {request}
    assert answer == check
    return problem, str(answer), verifier

DOMAIN = "{domain}"
"""
    metadata = {
        "lineage_root_id": root,
        "problem_type": problem_type,
        "descriptor_contract": {
            "domain_authority": "source_exact_one_literal",
            "problem_type_authority": "deterministic_statement_and_verifier",
            "problem_type_ruleset": PROBLEM_TYPE_RULESET,
            "problem_type_ruleset_sha256": problem_type_ruleset_sha256(),
            "verified_seeds": 5,
            "domain": domain,
            "problem_type": problem_type,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        },
    }
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


def test_legacy_v2_config_is_rejected_instead_of_becoming_new_production():
    with pytest.raises(ValueError, match="relabel_skill is retired"):
        load_config("configs/rq_evolve_4b_4gpu_structural_inspiration_v2.yaml")

    fresh = load_config("configs/rq_evolve_4b_8gpu_domain_type.yaml")
    assert not hasattr(fresh.evolution, "use_evaluator")
    assert not hasattr(fresh.evolution, "evaluator_provider")
    assert fresh.evolution.target_cell_injection is False
    assert fresh.evolution.relabel_skill is False
    assert fresh.evolution.independent_domain_labeling is True
    assert fresh.archive.require_domain_labeling is True
    assert fresh.evolution.manual_certified_seed_files == ()
    raw_fresh = load_raw_config("configs/rq_evolve_4b_8gpu_domain_type.yaml")
    assert raw_fresh.verl_config.trainer.resume_mode == "disable"


@pytest.mark.parametrize(
    ("path", "gpu_count", "output_suffix"),
    [
        (
            "configs/rq_evolve_4b_4gpu_domain_type.yaml",
            4,
            "rq_evolve_4b_domain_type_35cell_4gpu",
        ),
        (
            "configs/rq_evolve_4b_8gpu_domain_type.yaml",
            8,
            "rq_evolve_4b_domain_type_35cell_8gpu",
        ),
    ],
)
def test_domain_type_production_configs_share_descriptor_contract(
    path, gpu_count, output_suffix
):
    cfg = load_config(path)
    raw = load_raw_config(path)

    assert cfg.evolution.seed_programs_dir == "seed_programs_domain_type"
    assert cfg.evolution.two_stage_mutation is True
    assert cfg.evolution.target_cell_injection is False
    assert cfg.evolution.relabel_skill is False
    assert cfg.evolution.independent_domain_labeling is True
    assert cfg.archive.require_domain_labeling is True
    assert cfg.evolution.structural_inspiration is False
    assert cfg.training_data.training_budget == 32
    assert raw.verl_config.trainer.n_gpus_per_node == gpu_count
    assert raw.verl_config.trainer.save_freq == 32
    assert raw.verl_config.trainer.max_actor_ckpt_to_keep is None
    assert raw.verl_config.trainer.resume_mode == "disable"
    assert str(raw.verl_config.trainer.default_local_dir).endswith(output_suffix)


@pytest.mark.parametrize(
    ("path", "fitness_mode", "output_suffix"),
    [
        (
            "configs/rq_evolve_4b_8gpu_domain_type_reverse_u.yaml",
            "reverse_u",
            "rq_evolve_4b_domain_type_reverse_u_35cell_8gpu",
        ),
        (
            "configs/rq_evolve_4b_8gpu_domain_type_no_u.yaml",
            "no_u",
            "rq_evolve_4b_domain_type_no_u_35cell_8gpu",
        ),
    ],
)
def test_domain_type_fitness_ablation_configs_change_only_the_named_score_arm(
    path, fitness_mode, output_suffix
):
    baseline = load_config("configs/rq_evolve_4b_8gpu_domain_type.yaml")
    cfg = load_config(path)
    raw = load_raw_config(path)

    assert baseline.evolution.rq_fitness_mode == "standard"
    assert cfg.evolution.rq_fitness_mode == fitness_mode
    assert cfg.evolution.rq_reverse_u_constant == pytest.approx(2.0)
    assert cfg.evolution.seed_programs_dir == baseline.evolution.seed_programs_dir
    assert cfg.evolution.group_size == baseline.evolution.group_size
    assert cfg.evolution.inner_iterations == baseline.evolution.inner_iterations
    assert cfg.training_data == baseline.training_data
    assert raw.verl_config.trainer.n_gpus_per_node == 8
    assert raw.verl_config.trainer.max_actor_ckpt_to_keep is None
    assert str(raw.verl_config.trainer.default_local_dir).endswith(output_suffix)
    assert raw.verl_config.trainer.resume_mode == "disable"


def test_uncertified_children_can_enter_map_but_not_donor_pool():
    archive = MAPElitesArchive()
    program = _program(
        "A finite algebra problem.",
        "algebra",
        "function",
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
        "discrete_mathematics",
        "decision",
        root="good",
        certified=True,
        rq=0.1,
    )
    zero = _program(
        "Certified but zero donor.",
        "geometry",
        "function",
        root="zero",
        certified=True,
        rq=0.0,
    )
    uncertified = _program(
        "Positive but uncertified donor.",
        "number_theory",
        "search",
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
        "discrete_mathematics",
        "decision",
        root="seed",
        certified=True,
        rq=0.1,
    )
    replacement = _program(
        "Later generated champion.",
        "discrete_mathematics",
        "decision",
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
        "applied_mathematics",
        "counting",
        root="donor",
        certified=True,
        rq=0.1,
    )
    child = _program(
        "For every ordered pair of positive integers call it feasible if real numbers satisfy the product and sum constraints.",
        "algebra",
        "function",
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


def test_only_allowlisted_fresh_seeds_receive_structural_donor_certification(
    tmp_path,
):
    allowed = _program(
        "Allowed seed.",
        "calculus",
        "function",
        root="allowed",
        certified=False,
        rq=0.0,
    )
    ordinary = _program(
        "Ordinary seed.",
        "precalculus",
        "optimization",
        root="ordinary",
        certified=False,
        rq=0.0,
    )
    (tmp_path / "allowed.py").write_text(allowed.source_code, encoding="utf-8")
    (tmp_path / "ordinary.py").write_text(ordinary.source_code, encoding="utf-8")
    config = EvolutionConfig(manual_certified_seed_files=("allowed.py",))
    evolver = RQEvolver(
        archive=MAPElitesArchive(), backend=None, evolution_config=config
    )

    seeds = evolver.load_seed_programs(tmp_path)

    certified = [
        p
        for p in seeds
        if p.metadata.get("structural_donor_certification", {}).get("passed")
    ]
    assert len(seeds) == 2
    assert [p.declared_domain() for p in certified] == ["calculus"]
    assert len(certified) == 1
    assert all(
        p.metadata["structural_donor_certification"]["source"]
        == "manual_seed_allowlist"
        for p in certified
    )
