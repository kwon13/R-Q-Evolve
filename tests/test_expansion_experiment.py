from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from rq_evolve.expansion_experiment import (
    audit_generator_pair_design,
    audit_generation_sufficiency,
    numeric_summary,
    operator_contract_valid,
    prepare_comparison_manifests,
    read_jsonl,
    representation_input_row,
    select_child_instances,
    validate_heldout_rows,
)
from rq_evolve.expansion_capability import (
    HeldoutSchemaError,
    audit_paired_evaluation_rows,
    audit_paired_training_budgets,
    evaluate_checkpoint_vllm,
    validate_heldout_rows as validate_capability_heldout_rows,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_expansion_script():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_expansion_hypothesis.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_expansion_hypothesis_merge_test",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _comparison_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "comparison"
    root.mkdir(parents=True)
    parent_program = tmp_path / "parent.py"
    parent_program.write_text(
        "\n".join(
            [
                "def generate(seed):",
                "    return f'Compute {seed} + 1.', str(seed + 1)",
                'CONCEPT_GROUP = "algebra"',
                'CONCEPT_TYPE = "algebra.linear_sum"',
            ]
        ),
        encoding="utf-8",
    )
    parent_seed = {
        "status": "ok",
        "seed": 0,
        "problem": "Compute 0 + 1.",
        "answer": "1",
        "num_rollouts": 2,
        "num_correct": 1,
        "p_hat": 0.5,
        "uncertainty_proxy": 0.2,
        "rq_proxy": 0.05,
    }
    parent = {
        "status": "ok",
        "per_seed": [parent_seed],
        "p_hat": 0.5,
        "rq_proxy": 0.05,
    }
    _write_json(root / "parent_scores.json", parent)
    _write_json(root / "parent_rollouts.json", {**parent, "per_seed": []})

    child_seed = {
        "status": "ok",
        "seed": 0,
        "problem": "Count 3 selected objects.",
        "answer": "3",
        "num_rollouts": 2,
        "num_correct": 1,
        "p_hat": 0.5,
        "uncertainty_proxy": 0.4,
        "rq_proxy": 0.1,
    }
    evaluator = {
        "seed": 0,
        "problem": child_seed["problem"],
        "answer": "3",
        "valid": True,
        "reason": "valid",
    }
    plain = {
        "operator": "in_breadth",
        "method": "legacy",
        "method_dir": "in_breadth/legacy",
        "status": "ok",
        "program_id": "plain-program",
        "concept_group": "combinatorics",
        "concept_type": "combinatorics.count",
        "generated_instances": [
            {"seed": 0, "problem": child_seed["problem"], "answer": "3"}
        ],
        "seed_scores": [child_seed],
        "evaluator_by_seed": [evaluator],
        "evaluator_valid": True,
        "evaluator_passed": 1,
        "evaluator_total": 1,
    }
    reasoning = {
        **plain,
        "method": "metacognitive",
        "method_dir": "in_breadth/metacognitive",
        "program_id": "reasoning-program",
        "concept_group": "algebra",
        "concept_type": "algebra.linear_sum",
        "plan_id": "plan",
        "plan": {"target_reasoning_move": "check a boundary case"},
    }
    invalid = {
        "operator": "in_depth",
        "method": "legacy",
        "method_dir": "in_depth/legacy",
        "status": "invalid_code",
        "reason": "bad code",
    }
    _write_json(root / "summaries.json", [plain, reasoning, invalid])
    return root, parent_program


def _merge_sources(tmp_path: Path) -> tuple[object, list[Path]]:
    module = _load_expansion_script()
    comparison, parent = _comparison_fixture(tmp_path)
    config_bytes = b'{"analysis_status":"preregistered"}\n'
    import hashlib

    config_hash = hashlib.sha256(config_bytes).hexdigest()
    solver_hash = "a" * 64
    prompt_hash = "b" * 64
    source_dirs: list[Path] = []
    for draw in (7, 8):
        source = tmp_path / f"normalized-{draw}"
        manifest = prepare_comparison_manifests(
            comparison,
            output_dir=source,
            parent_program=parent,
            run_id="run-shared",
            generator_draw_idx=draw,
            solver_checkpoint_id="solver",
            solver_checkpoint_sha256=solver_hash,
            generator_checkpoint_id="generator",
        )
        source_hash = f"{draw:064x}"
        paired_seeds = {
            "in_depth": {"plan": draw * 10 + 1, "code": draw * 10 + 2},
            "in_breadth": {"plan": draw * 10 + 3, "code": draw * 10 + 4},
        }
        manifest.update(
            {
                "preregistered_config_sha256": config_hash,
                "preregistered_config_path": str(
                    (source / "preregistered_config.json").resolve()
                ),
                "source_run_manifest_sha256": source_hash,
                "source_comparison_design": "two_stage_plain_vs_reasoning_v1",
                "source_evidence_gate_valid": True,
                "source_evidence_gate_issues": [],
                "source_evidence_gate_version": "clean_live_parent_v1",
                "source_prompt_files": [
                    {
                        "path": "/repo/prompt_templates/metacognitive_plan.txt",
                        "sha256": prompt_hash,
                    }
                ],
                "source_sampling": {
                    "plan_temperature": 0.3,
                    "plan_top_p": 0.95,
                    "code_temperature": 0.2,
                    "code_top_p": 0.95,
                    "evaluation_seeds": [0],
                },
                "source_sampling_seed_policy": (
                    "recorded_vllm_engine_seed_per_run"
                ),
                "source_paired_request_seed_policy": (
                    "sha256_llm_seed_operator_stage_v1"
                ),
                "source_paired_request_seeds": paired_seeds,
                "source_instance_seed_policy": (
                    "explicit_evaluation_seed_list"
                ),
                "source_llm_seed": draw,
                "source_llm_seed_range": [draw, draw],
                "source_code_contract_version": 4,
                "source_code_contract": (
                    "single_answer_target_move_necessity"
                ),
            }
        )
        (source / "preregistered_config.json").write_bytes(config_bytes)
        generators = read_jsonl(source / "generator_manifest.jsonl")
        for row in generators:
            row["source_run_manifest_sha256"] = source_hash
            row["llm_seed"] = draw
            row["generator_sampling_seed_range"] = [draw, draw]
            row["sampling_seed_policy"] = (
                "recorded_vllm_engine_seed_per_run"
            )
            row["instance_seed_policy"] = "explicit_evaluation_seed_list"
            if row.get("condition") in {"plain", "reasoning"}:
                row["paired_request_seed_policy"] = (
                    "sha256_llm_seed_operator_stage_v1"
                )
                seeds = paired_seeds[row["operator"]]
                row["plan_sampling_seed"] = seeds["plan"]
                row["code_sampling_seed"] = seeds["code"]
        _write_jsonl(source / "generator_manifest.jsonl", generators)
        parent_instance = next(
            row
            for row in read_jsonl(source / "instance_manifest.jsonl")
            if row["role"] == "parent"
        )
        if draw == 8:
            # Parent-score telemetry is a repeated Solver measurement, so it
            # may vary across generator draws without changing parent identity.
            instance_rows = read_jsonl(source / "instance_manifest.jsonl")
            varying_parent = next(
                row for row in instance_rows if row["role"] == "parent"
            )
            varying_parent.update(
                {
                    "num_correct": 0,
                    "p_hat": 0.0,
                    "uncertainty_proxy": 0.4,
                    "rq": 0.0,
                }
            )
            _write_jsonl(source / "instance_manifest.jsonl", instance_rows)
        _write_jsonl(
            source / "rollout_manifest.jsonl",
            [
                {
                    "schema_version": 1,
                    "instance_id": parent_instance["item_id"],
                    "generator_unit_id": parent_instance[
                        "generator_unit_id"
                    ],
                    "condition": "parent",
                    "instance_seed": 0,
                    "rollout_idx": 0,
                    "solver_sampling_seed": draw,
                    "response": f"draw {draw}",
                    "cleaned_response": f"draw {draw}",
                    "predicted_answer": "1" if draw == 7 else "0",
                    "correct": draw == 7,
                    "solver_checkpoint_id": "solver",
                    "source_run_manifest_sha256": source_hash,
                    "source_path": f"/source/{draw}/parent_rollouts.json",
                }
            ],
        )
        _write_json(source / "experiment_manifest.json", manifest)
        source_dirs.append(source)
    return module, source_dirs


def test_prepare_preserves_invalid_and_blocks_bad_operator_pair(tmp_path: Path) -> None:
    root, parent = _comparison_fixture(tmp_path)
    output = tmp_path / "normalized"
    manifest = prepare_comparison_manifests(
        root,
        output_dir=output,
        parent_program=parent,
        run_id="run-1",
        solver_checkpoint_sha256="1" * 64,
    )

    generators = read_jsonl(output / "generator_manifest.jsonl")
    instances = read_jsonl(output / "instance_manifest.jsonl")
    rep_inputs = read_jsonl(output / "representation_inputs.jsonl")

    assert len(generators) == 4  # parent plus all three attempted generators
    assert any(row["overall_status"] == "invalid_code" for row in generators)
    reasoning = next(row for row in generators if row["condition"] == "reasoning")
    assert reasoning["operator_contract_valid"] is False
    assert manifest["sufficiency"]["confirmatory_ready"] is False
    assert manifest["sufficiency"]["strict_matched_generator_pairs"] == 0
    assert manifest["solver_checkpoint_sha256"] == "1" * 64
    assert all(
        row["solver_checkpoint_sha256"] == "1" * 64
        for row in generators
    )

    assert {row["role"] for row in instances} == {"parent", "child"}
    forbidden = {
        "answer",
        "target_reasoning_move",
        "evaluator_reason",
        "rq",
        "source_path",
    }
    assert all(forbidden.isdisjoint(row) for row in rep_inputs)


def test_merge_dedupes_identical_parent_rows_and_preserves_draw_lineage(
    tmp_path: Path,
) -> None:
    module, source_dirs = _merge_sources(tmp_path)
    output = tmp_path / "merged"

    assert module.cmd_merge(
        Namespace(
            inputs=[str(path) for path in source_dirs],
            output_dir=str(output),
        )
    ) == 0

    generators = read_jsonl(output / "generator_manifest.jsonl")
    instances = read_jsonl(output / "instance_manifest.jsonl")
    representations = read_jsonl(output / "representation_inputs.jsonl")
    rollouts = read_jsonl(output / "rollout_manifest.jsonl")
    manifest = json.loads(
        (output / "experiment_manifest.json").read_text(encoding="utf-8")
    )

    assert sum(row["condition"] == "parent" for row in generators) == 1
    assert sum(row["role"] == "parent" for row in instances) == 1
    assert sum(row["role"] == "parent" for row in representations) == 1
    merged_parent = next(row for row in instances if row["role"] == "parent")
    assert merged_parent["num_rollouts"] == 4
    assert merged_parent["num_correct"] == 1
    assert merged_parent["p_hat"] == pytest.approx(0.25)
    assert merged_parent["uncertainty_proxy"] == pytest.approx(0.3)
    assert merged_parent["rq"] == pytest.approx(0.25 * 0.75 * 0.3)
    assert merged_parent["lineage_source"] == "merged_parent_rollout_pool_v1"
    child_ids = [
        row["generator_unit_id"]
        for row in generators
        if row["condition"] in {"plain", "reasoning"}
    ]
    assert len(child_ids) == len(set(child_ids))
    assert len(rollouts) == 2
    assert len(
        {row["source_run_manifest_sha256"] for row in rollouts}
    ) == 2
    assert manifest["source_comparison_design"] == (
        "two_stage_plain_vs_reasoning_v1"
    )
    assert manifest["source_evidence_gate_valid"] is True
    assert manifest["source_evidence_gate_version"] == "clean_live_parent_v1"
    assert len(manifest["source_run_manifest_sha256s"]) == 2
    assert len(manifest["sampling_provenance"]["source_draws"]) == 2
    assert [
        row["parent_score_telemetry"][0]["p_hat"]
        for row in manifest["source_manifests"]
    ] == [0.5, 0.0]
    assert manifest["deduplication"] == {
        "child_rows_removed": 0,
        "identical_parent_rollouts_removed": 0,
        "parent_generators_removed": 1,
        "parent_instances_removed": 1,
        "parent_representation_inputs_removed": 1,
    }


def test_merge_rejects_parent_semantic_conflict_and_child_id_collision(
    tmp_path: Path,
) -> None:
    module, source_dirs = _merge_sources(tmp_path)
    conflicting_instances = read_jsonl(
        source_dirs[1] / "instance_manifest.jsonl"
    )
    parent = next(row for row in conflicting_instances if row["role"] == "parent")
    parent["problem_text"] = "A conflicting parent problem."
    _write_jsonl(
        source_dirs[1] / "instance_manifest.jsonl",
        conflicting_instances,
    )
    with pytest.raises(ValueError, match="conflicting parent instance"):
        module.cmd_merge(
            Namespace(
                inputs=[str(path) for path in source_dirs],
                output_dir=str(tmp_path / "conflict-output"),
            )
        )

    module, source_dirs = _merge_sources(tmp_path / "child-collision")
    first_children = [
        row
        for row in read_jsonl(source_dirs[0] / "generator_manifest.jsonl")
        if row["condition"] in {"plain", "reasoning"}
    ]
    second_rows = read_jsonl(source_dirs[1] / "generator_manifest.jsonl")
    second_child = next(
        row
        for row in second_rows
        if row["condition"] == first_children[0]["condition"]
        and row["operator"] == first_children[0]["operator"]
    )
    second_child["generator_unit_id"] = first_children[0][
        "generator_unit_id"
    ]
    _write_jsonl(source_dirs[1] / "generator_manifest.jsonl", second_rows)
    with pytest.raises(ValueError, match="generator_unit_id values are not unique"):
        module.cmd_merge(
            Namespace(
                inputs=[str(path) for path in source_dirs],
                output_dir=str(tmp_path / "collision-output"),
            )
        )


def test_merge_rejects_source_protocol_provenance_conflict(
    tmp_path: Path,
) -> None:
    module, source_dirs = _merge_sources(tmp_path)
    manifest_path = source_dirs[1] / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_evidence_gate_version"] = "different-gate"
    _write_json(manifest_path, manifest)

    with pytest.raises(
        ValueError,
        match="disagree on source_evidence_gate_version",
    ):
        module.cmd_merge(
            Namespace(
                inputs=[str(path) for path in source_dirs],
                output_dir=str(tmp_path / "provenance-output"),
            )
        )


def test_validity_policy_never_imputes_invalid_as_zero(tmp_path: Path) -> None:
    root, parent = _comparison_fixture(tmp_path)
    output = tmp_path / "normalized"
    prepare_comparison_manifests(
        root,
        output_dir=output,
        parent_program=parent,
        solver_checkpoint_sha256="1" * 64,
    )
    rows = read_jsonl(output / "instance_manifest.jsonl")
    strict = select_child_instances(rows, validity_policy="strict")
    diagnostic = select_child_instances(rows, validity_policy="code_valid")

    assert [row["condition"] for row in strict] == ["plain"]
    assert {row["condition"] for row in diagnostic} == {"plain", "reasoning"}
    assert not any(row.get("generation_status") == "invalid_code" for row in diagnostic)


def test_prepare_rejects_malformed_external_checkpoint_digest(
    tmp_path: Path,
) -> None:
    root, parent = _comparison_fixture(tmp_path)
    with pytest.raises(ValueError, match="64 hexadecimal"):
        prepare_comparison_manifests(
            root,
            output_dir=tmp_path / "normalized",
            parent_program=parent,
            solver_checkpoint_sha256="not-a-checkpoint-digest",
        )


def test_representation_row_cannot_receive_analysis_fields() -> None:
    source = {
        "item_id": "sample",
        "role": "child",
        "problem_text": "What is 2 + 2?",
        "common_io_spec": "Return a boxed answer.",
        "independent_run_id": "run",
        "parent_program_id": "parent",
        "generator_unit_id": "generator",
        "instance_seed": 0,
        "answer": "4",
        "target_reasoning_move": "add",
        "source_code": "malicious leakage",
    }
    result = representation_input_row(source)
    assert result["problem_text"] == "What is 2 + 2?"
    assert "answer" not in result
    assert "target_reasoning_move" not in result
    assert "source_code" not in result


def test_numeric_controls_are_surface_derived() -> None:
    result = numeric_summary("Use -3, 2.5, and 10.")
    assert result["numeric_count"] == 3
    assert result["numeric_min"] == -3
    assert result["numeric_max"] == 10
    assert result["numeric_span"] == 13
    assert result["numeric_max_abs"] == 10


@pytest.mark.parametrize(
    ("operator", "child_group", "child_type", "expected"),
    [
        ("in_depth", "algebra", "algebra.linear", True),
        ("in_depth", "algebra", "algebra.other", False),
        ("in_breadth", "sequence", "sequence.sum", True),
        ("in_breadth", "algebra", "algebra.other", False),
    ],
)
def test_operator_contract(
    operator: str,
    child_group: str,
    child_type: str,
    expected: bool,
) -> None:
    assert (
        operator_contract_valid(
            operator,
            parent_group="algebra",
            parent_type="algebra.linear",
            child_group=child_group,
            child_type=child_type,
        )
        is expected
    )


def test_one_run_is_descriptive_only() -> None:
    generators = [
        {
            "condition": condition,
            "overall_status": "ok",
            "code_valid": True,
            "operator_contract_valid": True,
            "evaluator_passed": 5,
            "evaluator_total": 5,
            "generator_pair_id": "pair",
            "parent_program_id": "parent",
            "independent_run_id": "run",
        }
        for condition in ("plain", "reasoning")
    ]
    audit = audit_generation_sufficiency(generators, [])
    assert audit["strict_matched_generator_pairs"] == 1
    assert audit["descriptive_only"] is True
    assert "insufficient_independent_units" in audit["warnings"][0]


def test_pair_design_audit_uses_actual_llm_calls_and_token_budget() -> None:
    common = {
        "generator_pair_id": "pair",
        "solver_checkpoint_id": "solver",
        "solver_checkpoint_sha256": "1" * 64,
        "generator_checkpoint_id": "generator",
        "source_model_id": "generator",
        "source_run_manifest_sha256": "a" * 64,
        "generator_candidate_draw_count": 1,
        "sampling_seed_policy": "recorded_vllm_engine_seed_per_run",
        "generator_sampling_seed_range": [7, 7],
        "llm_seed": 7,
        "generator_draw_idx": 7,
        "instance_seed_policy": "explicit_evaluation_seed_list",
        "instance_seed_range": [0, 4],
        "evaluation_seeds": [0, 1, 2, 3, 4],
        "requested_instance_budget": 5,
        "child_rollout_budget": 10,
        "acceptance_budget": None,
        "code_stage_max_tokens": 5000,
        "plan_temperature": 0.3,
        "plan_top_p": 0.95,
        "code_temperature": 0.2,
        "code_top_p": 0.95,
    }
    rows = [
        {
            **common,
            "condition": "plain",
            "llm_generation_call_count": 1,
            "plan_stage_max_tokens": 0,
            "total_generator_max_token_budget": 5000,
        },
        {
            **common,
            "condition": "reasoning",
            "llm_generation_call_count": 2,
            "plan_stage_max_tokens": 1024,
            "total_generator_max_token_budget": 6024,
        },
    ]
    audit = audit_generator_pair_design(rows)
    assert audit["passed"] is False
    assert any(
        "llm_generation_call_count differs" in failure
        for failure in audit["failures"]
    )
    assert any(
        "total_generator_max_token_budget differs" in failure
        for failure in audit["failures"]
    )


def test_pair_design_accepts_equal_two_stage_generation() -> None:
    common = {
        "generator_pair_id": "pair",
        "solver_checkpoint_id": "solver",
        "solver_checkpoint_sha256": "1" * 64,
        "generator_checkpoint_id": "generator",
        "source_model_id": "generator",
        "source_run_manifest_sha256": "a" * 64,
        "generator_candidate_draw_count": 1,
        "llm_generation_call_count": 2,
        "actual_llm_generation_call_count": 2,
        "total_generator_max_token_budget": 6024,
        "plan_stage_max_tokens": 1024,
        "code_stage_max_tokens": 5000,
        "plan_temperature": 0.3,
        "plan_top_p": 0.95,
        "code_temperature": 0.2,
        "code_top_p": 0.95,
        "sampling_seed_policy": "recorded_vllm_engine_seed_per_run",
        "generator_sampling_seed_range": [7, 7],
        "llm_seed": 7,
        "generator_draw_idx": 7,
        "instance_seed_policy": "explicit_evaluation_seed_list",
        "instance_seed_range": [0, 4],
        "evaluation_seeds": [0, 1, 2, 3, 4],
        "requested_instance_budget": 5,
        "child_rollout_budget": 10,
        "acceptance_budget": 1,
        "paired_request_seed_policy": "sha256_llm_seed_operator_stage_v1",
        "plan_sampling_seed": 101,
        "code_sampling_seed": 202,
        "evaluator_sampling_seed_policy": (
            "sha256_llm_seed_operator_instance_seed_v1"
        ),
        "evaluator_sampling_seed_sha256": "b" * 64,
        "evaluator_sampling_seed_range": [303, 404],
        "child_solver_sampling_seed_policy": (
            "sha256_llm_seed_operator_instance_seed_rollout_idx_v1"
        ),
        "child_solver_sampling_seed_sha256": "c" * 64,
        "child_solver_sampling_seed_range": [505, 606],
        "plan_input_token_count": 800,
        "plan_input_token_target": 800,
        "plan_input_token_delta": 0,
        "code_input_token_count": 900,
    }
    rows = [
        {**common, "condition": "plain"},
        {**common, "condition": "reasoning"},
    ]

    audit = audit_generator_pair_design(rows)

    assert audit["passed"] is True

    malformed = [dict(row) for row in rows]
    malformed[1]["solver_checkpoint_sha256"] = "not-a-digest"
    malformed_audit = audit_generator_pair_design(malformed)
    assert malformed_audit["passed"] is False
    assert any(
        "solver_checkpoint_sha256" in failure
        for failure in malformed_audit["failures"]
    )
    mismatched = [dict(row) for row in rows]
    mismatched[1]["child_solver_sampling_seed_sha256"] = "d" * 64
    seed_audit = audit_generator_pair_design(mismatched)
    assert seed_audit["passed"] is False
    assert any(
        "child_solver_sampling_seed_sha256 differs" in failure
        for failure in seed_audit["failures"]
    )
    assert audit["n_passed_pairs"] == 1


def test_heldout_schema_rejects_leakage_prone_duplicates() -> None:
    row = {
        "problem_id": "p1",
        "problem_text": "Problem",
        "answer": "1",
        "target_reasoning_move": "verify consistency",
        "transfer_level": "structural",
        "family_id": "family",
        "independent_run_id": "run",
    }
    validate_heldout_rows([row])
    with pytest.raises(ValueError, match="duplicate"):
        validate_heldout_rows([row, row])


def test_capability_heldout_schema_preserves_transfer_metadata() -> None:
    row = {
        "problem_id": 1,
        "problem_text": "A blinded problem",
        "answer": 2,
        "target_reasoning_move": "test a necessary condition",
        "transfer_level": "cross_domain",
        "family_id": "family",
        "independent_run_id": "run",
        "construction_seed": 991,
        "split_frozen_before_training": True,
        "heldout_provenance": {
            "provenance_id": "heldout-v1",
            "construction_method": "blinded independent generator",
            "frozen_at_utc": "2026-07-01T00:00:00Z",
            "freeze_manifest_sha256": "a" * 64,
        },
    }
    validated = validate_capability_heldout_rows([row], mode="confirmatory")
    assert validated[0]["problem_id"] == "1"
    assert validated[0]["answer"] == "2"
    assert validated[0]["construction_seed"] == "991"
    with pytest.raises(HeldoutSchemaError, match="transfer_level"):
        validate_capability_heldout_rows(
            [{**row, "transfer_level": "latent-nearest"}]
        )
    with pytest.raises(HeldoutSchemaError, match="frozen before training"):
        validate_capability_heldout_rows(
            [{**row, "split_frozen_before_training": False}],
            mode="confirmatory",
        )


def test_checkpoint_evaluation_defaults_to_native_pytorch_sampler(
    monkeypatch,
) -> None:
    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLLM:
        def __init__(self, **kwargs):
            assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
            self.kwargs = kwargs

        def chat(self, conversations, **kwargs):
            assert len(conversations) == 1
            assert kwargs["sampling_params"].kwargs["temperature"] == 0.0
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[
                        SimpleNamespace(
                            text=r"\boxed{2}",
                            token_ids=[3, 4],
                            finish_reason="stop",
                        )
                    ],
                )
            ]

    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
    monkeypatch.setitem(
        sys.modules,
        "rq_evolve.math_eval",
        SimpleNamespace(grade_eval=lambda response, answer: answer == "2"),
    )
    heldout = {
        "problem_id": "p1",
        "problem_text": "Compute 1 + 1.",
        "answer": "2",
        "target_reasoning_move": "verify the arithmetic",
        "transfer_level": "in_family",
        "family_id": "addition",
        "independent_run_id": "run1",
        "construction_seed": "17",
        "split_frozen_before_training": False,
        "heldout_provenance": {},
    }

    rows = evaluate_checkpoint_vllm(
        [heldout],
        checkpoint="/fake/checkpoint",
        condition="base",
        checkpoint_run_id="run1",
        evaluation_mode="pilot",
        max_tokens=8,
        use_tqdm=False,
    )

    assert rows[0]["correct"] is True
    assert rows[0]["decoding_parameters"]["vllm_sampler_backend"] == "pytorch"
    assert (
        rows[0]["decoding_parameters"]["vllm_use_flashinfer_sampler"] == "0"
    )


def test_capability_evaluation_audit_blocks_cross_condition_mismatch() -> None:
    import hashlib
    import json

    def digest_text(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    def digest_json(value) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    problem_text = "Compute 1 + 1."
    answer = "2"
    provenance = {
        "provenance_id": "heldout-v1",
        "construction_method": "blinded",
        "frozen_at_utc": "2026-07-01T00:00:00Z",
        "freeze_manifest_sha256": "a" * 64,
    }
    problem_contract = {
        "problem_text_sha256": digest_text(problem_text),
        "answer_sha256": digest_text(answer),
        "target_reasoning_move": "verify",
        "transfer_level": "structural",
        "family_id": "family",
        "construction_seed": "7",
    }
    common = {
        "evaluation_schema_version": 2,
        "evaluation_mode": "confirmatory",
        "independent_run_id": "run",
        "checkpoint_run_id": "run",
        "problem_id": "p",
        "problem_text": problem_text,
        "answer": answer,
        "problem_text_sha256": digest_text(problem_text),
        "answer_sha256": digest_text(answer),
        "problem_contract_sha256": digest_json(problem_contract),
        "target_reasoning_move": "verify",
        "family_id": "family",
        "transfer_level": "structural",
        "construction_seed": "7",
        "split_frozen_before_training": True,
        "heldout_provenance": provenance,
        "heldout_provenance_sha256": digest_json(provenance),
        "heldout_input_sha256": "b" * 64,
        "heldout_content_sha256": "c" * 64,
        "decoding_parameters": {"temperature": 0.0, "max_tokens": 32},
        "evaluation_contract_sha256": "d" * 64,
        "tokenizer_id": "tokenizer",
        "solver_system_prompt_sha256": "e" * 64,
        "correct": True,
    }
    rows = [
        {**common, "condition": condition, "checkpoint_id": f"{condition}-ckpt"}
        for condition in ("base", "plain", "reasoning")
    ]
    assert audit_paired_evaluation_rows(rows)["passed"] is True

    mismatched = [dict(row) for row in rows]
    mismatched[-1]["decoding_parameters"] = {
        "temperature": 1.0,
        "max_tokens": 32,
    }
    audit = audit_paired_evaluation_rows(mismatched)
    assert audit["passed"] is False
    assert audit["mismatched_units"][0]["field_mismatches"][
        "decoding_parameters"
    ]


def test_static_budget_audit_requires_equal_tokens_per_run() -> None:
    plain = [
        {
            "independent_run_id": "run",
            "prompt_token_count": 10,
        }
    ]
    reasoning = [
        {
            "independent_run_id": "run",
            "prompt_token_count": 11,
        }
    ]
    audit = audit_paired_training_budgets(
        plain,
        reasoning,
        token_count_fields=("prompt_token_count",),
    )
    assert audit["budget_equal"] is False
    assert "training token totals differ" in audit["issues"]


def test_capability_compute_manifests_link_runs_files_and_checkpoints(
    tmp_path: Path,
) -> None:
    import hashlib
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_expansion_hypothesis.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_expansion_hypothesis_test",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def write_artifact(name: str, text: str) -> tuple[Path, str]:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    plain_data, plain_data_hash = write_artifact(
        "plain.jsonl",
        json.dumps(
            {
                "sample_id": "plain-train",
                "independent_run_id": "run-1",
                "problem": "Compute 3 + 4.",
                "seed": 3,
            }
        )
        + "\n",
    )
    reasoning_data, reasoning_data_hash = write_artifact(
        "reasoning.jsonl",
        json.dumps(
            {
                "sample_id": "reasoning-train",
                "independent_run_id": "run-1",
                "problem": "Compute 5 + 6.",
                "seed": 5,
            }
        )
        + "\n",
    )
    plain_log, plain_log_hash = write_artifact("plain.log", "plain")
    reasoning_log, reasoning_log_hash = write_artifact(
        "reasoning.log",
        "reasoning",
    )
    common = {
        "independent_run_id": "run-1",
        "base_checkpoint": "base-ckpt",
        "base_checkpoint_sha256": "a" * 64,
        "training_instance_count": 1,
        "training_token_count": 10,
        "optimizer": "adamw",
        "learning_rate": 1e-6,
        "update_steps": 1,
        "batch_size": 1,
        "batch_composition": "fixed",
        "verifier": "grader",
        "max_rollout_length": 32,
        "total_compute": "fixed",
        "resume_mode": "disable",
    }
    manifest = {
        "conditions": {
            "plain": {
                **common,
                "training_data": str(plain_data),
                "training_data_sha256": plain_data_hash,
                "training_log": str(plain_log),
                "training_log_sha256": plain_log_hash,
                "output_checkpoint": "plain-ckpt",
                "output_checkpoint_sha256": "b" * 64,
            },
            "reasoning": {
                **common,
                "training_data": str(reasoning_data),
                "training_data_sha256": reasoning_data_hash,
                "training_log": str(reasoning_log),
                "training_log_sha256": reasoning_log_hash,
                "output_checkpoint": "reasoning-ckpt",
                "output_checkpoint_sha256": "c" * 64,
            },
        }
    }
    manifest_path = tmp_path / "compute.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    evaluation_rows = [
        {
            "condition": condition,
            "independent_run_id": "run-1",
            "checkpoint_id": checkpoint,
        }
        for condition, checkpoint in (
            ("base", "base-ckpt"),
            ("plain", "plain-ckpt"),
            ("reasoning", "reasoning-ckpt"),
        )
    ]
    audit, training = module._audit_capability_compute_manifests(
        [str(manifest_path)],
        evaluation_rows,
    )
    assert audit["passed"] is True
    assert len(training["plain"]) == len(training["reasoning"]) == 1

    manifest["conditions"]["reasoning"]["training_data_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    broken, _ = module._audit_capability_compute_manifests(
        [str(manifest_path)],
        evaluation_rows,
    )
    assert broken["passed"] is False
    assert any(
        issue.get("issue") == "declared SHA-256 does not match local file"
        for issue in broken["per_run"][0][
            "local_artifact_and_checkpoint_issues"
        ]
    )


def test_generation_capability_linkage_is_exact_and_gates_strong_claim() -> None:
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_expansion_hypothesis.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_expansion_hypothesis_linkage_test",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report_hash = "a" * 64
    experiment_hash = "b" * 64
    generator_hash = "c" * 64
    instance_hash = "d" * 64
    source_hash = "e" * 64
    solver_hash = "1" * 64
    generation_report = {
        "analysis_kind": "generation_space",
        "experiment_provenance": {
            "experiment_id": "experiment",
            "independent_run_ids": ["run-1"],
            "solver_checkpoint_id": "base-ckpt",
            "observed_solver_checkpoint_ids": ["base-ckpt"],
            "solver_checkpoint_sha256": solver_hash,
            "observed_solver_checkpoint_sha256s": [solver_hash],
            "source_run_manifest_sha256_by_run": {
                "run-1": [source_hash],
            },
            "experiment_manifest_sha256": experiment_hash,
            "generator_manifest_sha256": generator_hash,
            "instance_manifest_sha256": instance_hash,
        },
        "claim_criteria": {
            "limited_generation_space_claim_allowed": True,
            "layer_pooling_direction_consistent": True,
            "independent_runs_and_parents_gate": True,
        },
    }
    linkage = {
        "independent_run_id": "run-1",
        "solver_checkpoint_id": "base-ckpt",
        "solver_checkpoint_sha256": solver_hash,
        "generation_report_sha256": report_hash,
        "generation_experiment_manifest_sha256": experiment_hash,
        "generation_generator_manifest_sha256": generator_hash,
        "generation_instance_manifest_sha256": instance_hash,
        "source_run_manifest_sha256": [source_hash],
    }
    compute_audit = {
        "passed": True,
        "artifact_provenance_complete": True,
        "compute_manifest_runs": ["run-1"],
        "per_run": [
            {
                "independent_run_id": "run-1",
                "path": "/compute.json",
                "training_base_checkpoints": ["base-ckpt"],
                "training_base_checkpoint_sha256s": [solver_hash],
                "training_base_checkpoint_sha256_by_condition": {
                    "plain": solver_hash,
                    "reasoning": solver_hash,
                },
                "generation_linkage": linkage,
            }
        ],
    }
    evaluation_rows = [{"independent_run_id": "run-1"}]
    audit = module._audit_generation_capability_linkage(
        generation_report=generation_report,
        generation_report_sha256=report_hash,
        evaluation_rows=evaluation_rows,
        compute_audit=compute_audit,
    )
    assert audit["passed"] is True

    wrong_base_digest = {
        **compute_audit,
        "per_run": [
            {
                **compute_audit["per_run"][0],
                "training_base_checkpoint_sha256_by_condition": {
                    "plain": solver_hash,
                    "reasoning": "2" * 64,
                },
            }
        ],
    }
    digest_mismatch = module._audit_generation_capability_linkage(
        generation_report=generation_report,
        generation_report_sha256=report_hash,
        evaluation_rows=evaluation_rows,
        compute_audit=wrong_base_digest,
    )
    assert digest_mismatch["passed"] is False
    assert any(
        "each_condition_base_checkpoint_sha256_matches" in failure
        for failure in digest_mismatch["failures"]
    )

    mismatched_compute = {
        **compute_audit,
        "per_run": [
            {
                **compute_audit["per_run"][0],
                "generation_linkage": {
                    **linkage,
                    "source_run_manifest_sha256": ["f" * 64],
                },
            }
        ],
    }
    mismatched = module._audit_generation_capability_linkage(
        generation_report=generation_report,
        generation_report_sha256=report_hash,
        evaluation_rows=evaluation_rows,
        compute_audit=mismatched_compute,
    )
    assert mismatched["passed"] is False
    assert any(
        "source_run_hashes_match" in failure
        for failure in mismatched["failures"]
    )

    positive = {
        "inferential_valid": True,
        "ci_low": 0.1,
        "n_runs": 3,
    }
    criteria = module._capability_claim_criteria(
        {
            "overall": positive,
            "all_condition_units_complete": True,
            "incomplete_units": [],
            "by_transfer_level": {
                "in_family": positive,
                "structural": positive,
            },
            "by_target_reasoning_move": {"verify": positive},
        },
        forgetting_did={
            "by_transfer_level": {
                "archive": {
                    "inferential_valid": True,
                    "reasoning_gain_ci_low": 0.0,
                }
            }
        },
        compute_audit=compute_audit,
        evaluation_audit={"passed": True},
        disjointness_audit={"passed": True},
        generation_report=generation_report,
        generation_linkage_audit=mismatched,
        max_forgetting=0.02,
        minimum_independent_runs=3,
    )
    assert criteria["generation_capability_provenance_linked"] is False
    assert criteria["strong_capability_expansion_claim_allowed"] is False


def test_generation_report_provenance_records_run_source_and_checkpoint(
    tmp_path: Path,
) -> None:
    import hashlib
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_expansion_hypothesis.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_expansion_hypothesis_provenance_test",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    experiment_path = tmp_path / "experiment_manifest.json"
    generator_path = tmp_path / "generator_manifest.jsonl"
    instance_path = tmp_path / "instance_manifest.jsonl"
    experiment_path.write_text("{}", encoding="utf-8")
    generator_path.write_text("{}\n", encoding="utf-8")
    instance_path.write_text("{}\n", encoding="utf-8")
    source_hash = "e" * 64
    solver_hash = "1" * 64
    provenance = module._generation_experiment_provenance(
        experiment_manifest={
            "experiment_id": "exp",
            "solver_checkpoint_id": "solver",
            "solver_checkpoint_sha256": solver_hash,
        },
        experiment_manifest_path=experiment_path,
        generator_rows=[
            {
                "condition": condition,
                "independent_run_id": "run-1",
                "solver_checkpoint_id": "solver",
                "solver_checkpoint_sha256": solver_hash,
                "source_run_manifest_sha256": source_hash,
            }
            for condition in ("plain", "reasoning")
        ],
        generator_manifest_path=generator_path,
        instance_manifest_path=instance_path,
    )
    assert provenance["experiment_id"] == "exp"
    assert provenance["independent_run_ids"] == ["run-1"]
    assert provenance["solver_checkpoint_id"] == "solver"
    assert provenance["solver_checkpoint_sha256"] == solver_hash
    assert provenance["observed_solver_checkpoint_sha256s"] == [solver_hash]
    assert provenance["source_run_manifest_sha256_by_run"] == {
        "run-1": [source_hash]
    }
    assert provenance["experiment_manifest_sha256"] == hashlib.sha256(
        experiment_path.read_bytes()
    ).hexdigest()
