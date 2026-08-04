from __future__ import annotations

import numpy as np

from rq_evolve.expansion_analysis import analyze_generation_space
from rq_evolve.expansion_capability import audit_training_heldout_disjointness
from rq_evolve.expansion_repr import (
    RepresentationArtifact,
    l2_normalize as normalize_representation,
    render_representation_prompt,
    resolve_layers,
)
from rq_evolve.expansion_stats import (
    aligned_orthogonal_norms,
    analyze_capability_did,
    audit_same_compute_manifest,
    cosine_coverage,
    fit_plain_pca_subspace,
    hierarchical_bootstrap_adjusted_paired_ridge,
    hierarchical_bootstrap_paired_effect,
    leave_one_out_kth_epsilon,
)
from rq_evolve.expansion_trajectory import (
    compute_stalt,
    correct_wrong_divergence_onset,
)


def test_plain_only_subspace_finds_reasoning_orthogonal_axis() -> None:
    plain = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    subspace = fit_plain_pca_subspace(plain, variance_threshold=0.95)
    aligned, orthogonal = aligned_orthogonal_norms(
        np.asarray([[2.0, 0.0], [0.0, 2.0]]),
        subspace.basis,
    )
    assert aligned[0] == 2.0
    assert orthogonal[0] < 1e-10
    assert aligned[1] < 1e-10
    assert orthogonal[1] == 2.0

    # Reasoning observations are never an input to the fitted basis.
    wildly_different_reasoning = np.asarray([[0.0, 1e9]])
    _ = wildly_different_reasoning
    repeated = fit_plain_pca_subspace(plain, variance_threshold=0.95)
    np.testing.assert_allclose(subspace.basis, repeated.basis)


def test_uncentered_primary_includes_mean_plain_expansion_direction() -> None:
    plain = np.asarray(
        [
            [10.0, -0.1, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.1, 0.0],
        ]
    )
    primary = fit_plain_pca_subspace(
        plain,
        variance_threshold=0.95,
        centered=False,
    )
    _, orthogonal = aligned_orthogonal_norms(plain, primary.basis)
    assert np.max(orthogonal) < 0.11

    centered_sensitivity = fit_plain_pca_subspace(
        plain,
        variance_threshold=0.95,
        centered=True,
    )
    _, centered_orthogonal = aligned_orthogonal_norms(
        plain,
        centered_sensitivity.basis,
    )
    assert np.min(centered_orthogonal) > 9.9


def test_representation_prompt_has_solver_boundary_without_label_leakage() -> None:
    class FakeTokenizer:
        chat_template = "present"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["add_generation_prompt"] is True
            assert [message["role"] for message in messages] == ["system", "user"]
            assert messages[1]["content"] == "Visible problem"
            return "rendered solver prompt"

    assert (
        render_representation_prompt("Visible problem", FakeTokenizer())
        == "rendered solver prompt"
    )
    assert resolve_layers(36).primary == 23
    assert resolve_layers(36).adjacent == (22, 24)


def test_representation_normalization_rejects_zero_vector() -> None:
    with np.testing.assert_raises(ValueError):
        normalize_representation(np.zeros((1, 3), dtype=np.float32))


def test_generator_balanced_pca_weights_limit_seed_duplication() -> None:
    small_generator = np.asarray([[1.0, 0.0], [2.0, 0.0]])
    large_generator = np.repeat([[0.0, 1.0]], repeats=100, axis=0)
    rows = np.concatenate([small_generator, large_generator])
    weights = np.asarray([0.5, 0.5, *([0.01] * 100)])
    weighted = fit_plain_pca_subspace(
        rows,
        variance_threshold=0.8,
        sample_weights=weights,
    )
    assert weighted.n_samples == 102
    assert weighted.sample_weighted is True
    assert weighted.effective_sample_size < 10


def test_archive_loo_excludes_self_and_coverage_uses_strict_threshold() -> None:
    archive = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    epsilon = leave_one_out_kth_epsilon(
        archive,
        k=1,
        quantile=0.95,
        mode="confirmatory",
    )
    assert np.all(epsilon["leave_one_out_kth_distances"] > 0)

    query = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    coverage = cosine_coverage(
        query,
        archive,
        k=1,
        quantile=0.95,
        mode="confirmatory",
    )
    # The first query lands exactly on epsilon and is therefore not covered:
    # the experiment contract is d_k > epsilon, not >=.
    assert coverage["query_kth_distances"][0] == coverage["epsilon"]
    assert coverage["covered"].tolist() == [False, False]


def test_one_run_bootstrap_does_not_fabricate_confidence_interval() -> None:
    result = hierarchical_bootstrap_paired_effect(
        [
            {
                "run_id": "run",
                "parent_program_id": "parent",
                "generator_pair_id": "pair",
                "orthogonal_displacement_difference": 1.0,
            }
        ],
        difference_field="orthogonal_displacement_difference",
        n_resamples=10,
    )
    assert result["mean_difference"] == 1.0
    assert result["inferential_valid"] is False
    assert result["ci_low"] is None
    assert result["ci_high"] is None


def test_adjusted_ridge_claim_has_hierarchical_bootstrap_interval() -> None:
    rows = [
        {
            "run_id": run,
            "parent_program_id": parent,
            "orthogonal_displacement_difference": value,
            "rq_difference": 0.0,
        }
        for run, parent, value in (
            ("run-1", "parent-1", 1.0),
            ("run-1", "parent-2", 1.2),
            ("run-2", "parent-1", 0.8),
            ("run-2", "parent-2", 1.1),
        )
    ]
    result = hierarchical_bootstrap_adjusted_paired_ridge(
        rows,
        outcome_difference_field="orthogonal_displacement_difference",
        covariate_difference_fields=("rq_difference",),
        n_resamples=50,
        seed=9,
    )
    assert result["inferential_valid"] is True
    assert result["ci_low"] > 0


def test_capability_did_is_stratified_and_run_gated() -> None:
    rows = []
    values = {
        "base": [0.2, 0.3],
        "plain": [0.3, 0.4],
        "reasoning": [0.7, 0.8],
    }
    for run_index, run_id in enumerate(("run-1", "run-2")):
        for condition, condition_values in values.items():
            rows.append(
                {
                    "run_id": run_id,
                    "problem_id": f"problem-{run_index}",
                    "transfer_level": "structural",
                    "target_reasoning_move": "verify consistency",
                    "family_id": "structural-family",
                    "condition": condition,
                    "accuracy": condition_values[run_index],
                }
            )
    result = analyze_capability_did(rows, n_resamples=50, seed=4)
    assert np.isclose(result["overall"]["delta_cap"], 0.4)
    assert result["overall"]["inferential_valid"] is True
    assert "structural" in result["by_transfer_level"]
    assert "verify consistency" in result["by_target_reasoning_move"]
    assert result["overall"]["n_families"] == 1


def test_compute_manifest_requires_every_prespecified_field_equal() -> None:
    common = {
        "independent_run_id": "run-1",
        "base_checkpoint": "base",
        "base_checkpoint_sha256": "a" * 64,
        "training_data_sha256": "b" * 64,
        "training_log_sha256": "c" * 64,
        "output_checkpoint_sha256": "d" * 64,
        "training_instance_count": 100,
        "training_token_count": 1000,
        "optimizer": "adamw",
        "learning_rate": 1e-6,
        "update_steps": 128,
        "batch_size": 32,
        "batch_composition": "fixed",
        "verifier": "math_verify",
        "max_rollout_length": 4096,
        "total_compute": "8xA100x128steps",
        "resume_mode": "disable",
    }
    manifest = {
        "conditions": {
            "plain": {
                **common,
                "training_data_sha256": "b" * 64,
                "training_log_sha256": "c" * 64,
                "output_checkpoint_sha256": "d" * 64,
            },
            "reasoning": {
                **common,
                "learning_rate": 2e-6,
                "training_data_sha256": "e" * 64,
                "training_log_sha256": "f" * 64,
                "output_checkpoint_sha256": "1" * 64,
            },
        }
    }
    result = audit_same_compute_manifest(manifest)
    assert result["passed"] is False
    assert result["failed_fields"] == ["learning_rate"]
    assert result["artifact_provenance_complete"] is True


def test_compute_manifest_blocks_missing_artifact_provenance() -> None:
    common = {
        "independent_run_id": "run-1",
        "base_checkpoint": "base",
        "base_checkpoint_sha256": "a" * 64,
        "training_data_sha256": "b" * 64,
        "training_log_sha256": "c" * 64,
        "training_instance_count": 100,
        "training_token_count": 1000,
        "optimizer": "adamw",
        "learning_rate": 1e-6,
        "update_steps": 10,
        "batch_size": 4,
        "batch_composition": "fixed",
        "verifier": "grader",
        "max_rollout_length": 128,
        "total_compute": "fixed",
        "resume_mode": "disable",
    }
    result = audit_same_compute_manifest(
        {"conditions": {"plain": common, "reasoning": dict(common)}}
    )
    assert result["passed"] is False
    assert "plain.output_checkpoint" in result["failed_provenance"]
    assert "reasoning.output_checkpoint" in result["failed_provenance"]


def test_capability_did_marks_incomplete_condition_units() -> None:
    complete = [
        {
            "run_id": "run-1",
            "problem_id": "complete",
            "transfer_level": "in_family",
            "target_reasoning_move": "verify",
            "family_id": "family",
            "condition": condition,
            "accuracy": value,
        }
        for condition, value in (
            ("base", 0.0),
            ("plain", 0.0),
            ("reasoning", 1.0),
        )
    ]
    incomplete = [
        {
            "run_id": "run-1",
            "problem_id": "missing-reasoning",
            "transfer_level": "in_family",
            "target_reasoning_move": "verify",
            "family_id": "family",
            "condition": condition,
            "accuracy": 0.0,
        }
        for condition in ("base", "plain")
    ]
    result = analyze_capability_did(
        [*complete, *incomplete],
        n_resamples=10,
    )
    assert result["all_condition_units_complete"] is False
    assert result["incomplete_units"][0]["missing_conditions"] == ["reasoning"]


def test_training_heldout_disjointness_detects_text_seed_and_numeric_leakage() -> None:
    heldout = [
        {
            "condition": condition,
            "independent_run_id": "run-1",
            "problem_id": "heldout-1",
            "problem_text": "Find the sum of 2, 4, and 6.",
            "construction_seed": "99",
        }
        for condition in ("base", "plain", "reasoning")
    ]
    clean_training = [
        {
            "sample_id": "train-1",
            "independent_run_id": "run-1",
            "problem": "Find the product of 3 and 5.",
            "seed": 7,
        }
    ]
    clean = audit_training_heldout_disjointness(
        heldout,
        {"plain": clean_training, "reasoning": clean_training},
    )
    assert clean["passed"] is True

    leaked_training = [
        {
            "sample_id": "train-2",
            "independent_run_id": "run-1",
            "problem": "Find the sum of 2, 4, and 6.",
            "seed": 99,
        }
    ]
    leaked = audit_training_heldout_disjointness(
        heldout,
        {"plain": leaked_training, "reasoning": clean_training},
    )
    assert leaked["passed"] is False
    issues = {
        issue["issue"]
        for issue in leaked["condition_reports"][0]["issues"]
    }
    assert "normalized surface problem text overlaps training" in issues
    assert "construction/sampling seed overlaps training" in issues
    assert "complete numeric-literal signature overlaps training" in issues


def _unit(vector: list[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    return value / np.linalg.norm(value)


def test_generation_orchestrator_reports_pilot_without_inference() -> None:
    sample_ids = [
        "parent-0",
        "parent-1",
        "plain-0",
        "plain-1",
        "reasoning-0",
        "reasoning-1",
    ]
    vectors = np.stack(
        [
            _unit([1, 0, 0]),
            _unit([1, 0, 0]),
            _unit([1, 0.1, 0]),
            _unit([1, 0.2, 0]),
            _unit([1, 0.1, 0.8]),
            _unit([1, 0.2, 0.8]),
        ]
    )
    artifact = RepresentationArtifact(
        arrays={"layer_023__masked_mean": vectors},
        metadata={"token_counts": [10, 10, 12, 13, 12, 13]},
    )
    rows = []
    for seed in (0, 1):
        rows.append(
            {
                "item_id": f"parent-{seed}",
                "role": "parent",
                "instance_seed": seed,
            }
        )
        for condition in ("plain", "reasoning"):
            rows.append(
                {
                    "item_id": f"{condition}-{seed}",
                    "role": "child",
                    "independent_run_id": "run",
                    "parent_program_id": "parent",
                    "parent_instance_id": f"parent-{seed}",
                    "generator_pair_id": "pair",
                    "generator_unit_id": f"{condition}-generator",
                    "instance_seed": seed,
                    "condition": condition,
                    "operator": "in_depth",
                    "evolution_iteration": 0,
                    "code_valid": True,
                    "valid_for_confirmatory": True,
                    "operator_contract_valid": True,
                    "concept_group": "algebra",
                    "concept_type": "algebra.linear",
                    "rq": 0.1,
                    "numeric_count": 2,
                    "numeric_span": 3,
                    "numeric_max_abs": 3,
                }
            )

    result = analyze_generation_space(
        rows,
        representation_artifact=artifact,
        representation_sample_ids=sample_ids,
        representation_key="layer_023__masked_mean",
        mode="pilot",
        validity_policy="code_valid",
        bootstrap_replicates=10,
    )
    assert result["status"] == "ok_descriptive"
    assert result["inferential_valid"] is False
    assert result["unadjusted_effect"]["mean_difference"] > 0
    assert result["rq_control_audit"]["genuine_scalar_rq_available"] is False
    assert result["controls_adjusted_effect"] is None
    assert any("pilot_fit_evaluation_reuse" in warning for warning in result["warnings"])


def test_stalt_matches_layer_weighted_temporal_definition() -> None:
    hidden = np.asarray(
        [
            [[0.0], [0.0], [0.0]],
            [[0.0], [1.0], [2.0]],
            [[0.0], [3.0], [6.0]],
        ]
    )
    result = compute_stalt(hidden, tau=1.0)
    np.testing.assert_allclose(result["token_wise_amplitude"], [1.5, 3.0])
    assert result["stalt"] == 2.25
    assert result["mean_layer_concentration_hhi"] == 0.5

    wrong = hidden * 0.1
    onset = correct_wrong_divergence_onset(
        hidden,
        wrong,
        absolute_threshold=1.0,
        bins=10,
    )
    assert onset["divergence_metric"].startswith("rms_l2_hidden_vector")
    assert onset["divergence_onset_relative_position"] == 4 / 9
    assert onset["interpolated_hidden_trajectory_distance"][0] == 0.0
    assert onset["interpolated_stalt_amplitude_difference"][0] > 1.0

    # Equal transition amplitudes in opposite hidden directions must still
    # register as trajectory divergence.
    opposite = -hidden
    directional = correct_wrong_divergence_onset(
        hidden,
        opposite,
        absolute_threshold=1.0,
        bins=10,
    )
    np.testing.assert_allclose(
        directional["interpolated_stalt_amplitude_difference"],
        0.0,
    )
    assert directional["divergence_onset_relative_position"] is not None
