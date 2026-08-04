#!/usr/bin/env python3
"""Run the reasoning-informed Evolver expansion experiment.

The CLI intentionally separates:

* generation-space geometry under one frozen Solver; and
* capability-space transfer after equal-compute training.

It never converts instance seeds or rollout replicas into independent runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.expansion_analysis import analyze_generation_space  # noqa: E402
from rq_evolve.expansion_capability import (  # noqa: E402
    audit_paired_evaluation_rows,
    audit_paired_training_jsonl,
    audit_training_heldout_disjointness,
    evaluate_checkpoint_vllm,
    load_heldout_jsonl,
    read_jsonl as read_capability_jsonl,
    write_jsonl as write_capability_jsonl,
)
from rq_evolve.expansion_experiment import (  # noqa: E402
    audit_generator_pair_design,
    audit_generation_sufficiency,
    prepare_comparison_manifests,
    read_json,
    read_jsonl,
    representation_input_row,
    select_child_instances,
    write_json,
    write_jsonl,
)
from rq_evolve.expansion_repr import (  # noqa: E402
    HFSelectedLayerExtractor,
    RepresentationArtifact,
    load_representation_artifact,
    save_representation_artifact,
)
from rq_evolve.expansion_stats import (  # noqa: E402
    analyze_capability_did,
    audit_same_compute_manifest,
)
from rq_evolve.prompts import SOLVER_SYSTEM_PROMPT  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "expansion_hypothesis.json"
_SHA256_HEX = frozenset("0123456789abcdefABCDEF")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_external_sha256(value: Any, *, field: str) -> str:
    """Validate a caller/registry supplied digest without hashing the model."""

    if (
        not isinstance(value, str)
        or len(value.strip()) != 64
        or any(character not in _SHA256_HEX for character in value.strip())
    ):
        raise ValueError(
            f"{field} must be an externally supplied 64-character SHA-256 "
            "digest; this command does not hash large checkpoint files"
        )
    return value.strip().lower()


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _load_config(path: str | Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sample_ids(artifact: RepresentationArtifact) -> list[str]:
    values = artifact.metadata.get("sample_ids")
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise ValueError(
            "representation metadata has no sample_ids; re-run this script's "
            "'extract' command with the matching representation input manifest"
        )
    return values


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    scalar_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def _primary_representation_key(artifact: RepresentationArtifact) -> str:
    layer = artifact.metadata.get("layer_indices", {}).get("primary")
    if layer is None:
        raise ValueError("representation metadata has no primary layer index")
    key = f"layer_{int(layer):03d}__masked_mean"
    if key not in artifact.arrays:
        raise ValueError(f"primary representation array is missing: {key}")
    return key


def _preregistered_representation_keys(
    config: Mapping[str, Any],
) -> tuple[str, list[str]]:
    representation = config.get("representation")
    if not isinstance(representation, Mapping):
        raise ValueError("config.representation must be an object")
    primary = representation.get("primary_decoder_block_zero_based")
    if isinstance(primary, bool) or not isinstance(primary, int) or primary < 0:
        raise ValueError(
            "config.representation.primary_decoder_block_zero_based must be "
            "a non-negative integer"
        )
    offsets = representation.get("robustness_decoder_block_offsets")
    poolings = representation.get("robustness_poolings")
    if not isinstance(offsets, list) or not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in offsets
    ):
        raise ValueError(
            "config.representation.robustness_decoder_block_offsets must be "
            "an integer list"
        )
    pooling_names = {
        "mean_prompt_tokens": "masked_mean",
        "last_prompt_token": "last_prompt_token",
    }
    if not isinstance(poolings, list) or any(
        value not in pooling_names for value in poolings
    ):
        raise ValueError(
            "config.representation.robustness_poolings contains an unknown pooling"
        )
    primary_pooling = representation.get("primary_pooling")
    if primary_pooling not in pooling_names:
        raise ValueError("config.representation.primary_pooling is unknown")
    primary_key = (
        f"layer_{primary:03d}__{pooling_names[str(primary_pooling)]}"
    )
    layers = sorted({primary, *(primary + int(offset) for offset in offsets)})
    if layers[0] < 0:
        raise ValueError("preregistered robustness layer index is negative")
    keys = sorted(
        f"layer_{layer:03d}__{pooling_names[str(pooling)]}"
        for layer in layers
        for pooling in poolings
    )
    if primary_key not in keys:
        keys.append(primary_key)
        keys.sort()
    return primary_key, keys


def _parent_surrogate_archive(
    artifact: RepresentationArtifact,
    sample_ids: Sequence[str],
    instance_rows: Sequence[Mapping[str, Any]],
) -> tuple[RepresentationArtifact, list[str]]:
    parent_ids = {
        str(row["item_id"])
        for row in instance_rows
        if row.get("role") == "parent"
    }
    indices = [
        index
        for index, sample_id in enumerate(sample_ids)
        if sample_id in parent_ids
    ]
    if len(indices) < 2:
        raise ValueError("parent surrogate archive requires at least two parent instances")
    subset = {
        key: value[indices]
        for key, value in artifact.arrays.items()
    }
    metadata = dict(artifact.metadata)
    metadata["num_problems"] = len(indices)
    metadata["token_counts"] = [
        artifact.metadata["token_counts"][index] for index in indices
    ]
    metadata["archive_kind"] = "parent_instances_surrogate_not_archive_snapshot"
    return (
        RepresentationArtifact(arrays=subset, metadata=metadata),
        [sample_ids[index] for index in indices],
    )


def _generation_claim_criteria(
    primary: Mapping[str, Any],
    robustness: Sequence[Mapping[str, Any]],
    protocol_audit: Mapping[str, Any],
) -> dict[str, Any]:
    effect = primary.get("unadjusted_effect") or {}
    adjusted = primary.get("controls_adjusted_effect") or {}
    paired = primary.get("paired_metrics") or []
    inferentially_positive_layers = [
        result
        for result in robustness
        if (
            result.get("inferential_valid")
            and (result.get("unadjusted_effect") or {}).get("inferential_valid")
            and (result.get("unadjusted_effect") or {}).get("ci_low") is not None
            and result["unadjusted_effect"]["ci_low"] > 0
        )
    ]
    rq_audit = primary.get("rq_control_audit") or {}
    criteria = {
        "preregistered_protocol_and_pair_design_passed": bool(
            protocol_audit.get("passed")
        ),
        "orthogonal_effect_positive_with_ci_above_zero": bool(
            primary.get("inferential_valid")
            and
            effect.get("inferential_valid")
            and effect.get("ci_low") is not None
            and effect["ci_low"] > 0
        ),
        "controls_adjusted_effect_positive_with_ci_above_zero": bool(
            primary.get("inferential_valid")
            and rq_audit.get("genuine_scalar_rq_available")
            and adjusted.get("inferential_valid")
            and adjusted.get("ci_low") is not None
            and adjusted["ci_low"] > 0
        ),
        "strict_valid_matched_generators_present": bool(
            primary.get("validity_policy") == "strict" and paired
        ),
        "independent_runs_and_parents_gate": bool(primary.get("inferential_valid")),
        "layer_pooling_direction_consistent": bool(
            len(robustness) >= 3
            and len(inferentially_positive_layers) == len(robustness)
        ),
    }
    criteria["limited_generation_space_claim_allowed"] = all(criteria.values())
    return criteria


def _confirmatory_protocol_audit(
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    experiment_manifest: Mapping[str, Any] | None,
    experiment_manifest_path: Path,
    generator_rows: Sequence[Mapping[str, Any]],
    instance_rows: Sequence[Mapping[str, Any]],
    representation_input_rows: Sequence[Mapping[str, Any]],
    representation_input_path: Path,
    artifact: RepresentationArtifact,
    archive_artifact: RepresentationArtifact | None,
    archive_kind: str | None,
    actual_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify preregistration, frozen-model provenance, and paired compute."""

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    config_hash = _sha256_file(config_path)
    add(
        "experiment_manifest_present",
        experiment_manifest is not None,
        str(experiment_manifest_path),
    )
    if experiment_manifest is None:
        pair_audit = audit_generator_pair_design(generator_rows)
        return {
            "passed": False,
            "checks": checks,
            "generator_pair_design": pair_audit,
            "failures": ["experiment_manifest_present"],
        }

    declared_hash = experiment_manifest.get("preregistered_config_sha256")
    add(
        "supplied_config_matches_preregistered_hash",
        declared_hash is not None and str(declared_hash) == config_hash,
        {"declared": declared_hash, "actual": config_hash},
    )
    prereg_path = experiment_manifest.get("preregistered_config_path")
    if prereg_path:
        prereg_path = Path(str(prereg_path))
        prereg_ok = prereg_path.exists() and _sha256_file(prereg_path) == declared_hash
        add(
            "stored_preregistered_config_integrity",
            prereg_ok,
            str(prereg_path),
        )
    add(
        "config_declares_preregistered_before_extraction",
        config.get("analysis_status")
        == "preregistered_before_representation_extraction",
        config.get("analysis_status"),
    )

    warnings = [str(value) for value in experiment_manifest.get("warnings", [])]
    source_missing = (
        experiment_manifest.get("source_run_manifest_sha256") is None
        and not experiment_manifest.get("merged_sources")
    ) or any("source_manifest_missing" in warning for warning in warnings)
    add(
        "source_generation_manifest_present",
        not source_missing,
        experiment_manifest.get("source_run_manifest_path"),
    )
    source_design = experiment_manifest.get("source_comparison_design")
    supported_source_design = source_design in {
        "two_stage_plain_vs_reasoning_v1",
        "hybrid_compiler_plain_vs_reasoning_v2",
    }
    add(
        "paired_source_design",
        supported_source_design,
        source_design,
    )
    if source_design == "hybrid_compiler_plain_vs_reasoning_v2":
        add(
            "hybrid_registered_route_parity",
            (
                experiment_manifest.get("source_mutation_code_backend")
                in {"hybrid", "compiler_only"}
                and experiment_manifest.get("source_plan_schema_version") == 5
                and experiment_manifest.get(
                    "source_compiler_registry_version"
                )
                is not None
                and experiment_manifest.get("source_paired_route_valid")
                is True
            ),
            {
                "mutation_code_backend": experiment_manifest.get(
                    "source_mutation_code_backend"
                ),
                "plan_schema_version": experiment_manifest.get(
                    "source_plan_schema_version"
                ),
                "compiler_registry_version": experiment_manifest.get(
                    "source_compiler_registry_version"
                ),
                "paired_route_valid": experiment_manifest.get(
                    "source_paired_route_valid"
                ),
            },
        )
    add(
        "clean_reasoning_evidence_gate_passed",
        experiment_manifest.get("source_evidence_gate_valid") is True,
        {
            "valid": experiment_manifest.get("source_evidence_gate_valid"),
            "issues": experiment_manifest.get("source_evidence_gate_issues"),
        },
    )
    add(
        "manifest_instance_count_matches",
        (experiment_manifest.get("counts") or {}).get("instances")
        == len(instance_rows),
        {
            "declared": (experiment_manifest.get("counts") or {}).get("instances"),
            "observed": len(instance_rows),
        },
    )
    add(
        "manifest_generator_count_matches",
        (experiment_manifest.get("counts") or {}).get("generators")
        == len(generator_rows),
        {
            "declared": (experiment_manifest.get("counts") or {}).get("generators"),
            "observed": len(generator_rows),
        },
    )
    add(
        "representation_input_manifest_present",
        representation_input_path.is_file(),
        str(representation_input_path),
    )
    expected_representation_rows = {
        str(row["item_id"]): representation_input_row(row)
        for row in instance_rows
    }
    observed_representation_rows = {
        str(row.get("sample_id")): dict(row)
        for row in representation_input_rows
        if row.get("sample_id") is not None
    }
    representation_contract_matches = bool(
        len(observed_representation_rows) == len(representation_input_rows)
        and observed_representation_rows == expected_representation_rows
    )
    add(
        "representation_inputs_match_instance_manifest_and_leakage_contract",
        representation_contract_matches,
        {
            "expected_rows": len(expected_representation_rows),
            "observed_rows": len(observed_representation_rows),
        },
    )
    declared_representation_count = (
        experiment_manifest.get("counts") or {}
    ).get("representation_inputs")
    add(
        "manifest_representation_input_count_matches",
        declared_representation_count == len(representation_input_rows),
        {
            "declared": declared_representation_count,
            "observed": len(representation_input_rows),
        },
    )
    recorded_representation_input_hash = artifact.metadata.get(
        "representation_input_sha256"
    )
    actual_representation_input_hash = (
        _sha256_file(representation_input_path)
        if representation_input_path.is_file()
        else None
    )
    add(
        "representation_artifact_input_hash_matches",
        recorded_representation_input_hash is not None
        and recorded_representation_input_hash
        == actual_representation_input_hash,
        {
            "artifact": recorded_representation_input_hash,
            "input_manifest": actual_representation_input_hash,
        },
    )

    expected_checkpoint = experiment_manifest.get("solver_checkpoint_id")
    observed_checkpoint = artifact.metadata.get("checkpoint_source")
    add(
        "frozen_solver_checkpoint_matches",
        expected_checkpoint is not None
        and observed_checkpoint is not None
        and str(expected_checkpoint) == str(observed_checkpoint),
        {"manifest": expected_checkpoint, "representation": observed_checkpoint},
    )
    frozen = config.get("frozen_solver") or {}
    declared_checkpoint_sha256 = experiment_manifest.get(
        "solver_checkpoint_sha256"
    )
    preregistered_checkpoint_sha256 = frozen.get("checkpoint_sha256")
    checkpoint_sha256_source = experiment_manifest.get(
        "solver_checkpoint_sha256_source"
    )
    add(
        "solver_checkpoint_sha256_is_immutable_digest",
        _is_sha256(declared_checkpoint_sha256),
        declared_checkpoint_sha256,
    )
    add(
        "solver_checkpoint_sha256_matches_declared_input",
        _is_sha256(declared_checkpoint_sha256)
        and (
            (
                _is_sha256(preregistered_checkpoint_sha256)
                and str(declared_checkpoint_sha256).lower()
                == str(preregistered_checkpoint_sha256).lower()
            )
            or (
                preregistered_checkpoint_sha256 is None
                and checkpoint_sha256_source
                in {
                    "cli",
                    "merged_prepared_manifests",
                    "normalization_argument_or_source_manifest",
                }
            )
        ),
        {
            "manifest": declared_checkpoint_sha256,
            "preregistered": preregistered_checkpoint_sha256,
            "source": checkpoint_sha256_source,
        },
    )
    add(
        "frozen_solver_matches_preregistered_checkpoint",
        frozen.get("checkpoint") is not None
        and str(frozen.get("checkpoint")) == str(observed_checkpoint),
        {"preregistered": frozen.get("checkpoint"), "observed": observed_checkpoint},
    )
    if archive_artifact is not None:
        archive_checkpoint = archive_artifact.metadata.get("checkpoint_source")
        add(
            "archive_uses_same_frozen_solver_checkpoint",
            archive_checkpoint is not None
            and str(archive_checkpoint) == str(observed_checkpoint),
            {
                "archive": archive_checkpoint,
                "child_representation": observed_checkpoint,
            },
        )
        add(
            "archive_is_labelled_frozen_snapshot",
            archive_kind == "frozen_pi_t_archive_snapshot",
            archive_kind,
        )
    add(
        "representation_eval_mode",
        frozen.get("evaluation_mode") is True
        and artifact.metadata.get("model_mode") == "eval",
        {
            "preregistered": frozen.get("evaluation_mode"),
            "observed": artifact.metadata.get("model_mode"),
        },
    )
    add(
        "representation_attention_implementation_matches",
        frozen.get("attention_implementation") is not None
        and artifact.metadata.get("attention_implementation")
        == frozen.get("attention_implementation"),
        {
            "preregistered": frozen.get("attention_implementation"),
            "observed": artifact.metadata.get("attention_implementation"),
        },
    )
    add(
        "representation_dtype_matches",
        frozen.get("dtype") is not None
        and artifact.metadata.get("dtype") == frozen.get("dtype"),
        {
            "preregistered": frozen.get("dtype"),
            "observed": artifact.metadata.get("dtype"),
        },
    )
    add(
        "representation_cache_disabled",
        artifact.metadata.get("use_cache") is False,
        artifact.metadata.get("use_cache"),
    )
    add(
        "representation_l2_normalized",
        artifact.metadata.get("normalization") == "fp32_l2",
        artifact.metadata.get("normalization"),
    )

    preregistered_primary_key, preregistered_keys = (
        _preregistered_representation_keys(config)
    )
    frozen_layer_count = frozen.get("num_hidden_layers")
    representation_config = config.get("representation") or {}
    explicit_primary_layer = representation_config.get(
        "primary_decoder_block_zero_based"
    )
    rule_primary_layer = (
        (2 * int(frozen_layer_count)) // 3 - 1
        if isinstance(frozen_layer_count, int)
        and not isinstance(frozen_layer_count, bool)
        and frozen_layer_count > 0
        else None
    )
    add(
        "primary_layer_rule_matches_explicit_index",
        rule_primary_layer is not None
        and rule_primary_layer == explicit_primary_layer,
        {
            "num_hidden_layers": frozen_layer_count,
            "rule_result": rule_primary_layer,
            "explicit": explicit_primary_layer,
        },
    )
    missing_representation_arrays = sorted(
        set(preregistered_keys).difference(artifact.arrays)
    )
    add(
        "preregistered_layer_pooling_arrays_present",
        not missing_representation_arrays,
        {"missing": missing_representation_arrays},
    )

    expected_settings = {
        "validity_policy": (config.get("generation_space") or {}).get(
            "validity_policy"
        ),
        "primary_representation_key": preregistered_primary_key,
        "analyzed_representation_keys": preregistered_keys,
        "pca_variance_threshold": (config.get("generation_space") or {}).get(
            "pca_cumulative_variance"
        ),
        "pca_centered": (config.get("generation_space") or {}).get("pca_center"),
        "knn_k": (config.get("generation_space") or {}).get("knn_k"),
        "coverage_quantile": (config.get("generation_space") or {}).get(
            "coverage_quantile"
        ),
        "bootstrap_replicates": (config.get("generation_space") or {}).get(
            "bootstrap_replicates"
        ),
    }
    for name, expected in expected_settings.items():
        observed = actual_settings.get(name)
        add(
            f"preregistered_primary_setting.{name}",
            expected is not None and observed == expected,
            {"preregistered": expected, "observed": observed},
        )

    pair_audit = audit_generator_pair_design(generator_rows)
    add(
        "paired_generator_design_parity",
        pair_audit["passed"],
        {
            "pairs": pair_audit["n_generator_pairs"],
            "passed_pairs": pair_audit["n_passed_pairs"],
        },
    )
    failures = [check["name"] for check in checks if not check["passed"]]
    return {
        "passed": not failures,
        "checks": checks,
        "generator_pair_design": pair_audit,
        "failures": failures,
    }


def _generation_experiment_provenance(
    *,
    experiment_manifest: Mapping[str, Any] | None,
    experiment_manifest_path: Path,
    generator_rows: Sequence[Mapping[str, Any]],
    generator_manifest_path: Path,
    instance_manifest_path: Path,
) -> dict[str, Any]:
    """Freeze the generation artifacts needed by the capability claim.

    A capability report may only be combined with the exact generation run
    that supplied its training data.  Keep both aggregate experiment identity
    and each independent run's immutable source-comparison manifest digest.
    """

    child_rows = [
        row
        for row in generator_rows
        if row.get("condition") in {"plain", "reasoning"}
    ]
    run_ids = sorted(
        {
            str(row["independent_run_id"])
            for row in child_rows
            if row.get("independent_run_id") is not None
        }
    )
    source_hashes_by_run: dict[str, list[str]] = {}
    for run_id in run_ids:
        source_hashes_by_run[run_id] = sorted(
            {
                str(row["source_run_manifest_sha256"]).lower()
                for row in child_rows
                if str(row.get("independent_run_id")) == run_id
                and row.get("source_run_manifest_sha256") is not None
            }
        )
    solver_ids = sorted(
        {
            str(row["solver_checkpoint_id"])
            for row in child_rows
            if row.get("solver_checkpoint_id") is not None
        }
    )
    solver_sha256s = sorted(
        {
            str(row["solver_checkpoint_sha256"]).lower()
            for row in child_rows
            if row.get("solver_checkpoint_sha256") is not None
        }
    )
    manifest_solver = (
        experiment_manifest.get("solver_checkpoint_id")
        if isinstance(experiment_manifest, Mapping)
        else None
    )
    solver_checkpoint_id = (
        str(manifest_solver)
        if manifest_solver is not None
        else (solver_ids[0] if len(solver_ids) == 1 else None)
    )
    manifest_solver_sha256 = (
        experiment_manifest.get("solver_checkpoint_sha256")
        if isinstance(experiment_manifest, Mapping)
        else None
    )
    solver_checkpoint_sha256 = (
        str(manifest_solver_sha256).lower()
        if manifest_solver_sha256 is not None
        else (solver_sha256s[0] if len(solver_sha256s) == 1 else None)
    )
    return {
        "experiment_id": (
            experiment_manifest.get("experiment_id")
            if isinstance(experiment_manifest, Mapping)
            else None
        ),
        "independent_run_ids": run_ids,
        "solver_checkpoint_id": solver_checkpoint_id,
        "observed_solver_checkpoint_ids": solver_ids,
        "solver_checkpoint_sha256": solver_checkpoint_sha256,
        "observed_solver_checkpoint_sha256s": solver_sha256s,
        "source_run_manifest_sha256_by_run": source_hashes_by_run,
        "experiment_manifest_sha256": (
            _sha256_file(experiment_manifest_path)
            if experiment_manifest_path.is_file()
            else None
        ),
        "generator_manifest_sha256": (
            _sha256_file(generator_manifest_path)
            if generator_manifest_path.is_file()
            else None
        ),
        "instance_manifest_sha256": _sha256_file(instance_manifest_path),
    }


def _apply_confirmatory_protocol_gate(
    result: dict[str, Any],
    *,
    mode: str,
    audit: Mapping[str, Any],
) -> None:
    result["confirmatory_protocol_audit"] = dict(audit)
    if mode != "confirmatory" or audit.get("passed"):
        return
    result["inferential_valid_before_protocol_audit"] = bool(
        result.get("inferential_valid")
    )
    result["inferential_valid"] = False
    result["descriptive_only"] = True
    if str(result.get("status", "")).startswith("ok_"):
        result["status"] = "ok_descriptive"
    result.setdefault("warnings", []).append(
        "confirmatory_protocol_gate_failed: " + ", ".join(audit.get("failures", []))
    )


def _plot_generation_primary(
    result: Mapping[str, Any],
    output_path: Path,
) -> None:
    generator_rows = result.get("generator_metrics") or []
    if not generator_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"plain": "#777777", "reasoning": "#2b6cb0"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for condition in ("plain", "reasoning"):
        rows = [row for row in generator_rows if row["condition"] == condition]
        axes[0].scatter(
            [row["aligned_displacement"] for row in rows],
            [row["orthogonal_displacement"] for row in rows],
            label=condition,
            color=colors[condition],
            alpha=0.8,
        )
        axes[1].scatter(
            [condition] * len(rows),
            [row["orthogonal_displacement"] for row in rows],
            color=colors[condition],
            alpha=0.8,
        )
    axes[0].set_xlabel("Aligned displacement A")
    axes[0].set_ylabel("Orthogonal displacement O")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Generator-mean O")
    axes[1].set_xlabel("Condition")
    fig.suptitle(
        "Generation-space diagnostic"
        + (" (descriptive only)" if result.get("descriptive_only") else "")
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _generation_sensitivity_summary(
    result: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    effect = result.get("unadjusted_effect") or {}
    adjusted = result.get("controls_adjusted_effect") or {}
    pca = result.get("pca") or {}
    novelty = result.get("novelty") or {}
    return {
        **dict(settings),
        "status": result.get("status"),
        "inferential_valid": result.get("inferential_valid"),
        "mean_o_difference": effect.get("mean_difference"),
        "o_ci_low": effect.get("ci_low"),
        "o_ci_high": effect.get("ci_high"),
        "adjusted_o_difference": adjusted.get("adjusted_intercept"),
        "adjusted_ci_low": adjusted.get("ci_low"),
        "adjusted_ci_high": adjusted.get("ci_high"),
        "pca_n_components": pca.get("n_components"),
        "pca_cumulative_explained_variance": pca.get(
            "cumulative_explained_variance"
        ),
        "novelty_effective_k": novelty.get("novelty_effective_k"),
        "coverage_epsilon": novelty.get("coverage_epsilon"),
        "warnings": result.get("warnings") or [],
        "claim_eligible": False,
        "note": "Prespecified sensitivity only; never replaces the primary analysis.",
    }


def _write_generation_markdown(
    path: Path,
    report: Mapping[str, Any],
) -> None:
    primary = report["primary"]
    effect = primary.get("unadjusted_effect") or {}
    criteria = report["claim_criteria"]
    warnings = primary.get("warnings") or []
    lines = [
        "# Reasoning-informed expansion: generation-space report",
        "",
        f"- Status: `{primary.get('status')}`",
        f"- Mode: `{primary.get('mode')}`",
        f"- Validity policy: `{primary.get('validity_policy')}`",
        f"- Primary representation: `{primary.get('representation_key')}`",
        f"- Inferentially valid: `{primary.get('inferential_valid')}`",
        (
            "- Mean paired O difference (reasoning - plain): "
            f"`{effect.get('mean_difference')}`"
        ),
        f"- CI: `[{effect.get('ci_low')}, {effect.get('ci_high')}]`",
        (
            "- Limited generation-space claim allowed: "
            f"`{criteria['limited_generation_space_claim_allowed']}`"
        ),
        "",
        "## Claim gates",
        "",
    ]
    for name, passed in criteria.items():
        lines.append(f"- `{name}`: `{passed}`")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "> Orthogonal projection is this experiment's operational "
            "definition. It is not a metric proposed by Manifold Bandits.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_prepare(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    frozen_solver = config["frozen_solver"]
    checkpoint = args.solver_checkpoint or frozen_solver["checkpoint"]
    checkpoint_sha256 = _validated_external_sha256(
        args.solver_checkpoint_sha256
        or frozen_solver.get("checkpoint_sha256"),
        field="solver_checkpoint_sha256",
    )
    manifest = prepare_comparison_manifests(
        args.comparison_root,
        output_dir=args.output_dir,
        parent_program=args.parent_program,
        run_id=args.run_id,
        evolution_iteration=args.evolution_iteration,
        generator_draw_idx=args.generator_draw_idx,
        solver_checkpoint_id=str(checkpoint),
        solver_checkpoint_sha256=checkpoint_sha256,
        generator_checkpoint_id=args.generator_checkpoint,
    )
    manifest["solver_checkpoint_sha256_source"] = (
        "cli" if args.solver_checkpoint_sha256 else "config"
    )
    output = Path(args.output_dir)
    config_copy = output / "preregistered_config.json"
    config_copy.parent.mkdir(parents=True, exist_ok=True)
    config_copy.write_bytes(Path(args.config).resolve().read_bytes())
    manifest["preregistered_config_path"] = str(config_copy.resolve())
    manifest["preregistered_config_sha256"] = _sha256_file(config_copy)
    write_json(output / "experiment_manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output.resolve()),
                "counts": manifest["counts"],
                "sufficiency": manifest["sufficiency"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    source_dirs = [Path(path).resolve() for path in args.inputs]
    if not source_dirs:
        raise ValueError("merge requires at least one input directory")
    manifests = [
        read_json(path / "experiment_manifest.json") for path in source_dirs
    ]
    if not all(isinstance(manifest, dict) for manifest in manifests):
        raise ValueError("every experiment_manifest.json must contain an object")
    checkpoints = {
        str(manifest.get("solver_checkpoint_id"))
        for manifest in manifests
    }
    checkpoint_sha256s = {
        str(manifest.get("solver_checkpoint_sha256")).lower()
        for manifest in manifests
    }
    config_hashes = {
        str(manifest.get("preregistered_config_sha256"))
        for manifest in manifests
    }
    if len(checkpoints) != 1:
        raise ValueError(
            "all merged runs must use the same frozen Solver checkpoint: "
            f"{sorted(checkpoints)}"
        )
    if (
        len(checkpoint_sha256s) != 1
        or not all(_is_sha256(value) for value in checkpoint_sha256s)
    ):
        raise ValueError(
            "all merged runs must provide the same immutable 64-hex frozen "
            f"Solver checkpoint digest: {sorted(checkpoint_sha256s)}"
        )
    if len(config_hashes) != 1:
        raise ValueError(
            "all merged runs must use the same preregistered config: "
            f"{sorted(config_hashes)}"
        )

    def canonical(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    missing = {"__merge_missing_field__": True}

    def consistent_value(field: str, *, normalize=None) -> Any:
        values: list[Any] = []
        for manifest in manifests:
            value = manifest[field] if field in manifest else missing
            values.append(normalize(value) if normalize is not None else value)
        if len({canonical(value) for value in values}) != 1:
            raise ValueError(
                f"merged source manifests disagree on {field}: {values}"
            )
        return None if values[0] == missing else values[0]

    def normalized_prompt_hashes(value: Any) -> Any:
        if value == missing or value is None:
            return value
        if not isinstance(value, list):
            raise ValueError("source_prompt_files must be a list")
        result = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("source_prompt_files entries must be objects")
            digest = item.get("sha256")
            if not isinstance(digest, str) or not _is_sha256(digest.lower()):
                raise ValueError("source prompt entry has no valid sha256")
            path = item.get("path")
            result.append(
                {
                    "name": (
                        "/".join(Path(str(path)).parts[-2:])
                        if path is not None
                        else None
                    ),
                    "sha256": digest.lower(),
                }
            )
        return sorted(result, key=canonical)

    consistent_fields = (
        "schema_version",
        "generator_checkpoint_id",
        "source_comparison_design",
        "source_evidence_gate_valid",
        "source_evidence_gate_issues",
        "source_evidence_gate_version",
        "source_sampling",
        "source_sampling_seed_policy",
        "source_paired_request_seed_policy",
        "source_instance_seed_policy",
        "source_code_contract_version",
        "source_code_contract",
        "source_mutation_code_backend",
        "source_plan_schema_version",
        "source_compiler_registry_version",
        "source_paired_route_valid",
    )
    common = {
        field: consistent_value(field)
        for field in consistent_fields
    }
    prompt_hashes = consistent_value(
        "source_prompt_files",
        normalize=normalized_prompt_hashes,
    )

    generators: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    representation_inputs: list[dict[str, Any]] = []
    warnings: list[str] = []
    source_provenance: list[dict[str, Any]] = []
    for source, manifest in zip(source_dirs, manifests, strict=True):
        source_generators = read_jsonl(source / "generator_manifest.jsonl")
        source_instances = read_jsonl(source / "instance_manifest.jsonl")
        source_rollouts = read_jsonl(source / "rollout_manifest.jsonl")
        source_representations = read_jsonl(
            source / "representation_inputs.jsonl"
        )
        source_hash = manifest.get("source_run_manifest_sha256")
        for row in source_generators:
            expected_fields = {
                "source_run_manifest_sha256": source_hash,
                "llm_seed": manifest.get("source_llm_seed"),
                "generator_sampling_seed_range": manifest.get(
                    "source_llm_seed_range"
                ),
                "sampling_seed_policy": manifest.get(
                    "source_sampling_seed_policy"
                ),
                "instance_seed_policy": manifest.get(
                    "source_instance_seed_policy"
                ),
            }
            for field, expected in expected_fields.items():
                if (
                    expected is not None
                    and canonical(row.get(field)) != canonical(expected)
                ):
                    raise ValueError(
                        f"{source}: generator {row.get('generator_unit_id')} "
                        f"disagrees with source manifest field {field}"
                    )
            if row.get("condition") in {"plain", "reasoning"}:
                expected_policy = manifest.get(
                    "source_paired_request_seed_policy"
                )
                if (
                    expected_policy is not None
                    and row.get("paired_request_seed_policy") != expected_policy
                ):
                    raise ValueError(
                        f"{source}: child generator paired seed policy mismatch"
                    )
                paired = manifest.get("source_paired_request_seeds")
                if isinstance(paired, dict):
                    expected_seeds = paired.get(str(row.get("operator")))
                    if isinstance(expected_seeds, dict):
                        for stage in ("plan", "code"):
                            field = f"{stage}_sampling_seed"
                            if canonical(row.get(field)) != canonical(
                                expected_seeds.get(stage)
                            ):
                                raise ValueError(
                                    f"{source}: child {field} conflicts with "
                                    "paired-request provenance"
                                )
        for row in source_rollouts:
            recorded_hash = row.get("source_run_manifest_sha256")
            if (
                recorded_hash is not None
                and source_hash is not None
                and recorded_hash != source_hash
            ):
                raise ValueError(
                    f"{source}: rollout source manifest hash conflicts"
                )
            row.setdefault("source_run_manifest_sha256", source_hash)
        generators.extend(source_generators)
        instances.extend(source_instances)
        rollouts.extend(source_rollouts)
        representation_inputs.extend(source_representations)
        warnings.extend(str(value) for value in manifest.get("warnings", []))
        source_provenance.append(
            {
                "normalized_source_dir": str(source),
                "normalized_experiment_manifest_sha256": _sha256_file(
                    source / "experiment_manifest.json"
                ),
                "source_run_manifest_path": manifest.get(
                    "source_run_manifest_path"
                ),
                "source_run_manifest_sha256": source_hash,
                "independent_run_id": manifest.get("independent_run_id"),
                "parent_program_id": (
                    (manifest.get("parent") or {}).get("parent_program_id")
                    if isinstance(manifest.get("parent"), dict)
                    else None
                ),
                "evolution_iteration": manifest.get("evolution_iteration"),
                "generator_draw_indices": sorted(
                    {
                        int(row["generator_draw_idx"])
                        for row in source_generators
                        if row.get("condition") in {"plain", "reasoning"}
                        and row.get("generator_draw_idx") is not None
                    }
                ),
                "llm_seed": manifest.get("source_llm_seed"),
                "llm_seed_range": manifest.get("source_llm_seed_range"),
                "paired_request_seeds": manifest.get(
                    "source_paired_request_seeds"
                ),
                "parent_score_telemetry": [
                    {
                        key: row.get(key)
                        for key in (
                            "item_id",
                            "instance_seed",
                            "num_rollouts",
                            "num_correct",
                            "p_hat",
                            "uncertainty_proxy",
                            "rq",
                            "rq_kind",
                        )
                    }
                    for row in source_instances
                    if row.get("role") == "parent"
                ],
            }
        )

    def require_unique(rows, key: str, *, label: str) -> None:
        values = [row.get(key) for row in rows]
        if any(value is None for value in values):
            raise ValueError(f"merged {label} contains a missing {key}")
        duplicates = [
            value
            for value, count in Counter(values).items()
            if count > 1
        ]
        if duplicates:
            raise ValueError(
                f"merged {label} {key} values are not unique; check duplicate input "
                f"directories/runs: {duplicates[:10]}"
            )

    def dedupe_parents(
        rows: list[dict[str, Any]],
        *,
        key: str,
        is_parent,
        ignored: set[str],
        label: str,
    ) -> tuple[list[dict[str, Any]], int]:
        result: list[dict[str, Any]] = []
        seen: dict[str, str] = {}
        removed = 0
        for row in rows:
            if not is_parent(row):
                result.append(row)
                continue
            identity = row.get(key)
            if identity is None:
                raise ValueError(f"parent {label} row is missing {key}")
            semantic = {
                name: value for name, value in row.items() if name not in ignored
            }
            signature = canonical(semantic)
            prior = seen.get(str(identity))
            if prior is None:
                seen[str(identity)] = signature
                result.append(row)
            elif prior == signature:
                removed += 1
            else:
                raise ValueError(
                    f"conflicting parent {label} rows for {key}={identity!r}"
                )
        return result, removed

    generators, removed_parent_generators = dedupe_parents(
        generators,
        key="generator_unit_id",
        is_parent=lambda row: row.get("condition") == "parent",
        ignored={
            "experiment_id",
            "source_run_manifest_sha256",
            "artifact_dir",
            "llm_seed",
            "generator_sampling_seed_range",
        },
        label="generator",
    )
    parent_score_fields = {
        "num_correct",
        "p_hat",
        "uncertainty_proxy",
        "rq",
    }
    parent_score_rows: dict[str, list[dict[str, Any]]] = {}
    for row in instances:
        if row.get("role") == "parent" and row.get("item_id") is not None:
            parent_score_rows.setdefault(str(row["item_id"]), []).append(row)

    instances, removed_parent_instances = dedupe_parents(
        instances,
        key="item_id",
        is_parent=lambda row: row.get("role") == "parent",
        # These are repeated Solver-rollout measurements of the same immutable
        # parent, not parent identity. They are pooled below and retained in
        # source_manifests for draw-level provenance.
        ignored={"source_path", *parent_score_fields},
        label="instance",
    )
    for row in instances:
        if row.get("role") != "parent":
            continue
        score_rows = parent_score_rows.get(str(row["item_id"]), [])
        rollout_counts = [int(item.get("num_rollouts", 0) or 0) for item in score_rows]
        correct_counts = [int(item.get("num_correct", 0) or 0) for item in score_rows]
        total_rollouts = sum(rollout_counts)
        total_correct = sum(correct_counts)
        if total_rollouts <= 0:
            continue
        weighted_uncertainty = sum(
            float(item.get("uncertainty_proxy", 0.0) or 0.0) * count
            for item, count in zip(score_rows, rollout_counts, strict=True)
        ) / total_rollouts
        pooled_p = total_correct / total_rollouts
        row.update(
            {
                "num_rollouts": total_rollouts,
                "num_correct": total_correct,
                "p_hat": pooled_p,
                "uncertainty_proxy": weighted_uncertainty,
                "rq": pooled_p * (1.0 - pooled_p) * weighted_uncertainty,
                "lineage_source": "merged_parent_rollout_pool_v1",
            }
        )
    representation_inputs, removed_parent_representations = dedupe_parents(
        representation_inputs,
        key="sample_id",
        is_parent=lambda row: row.get("role") == "parent",
        ignored=set(),
        label="representation",
    )

    # Only parent identities may collapse. Every child remains an independent,
    # globally unique generator/instance lineage.
    require_unique(generators, "generator_unit_id", label="generator")
    require_unique(instances, "item_id", label="instance")
    require_unique(
        representation_inputs,
        "sample_id",
        label="representation",
    )
    generators_by_id = {
        str(row["generator_unit_id"]): row for row in generators
    }
    instances_by_id = {str(row["item_id"]): row for row in instances}
    for row in instances:
        if str(row.get("generator_unit_id")) not in generators_by_id:
            raise ValueError(
                "instance references unknown generator_unit_id "
                f"{row.get('generator_unit_id')!r}"
            )
    representations_by_id = {
        str(row["sample_id"]): row for row in representation_inputs
    }
    if set(representations_by_id) != set(instances_by_id):
        raise ValueError(
            "representation sample IDs do not exactly match merged instance IDs"
        )
    for item_id, instance in instances_by_id.items():
        if canonical(representations_by_id[item_id]) != canonical(
            representation_input_row(instance)
        ):
            raise ValueError(
                f"representation row conflicts with instance {item_id!r}"
            )

    merged_rollouts: list[dict[str, Any]] = []
    seen_rollouts: dict[tuple[Any, ...], str] = {}
    removed_parent_rollouts = 0
    for row in rollouts:
        instance_id = str(row.get("instance_id"))
        instance = instances_by_id.get(instance_id)
        if instance is None:
            raise ValueError(
                f"rollout references unknown instance_id {instance_id!r}"
            )
        if str(row.get("generator_unit_id")) != str(
            instance.get("generator_unit_id")
        ):
            raise ValueError(
                f"rollout generator lineage conflicts for {instance_id!r}"
            )
        parent = instance.get("role") == "parent"
        lineage = (
            "parent",
            instance_id,
            row.get("source_run_manifest_sha256"),
            row.get("rollout_idx"),
        ) if parent else (
            "child",
            instance_id,
            row.get("rollout_idx"),
        )
        semantic = {
            name: value for name, value in row.items() if name != "source_path"
        }
        signature = canonical(semantic)
        prior = seen_rollouts.get(lineage)
        if prior is None:
            seen_rollouts[lineage] = signature
            merged_rollouts.append(row)
        elif parent and prior == signature:
            removed_parent_rollouts += 1
        elif parent:
            raise ValueError(
                f"conflicting parent rollout lineage {lineage!r}"
            )
        else:
            raise ValueError(
                f"duplicate child rollout lineage {lineage!r}"
            )
    rollouts = merged_rollouts

    sufficiency = audit_generation_sufficiency(generators, instances)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    preregistered_config_path = None
    config_hash = next(iter(config_hashes))
    for source in source_dirs:
        candidate = source / "preregistered_config.json"
        if not candidate.is_file():
            continue
        if _sha256_file(candidate) != config_hash:
            raise ValueError(
                f"{candidate} does not match preregistered_config_sha256"
            )
        target = output / "preregistered_config.json"
        if not target.exists():
            target.write_bytes(candidate.read_bytes())
        elif _sha256_file(target) != _sha256_file(candidate):
            raise ValueError("source preregistration copies conflict")
        preregistered_config_path = str(target.resolve())

    source_hashes = sorted(
        {
            str(manifest["source_run_manifest_sha256"])
            for manifest in manifests
            if manifest.get("source_run_manifest_sha256") is not None
        }
    )
    merged = {
        "schema_version": 1,
        "merge_contract_version": 2,
        "experiment_id": output.name,
        "merged_sources": [str(path) for path in source_dirs],
        "solver_checkpoint_id": next(iter(checkpoints)),
        "solver_checkpoint_sha256": next(iter(checkpoint_sha256s)),
        "solver_checkpoint_sha256_source": "merged_prepared_manifests",
        "generator_checkpoint_id": common["generator_checkpoint_id"],
        "preregistered_config_sha256": config_hash,
        "preregistered_config_path": preregistered_config_path,
        "source_comparison_design": common["source_comparison_design"],
        "source_evidence_gate_valid": common["source_evidence_gate_valid"],
        "source_evidence_gate_issues": common[
            "source_evidence_gate_issues"
        ],
        "source_evidence_gate_version": common[
            "source_evidence_gate_version"
        ],
        "source_prompt_file_hashes": prompt_hashes,
        "source_sampling": common["source_sampling"],
        "source_sampling_seed_policy": common[
            "source_sampling_seed_policy"
        ],
        "source_paired_request_seed_policy": common[
            "source_paired_request_seed_policy"
        ],
        "source_instance_seed_policy": common[
            "source_instance_seed_policy"
        ],
        "source_code_contract_version": common[
            "source_code_contract_version"
        ],
        "source_code_contract": common["source_code_contract"],
        "source_mutation_code_backend": common[
            "source_mutation_code_backend"
        ],
        "source_plan_schema_version": common[
            "source_plan_schema_version"
        ],
        "source_compiler_registry_version": common[
            "source_compiler_registry_version"
        ],
        "source_paired_route_valid": common[
            "source_paired_route_valid"
        ],
        "source_run_manifest_sha256s": source_hashes,
        "source_normalized_experiment_manifest_sha256s": sorted(
            row["normalized_experiment_manifest_sha256"]
            for row in source_provenance
        ),
        "source_manifests": source_provenance,
        "sampling_provenance": {
            "sampling_seed_policy": common["source_sampling_seed_policy"],
            "paired_request_seed_policy": common[
                "source_paired_request_seed_policy"
            ],
            "instance_seed_policy": common["source_instance_seed_policy"],
            "source_draws": [
                {
                    key: row.get(key)
                    for key in (
                        "normalized_experiment_manifest_sha256",
                        "source_run_manifest_sha256",
                        "llm_seed",
                        "llm_seed_range",
                        "paired_request_seeds",
                        "generator_draw_indices",
                    )
                }
                for row in source_provenance
            ],
        },
        "counts": {
            "generators": len(generators),
            "instances": len(instances),
            "rollouts": len(rollouts),
            "representation_inputs": len(representation_inputs),
        },
        "deduplication": {
            "parent_generators_removed": removed_parent_generators,
            "parent_instances_removed": removed_parent_instances,
            "parent_representation_inputs_removed": (
                removed_parent_representations
            ),
            "identical_parent_rollouts_removed": removed_parent_rollouts,
            "child_rows_removed": 0,
        },
        "sufficiency": sufficiency,
        "warnings": list(dict.fromkeys([*warnings, *sufficiency["warnings"]])),
    }
    write_json(output / "experiment_manifest.json", merged)
    write_jsonl(output / "generator_manifest.jsonl", generators)
    write_jsonl(output / "instance_manifest.jsonl", instances)
    write_jsonl(output / "rollout_manifest.jsonl", rollouts)
    write_jsonl(output / "representation_inputs.jsonl", representation_inputs)
    print(json.dumps(merged, ensure_ascii=False, indent=2))
    return 0


def cmd_prepare_archive(args: argparse.Namespace) -> int:
    source_rows = read_jsonl(args.archive_jsonl)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(source_rows):
        sample_id = source.get("sample_id") or source.get("item_id") or source.get(
            "problem_id"
        )
        problem = source.get("problem_text") or source.get("problem")
        if sample_id is None or not str(sample_id).strip():
            raise ValueError(f"archive row {index} has no stable sample ID")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"archive row {index} has no problem text")
        sample_id = str(sample_id)
        if sample_id in seen:
            raise ValueError(f"duplicate archive sample_id: {sample_id}")
        seen.add(sample_id)
        rows.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "role": "archive",
                "independent_run_id": str(
                    source.get("independent_run_id", "archive")
                ),
                "parent_program_id": str(
                    source.get("parent_program_id", "archive")
                ),
                "generator_unit_id": str(
                    source.get("archive_generator_id", "unknown_archive_generator")
                ),
                "instance_seed": source.get("instance_seed", index),
                "common_io_spec": SOLVER_SYSTEM_PROMPT,
                "problem_text": problem,
            }
        )
    write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "archive_inputs": len(rows),
                "output": str(Path(args.output).resolve()),
                "leakage_fields_removed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.representation_inputs)
    if not rows:
        raise ValueError("representation input manifest is empty")
    for index, row in enumerate(rows):
        if row.get("common_io_spec") != SOLVER_SYSTEM_PROMPT:
            raise ValueError(
                f"row {index} common_io_spec differs from SOLVER_SYSTEM_PROMPT"
            )
        forbidden = {
            "answer",
            "target_reasoning_move",
            "source_code",
            "evaluator_reason",
            "rq",
            "mutation_plan",
        }
        leaked = sorted(forbidden.intersection(row))
        if leaked:
            raise ValueError(
                f"row {index} violates representation input contract: {leaked}"
            )

    extractor = HFSelectedLayerExtractor.from_pretrained(
        args.model,
        tokenizer_source=args.tokenizer,
        dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    artifact = extractor.extract(
        [str(row["problem_text"]) for row in rows],
        batch_size=args.batch_size,
        max_length=args.max_prompt_tokens,
    )
    artifact.metadata.update(
        {
            "sample_ids": [str(row["sample_id"]) for row in rows],
            "representation_input_path": str(
                Path(args.representation_inputs).resolve()
            ),
            "representation_input_sha256": _sha256_file(
                args.representation_inputs
            ),
        }
    )
    npz_path, json_path = save_representation_artifact(
        args.output,
        artifact,
        extractor=extractor,
    )
    print(
        json.dumps(
            {
                "representations": str(npz_path.resolve()),
                "metadata": str(json_path.resolve()),
                "arrays": {
                    key: list(value.shape)
                    for key, value in artifact.arrays.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_analyze_generation(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    generation_config = config["generation_space"]
    instances = read_jsonl(args.instance_manifest)
    manifest_path = Path(
        args.experiment_manifest
        or Path(args.instance_manifest).with_name("experiment_manifest.json")
    ).resolve()
    generator_manifest_path = Path(
        args.generator_manifest
        or Path(args.instance_manifest).with_name("generator_manifest.jsonl")
    ).resolve()
    representation_input_path = Path(
        args.representation_input_manifest
        or Path(args.instance_manifest).with_name("representation_inputs.jsonl")
    ).resolve()
    experiment_manifest = (
        read_json(manifest_path) if manifest_path.exists() else None
    )
    generators = (
        read_jsonl(generator_manifest_path)
        if generator_manifest_path.exists()
        else []
    )
    representation_inputs = (
        read_jsonl(representation_input_path)
        if representation_input_path.exists()
        else []
    )
    artifact = load_representation_artifact(args.representations)
    sample_ids = _sample_ids(artifact)

    if args.archive_representations:
        archive = load_representation_artifact(args.archive_representations)
        archive_ids = _sample_ids(archive)
        archive_kind = args.archive_kind or "frozen_solver_archive_snapshot"
    elif args.use_parent_surrogate_archive:
        archive, archive_ids = _parent_surrogate_archive(
            artifact,
            sample_ids,
            instances,
        )
        archive_kind = "parent_instances_surrogate_not_archive_snapshot"
    else:
        archive = None
        archive_ids = None
        archive_kind = None

    preregistered_primary_key, preregistered_keys = (
        _preregistered_representation_keys(config)
    )
    primary_key = args.representation_key or preregistered_primary_key
    pca_variance_threshold = (
        float(args.pca_variance_threshold)
        if args.pca_variance_threshold is not None
        else float(generation_config["pca_cumulative_variance"])
    )
    pca_centered = (
        bool(args.pca_centered)
        if args.pca_centered is not None
        else bool(generation_config["pca_center"])
    )
    knn_k = (
        int(args.knn_k)
        if args.knn_k is not None
        else int(generation_config["knn_k"])
    )
    coverage_quantile = (
        float(args.coverage_quantile)
        if args.coverage_quantile is not None
        else float(generation_config["coverage_quantile"])
    )
    bootstrap_replicates = (
        int(args.bootstrap_replicates)
        if args.bootstrap_replicates is not None
        else int(generation_config["bootstrap_replicates"])
    )
    keys = (
        sorted(artifact.arrays)
        if args.all_representations
        else [primary_key]
    )
    if primary_key not in keys:
        keys.insert(0, primary_key)
    actual_settings = {
        "validity_policy": args.validity_policy,
        "primary_representation_key": primary_key,
        "analyzed_representation_keys": keys,
        "pca_variance_threshold": pca_variance_threshold,
        "pca_centered": pca_centered,
        "knn_k": knn_k,
        "coverage_quantile": coverage_quantile,
        "bootstrap_replicates": bootstrap_replicates,
    }
    protocol_audit = _confirmatory_protocol_audit(
        config=config,
        config_path=args.config,
        experiment_manifest=experiment_manifest,
        experiment_manifest_path=manifest_path,
        generator_rows=generators,
        instance_rows=instances,
        representation_input_rows=representation_inputs,
        representation_input_path=representation_input_path,
        artifact=artifact,
        archive_artifact=archive,
        archive_kind=archive_kind,
        actual_settings=actual_settings,
    )
    analyses: dict[str, Any] = {}
    for key in keys:
        analyses[key] = analyze_generation_space(
            instances,
            representation_artifact=artifact,
            representation_sample_ids=sample_ids,
            representation_key=key,
            mode=args.mode,
            validity_policy=args.validity_policy,
            calibration_fraction=float(
                generation_config["calibration_fraction"]
            ),
            split_seed=int(generation_config["split_seed"]),
            pca_variance_threshold=pca_variance_threshold,
            pca_centered=pca_centered,
            archive_artifact=archive,
            archive_sample_ids=archive_ids,
            archive_representation_key=key if archive is not None else None,
            archive_kind=archive_kind,
            knn_k=knn_k,
            coverage_quantile=coverage_quantile,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=int(generation_config["bootstrap_seed"]),
            ridge_alpha=float(config["controls"]["ridge_alpha"]),
            minimum_independent_runs=int(
                config["inference_gates"]["minimum_independent_runs"]
            ),
            minimum_evaluation_parents=int(
                config["inference_gates"]["minimum_heldout_parents"]
            ),
            minimum_paired_generators_per_parent=int(
                config["inference_gates"][
                    "minimum_paired_generators_per_parent"
                ]
            ),
        )
        _apply_confirmatory_protocol_gate(
            analyses[key],
            mode=args.mode,
            audit=protocol_audit,
        )

    primary = analyses[primary_key]
    robustness = [analyses[key] for key in keys]
    sensitivity_rows: list[dict[str, Any]] = []
    if args.run_sensitivity:
        sensitivity = config.get("sensitivity") or {}
        thresholds = list(
            dict.fromkeys(
                float(value)
                for value in sensitivity.get(
                    "pca_cumulative_variance",
                    [pca_variance_threshold],
                )
            )
        )
        centerings = list(
            dict.fromkeys(
                bool(value)
                for value in sensitivity.get("pca_center", [pca_centered])
            )
        )
        if archive is None:
            knn_values = [knn_k]
            coverage_values = [coverage_quantile]
        else:
            knn_values = list(
                dict.fromkeys(
                    int(value)
                    for value in sensitivity.get("knn_k", [knn_k])
                )
            )
            coverage_values = list(
                dict.fromkeys(
                    float(value)
                    for value in sensitivity.get(
                        "coverage_quantile",
                        [coverage_quantile],
                    )
                )
            )
        for threshold, centered, sensitivity_k, sensitivity_quantile in product(
            thresholds,
            centerings,
            knn_values,
            coverage_values,
        ):
            settings = {
                "representation_key": primary_key,
                "pca_variance_threshold": threshold,
                "pca_centered": centered,
                "knn_k": sensitivity_k,
                "coverage_quantile": sensitivity_quantile,
                "distance": generation_config["distance"],
            }
            sensitivity_result = analyze_generation_space(
                instances,
                representation_artifact=artifact,
                representation_sample_ids=sample_ids,
                representation_key=primary_key,
                mode=args.mode,
                validity_policy=args.validity_policy,
                calibration_fraction=float(
                    generation_config["calibration_fraction"]
                ),
                split_seed=int(generation_config["split_seed"]),
                pca_variance_threshold=threshold,
                pca_centered=centered,
                archive_artifact=archive,
                archive_sample_ids=archive_ids,
                archive_representation_key=(
                    primary_key if archive is not None else None
                ),
                archive_kind=archive_kind,
                knn_k=sensitivity_k,
                coverage_quantile=sensitivity_quantile,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=int(generation_config["bootstrap_seed"]),
                ridge_alpha=float(config["controls"]["ridge_alpha"]),
                minimum_independent_runs=int(
                    config["inference_gates"]["minimum_independent_runs"]
                ),
                minimum_evaluation_parents=int(
                    config["inference_gates"]["minimum_heldout_parents"]
                ),
                minimum_paired_generators_per_parent=int(
                    config["inference_gates"][
                        "minimum_paired_generators_per_parent"
                    ]
                ),
            )
            sensitivity_rows.append(
                _generation_sensitivity_summary(
                    sensitivity_result,
                    settings=settings,
                )
            )
    experiment_provenance = _generation_experiment_provenance(
        experiment_manifest=experiment_manifest,
        experiment_manifest_path=manifest_path,
        generator_rows=generators,
        generator_manifest_path=generator_manifest_path,
        instance_manifest_path=Path(args.instance_manifest).resolve(),
    )
    report = {
        "schema_version": 1,
        "analysis_kind": "generation_space",
        "experiment_provenance": experiment_provenance,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": _sha256_file(args.config),
        "experiment_manifest_path": str(manifest_path),
        "generator_manifest_path": str(generator_manifest_path),
        "representation_input_manifest_path": str(representation_input_path),
        "confirmatory_protocol_audit": protocol_audit,
        "actual_primary_settings": actual_settings,
        "instance_manifest_path": str(Path(args.instance_manifest).resolve()),
        "instance_manifest_sha256": _sha256_file(args.instance_manifest),
        "representation_metadata": artifact.metadata,
        "primary_representation_key": primary_key,
        "primary": primary,
        "robustness": analyses,
        "sensitivity_analysis": {
            "status": "completed" if sensitivity_rows else "not_run",
            "primary_claim_uses_sensitivity": False,
            "rows": sensitivity_rows,
        },
        "claim_criteria": _generation_claim_criteria(
            primary,
            robustness,
            protocol_audit,
        ),
        "capability_space_claim_allowed": False,
        "capability_note": (
            "Capability expansion requires equal-compute training plus "
            "independent held-out transfer and is never inferred from this report."
        ),
        "internal_trajectory_analysis": {
            "status": "not_run",
            "reason": (
                "The comparison artifacts contain generated text/token IDs but "
                "not per-token, per-layer response hidden states."
            ),
            "role": (
                "Optional StALT/internal-dynamics diagnostic only; never a "
                "semantic novelty or capability-expansion endpoint."
            ),
            "implementation": "rq_evolve.expansion_trajectory",
        },
    }
    output = Path(args.output_dir)
    write_json(output / "generation_report.json", report)
    _write_generation_markdown(output / "GENERATION_REPORT.md", report)
    write_jsonl(output / "instance_metrics.jsonl", primary.get("instance_metrics", []))
    write_jsonl(output / "generator_metrics.jsonl", primary.get("generator_metrics", []))
    write_jsonl(output / "paired_metrics.jsonl", primary.get("paired_metrics", []))
    _write_csv(output / "generator_metrics.csv", primary.get("generator_metrics", []))
    _write_csv(output / "paired_metrics.csv", primary.get("paired_metrics", []))
    _write_csv(output / "sensitivity_summary.csv", sensitivity_rows)
    write_jsonl(output / "sensitivity_summary.jsonl", sensitivity_rows)
    _plot_generation_primary(primary, output / "generation_primary.png")
    print(
        json.dumps(
            {
                "status": primary.get("status"),
                "inferential_valid": primary.get("inferential_valid"),
                "claim_criteria": report["claim_criteria"],
                "output_dir": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _tokenize_training_rows(
    rows: list[dict[str, Any]],
    *,
    tokenizer_source: str,
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    result: list[dict[str, Any]] = []
    for row in rows:
        messages = [
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": row["problem_text"]},
        ]
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        result.append(
            {
                "sample_id": row["item_id"],
                "problem": row["problem_text"],
                "answer": row["answer"],
                "program_id": row["generator_unit_id"],
                "generator_unit_id": row["generator_unit_id"],
                "seed": row["instance_seed"],
                "independent_run_id": row["independent_run_id"],
                "parent_program_id": row["parent_program_id"],
                "generator_pair_id": row["generator_pair_id"],
                "target_reasoning_move": row.get("target_reasoning_move"),
                "prompt_token_count": len(ids),
            }
        )
    return result


def cmd_prepare_training(args: argparse.Namespace) -> int:
    instances = read_jsonl(args.instance_manifest)
    selected = select_child_instances(
        instances,
        validity_policy=args.validity_policy,
    )
    output = Path(args.output_dir)
    condition_rows: dict[str, list[dict[str, Any]]] = {}
    for condition in ("plain", "reasoning"):
        source = [row for row in selected if row["condition"] == condition]
        condition_rows[condition] = _tokenize_training_rows(
            source,
            tokenizer_source=args.tokenizer,
        )
        write_jsonl(
            output / f"training_{condition}.jsonl",
            condition_rows[condition],
        )
    audit = audit_paired_training_jsonl(
        output / "training_plain.jsonl",
        output / "training_reasoning.jsonl",
        token_count_fields=("prompt_token_count",),
    )
    report = {
        "schema_version": 1,
        "validity_policy": args.validity_policy,
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "static_training_only": True,
        "budget_audit": audit,
        "training_allowed": bool(
            args.validity_policy == "strict" and audit["budget_equal"]
        ),
        "note": (
            "This audit checks fixed accepted rows and observed prompt tokens. "
            "The training compute manifest must additionally match optimizer, "
            "learning rate, update steps, batch composition, verifier, and "
            "maximum rollout length."
        ),
    }
    write_json(output / "training_data_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_audit_training(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {"schema_version": 1}
    if args.compute_manifest:
        report["compute_manifest"] = audit_same_compute_manifest(
            read_json(args.compute_manifest)
        )
    if args.plain_jsonl or args.reasoning_jsonl:
        if not args.plain_jsonl or not args.reasoning_jsonl:
            raise ValueError(
                "--plain-jsonl and --reasoning-jsonl must be supplied together"
            )
        report["data_budget"] = audit_paired_training_jsonl(
            args.plain_jsonl,
            args.reasoning_jsonl,
            token_count_fields=tuple(args.token_count_field),
        )
    if len(report) == 1:
        raise ValueError("provide a compute manifest and/or paired training JSONLs")
    checks = [
        value.get("passed", value.get("budget_equal", False))
        for key, value in report.items()
        if key != "schema_version"
    ]
    report["training_allowed"] = bool(checks and all(checks))
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_evaluate_heldout(args: argparse.Namespace) -> int:
    heldout_path = Path(args.heldout).resolve()
    heldout_input_sha256 = _sha256_file(heldout_path)
    heldout = load_heldout_jsonl(heldout_path, mode=args.mode)
    rows = evaluate_checkpoint_vllm(
        heldout,
        checkpoint=args.checkpoint,
        condition=args.condition,
        checkpoint_id=args.checkpoint_id,
        checkpoint_run_id=args.checkpoint_run_id,
        checkpoint_provenance=args.checkpoint_provenance,
        tokenizer=args.tokenizer,
        tokenizer_id=args.tokenizer_id,
        heldout_input_sha256=heldout_input_sha256,
        evaluation_mode=args.mode,
        n=args.rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        sampler_backend=args.vllm_sampler_backend,
        use_tqdm=not args.no_progress,
    )
    write_capability_jsonl(args.output, rows)
    output_path = Path(args.output).resolve()
    evaluation_manifest_path = (
        Path(args.evaluation_manifest).resolve()
        if args.evaluation_manifest
        else Path(f"{output_path}.manifest.json")
    )
    problem_hashes = [
        {
            "independent_run_id": row["independent_run_id"],
            "problem_id": row["problem_id"],
            "problem_text_sha256": row["problem_text_sha256"],
            "answer_sha256": row["answer_sha256"],
            "problem_contract_sha256": row["problem_contract_sha256"],
        }
        for row in rows[:: max(1, args.rollouts)]
    ]
    evaluation_manifest = {
        "schema_version": 2,
        "evaluation_mode": args.mode,
        "condition": args.condition,
        "checkpoint_id": rows[0]["checkpoint_id"] if rows else args.checkpoint_id,
        "checkpoint_run_id": (
            rows[0]["checkpoint_run_id"] if rows else args.checkpoint_run_id
        ),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_provenance": args.checkpoint_provenance,
        "heldout_input_path": str(heldout_path),
        "heldout_input_sha256": heldout_input_sha256,
        "heldout_content_sha256": (
            rows[0]["heldout_content_sha256"] if rows else None
        ),
        "decoding_parameters": (
            rows[0]["decoding_parameters"] if rows else None
        ),
        "evaluation_contract_sha256": (
            rows[0]["evaluation_contract_sha256"] if rows else None
        ),
        "problem_hashes": problem_hashes,
        "result_path": str(output_path),
        "result_sha256": _sha256_file(output_path),
        "problem_count": len(heldout),
        "rollout_count": len(rows),
    }
    write_json(evaluation_manifest_path, evaluation_manifest)
    summary = {
        "condition": args.condition,
        "checkpoint": str(args.checkpoint),
        "problems": len(heldout),
        "rollouts": len(rows),
        "accuracy": (
            sum(bool(row["correct"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "heldout_input_sha256": heldout_input_sha256,
        "checkpoint_run_id": evaluation_manifest["checkpoint_run_id"],
        "output": str(output_path),
        "evaluation_manifest": str(evaluation_manifest_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _audit_generation_capability_linkage(
    *,
    generation_report: Mapping[str, Any] | None,
    generation_report_sha256: str | None,
    evaluation_rows: Sequence[Mapping[str, Any]],
    compute_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require one immutable lineage across generation, training, and eval."""

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    provenance = (
        generation_report.get("experiment_provenance")
        if isinstance(generation_report, Mapping)
        else None
    )
    add(
        "generation_experiment_provenance_present",
        isinstance(provenance, Mapping),
        provenance,
    )
    if not isinstance(provenance, Mapping):
        return {
            "schema_version": 1,
            "passed": False,
            "checks": checks,
            "failures": [
                check["name"] for check in checks if not check["passed"]
            ],
        }

    generation_runs = {
        str(value)
        for value in provenance.get("independent_run_ids", [])
    }
    evaluation_runs = {
        str(row["independent_run_id"])
        for row in evaluation_rows
        if row.get("independent_run_id") is not None
    }
    compute_runs = {
        str(value)
        for value in (
            compute_audit.get("compute_manifest_runs", [])
            if isinstance(compute_audit, Mapping)
            else []
        )
    }
    add(
        "independent_run_ids_match_exactly",
        bool(generation_runs)
        and generation_runs == evaluation_runs == compute_runs,
        {
            "generation": sorted(generation_runs),
            "evaluation": sorted(evaluation_runs),
            "training_manifests": sorted(compute_runs),
        },
    )

    solver_checkpoint = provenance.get("solver_checkpoint_id")
    observed_solver_checkpoints = {
        str(value)
        for value in provenance.get("observed_solver_checkpoint_ids", [])
    }
    add(
        "generation_solver_checkpoint_is_unambiguous",
        solver_checkpoint is not None
        and observed_solver_checkpoints == {str(solver_checkpoint)},
        {
            "declared": solver_checkpoint,
            "observed": sorted(observed_solver_checkpoints),
        },
    )
    solver_checkpoint_sha256 = provenance.get("solver_checkpoint_sha256")
    observed_solver_sha256s = {
        str(value).lower()
        for value in provenance.get(
            "observed_solver_checkpoint_sha256s",
            [],
        )
    }
    add(
        "generation_solver_checkpoint_sha256_is_unambiguous",
        _is_sha256(solver_checkpoint_sha256)
        and observed_solver_sha256s
        == {str(solver_checkpoint_sha256).lower()},
        {
            "declared": solver_checkpoint_sha256,
            "observed": sorted(observed_solver_sha256s),
        },
    )

    expected_hash_fields = {
        "generation_report_sha256": generation_report_sha256,
        "generation_experiment_manifest_sha256": provenance.get(
            "experiment_manifest_sha256"
        ),
        "generation_generator_manifest_sha256": provenance.get(
            "generator_manifest_sha256"
        ),
        "generation_instance_manifest_sha256": provenance.get(
            "instance_manifest_sha256"
        ),
    }
    source_by_run = provenance.get("source_run_manifest_sha256_by_run")
    if not isinstance(source_by_run, Mapping):
        source_by_run = {}
    per_run_reports = (
        compute_audit.get("per_run", [])
        if isinstance(compute_audit, Mapping)
        else []
    )
    observed_compute_reports = {
        str(report.get("independent_run_id")): report
        for report in per_run_reports
        if isinstance(report, Mapping)
        and report.get("independent_run_id") is not None
    }

    def normalized_hashes(value: Any) -> list[str]:
        values = value if isinstance(value, list) else [value]
        return sorted(
            {
                str(item).lower()
                for item in values
                if isinstance(item, str) and item
            }
        )

    for run_id in sorted(generation_runs):
        compute_report = observed_compute_reports.get(run_id)
        add(
            f"training_manifest_present.{run_id}",
            compute_report is not None,
            compute_report.get("path") if compute_report else None,
        )
        if compute_report is None:
            continue
        base_checkpoints = {
            str(value)
            for value in compute_report.get("training_base_checkpoints", [])
        }
        add(
            f"base_checkpoint_matches_generation_solver.{run_id}",
            solver_checkpoint is not None
            and base_checkpoints == {str(solver_checkpoint)},
            {
                "generation": solver_checkpoint,
                "training": sorted(base_checkpoints),
            },
        )
        base_checkpoint_sha256s = {
            str(value).lower()
            for value in compute_report.get(
                "training_base_checkpoint_sha256s",
                [],
            )
        }
        add(
            f"base_checkpoint_sha256_matches_generation_solver.{run_id}",
            _is_sha256(solver_checkpoint_sha256)
            and base_checkpoint_sha256s
            == {str(solver_checkpoint_sha256).lower()},
            {
                "generation": solver_checkpoint_sha256,
                "training": sorted(base_checkpoint_sha256s),
            },
        )
        condition_base_sha256s = compute_report.get(
            "training_base_checkpoint_sha256_by_condition",
            {},
        )
        add(
            f"each_condition_base_checkpoint_sha256_matches.{run_id}",
            isinstance(condition_base_sha256s, Mapping)
            and set(condition_base_sha256s) == {"plain", "reasoning"}
            and all(
                _is_sha256(condition_base_sha256s.get(condition))
                and str(condition_base_sha256s[condition]).lower()
                == str(solver_checkpoint_sha256).lower()
                for condition in ("plain", "reasoning")
            ),
            condition_base_sha256s,
        )
        linkage = compute_report.get("generation_linkage")
        add(
            f"training_manifest_generation_linkage_present.{run_id}",
            isinstance(linkage, Mapping),
            linkage,
        )
        if not isinstance(linkage, Mapping):
            continue
        add(
            f"training_manifest_run_id_matches.{run_id}",
            str(linkage.get("independent_run_id")) == run_id,
            linkage.get("independent_run_id"),
        )
        add(
            f"training_manifest_solver_checkpoint_matches.{run_id}",
            solver_checkpoint is not None
            and str(linkage.get("solver_checkpoint_id"))
            == str(solver_checkpoint),
            linkage.get("solver_checkpoint_id"),
        )
        add(
            f"training_manifest_solver_checkpoint_sha256_matches.{run_id}",
            _is_sha256(solver_checkpoint_sha256)
            and str(linkage.get("solver_checkpoint_sha256")).lower()
            == str(solver_checkpoint_sha256).lower(),
            linkage.get("solver_checkpoint_sha256"),
        )
        for field, expected in expected_hash_fields.items():
            add(
                f"training_manifest_{field}_matches.{run_id}",
                expected is not None
                and str(linkage.get(field)).lower() == str(expected).lower(),
                {
                    "expected": expected,
                    "observed": linkage.get(field),
                },
            )
        expected_source_hashes = normalized_hashes(source_by_run.get(run_id))
        observed_source_hashes = normalized_hashes(
            linkage.get("source_run_manifest_sha256")
        )
        add(
            f"training_manifest_source_run_hashes_match.{run_id}",
            bool(expected_source_hashes)
            and observed_source_hashes == expected_source_hashes,
            {
                "expected": expected_source_hashes,
                "observed": observed_source_hashes,
            },
        )

    failures = [check["name"] for check in checks if not check["passed"]]
    return {
        "schema_version": 1,
        "passed": not failures,
        "generation_runs": sorted(generation_runs),
        "evaluation_runs": sorted(evaluation_runs),
        "training_manifest_runs": sorted(compute_runs),
        "checks": checks,
        "failures": failures,
    }


def _capability_claim_criteria(
    did: Mapping[str, Any],
    *,
    forgetting_did: Mapping[str, Any] | None,
    compute_audit: Mapping[str, Any] | None,
    evaluation_audit: Mapping[str, Any],
    disjointness_audit: Mapping[str, Any],
    generation_report: Mapping[str, Any] | None,
    generation_linkage_audit: Mapping[str, Any],
    max_forgetting: float,
    minimum_independent_runs: int,
) -> dict[str, Any]:
    by_level = did.get("by_transfer_level") or {}
    in_family = by_level.get("in_family")
    structural = by_level.get("structural")
    cross_domain = by_level.get("cross_domain")
    transfer = [value for value in (structural, cross_domain) if value]
    forgetting_by_level = (
        (forgetting_did or {}).get("by_transfer_level") or {}
    )
    forgetting = [
        forgetting_by_level[level]
        for level in ("archive", "benchmark")
        if level in forgetting_by_level
    ]
    move_results = list(
        (did.get("by_target_reasoning_move") or {}).values()
    )

    def positive_with_ci(result: Mapping[str, Any] | None) -> bool:
        return bool(
            result
            and result.get("inferential_valid")
            and result.get("ci_low") is not None
            and result["ci_low"] > 0
        )

    generation_criteria = (
        generation_report.get("claim_criteria", {})
        if isinstance(generation_report, Mapping)
        else {}
    )
    capability_gates = {
        "evaluation_provenance_and_pairing_passed": bool(
            evaluation_audit.get("passed")
        ),
        "training_heldout_disjointness_passed": bool(
            disjointness_audit.get("passed")
        ),
        "all_condition_units_complete": bool(
            did.get("all_condition_units_complete")
            and not did.get("incomplete_units")
        ),
        "same_compute_audit_passed": bool(
            compute_audit and compute_audit.get("passed")
        ),
        "compute_artifact_provenance_complete": bool(
            compute_audit
            and compute_audit.get("artifact_provenance_complete")
        ),
        "independent_run_reproduction_gate": bool(
            did["overall"].get("inferential_valid")
            and did["overall"].get("n_runs", 0) >= minimum_independent_runs
        ),
        "overall_delta_cap_positive_with_ci": positive_with_ci(did["overall"]),
        "in_family_transfer_positive_with_ci": positive_with_ci(in_family),
        "structural_or_cross_domain_transfer_positive_with_ci": bool(
            transfer
            and all(positive_with_ci(result) for result in transfer)
        ),
        "target_reasoning_move_effects_positive_with_ci": bool(
            move_results
            and all(positive_with_ci(result) for result in move_results)
        ),
        "no_serious_forgetting": bool(
            forgetting
            and all(
                result.get("inferential_valid")
                and result.get("reasoning_gain_ci_low") is not None
                and result["reasoning_gain_ci_low"] >= -abs(max_forgetting)
                for result in forgetting
            )
        ),
    }
    capability_evidence_allowed = all(capability_gates.values())
    generation_gates = {
        "generation_report_supplied": isinstance(generation_report, Mapping),
        "generation_capability_provenance_linked": bool(
            generation_linkage_audit.get("passed")
        ),
        "generation_space_claim_gates_passed": bool(
            generation_criteria.get("limited_generation_space_claim_allowed")
        ),
        "generation_layer_robustness_passed": bool(
            generation_criteria.get("layer_pooling_direction_consistent")
        ),
        "generation_independent_reproduction_passed": bool(
            generation_criteria.get("independent_runs_and_parents_gate")
        ),
    }
    strong_claim_allowed = bool(
        capability_evidence_allowed and all(generation_gates.values())
    )
    criteria = {
        **capability_gates,
        "structural_transfer_positive_with_ci": positive_with_ci(structural),
        "cross_domain_transfer_positive_with_ci": positive_with_ci(cross_domain),
        **generation_gates,
        "capability_evidence_gates_passed": capability_evidence_allowed,
        "strong_capability_expansion_claim_allowed": strong_claim_allowed,
        # Backward-compatible key now deliberately means the final combined
        # claim, not capability-only evidence.
        "capability_space_claim_allowed": strong_claim_allowed,
    }
    return criteria


def _audit_capability_compute_manifests(
    manifest_paths: Sequence[str],
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Link each independent run's immutable training artifacts to evaluation."""

    evaluation_runs = {
        str(row.get("independent_run_id"))
        for row in evaluation_rows
        if row.get("independent_run_id") is not None
    }
    observed_checkpoints: dict[tuple[str, str], dict[str, set[str]]] = {}
    for row in evaluation_rows:
        condition = str(row.get("condition", ""))
        run_id = str(row.get("independent_run_id", ""))
        group = observed_checkpoints.setdefault(
            (condition, run_id),
            {"ids": set(), "provenance": set()},
        )
        if row.get("checkpoint_id") is not None:
            group["ids"].add(str(row["checkpoint_id"]))
        if row.get("checkpoint_provenance") is not None:
            group["provenance"].add(str(row["checkpoint_provenance"]))

    per_run: list[dict[str, Any]] = []
    training_rows: dict[str, list[dict[str, Any]]] = {
        "plain": [],
        "reasoning": [],
    }
    seen_runs: set[str] = set()
    for manifest_value in manifest_paths:
        manifest_path = Path(manifest_value).resolve()
        manifest = read_json(manifest_path)
        audit = audit_same_compute_manifest(manifest)
        conditions = (
            manifest.get("conditions")
            if isinstance(manifest, Mapping)
            else None
        )
        plain = (
            conditions.get("plain") if isinstance(conditions, Mapping) else None
        )
        reasoning = (
            conditions.get("reasoning")
            if isinstance(conditions, Mapping)
            else None
        )
        run_id = (
            str(plain.get("independent_run_id"))
            if isinstance(plain, Mapping)
            and plain.get("independent_run_id") is not None
            else ""
        )
        issues: list[dict[str, Any]] = []
        if not run_id:
            issues.append({"issue": "manifest has no independent_run_id"})
        elif run_id in seen_runs:
            issues.append({"issue": "duplicate compute manifest for run"})
        seen_runs.add(run_id)

        for condition, condition_manifest in (
            ("plain", plain),
            ("reasoning", reasoning),
        ):
            if not isinstance(condition_manifest, Mapping):
                issues.append(
                    {
                        "condition": condition,
                        "issue": "missing condition manifest",
                    }
                )
                continue
            for artifact in ("training_data", "training_log"):
                path_value = condition_manifest.get(artifact)
                declared_digest = condition_manifest.get(f"{artifact}_sha256")
                if not isinstance(path_value, str) or not path_value:
                    issues.append(
                        {
                            "condition": condition,
                            "artifact": artifact,
                            "issue": "artifact path is missing",
                        }
                    )
                    continue
                artifact_path = Path(path_value)
                if not artifact_path.is_absolute():
                    artifact_path = manifest_path.parent / artifact_path
                if not artifact_path.is_file():
                    issues.append(
                        {
                            "condition": condition,
                            "artifact": artifact,
                            "issue": "artifact is not a readable file",
                            "path": str(artifact_path),
                        }
                    )
                    continue
                actual_digest = _sha256_file(artifact_path)
                if declared_digest != actual_digest:
                    issues.append(
                        {
                            "condition": condition,
                            "artifact": artifact,
                            "issue": "declared SHA-256 does not match local file",
                            "declared": declared_digest,
                            "actual": actual_digest,
                        }
                    )
                if artifact == "training_data":
                    training_rows[condition].extend(read_jsonl(artifact_path))

        # Tie the exact checkpoints evaluated for this run to the training
        # manifest.  Path IDs are the default; immutable artifact IDs are also
        # accepted when explicitly supplied to evaluate-heldout.
        for condition in ("base", "plain", "reasoning"):
            condition_manifest = plain if condition == "base" else (
                plain if condition == "plain" else reasoning
            )
            if not isinstance(condition_manifest, Mapping) or not run_id:
                continue
            artifact = (
                "base_checkpoint" if condition == "base" else "output_checkpoint"
            )
            allowed_ids = {
                str(value)
                for value in (
                    condition_manifest.get(artifact),
                    (
                        condition_manifest.get(f"{artifact}_provenance") or {}
                    ).get("artifact_id")
                    if isinstance(
                        condition_manifest.get(f"{artifact}_provenance"),
                        Mapping,
                    )
                    else None,
                )
                if value is not None
            }
            allowed_provenance = {
                str(value)
                for value in (
                    condition_manifest.get(f"{artifact}_sha256"),
                    (
                        condition_manifest.get(f"{artifact}_provenance") or {}
                    ).get("immutable_ref")
                    if isinstance(
                        condition_manifest.get(f"{artifact}_provenance"),
                        Mapping,
                    )
                    else None,
                )
                if value is not None
            }
            observed = observed_checkpoints.get(
                (condition, run_id),
                {"ids": set(), "provenance": set()},
            )
            id_matches = bool(observed["ids"] & allowed_ids)
            provenance_matches = bool(
                observed["provenance"] & allowed_provenance
            )
            if not id_matches and not provenance_matches:
                issues.append(
                    {
                        "condition": condition,
                        "artifact": artifact,
                        "issue": "evaluated checkpoint is not linked to compute manifest",
                        "observed_ids": sorted(observed["ids"]),
                        "allowed_ids": sorted(allowed_ids),
                        "observed_provenance": sorted(observed["provenance"]),
                    }
                )

        training_base_checkpoints = sorted(
            {
                str(condition_manifest["base_checkpoint"])
                for condition_manifest in (plain, reasoning)
                if isinstance(condition_manifest, Mapping)
                and condition_manifest.get("base_checkpoint") is not None
            }
        )
        training_base_checkpoint_sha256s = sorted(
            {
                str(condition_manifest["base_checkpoint_sha256"]).lower()
                for condition_manifest in (plain, reasoning)
                if isinstance(condition_manifest, Mapping)
                and condition_manifest.get("base_checkpoint_sha256") is not None
            }
        )
        training_base_checkpoint_sha256_by_condition = {
            condition: (
                str(condition_manifest.get("base_checkpoint_sha256")).lower()
                if isinstance(condition_manifest, Mapping)
                and condition_manifest.get("base_checkpoint_sha256") is not None
                else None
            )
            for condition, condition_manifest in (
                ("plain", plain),
                ("reasoning", reasoning),
            )
        }
        generation_linkage = manifest.get("generation_linkage")
        per_run.append(
            {
                "independent_run_id": run_id,
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
                "training_base_checkpoints": training_base_checkpoints,
                "training_base_checkpoint_sha256s": (
                    training_base_checkpoint_sha256s
                ),
                "training_base_checkpoint_sha256_by_condition": (
                    training_base_checkpoint_sha256_by_condition
                ),
                "generation_linkage": (
                    dict(generation_linkage)
                    if isinstance(generation_linkage, Mapping)
                    else None
                ),
                "same_compute_audit": audit,
                "local_artifact_and_checkpoint_issues": issues,
                "passed": bool(audit.get("passed") and not issues),
            }
        )

    missing_runs = sorted(evaluation_runs.difference(seen_runs))
    extra_runs = sorted(seen_runs.difference(evaluation_runs))
    passed = bool(
        per_run
        and all(report["passed"] for report in per_run)
        and not missing_runs
        and not extra_runs
    )
    return (
        {
            "schema_version": 2,
            "passed": passed,
            "artifact_provenance_complete": bool(
                per_run
                and all(
                    report["same_compute_audit"].get(
                        "artifact_provenance_complete"
                    )
                    for report in per_run
                )
            ),
            "evaluation_runs": sorted(evaluation_runs),
            "compute_manifest_runs": sorted(seen_runs),
            "missing_compute_manifest_runs": missing_runs,
            "extra_compute_manifest_runs": extra_runs,
            "per_run": per_run,
        },
        training_rows,
    )


def cmd_analyze_capability(args: argparse.Namespace) -> int:
    if args.minimum_independent_runs < 2:
        raise ValueError("--minimum-independent-runs must be at least 2")
    rows: list[dict[str, Any]] = []
    result_provenance: list[dict[str, Any]] = []
    for path in args.results:
        rows.extend(read_capability_jsonl(path))
        result_provenance.append(
            {
                "path": str(Path(path).resolve()),
                "sha256": _sha256_file(path),
            }
        )
    evaluation_audit = audit_paired_evaluation_rows(rows, mode=args.mode)
    new_rows = [
        row
        for row in rows
        if row.get("transfer_level")
        in {"in_family", "structural", "cross_domain"}
    ]
    if not new_rows:
        raise ValueError(
            "capability analysis requires in_family, structural, or "
            "cross_domain held-out rows"
        )
    did = analyze_capability_did(
        new_rows,
        value_field="correct",
        run_key="independent_run_id",
        n_resamples=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    forgetting_rows = [
        row
        for row in rows
        if row.get("transfer_level") in {"archive", "benchmark"}
    ]
    forgetting_did = (
        analyze_capability_did(
            forgetting_rows,
            value_field="correct",
            run_key="independent_run_id",
            n_resamples=args.bootstrap_replicates,
            seed=args.bootstrap_seed + 1,
        )
        if forgetting_rows
        else None
    )
    compute_manifest_paths = list(args.compute_manifest or [])
    if compute_manifest_paths:
        compute_audit, training_rows_by_condition = (
            _audit_capability_compute_manifests(
                compute_manifest_paths,
                rows,
            )
        )
    else:
        compute_audit = None
        training_rows_by_condition = {}
    disjointness_audit = audit_training_heldout_disjointness(
        rows,
        training_rows_by_condition,
    )
    generation_report = (
        read_json(args.generation_report) if args.generation_report else None
    )
    if generation_report is not None and not isinstance(generation_report, Mapping):
        raise ValueError("--generation-report must contain a JSON object")
    generation_report_sha256 = (
        _sha256_file(args.generation_report)
        if args.generation_report
        else None
    )
    generation_report_provenance = (
        {
            "path": str(Path(args.generation_report).resolve()),
            "sha256": generation_report_sha256,
        }
        if args.generation_report
        else None
    )
    generation_linkage_audit = _audit_generation_capability_linkage(
        generation_report=generation_report,
        generation_report_sha256=generation_report_sha256,
        evaluation_rows=rows,
        compute_audit=compute_audit,
    )
    criteria = _capability_claim_criteria(
        did,
        forgetting_did=forgetting_did,
        compute_audit=compute_audit,
        evaluation_audit=evaluation_audit,
        disjointness_audit=disjointness_audit,
        generation_report=generation_report,
        generation_linkage_audit=generation_linkage_audit,
        max_forgetting=args.max_forgetting,
        minimum_independent_runs=args.minimum_independent_runs,
    )
    report = {
        "schema_version": 2,
        "analysis_kind": "capability_space",
        "mode": args.mode,
        "results": result_provenance,
        "evaluation_provenance_audit": evaluation_audit,
        "training_heldout_disjointness_audit": disjointness_audit,
        "difference_in_differences": did,
        "forgetting_evaluation": forgetting_did,
        "same_compute_audit": compute_audit,
        "compute_manifest": (
            [
                {
                    "path": str(Path(path).resolve()),
                    "sha256": _sha256_file(path),
                }
                for path in compute_manifest_paths
            ]
        ),
        "generation_report": generation_report_provenance,
        "generation_capability_linkage_audit": generation_linkage_audit,
        "max_allowed_accuracy_forgetting": args.max_forgetting,
        "minimum_independent_runs": args.minimum_independent_runs,
        "claim_criteria": criteria,
        "generation_geometry_is_not_used_as_capability_evidence": True,
    }
    output = Path(args.output_dir)
    write_json(output / "capability_report.json", report)
    write_jsonl(
        output / "capability_problem_contrasts.jsonl",
        did["problem_contrasts"],
    )
    lines = [
        "# Reasoning-informed expansion: capability-space report",
        "",
        f"- Delta-cap: `{did['overall']['delta_cap']}`",
        f"- CI: `[{did['overall']['ci_low']}, {did['overall']['ci_high']}]`",
        f"- Inferentially valid: `{did['overall']['inferential_valid']}`",
        (
            "- Evaluation provenance/pairing passed: "
            f"`{evaluation_audit['passed']}`"
        ),
        (
            "- Generation/training/evaluation lineage linked: "
            f"`{generation_linkage_audit['passed']}`"
        ),
        (
            "- Capability evidence gates passed: "
            f"`{criteria['capability_evidence_gates_passed']}`"
        ),
        (
            "- Strong combined capability-expansion claim allowed: "
            f"`{criteria['strong_capability_expansion_claim_allowed']}`"
        ),
        "",
        "## Claim gates",
        "",
        *[f"- `{key}`: `{value}`" for key, value in criteria.items()],
        "",
    ]
    (output / "CAPABILITY_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "delta_cap": did["overall"]["delta_cap"],
                "claim_criteria": criteria,
                "output_dir": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Normalize comparison artifacts without changing validity labels.",
    )
    prepare.add_argument("--comparison-root", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--parent-program")
    prepare.add_argument("--run-id")
    prepare.add_argument("--evolution-iteration", type=int, default=0)
    prepare.add_argument(
        "--generator-draw-idx",
        type=int,
        default=None,
        help="Defaults to source manifest llm_seed, or 0 for older artifacts.",
    )
    prepare.add_argument("--solver-checkpoint")
    prepare.add_argument(
        "--solver-checkpoint-sha256",
        help=(
            "Immutable digest supplied by the checkpoint registry/operator. "
            "Falls back to frozen_solver.checkpoint_sha256 in --config; the "
            "checkpoint directory is never hashed automatically."
        ),
    )
    prepare.add_argument("--generator-checkpoint")
    prepare.add_argument("--config", default=str(DEFAULT_CONFIG))
    prepare.set_defaults(func=cmd_prepare)

    merge = subparsers.add_parser(
        "merge",
        help="Merge independently prepared run/parent manifests with provenance checks.",
    )
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output-dir", required=True)
    merge.set_defaults(func=cmd_merge)

    archive = subparsers.add_parser(
        "prepare-archive",
        help="Strip answers/metadata from a frozen archive JSONL before extraction.",
    )
    archive.add_argument("--archive-jsonl", required=True)
    archive.add_argument("--output", required=True)
    archive.set_defaults(func=cmd_prepare_archive)

    extract = subparsers.add_parser(
        "extract",
        help="Extract frozen-Solver intermediate prompt representations.",
    )
    extract.add_argument("--representation-inputs", required=True)
    extract.add_argument("--model", required=True)
    extract.add_argument("--tokenizer")
    extract.add_argument("--output", required=True)
    extract.add_argument("--device", default=None)
    extract.add_argument("--dtype", default="bfloat16")
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--max-prompt-tokens", type=int, default=2048)
    extract.add_argument("--trust-remote-code", action="store_true")
    extract.set_defaults(func=cmd_extract)

    generation = subparsers.add_parser(
        "analyze-generation",
        help="Measure plain-subspace aligned and orthogonal displacement.",
    )
    generation.add_argument("--instance-manifest", required=True)
    generation.add_argument(
        "--experiment-manifest",
        help="Defaults to experiment_manifest.json beside --instance-manifest.",
    )
    generation.add_argument(
        "--generator-manifest",
        help="Defaults to generator_manifest.jsonl beside --instance-manifest.",
    )
    generation.add_argument(
        "--representation-input-manifest",
        help="Defaults to representation_inputs.jsonl beside --instance-manifest.",
    )
    generation.add_argument("--representations", required=True)
    generation.add_argument("--output-dir", required=True)
    generation.add_argument("--config", default=str(DEFAULT_CONFIG))
    generation.add_argument(
        "--mode",
        choices=("pilot", "confirmatory"),
        default="confirmatory",
    )
    generation.add_argument(
        "--validity-policy",
        choices=("strict", "code_valid"),
        default="strict",
    )
    generation.add_argument("--representation-key")
    generation.add_argument(
        "--all-representations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    generation.add_argument("--archive-representations")
    generation.add_argument("--archive-kind")
    generation.add_argument(
        "--use-parent-surrogate-archive",
        action="store_true",
        help="Pilot only: label parent instances as a surrogate, never a true archive.",
    )
    generation.add_argument("--pca-variance-threshold", type=float)
    generation.add_argument(
        "--pca-centered",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    generation.add_argument("--knn-k", type=int)
    generation.add_argument("--coverage-quantile", type=float)
    generation.add_argument("--bootstrap-replicates", type=int)
    generation.add_argument(
        "--run-sensitivity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the full prespecified PCA/k/coverage grid as non-primary output.",
    )
    generation.set_defaults(func=cmd_analyze_generation)

    training = subparsers.add_parser(
        "prepare-training",
        help="Write fixed accepted datasets and audit count/prompt-token parity.",
    )
    training.add_argument("--instance-manifest", required=True)
    training.add_argument("--tokenizer", required=True)
    training.add_argument("--output-dir", required=True)
    training.add_argument(
        "--validity-policy",
        choices=("strict", "code_valid"),
        default="strict",
    )
    training.set_defaults(func=cmd_prepare_training)

    audit = subparsers.add_parser(
        "audit-training",
        help="Block unequal-compute training before either condition starts.",
    )
    audit.add_argument("--compute-manifest")
    audit.add_argument("--plain-jsonl")
    audit.add_argument("--reasoning-jsonl")
    audit.add_argument(
        "--token-count-field",
        action="append",
        default=["prompt_token_count"],
    )
    audit.add_argument("--output")
    audit.set_defaults(func=cmd_audit_training)

    evaluate = subparsers.add_parser(
        "evaluate-heldout",
        help="Evaluate one base/plain/reasoning checkpoint on a fixed held-out set.",
    )
    evaluate.add_argument("--heldout", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument(
        "--condition",
        choices=("base", "plain", "reasoning"),
        required=True,
    )
    evaluate.add_argument("--checkpoint-id")
    evaluate.add_argument(
        "--checkpoint-run-id",
        help=(
            "Independent run served by this checkpoint; inferred only when "
            "the held-out file contains exactly one run."
        ),
    )
    evaluate.add_argument(
        "--checkpoint-provenance",
        help="Immutable checkpoint digest, artifact URI, or registry reference.",
    )
    evaluate.add_argument("--tokenizer")
    evaluate.add_argument("--tokenizer-id")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--evaluation-manifest")
    evaluate.add_argument(
        "--mode",
        choices=("pilot", "confirmatory"),
        default="confirmatory",
    )
    evaluate.add_argument("--rollouts", type=int, default=1)
    evaluate.add_argument("--temperature", type=float, default=0.0)
    evaluate.add_argument("--top-p", type=float, default=1.0)
    evaluate.add_argument("--max-tokens", type=int, default=4096)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--tensor-parallel-size", type=int, default=1)
    evaluate.add_argument("--dtype", default="auto")
    evaluate.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    evaluate.add_argument("--max-model-len", type=int)
    evaluate.add_argument("--trust-remote-code", action="store_true")
    evaluate.add_argument(
        "--vllm-sampler-backend",
        choices=("pytorch", "auto", "flashinfer"),
        default="pytorch",
        help=(
            "Top-k/top-p sampler implementation; pytorch avoids FlashInfer "
            "runtime JIT/toolkit compatibility failures."
        ),
    )
    evaluate.add_argument("--no-progress", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate_heldout)

    capability = subparsers.add_parser(
        "analyze-capability",
        help="Compute held-out capability difference-in-differences.",
    )
    capability.add_argument("--results", nargs="+", required=True)
    capability.add_argument("--output-dir", required=True)
    capability.add_argument(
        "--compute-manifest",
        nargs="+",
        help="One immutable same-compute manifest per independent run.",
    )
    capability.add_argument(
        "--generation-report",
        help=(
            "Optional generation_report.json; required for the final combined "
            "capability-expansion claim."
        ),
    )
    capability.add_argument(
        "--mode",
        choices=("pilot", "confirmatory"),
        default="confirmatory",
    )
    capability.add_argument(
        "--minimum-independent-runs",
        type=int,
        default=3,
    )
    capability.add_argument("--bootstrap-replicates", type=int, default=10000)
    capability.add_argument("--bootstrap-seed", type=int, default=271828)
    capability.add_argument("--max-forgetting", type=float, default=0.02)
    capability.set_defaults(func=cmd_analyze_capability)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
