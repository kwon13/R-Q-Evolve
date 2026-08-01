"""Orchestration for generation-space expansion measurements.

The mathematical primitives live in :mod:`rq_evolve.expansion_stats`; this
module joins them to normalized comparison manifests and representation
artifacts while enforcing pairing, calibration/evaluation splits, and claim
gates.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .expansion_experiment import (
    RQ_SCALAR_OBJECTIVE_KIND,
    select_child_instances,
)
from .expansion_repr import RepresentationArtifact
from .expansion_stats import (
    ExpansionStatsError,
    aggregate_generator_metrics,
    aligned_orthogonal_norms,
    cosine_coverage,
    cosine_knn_novelty,
    fit_plain_pca_subspace,
    hierarchical_bootstrap_adjusted_paired_ridge,
    hierarchical_bootstrap_paired_effect,
    paired_condition_differences,
)


PRIMARY_METRICS = ("aligned_displacement", "orthogonal_displacement")
BASE_CONTROL_METRICS = (
    "rq",
    "token_length",
    "numeric_count",
    "numeric_span",
    "numeric_max_abs",
)


def deterministic_parent_split(
    parent_program_id: str,
    *,
    calibration_fraction: float,
    split_seed: int,
) -> str:
    """Assign a parent to calibration/evaluation without inspecting outcomes."""

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    digest = hashlib.sha256(
        f"{int(split_seed)}\x1f{parent_program_id}".encode("utf-8")
    ).digest()
    unit = int.from_bytes(digest[:8], byteorder="big") / float(2**64)
    return "calibration" if unit < calibration_fraction else "evaluation"


def _representation_map(
    artifact: RepresentationArtifact,
    sample_ids: Sequence[str],
    representation_key: str,
) -> dict[str, np.ndarray]:
    if representation_key not in artifact.arrays:
        available = ", ".join(sorted(artifact.arrays))
        raise KeyError(
            f"representation key {representation_key!r} not found; "
            f"available: {available}"
        )
    matrix = np.asarray(artifact.arrays[representation_key], dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{representation_key} must be a 2-D array")
    if len(sample_ids) != matrix.shape[0]:
        raise ValueError(
            "sample ID count does not match representation rows: "
            f"{len(sample_ids)} != {matrix.shape[0]}"
        )
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("representation sample IDs contain duplicates")
    return {
        str(sample_id): matrix[index]
        for index, sample_id in enumerate(sample_ids)
    }


def _token_count_map(
    artifact: RepresentationArtifact,
    sample_ids: Sequence[str],
) -> dict[str, int]:
    counts = artifact.metadata.get("token_counts")
    if not isinstance(counts, list) or len(counts) != len(sample_ids):
        raise ValueError(
            "representation metadata must contain one token_count per sample"
        )
    return {
        str(sample_id): int(count)
        for sample_id, count in zip(sample_ids, counts, strict=True)
    }


def _matched_instance_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep shared seed pairs and report incomplete condition cells."""

    grouped: dict[
        tuple[str, int],
        dict[str, list[Mapping[str, Any]]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pair_id = str(row["generator_pair_id"])
        seed = int(row["instance_seed"])
        grouped[(pair_id, seed)][str(row["condition"])].append(row)

    matched: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for (pair_id, seed), conditions in grouped.items():
        missing = [
            condition
            for condition in ("plain", "reasoning")
            if not conditions.get(condition)
        ]
        duplicated = [
            condition
            for condition in ("plain", "reasoning")
            if len(conditions.get(condition, [])) > 1
        ]
        if missing or duplicated:
            incomplete.append(
                {
                    "generator_pair_id": pair_id,
                    "instance_seed": seed,
                    "missing_conditions": missing,
                    "duplicated_conditions": duplicated,
                }
            )
            continue
        matched.extend(
            dict(conditions[condition][0])
            for condition in ("plain", "reasoning")
        )
    return matched, incomplete


def _safe_category(value: Any) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    return cleaned or "missing"


def _add_control_features(
    rows: list[dict[str, Any]],
    *,
    numeric_fields: Sequence[str],
) -> tuple[list[str], list[str]]:
    group_values = sorted(
        {str(row.get("concept_group") or "missing") for row in rows}
    )
    type_values = sorted(
        {str(row.get("concept_type") or "missing") for row in rows}
    )
    categorical_fields = [
        *(f"concept_group__{_safe_category(value)}" for value in group_values),
        *(f"concept_type__{_safe_category(value)}" for value in type_values),
    ]
    for row in rows:
        group = str(row.get("concept_group") or "missing")
        concept_type = str(row.get("concept_type") or "missing")
        for value in group_values:
            row[f"concept_group__{_safe_category(value)}"] = float(group == value)
        for value in type_values:
            row[f"concept_type__{_safe_category(value)}"] = float(
                concept_type == value
            )
        row["operator_contract_indicator"] = float(
            row.get("operator_contract_valid") is True
        )
    return (
        [
            *numeric_fields,
            *categorical_fields,
            "operator_contract_indicator",
        ],
        categorical_fields,
    )


def _validate_pair_design(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    warnings: list[str] = []
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["generator_pair_id"])].append(row)
    invariants = ("parent_program_id", "evolution_iteration", "operator")
    for pair_id, pair_rows in by_pair.items():
        for field in invariants:
            values = {str(row.get(field)) for row in pair_rows}
            if len(values) != 1:
                warnings.append(
                    f"pair {pair_id} violates design invariant {field}: "
                    f"{sorted(values)}"
                )
    return warnings


def _generator_equal_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts = Counter(str(row["generator_unit_id"]) for row in rows)
    raw = np.asarray(
        [1.0 / counts[str(row["generator_unit_id"])] for row in rows],
        dtype=np.float64,
    )
    return raw / raw.mean()


def analyze_generation_space(
    instance_rows: Sequence[Mapping[str, Any]],
    *,
    representation_artifact: RepresentationArtifact,
    representation_sample_ids: Sequence[str],
    representation_key: str,
    mode: str = "confirmatory",
    validity_policy: str = "strict",
    calibration_fraction: float = 0.5,
    split_seed: int = 314159,
    pca_variance_threshold: float = 0.95,
    pca_centered: bool = False,
    archive_artifact: RepresentationArtifact | None = None,
    archive_sample_ids: Sequence[str] | None = None,
    archive_representation_key: str | None = None,
    archive_kind: str | None = None,
    knn_k: int = 5,
    coverage_quantile: float = 0.95,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 271828,
    ridge_alpha: float = 1.0,
    minimum_independent_runs: int = 3,
    minimum_evaluation_parents: int = 20,
    minimum_paired_generators_per_parent: int = 3,
) -> dict[str, Any]:
    """Run one layer/pooling generation-space analysis.

    Pilot mode reuses the observed parent(s) for PCA fitting and evaluation,
    and is therefore always labelled descriptive.  Confirmatory mode fits PCA
    only on preregistered calibration parents and measures paired effects on
    held-out evaluation parents.
    """

    if mode not in {"pilot", "confirmatory"}:
        raise ValueError("mode must be 'pilot' or 'confirmatory'")
    if mode == "confirmatory" and validity_policy != "strict":
        raise ValueError("confirmatory mode requires validity_policy='strict'")

    vector_map = _representation_map(
        representation_artifact,
        representation_sample_ids,
        representation_key,
    )
    token_counts = _token_count_map(
        representation_artifact,
        representation_sample_ids,
    )
    parent_rows = {
        str(row["item_id"]): dict(row)
        for row in instance_rows
        if row.get("role") == "parent"
    }
    selected = select_child_instances(
        instance_rows,
        validity_policy=validity_policy,
    )
    matched, incomplete_pairs = _matched_instance_rows(selected)
    matched_item_ids = {str(row["item_id"]) for row in matched}
    warnings = _validate_pair_design(matched)
    if mode == "pilot":
        warnings.append(
            "pilot_fit_evaluation_reuse: plain PCA is fitted and described on "
            "the same parent set; no inferential claim is permitted."
        )
    if validity_policy != "strict":
        warnings.append(
            "diagnostic_validity_policy: code-valid rows include rejected or "
            "operator-contract-invalid generators; selection/confounding "
            "precludes a causal method comparison."
        )
    if not matched:
        return {
            "status": "insufficient_matched_pairs",
            "mode": mode,
            "validity_policy": validity_policy,
            "representation_key": representation_key,
            "descriptive_only": True,
            "inferential_valid": False,
            "instance_metrics": [],
            "generator_metrics": [],
            "paired_metrics": [],
            "incomplete_instance_pairs": incomplete_pairs,
            "warnings": [
                *warnings,
                "No shared plain/reasoning instance pairs passed the validity rule.",
            ],
        }

    analysis_rows: list[dict[str, Any]] = []
    missing_vectors: list[str] = []
    # Calibration PCA may use every valid plain generator.  Reasoning validity
    # must not determine which plain directions define U_plain.  The matched
    # set is applied only to held-out condition comparisons below.
    for row in selected:
        child_id = str(row["item_id"])
        parent_id = str(row["parent_instance_id"])
        if child_id not in vector_map:
            missing_vectors.append(child_id)
            continue
        if parent_id not in vector_map:
            missing_vectors.append(parent_id)
            continue
        enriched = dict(row)
        enriched["run_id"] = str(row["independent_run_id"])
        enriched["token_length"] = float(token_counts[child_id])
        enriched["delta"] = vector_map[child_id] - vector_map[parent_id]
        enriched["split"] = (
            "all"
            if mode == "pilot"
            else deterministic_parent_split(
                str(row["parent_program_id"]),
                calibration_fraction=calibration_fraction,
                split_seed=split_seed,
            )
        )
        analysis_rows.append(enriched)
    if missing_vectors:
        raise ValueError(
            "manifest rows are missing representations for sample IDs: "
            + ", ".join(sorted(set(missing_vectors))[:20])
        )

    calibration_plain = [
        row
        for row in analysis_rows
        if row["condition"] == "plain"
        and (mode == "pilot" or row["split"] == "calibration")
    ]
    evaluation_rows = [
        row
        for row in analysis_rows
        if str(row["item_id"]) in matched_item_ids
        and (mode == "pilot" or row["split"] == "evaluation")
    ]
    if not calibration_plain or not evaluation_rows:
        return {
            "status": "insufficient_split_units",
            "mode": mode,
            "validity_policy": validity_policy,
            "representation_key": representation_key,
            "descriptive_only": True,
            "inferential_valid": False,
            "split_counts": {
                "calibration_plain_instances": len(calibration_plain),
                "evaluation_instances": len(evaluation_rows),
            },
            "instance_metrics": [],
            "generator_metrics": [],
            "paired_metrics": [],
            "incomplete_instance_pairs": incomplete_pairs,
            "warnings": [
                *warnings,
                "Calibration/evaluation parent split left an empty analysis cell.",
            ],
        }

    try:
        pca = fit_plain_pca_subspace(
            np.stack([row["delta"] for row in calibration_plain]),
            variance_threshold=pca_variance_threshold,
            centered=pca_centered,
            sample_weights=_generator_equal_weights(calibration_plain),
        )
    except ExpansionStatsError as exc:
        return {
            "status": "pca_unavailable",
            "mode": mode,
            "validity_policy": validity_policy,
            "representation_key": representation_key,
            "descriptive_only": True,
            "inferential_valid": False,
            "instance_metrics": [],
            "generator_metrics": [],
            "paired_metrics": [],
            "incomplete_instance_pairs": incomplete_pairs,
            "warnings": [*warnings, str(exc)],
        }

    evaluation_deltas = np.stack([row["delta"] for row in evaluation_rows])
    aligned, orthogonal = aligned_orthogonal_norms(
        evaluation_deltas,
        pca.basis,
    )
    for row, aligned_value, orthogonal_value in zip(
        evaluation_rows,
        aligned,
        orthogonal,
        strict=True,
    ):
        row["aligned_displacement"] = float(aligned_value)
        row["orthogonal_displacement"] = float(orthogonal_value)
        del row["delta"]

    novelty_report: dict[str, Any] | None = None
    metric_names = list(PRIMARY_METRICS)
    if archive_artifact is not None:
        if archive_sample_ids is None:
            raise ValueError("archive_sample_ids are required with archive_artifact")
        archive_key = archive_representation_key or representation_key
        archive_map = _representation_map(
            archive_artifact,
            archive_sample_ids,
            archive_key,
        )
        archive_matrix = np.stack(
            [archive_map[str(sample_id)] for sample_id in archive_sample_ids]
        )
        child_matrix = np.stack(
            [
                vector_map[str(row["item_id"])]
                for row in evaluation_rows
            ]
        )
        novelty = cosine_knn_novelty(
            child_matrix,
            archive_matrix,
            k=knn_k,
            mode=mode,
        )
        coverage = cosine_coverage(
            child_matrix,
            archive_matrix,
            k=knn_k,
            quantile=coverage_quantile,
            mode=mode,
        )
        for index, row in enumerate(evaluation_rows):
            row["novelty"] = float(novelty["novelty"][index])
            row["coverage_indicator"] = float(coverage["covered"][index])
        metric_names.extend(("novelty", "coverage_indicator"))
        novelty_report = {
            "archive_kind": archive_kind or "external",
            "archive_size": len(archive_sample_ids),
            "requested_k": knn_k,
            "novelty_effective_k": novelty["effective_k"],
            "coverage_effective_k": coverage["effective_k"],
            "coverage_epsilon": coverage["epsilon"],
            "coverage_quantile": coverage_quantile,
            "warning": novelty.get("warning") or coverage.get("warning"),
        }
        if novelty_report["warning"]:
            warnings.append(str(novelty_report["warning"]))

    observed_rq_kinds = sorted(
        {
            str(row.get("rq_kind"))
            for row in evaluation_rows
            if row.get("rq_kind") is not None
        }
    )
    genuine_rq_available = bool(evaluation_rows) and all(
        row.get("rq") is not None
        and row.get("rq_kind") == RQ_SCALAR_OBJECTIVE_KIND
        for row in evaluation_rows
    )
    missing_control_fields = [
        field
        for field in BASE_CONTROL_METRICS
        if any(row.get(field) is None for row in evaluation_rows)
    ]
    if not genuine_rq_available and "rq" not in missing_control_fields:
        missing_control_fields.append("rq")
        warnings.append(
            "genuine_scalar_rq_missing: rq is absent or is a mean-NLL proxy; "
            "the R_Q-controls-adjusted confirmatory claim is disabled."
        )
    available_numeric_controls = [
        field for field in BASE_CONTROL_METRICS if field not in missing_control_fields
    ]
    if missing_control_fields:
        warnings.append(
            "missing_prespecified_controls: "
            + ", ".join(missing_control_fields)
            + "; adjusted claims are disabled."
        )
    control_metrics, categorical_fields = _add_control_features(
        evaluation_rows,
        numeric_fields=available_numeric_controls,
    )
    all_metric_names = [*metric_names, *control_metrics]
    generator_metrics = aggregate_generator_metrics(
        evaluation_rows,
        metric_names=all_metric_names,
        group_keys=(
            "run_id",
            "parent_program_id",
            "generator_pair_id",
            "condition",
            "generator_unit_id",
        ),
    )
    paired_metrics = paired_condition_differences(
        generator_metrics,
        metric_names=all_metric_names,
        pair_keys=("run_id", "parent_program_id", "generator_pair_id"),
        require_complete=False,
    )

    primary_field = "orthogonal_displacement_difference"
    if not paired_metrics:
        warnings.append(
            "No generator pair retained both conditions after evaluation splitting."
        )
        effect = None
    else:
        effect = hierarchical_bootstrap_paired_effect(
            paired_metrics,
            difference_field=primary_field,
            n_resamples=bootstrap_replicates,
            seed=bootstrap_seed,
        )
    inferential_valid = bool(effect and effect.get("inferential_valid"))
    descriptive_only = mode == "pilot" or not inferential_valid

    evaluation_runs = {
        str(row["run_id"]) for row in paired_metrics
    }
    evaluation_parents = {
        str(row["parent_program_id"]) for row in paired_metrics
    }
    pairs_per_parent = Counter(
        str(row["parent_program_id"]) for row in paired_metrics
    )
    minimum_observed_pairs = min(pairs_per_parent.values(), default=0)
    design_gate = (
        len(evaluation_runs) >= int(minimum_independent_runs)
        and len(evaluation_parents) >= int(minimum_evaluation_parents)
        and minimum_observed_pairs
        >= int(minimum_paired_generators_per_parent)
    )
    if mode == "confirmatory" and not design_gate:
        inferential_valid = False
        descriptive_only = True
        if effect is not None:
            effect["hierarchical_bootstrap_valid_before_preregistered_gate"] = (
                bool(effect.get("inferential_valid"))
            )
            effect["inferential_valid"] = False
            effect["ci_low"] = None
            effect["ci_high"] = None
            effect["reason"] = (
                "preregistered independent-unit gate failed"
            )
        warnings.append(
            "confirmatory_independent_unit_gate_failed: "
            f"runs={len(evaluation_runs)}/{minimum_independent_runs}, "
            f"parents={len(evaluation_parents)}/{minimum_evaluation_parents}, "
            "minimum generator pairs per parent="
            f"{minimum_observed_pairs}/{minimum_paired_generators_per_parent}."
        )

    covariate_difference_fields = [
        f"{field}_difference" for field in control_metrics
    ]
    adjusted: dict[str, Any] | None = None
    if (
        inferential_valid
        and not missing_control_fields
        and len(paired_metrics) > len(covariate_difference_fields) + 1
    ):
        adjusted = hierarchical_bootstrap_adjusted_paired_ridge(
            paired_metrics,
            outcome_difference_field=primary_field,
            covariate_difference_fields=covariate_difference_fields,
            alpha=ridge_alpha,
            n_resamples=bootstrap_replicates,
            seed=bootstrap_seed + 1,
        )
    else:
        warnings.append(
            "controls_adjusted_effect_unavailable: too few independent paired "
            "generators (or inferential gate failed) for the prespecified controls."
        )

    plain_calibration_generators = len(
        {str(row["generator_unit_id"]) for row in calibration_plain}
    )
    required_calibration_generators = max(20, 5 * pca.n_components)
    if mode == "confirmatory" and (
        plain_calibration_generators < required_calibration_generators
    ):
        inferential_valid = False
        descriptive_only = True
        warnings.append(
            "plain_calibration_rank_gate_failed: "
            f"{plain_calibration_generators} generators for "
            f"{pca.n_components} retained components; require at least "
            f"{required_calibration_generators}."
        )
    if mode == "confirmatory" and missing_control_fields:
        inferential_valid = False
        descriptive_only = True
    if (
        mode == "confirmatory"
        and not inferential_valid
        and effect is not None
        and effect.get("inferential_valid")
    ):
        effect["hierarchical_bootstrap_valid_before_preregistered_gate"] = True
        effect["inferential_valid"] = False
        effect["ci_low"] = None
        effect["ci_high"] = None
        effect["reason"] = "one or more preregistered analysis gates failed"

    serializable_instance_rows = []
    for row in evaluation_rows:
        serializable_instance_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"answer", "evaluator_reason", "problem_text"}
            }
        )
    return {
        "status": "ok_descriptive" if descriptive_only else "ok_inferential",
        "mode": mode,
        "validity_policy": validity_policy,
        "representation_key": representation_key,
        "descriptive_only": descriptive_only,
        "inferential_valid": inferential_valid,
        "hypothesis": "O_reasoning > O_plain",
        "pca": {
            "fit_condition": "plain_only",
            "fit_split": "all_pilot" if mode == "pilot" else "calibration",
            "centered": pca.centered,
            "subspace_geometry": (
                "centered_variation_sensitivity"
                if pca.centered
                else "origin_anchored_raw_displacement"
            ),
            "variance_threshold": pca.variance_threshold,
            "n_components": pca.n_components,
            "cumulative_explained_variance": float(
                pca.cumulative_explained_variance[pca.n_components - 1]
            ),
            "n_plain_instances": pca.n_samples,
            "n_plain_generators": plain_calibration_generators,
            "required_plain_generators_for_rank_gate": (
                required_calibration_generators
            ),
            "generator_equal_weighting": True,
        },
        "split_counts": {
            "calibration_plain_instances": len(calibration_plain),
            "evaluation_instances": len(evaluation_rows),
            "evaluation_generator_pairs": len(paired_metrics),
            "evaluation_independent_runs": len(evaluation_runs),
            "evaluation_parents": len(evaluation_parents),
            "minimum_generator_pairs_per_parent": minimum_observed_pairs,
        },
        "independent_unit_gate": {
            "passed": design_gate if mode == "confirmatory" else False,
            "minimum_independent_runs": int(minimum_independent_runs),
            "minimum_evaluation_parents": int(minimum_evaluation_parents),
            "minimum_paired_generators_per_parent": int(
                minimum_paired_generators_per_parent
            ),
        },
        "unadjusted_effect": effect,
        "controls_adjusted_effect": adjusted,
        "rq_control_audit": {
            "genuine_scalar_rq_available": genuine_rq_available,
            "required_rq_kind": RQ_SCALAR_OBJECTIVE_KIND,
            "observed_rq_kinds": observed_rq_kinds,
            "claim_eligible": bool(genuine_rq_available and adjusted),
        },
        "control_note": (
            "Parent identity is removed by pairing; iteration/operator are "
            "checked as invariants. RQ, length, number range, and concept "
            "one-hot differences are robustness controls. A mean-NLL rq proxy "
            "is retained for diagnostics but cannot satisfy the scalar-R_Q gate."
        ),
        "categorical_control_fields": categorical_fields,
        "novelty": novelty_report,
        "instance_metrics": serializable_instance_rows,
        "generator_metrics": generator_metrics,
        "paired_metrics": paired_metrics,
        "incomplete_instance_pairs": incomplete_pairs,
        "warnings": list(dict.fromkeys(warnings)),
    }
