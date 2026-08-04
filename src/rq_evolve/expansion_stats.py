"""Statistics for the reasoning-informed Evolver expansion experiment.

This module deliberately has no dependency on scikit-learn, statsmodels, or
model-serving code.  Its inputs are NumPy arrays and row dictionaries so the
representation extraction and experiment orchestration layers can be tested
independently.

The functions enforce the experiment's unit of independence:

* instance metrics are aggregated to generators before comparing conditions;
* plain-condition displacements alone fit the reference PCA subspace;
* paired effects are formed at ``generator_pair_id``;
* inferential bootstrap intervals require multiple independent runs and
  multiple parent programs.

When a pilot has too little data, descriptive estimates are still returned,
but ``inferential_valid`` is false and no confidence interval is fabricated.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


_EPS = 1e-12


class ExpansionStatsError(ValueError):
    """Raised when an expansion statistic cannot be computed as specified."""


@dataclass(frozen=True)
class PCASubspace:
    """Plain-displacement PCA fit used as the reference expansion subspace.

    ``basis`` has shape ``(representation_dim, n_components)`` and orthonormal
    columns.  Because aligned/orthogonal norms project *raw displacement
    vectors* through an origin-anchored linear subspace, the confirmatory
    analysis uses uncentered SVD.  ``center`` is retained for the explicitly
    labelled centered sensitivity analysis only.
    """

    basis: np.ndarray
    center: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    singular_values: np.ndarray
    n_components: int
    variance_threshold: float
    centered: bool
    n_samples: int
    representation_dim: int
    sample_weighted: bool
    effective_sample_size: float


def _as_finite_2d(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ExpansionStatsError(f"{name} must be a 2-D array; got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ExpansionStatsError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ExpansionStatsError(f"{name} contains NaN or infinite values")
    return array


def l2_normalize(
    values: Any,
    *,
    axis: int = -1,
    eps: float = _EPS,
) -> np.ndarray:
    """Return an L2-normalized float64 array.

    Zero (or near-zero) vectors are rejected instead of silently returning
    arbitrary directions.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ExpansionStatsError("values must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ExpansionStatsError("values contains NaN or infinite values")
    norm = np.linalg.norm(array, axis=axis, keepdims=True)
    if np.any(norm <= eps):
        raise ExpansionStatsError("cannot L2-normalize a zero or near-zero vector")
    return array / norm


def fit_plain_pca_subspace(
    plain_deltas: Any,
    *,
    variance_threshold: float = 0.95,
    centered: bool = False,
    sample_weights: Any | None = None,
    rank_tolerance: float | None = None,
) -> PCASubspace:
    """Fit the scalar-objective expansion subspace from plain deltas only.

    Parameters
    ----------
    plain_deltas:
        Matrix of shape ``(n_plain_samples, representation_dim)``.  The
        explicit name is intentional: callers must not mix in reasoning
        condition displacements.
    variance_threshold:
        Smallest number of principal axes whose cumulative explained variance
        is at least this value.
    centered:
        Whether to subtract the plain-delta mean before the SVD.  The primary
        origin-anchored displacement analysis uses ``False`` so the mean plain
        expansion direction is part of the fitted linear subspace.  ``True``
        is a sensitivity analysis of variation around that mean and must not
        be interpreted as explaining raw displacement directions unless the
        mean direction is added separately.
    sample_weights:
        Optional non-negative weight per displacement.  To give every
        generator equal influence when generators contain different numbers
        of instances, pass ``1 / n_instances_in_generator`` for each instance.
        Weights are rescaled to have mean one, so uniform weights reproduce
        the unweighted fit exactly.
    rank_tolerance:
        Optional absolute cutoff for singular values.  The NumPy-style
        dimension-scaled tolerance is used by default.
    """

    deltas = _as_finite_2d(plain_deltas, name="plain_deltas")
    if not 0.0 < variance_threshold <= 1.0:
        raise ExpansionStatsError("variance_threshold must be in (0, 1]")
    if sample_weights is None:
        weights = np.ones(deltas.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.ndim != 1 or weights.shape[0] != deltas.shape[0]:
            raise ExpansionStatsError(
                "sample_weights must be a 1-D array with one value per plain delta"
            )
        if not np.all(np.isfinite(weights)):
            raise ExpansionStatsError("sample_weights contains NaN or infinite values")
        if np.any(weights < 0):
            raise ExpansionStatsError("sample_weights must be non-negative")
        if float(weights.sum()) <= 0:
            raise ExpansionStatsError("sample_weights must contain positive total weight")
    positive_weight_count = int(np.count_nonzero(weights > 0))
    if centered and positive_weight_count < 2:
        raise ExpansionStatsError(
            "centered PCA requires at least two positive-weight plain deltas"
        )

    center = (
        np.average(deltas, axis=0, weights=weights)
        if centered
        else np.zeros(deltas.shape[1])
    )
    # Mean-one scaling makes all-ones weights byte-for-byte equivalent to the
    # previous unweighted SVD while retaining the standard sqrt(weight)
    # weighted-covariance construction.
    scaled_weights = weights * (deltas.shape[0] / float(weights.sum()))
    fit_matrix = (deltas - center) * np.sqrt(scaled_weights)[:, None]
    _, singular_values, right_vectors = np.linalg.svd(
        fit_matrix,
        full_matrices=False,
    )
    if singular_values.size == 0:
        raise ExpansionStatsError("plain PCA produced no singular values")

    if rank_tolerance is None:
        rank_tolerance = (
            max(fit_matrix.shape)
            * np.finfo(singular_values.dtype).eps
            * float(singular_values[0])
        )
    if rank_tolerance < 0:
        raise ExpansionStatsError("rank_tolerance must be non-negative")
    positive = singular_values > rank_tolerance
    if not np.any(positive):
        raise ExpansionStatsError(
            "plain deltas have zero fitted variance; the PCA subspace is undefined"
        )

    singular_values = singular_values[positive]
    right_vectors = right_vectors[positive]
    variance = np.square(singular_values)
    explained_ratio = variance / variance.sum()
    cumulative = np.cumsum(explained_ratio)
    # At threshold=1.0, roundoff can leave the final cumulative value a few
    # ulps below one; never request more axes than the numerical rank.
    n_components = min(
        int(explained_ratio.size),
        int(np.searchsorted(cumulative, variance_threshold, side="left") + 1),
    )
    basis = right_vectors[:n_components].T.copy()

    return PCASubspace(
        basis=basis,
        center=center.copy(),
        explained_variance_ratio=explained_ratio.copy(),
        cumulative_explained_variance=cumulative.copy(),
        singular_values=singular_values.copy(),
        n_components=n_components,
        variance_threshold=float(variance_threshold),
        centered=bool(centered),
        n_samples=int(deltas.shape[0]),
        representation_dim=int(deltas.shape[1]),
        sample_weighted=sample_weights is not None,
        effective_sample_size=float(
            np.square(weights.sum()) / np.square(weights).sum()
        ),
    )


def aligned_orthogonal_norms(
    deltas: Any,
    basis: Any,
    *,
    orthonormal_tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute aligned and orthogonal displacement norms.

    For each row :math:`\\Delta z`, this returns
    ``||U.T @ delta||`` and ``||(I - U @ U.T) @ delta||``.
    """

    delta_matrix = _as_finite_2d(deltas, name="deltas")
    axes = np.asarray(basis, dtype=np.float64)
    if axes.ndim != 2:
        raise ExpansionStatsError(f"basis must be 2-D; got shape {axes.shape}")
    if axes.shape[0] != delta_matrix.shape[1]:
        raise ExpansionStatsError(
            "basis representation dimension does not match deltas: "
            f"{axes.shape[0]} != {delta_matrix.shape[1]}"
        )
    if axes.shape[1] == 0:
        raise ExpansionStatsError("basis must contain at least one axis")
    if not np.all(np.isfinite(axes)):
        raise ExpansionStatsError("basis contains NaN or infinite values")
    gram = axes.T @ axes
    if not np.allclose(
        gram,
        np.eye(axes.shape[1]),
        atol=orthonormal_tolerance,
        rtol=orthonormal_tolerance,
    ):
        raise ExpansionStatsError("basis columns must be orthonormal")

    aligned_coordinates = delta_matrix @ axes
    aligned_projection = aligned_coordinates @ axes.T
    orthogonal_projection = delta_matrix - aligned_projection
    aligned = np.linalg.norm(aligned_coordinates, axis=1)
    orthogonal = np.linalg.norm(orthogonal_projection, axis=1)
    return aligned, orthogonal


def _validate_knn_mode(mode: str) -> None:
    if mode not in {"pilot", "confirmatory"}:
        raise ExpansionStatsError("mode must be 'pilot' or 'confirmatory'")


def resolve_knn_k(
    requested_k: int,
    archive_size: int,
    *,
    leave_one_out: bool,
    mode: str,
) -> tuple[int, str | None]:
    """Resolve ``k`` and expose any pilot-only automatic reduction.

    Confirmatory analyses fail if the preregistered ``k`` is impossible.
    Pilot analyses reduce it to the largest legal value and return a warning.
    """

    _validate_knn_mode(mode)
    if isinstance(requested_k, bool) or int(requested_k) != requested_k:
        raise ExpansionStatsError("k must be a positive integer")
    requested_k = int(requested_k)
    if requested_k <= 0:
        raise ExpansionStatsError("k must be a positive integer")
    max_k = archive_size - 1 if leave_one_out else archive_size
    if max_k < 1:
        requirement = "at least two" if leave_one_out else "at least one"
        raise ExpansionStatsError(
            f"archive must contain {requirement} point(s) for this kNN calculation"
        )
    if requested_k <= max_k:
        return requested_k, None
    if mode == "confirmatory":
        suffix = " after leave-one-out exclusion" if leave_one_out else ""
        raise ExpansionStatsError(
            f"preregistered k={requested_k} exceeds {max_k} available neighbors{suffix}"
        )
    warning = (
        f"pilot_auto_k: requested k={requested_k}, using k={max_k} "
        f"for archive_size={archive_size}"
    )
    return max_k, warning


def _cosine_distance_matrix(query: Any, reference: Any) -> np.ndarray:
    query_matrix = l2_normalize(_as_finite_2d(query, name="query"), axis=1)
    reference_matrix = l2_normalize(
        _as_finite_2d(reference, name="reference"),
        axis=1,
    )
    if query_matrix.shape[1] != reference_matrix.shape[1]:
        raise ExpansionStatsError(
            "query and reference representation dimensions do not match"
        )
    # Numerical roundoff can otherwise produce tiny negative distances.
    return np.clip(1.0 - query_matrix @ reference_matrix.T, 0.0, 2.0)


def cosine_knn_novelty(
    query: Any,
    archive: Any,
    *,
    k: int = 5,
    mode: str = "confirmatory",
) -> dict[str, Any]:
    """Compute median cosine distance to each query's ``k`` archive neighbors."""

    archive_matrix = _as_finite_2d(archive, name="archive")
    effective_k, warning = resolve_knn_k(
        k,
        archive_matrix.shape[0],
        leave_one_out=False,
        mode=mode,
    )
    distances = _cosine_distance_matrix(query, archive_matrix)
    nearest = np.partition(distances, effective_k - 1, axis=1)[:, :effective_k]
    return {
        "novelty": np.median(nearest, axis=1),
        "effective_k": effective_k,
        "warning": warning,
    }


def leave_one_out_kth_epsilon(
    archive: Any,
    *,
    k: int = 5,
    quantile: float = 0.95,
    mode: str = "confirmatory",
) -> dict[str, Any]:
    """Estimate coverage epsilon from leave-one-out archive k-th distances."""

    archive_matrix = _as_finite_2d(archive, name="archive")
    if not 0.0 <= quantile <= 1.0:
        raise ExpansionStatsError("quantile must be in [0, 1]")
    effective_k, warning = resolve_knn_k(
        k,
        archive_matrix.shape[0],
        leave_one_out=True,
        mode=mode,
    )
    distances = _cosine_distance_matrix(archive_matrix, archive_matrix)
    np.fill_diagonal(distances, np.inf)
    kth = np.partition(distances, effective_k - 1, axis=1)[:, effective_k - 1]
    epsilon = float(np.quantile(kth, quantile))
    return {
        "epsilon": epsilon,
        "leave_one_out_kth_distances": kth,
        "effective_k": effective_k,
        "quantile": float(quantile),
        "warning": warning,
    }


def cosine_coverage(
    query: Any,
    archive: Any,
    *,
    k: int = 5,
    quantile: float = 0.95,
    mode: str = "confirmatory",
) -> dict[str, Any]:
    """Compute child coverage beyond the archive's leave-one-out threshold.

    The same effective ``k`` is used for the archive threshold and query
    distances.  In pilot mode it may be reduced once based on the stricter
    leave-one-out neighbor count.
    """

    archive_matrix = _as_finite_2d(archive, name="archive")
    epsilon_result = leave_one_out_kth_epsilon(
        archive_matrix,
        k=k,
        quantile=quantile,
        mode=mode,
    )
    effective_k = int(epsilon_result["effective_k"])
    distances = _cosine_distance_matrix(query, archive_matrix)
    kth = np.partition(distances, effective_k - 1, axis=1)[:, effective_k - 1]
    indicators = kth > float(epsilon_result["epsilon"])
    return {
        "coverage_gain": float(indicators.mean()),
        "covered": indicators,
        "query_kth_distances": kth,
        **epsilon_result,
    }


def _require_row_keys(
    row: Mapping[str, Any],
    keys: Iterable[str],
    *,
    row_index: int,
) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise ExpansionStatsError(f"row {row_index} is missing keys: {missing}")


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ExpansionStatsError(f"{label} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExpansionStatsError(f"{label} must be numeric") from exc
    if not np.isfinite(result):
        raise ExpansionStatsError(f"{label} must be finite")
    return result


def aggregate_generator_metrics(
    instance_rows: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str],
    group_keys: Sequence[str] = (
        "run_id",
        "parent_program_id",
        "generator_pair_id",
        "condition",
    ),
) -> list[dict[str, Any]]:
    """Aggregate nested instance metrics before any condition comparison.

    Each metric's generator mean is stored under its original name, with
    ``<metric>_sd`` and ``<metric>_n`` diagnostics.  Standard deviation is
    descriptive (population ``ddof=0``) because instances are not independent
    inferential samples.
    """

    if not instance_rows:
        return []
    if not metric_names:
        raise ExpansionStatsError("metric_names must be non-empty")
    if not group_keys:
        raise ExpansionStatsError("group_keys must be non-empty")

    groups: dict[tuple[Any, ...], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    required = tuple(group_keys) + tuple(metric_names)
    for index, row in enumerate(instance_rows):
        _require_row_keys(row, required, row_index=index)
        group = tuple(row[key] for key in group_keys)
        groups[group].append((index, row))

    aggregated: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        indexed_rows = groups[group]
        result = dict(zip(group_keys, group, strict=True))
        result["n_instances"] = len(indexed_rows)
        for metric in metric_names:
            values = np.asarray(
                [
                    _finite_float(row[metric], label=f"row {index} field {metric!r}")
                    for index, row in indexed_rows
                ],
                dtype=np.float64,
            )
            result[metric] = float(values.mean())
            result[f"{metric}_sd"] = float(values.std(ddof=0))
            result[f"{metric}_n"] = int(values.size)
        aggregated.append(result)
    return aggregated


def paired_condition_differences(
    generator_rows: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str],
    pair_keys: Sequence[str] = (
        "run_id",
        "parent_program_id",
        "generator_pair_id",
    ),
    condition_key: str = "condition",
    plain_label: str = "plain",
    reasoning_label: str = "reasoning",
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    """Form reasoning-minus-plain differences at the paired-generator level."""

    if not metric_names:
        raise ExpansionStatsError("metric_names must be non-empty")
    groups: dict[
        tuple[Any, ...],
        dict[str, list[tuple[int, Mapping[str, Any]]]],
    ] = defaultdict(lambda: defaultdict(list))
    required = tuple(pair_keys) + (condition_key,) + tuple(metric_names)
    for index, row in enumerate(generator_rows):
        _require_row_keys(row, required, row_index=index)
        condition = str(row[condition_key])
        if condition not in {plain_label, reasoning_label}:
            continue
        group = tuple(row[key] for key in pair_keys)
        groups[group][condition].append((index, row))

    paired: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        conditions = groups[group]
        missing = [
            label
            for label in (plain_label, reasoning_label)
            if label not in conditions
        ]
        if missing:
            if require_complete:
                raise ExpansionStatsError(
                    f"generator pair {group!r} is missing condition(s): {missing}"
                )
            continue
        result = dict(zip(pair_keys, group, strict=True))
        result["plain_generator_count"] = len(conditions[plain_label])
        result["reasoning_generator_count"] = len(conditions[reasoning_label])
        for metric in metric_names:
            plain_values = [
                _finite_float(row[metric], label=f"row {index} field {metric!r}")
                for index, row in conditions[plain_label]
            ]
            reasoning_values = [
                _finite_float(row[metric], label=f"row {index} field {metric!r}")
                for index, row in conditions[reasoning_label]
            ]
            plain_mean = float(np.mean(plain_values))
            reasoning_mean = float(np.mean(reasoning_values))
            result[f"{metric}_plain"] = plain_mean
            result[f"{metric}_reasoning"] = reasoning_mean
            result[f"{metric}_difference"] = reasoning_mean - plain_mean
        paired.append(result)
    return paired


def hierarchical_bootstrap_paired_effect(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    difference_field: str,
    run_key: str = "run_id",
    parent_key: str = "parent_program_id",
    pair_key: str = "generator_pair_id",
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap a paired effect via run -> parent -> generator-pair sampling.

    At least two independent runs and two distinct parent identities are
    required.  Otherwise the descriptive effect is returned without a CI.
    """

    if n_resamples <= 0:
        raise ExpansionStatsError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ExpansionStatsError("confidence must be in (0, 1)")
    required = (difference_field, run_key, parent_key, pair_key)
    clean_rows: list[dict[str, Any]] = []
    for index, row in enumerate(paired_rows):
        _require_row_keys(row, required, row_index=index)
        clean_rows.append(
            {
                "run": row[run_key],
                "parent": row[parent_key],
                "pair": row[pair_key],
                "value": _finite_float(
                    row[difference_field],
                    label=f"row {index} field {difference_field!r}",
                ),
            }
        )
    if not clean_rows:
        raise ExpansionStatsError("paired_rows must contain at least one row")

    values = np.asarray([row["value"] for row in clean_rows], dtype=np.float64)
    run_ids = list(dict.fromkeys(row["run"] for row in clean_rows))
    parent_ids = set(row["parent"] for row in clean_rows)
    parent_units = set((row["run"], row["parent"]) for row in clean_rows)
    result: dict[str, Any] = {
        "difference_field": difference_field,
        "mean_difference": float(values.mean()),
        "paired_effect_size_dz": (
            float(values.mean() / values.std(ddof=1))
            if values.size > 1 and values.std(ddof=1) > _EPS
            else None
        ),
        "n_runs": len(run_ids),
        "n_parent_identities": len(parent_ids),
        "n_parent_run_units": len(parent_units),
        "n_generator_pairs": len(clean_rows),
        "inferential_valid": False,
        "ci_low": None,
        "ci_high": None,
        "confidence": float(confidence),
        "n_resamples": 0,
        "reason": None,
    }
    gate_reasons: list[str] = []
    if len(run_ids) < 2:
        gate_reasons.append("requires at least two independent run_id values")
    if len(parent_ids) < 2:
        gate_reasons.append("requires at least two distinct parent_program_id values")
    if gate_reasons:
        result["reason"] = "; ".join(gate_reasons)
        return result

    nested: dict[Any, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in clean_rows:
        nested[row["run"]][row["parent"]].append(row["value"])

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_resamples, dtype=np.float64)
    for replicate in range(n_resamples):
        sampled_values: list[float] = []
        sampled_runs = rng.choice(run_ids, size=len(run_ids), replace=True)
        for run_id in sampled_runs:
            parents = list(nested[run_id])
            sampled_parents = rng.choice(parents, size=len(parents), replace=True)
            for parent_id in sampled_parents:
                pair_values = nested[run_id][parent_id]
                sampled_values.extend(
                    float(value)
                    for value in rng.choice(
                        pair_values,
                        size=len(pair_values),
                        replace=True,
                    )
                )
        bootstrap_means[replicate] = float(np.mean(sampled_values))

    tail = (1.0 - confidence) / 2.0
    result.update(
        {
            "inferential_valid": True,
            "ci_low": float(np.quantile(bootstrap_means, tail)),
            "ci_high": float(np.quantile(bootstrap_means, 1.0 - tail)),
            "n_resamples": int(n_resamples),
            "reason": None,
        }
    )
    return result


def controls_adjusted_paired_ridge(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    outcome_difference_field: str,
    covariate_difference_fields: Sequence[str],
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Estimate the condition effect as a paired ridge-regression intercept.

    The model is ``paired_outcome_diff = intercept + covariate_diffs @ beta``.
    Covariates are divided by their standard deviations but are *not centered*,
    so the returned intercept remains the estimated reasoning-minus-plain
    effect when every covariate difference is zero.  The intercept is not
    penalized; ``alpha`` is fixed rather than selected after looking at results.
    """

    if alpha < 0:
        raise ExpansionStatsError("alpha must be non-negative")
    required = (outcome_difference_field,) + tuple(covariate_difference_fields)
    if not paired_rows:
        raise ExpansionStatsError("paired_rows must contain at least one row")
    y_values: list[float] = []
    x_values: list[list[float]] = []
    for index, row in enumerate(paired_rows):
        _require_row_keys(row, required, row_index=index)
        y_values.append(
            _finite_float(
                row[outcome_difference_field],
                label=f"row {index} field {outcome_difference_field!r}",
            )
        )
        x_values.append(
            [
                _finite_float(row[field], label=f"row {index} field {field!r}")
                for field in covariate_difference_fields
            ]
        )

    y = np.asarray(y_values, dtype=np.float64)
    x = np.asarray(x_values, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(y.size, 0)
    if x.shape[1]:
        scale = x.std(axis=0, ddof=0)
        scale[scale <= _EPS] = 1.0
        x_scaled = x / scale
    else:
        scale = np.empty(0, dtype=np.float64)
        x_scaled = x
    design = np.column_stack([np.ones(y.size, dtype=np.float64), x_scaled])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    predictions = design @ coefficients
    residuals = y - predictions
    raw_coefficients = (
        coefficients[1:] / scale if scale.size else np.empty(0, dtype=np.float64)
    )
    return {
        "adjusted_intercept": float(coefficients[0]),
        "coefficients": {
            field: float(value)
            for field, value in zip(
                covariate_difference_fields,
                raw_coefficients,
                strict=True,
            )
        },
        "alpha": float(alpha),
        "n_pairs": int(y.size),
        "n_covariates": int(x.shape[1]),
        "residual_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "unadjusted_mean_difference": float(y.mean()),
    }


def hierarchical_bootstrap_adjusted_paired_ridge(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    outcome_difference_field: str,
    covariate_difference_fields: Sequence[str],
    alpha: float = 1.0,
    run_key: str = "run_id",
    parent_key: str = "parent_program_id",
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Attach a run→parent→generator-pair bootstrap CI to a ridge intercept."""

    if n_resamples <= 0:
        raise ExpansionStatsError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ExpansionStatsError("confidence must be in (0, 1)")
    required = (
        outcome_difference_field,
        run_key,
        parent_key,
        *tuple(covariate_difference_fields),
    )
    clean_rows: list[dict[str, Any]] = []
    for index, row in enumerate(paired_rows):
        _require_row_keys(row, required, row_index=index)
        clean_rows.append(dict(row))
    base = controls_adjusted_paired_ridge(
        clean_rows,
        outcome_difference_field=outcome_difference_field,
        covariate_difference_fields=covariate_difference_fields,
        alpha=alpha,
    )
    run_ids = list(dict.fromkeys(row[run_key] for row in clean_rows))
    parent_ids = {row[parent_key] for row in clean_rows}
    base.update(
        {
            "inferential_valid": False,
            "ci_low": None,
            "ci_high": None,
            "confidence": float(confidence),
            "n_resamples": 0,
            "n_runs": len(run_ids),
            "n_parent_identities": len(parent_ids),
            "reason": None,
        }
    )
    reasons: list[str] = []
    if len(run_ids) < 2:
        reasons.append("requires at least two independent run_id values")
    if len(parent_ids) < 2:
        reasons.append("requires at least two distinct parent_program_id values")
    if reasons:
        base["reason"] = "; ".join(reasons)
        return base

    nested: dict[Any, dict[Any, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in clean_rows:
        nested[row[run_key]][row[parent_key]].append(row)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    for replicate in range(n_resamples):
        sampled_rows: list[dict[str, Any]] = []
        sampled_runs = rng.choice(run_ids, size=len(run_ids), replace=True)
        for run_id in sampled_runs:
            parents = list(nested[run_id])
            sampled_parents = rng.choice(parents, size=len(parents), replace=True)
            for parent_id in sampled_parents:
                pair_rows = nested[run_id][parent_id]
                selected = rng.choice(
                    len(pair_rows),
                    size=len(pair_rows),
                    replace=True,
                )
                sampled_rows.extend(pair_rows[int(index)] for index in selected)
        fitted = controls_adjusted_paired_ridge(
            sampled_rows,
            outcome_difference_field=outcome_difference_field,
            covariate_difference_fields=covariate_difference_fields,
            alpha=alpha,
        )
        estimates[replicate] = float(fitted["adjusted_intercept"])

    tail = (1.0 - confidence) / 2.0
    base.update(
        {
            "inferential_valid": True,
            "ci_low": float(np.quantile(estimates, tail)),
            "ci_high": float(np.quantile(estimates, 1.0 - tail)),
            "n_resamples": int(n_resamples),
            "reason": None,
        }
    )
    return base


def _bootstrap_capability_effect(
    problem_rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str = "did",
    n_resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    # Held-out problems are commonly emitted by one structural generator
    # family for one target reasoning move.  Preserve that clustering rather
    # than treating every surface instance as an independent replicate.
    nested: dict[Any, dict[tuple[Any, Any], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in problem_rows:
        cluster = (
            row.get("target_reasoning_move", "__missing_move__"),
            row.get("family_id", "__missing_family__"),
        )
        nested[row["run_id"]][cluster].append(float(row[value_field]))
    runs = list(nested)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for replicate in range(n_resamples):
        sampled: list[float] = []
        for run_id in rng.choice(runs, size=len(runs), replace=True):
            clusters = list(nested[run_id])
            for cluster_index in rng.integers(
                0,
                len(clusters),
                size=len(clusters),
            ):
                cluster_key = clusters[int(cluster_index)]
                values = nested[run_id][cluster_key]
                sampled.extend(
                    float(value)
                    for value in rng.choice(
                        values,
                        size=len(values),
                        replace=True,
                    )
                )
        means[replicate] = float(np.mean(sampled))
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )


def _summarize_capability_stratum(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    did = np.asarray([float(row["did"]) for row in rows], dtype=np.float64)
    reasoning_gain = np.asarray(
        [float(row["reasoning_gain"]) for row in rows],
        dtype=np.float64,
    )
    plain_gain = np.asarray(
        [float(row["plain_gain"]) for row in rows],
        dtype=np.float64,
    )
    runs = set(row["run_id"] for row in rows)
    problems = set(row["problem_id"] for row in rows)
    moves = {
        row.get("target_reasoning_move")
        for row in rows
        if row.get("target_reasoning_move") is not None
    }
    families = {
        row.get("family_id")
        for row in rows
        if row.get("family_id") is not None
    }
    result: dict[str, Any] = {
        "delta_cap": float(did.mean()),
        "reasoning_gain_over_base": float(reasoning_gain.mean()),
        "plain_gain_over_base": float(plain_gain.mean()),
        "paired_effect_size_dz": (
            float(did.mean() / did.std(ddof=1))
            if did.size > 1 and did.std(ddof=1) > _EPS
            else None
        ),
        "n_runs": len(runs),
        "n_problems": len(problems),
        "n_run_problem_units": len(rows),
        "n_target_reasoning_moves": len(moves),
        "n_families": len(families),
        "inferential_valid": False,
        "ci_low": None,
        "ci_high": None,
        "reasoning_gain_ci_low": None,
        "reasoning_gain_ci_high": None,
        "plain_gain_ci_low": None,
        "plain_gain_ci_high": None,
        "confidence": float(confidence),
        "n_resamples": 0,
        "reason": None,
    }
    gate: list[str] = []
    if len(runs) < 2:
        gate.append("requires at least two independent training/evolution runs")
    if len(problems) < 2:
        gate.append("requires at least two held-out problems")
    if gate:
        result["reason"] = "; ".join(gate)
        return result
    ci_low, ci_high = _bootstrap_capability_effect(
        rows,
        value_field="did",
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
    reasoning_ci_low, reasoning_ci_high = _bootstrap_capability_effect(
        rows,
        value_field="reasoning_gain",
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed + 1_000_003,
    )
    plain_ci_low, plain_ci_high = _bootstrap_capability_effect(
        rows,
        value_field="plain_gain",
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed + 2_000_003,
    )
    result.update(
        {
            "inferential_valid": True,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "reasoning_gain_ci_low": reasoning_ci_low,
            "reasoning_gain_ci_high": reasoning_ci_high,
            "plain_gain_ci_low": plain_ci_low,
            "plain_gain_ci_high": plain_ci_high,
            "n_resamples": int(n_resamples),
        }
    )
    return result


def analyze_capability_did(
    problem_level_rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str = "accuracy",
    condition_key: str = "condition",
    run_key: str = "run_id",
    problem_key: str = "problem_id",
    transfer_key: str = "transfer_level",
    target_move_key: str = "target_reasoning_move",
    family_key: str = "family_id",
    base_label: str = "base",
    plain_label: str = "plain",
    reasoning_label: str = "reasoning",
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute capability difference-in-differences on held-out problems.

    Duplicate rows for the same run/problem/condition are averaged first (for
    example, multiple deterministic evaluation replicas).  Results are
    reported overall, by ``transfer_level``, and by target reasoning move.
    Bootstrap resampling preserves run -> (move, family) -> problem nesting.
    """

    if n_resamples <= 0:
        raise ExpansionStatsError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ExpansionStatsError("confidence must be in (0, 1)")
    required = (
        value_field,
        condition_key,
        run_key,
        problem_key,
        transfer_key,
        target_move_key,
        family_key,
    )
    grouped: dict[
        tuple[Any, Any],
        dict[str, list[tuple[float, Mapping[str, Any]]]],
    ] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(problem_level_rows):
        _require_row_keys(row, required, row_index=index)
        condition = str(row[condition_key])
        if condition not in {base_label, plain_label, reasoning_label}:
            continue
        key = (row[run_key], row[problem_key])
        grouped[key][condition].append(
            (
                _finite_float(
                    row[value_field],
                    label=f"row {index} field {value_field!r}",
                ),
                row,
            )
        )
    if not grouped:
        raise ExpansionStatsError("no recognized capability condition rows were provided")

    contrasts: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for (run_id, problem_id), conditions in grouped.items():
        missing = [
            label
            for label in (base_label, plain_label, reasoning_label)
            if label not in conditions
        ]
        if missing:
            incomplete.append(
                {
                    "run_id": run_id,
                    "problem_id": problem_id,
                    "missing_conditions": missing,
                }
            )
            continue
        metadata_fields = {
            "transfer_level": transfer_key,
            "target_reasoning_move": target_move_key,
            "family_id": family_key,
        }
        metadata: dict[str, Any] = {}
        metadata_mismatches: dict[str, list[Any]] = {}
        for output_field, source_field in metadata_fields.items():
            values = {
                str(row.get(source_field))
                for condition_rows in conditions.values()
                for _, row in condition_rows
            }
            if len(values) != 1:
                metadata_mismatches[output_field] = sorted(values)
            else:
                metadata[output_field] = next(
                    row.get(source_field)
                    for condition_rows in conditions.values()
                    for _, row in condition_rows
                )
        if metadata_mismatches:
            incomplete.append(
                {
                    "run_id": run_id,
                    "problem_id": problem_id,
                    "missing_conditions": [],
                    "metadata_mismatches": metadata_mismatches,
                }
            )
            continue
        base = float(np.mean([value for value, _ in conditions[base_label]]))
        plain = float(np.mean([value for value, _ in conditions[plain_label]]))
        reasoning = float(
            np.mean([value for value, _ in conditions[reasoning_label]])
        )
        contrasts.append(
            {
                "run_id": run_id,
                "problem_id": problem_id,
                **metadata,
                "base_accuracy": base,
                "plain_accuracy": plain,
                "reasoning_accuracy": reasoning,
                "reasoning_gain": reasoning - base,
                "plain_gain": plain - base,
                "did": (reasoning - base) - (plain - base),
            }
        )
    if not contrasts:
        raise ExpansionStatsError(
            "no run/problem has all base, plain, and reasoning conditions"
        )

    by_level: dict[str, dict[str, Any]] = {}
    levels = sorted(
        set(row["transfer_level"] for row in contrasts),
        key=str,
    )
    for level_index, level in enumerate(levels):
        level_rows = [row for row in contrasts if row["transfer_level"] == level]
        by_level[str(level)] = _summarize_capability_stratum(
            level_rows,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed + level_index + 1,
        )

    by_move: dict[str, dict[str, Any]] = {}
    moves = sorted(
        {
            str(row["target_reasoning_move"])
            for row in contrasts
            if row.get("target_reasoning_move") is not None
        }
    )
    for move_index, move in enumerate(moves):
        move_rows = [
            row
            for row in contrasts
            if str(row.get("target_reasoning_move")) == move
        ]
        by_move[move] = _summarize_capability_stratum(
            move_rows,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed + len(levels) + move_index + 1,
        )

    return {
        "overall": _summarize_capability_stratum(
            contrasts,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        ),
        "by_transfer_level": by_level,
        "by_target_reasoning_move": by_move,
        "problem_contrasts": contrasts,
        "incomplete_units": incomplete,
        "all_condition_units_complete": not incomplete,
    }


DEFAULT_COMPUTE_EQUAL_FIELDS: tuple[str, ...] = (
    "independent_run_id",
    "base_checkpoint",
    "training_instance_count",
    "training_token_count",
    "optimizer",
    "learning_rate",
    "update_steps",
    "batch_size",
    "batch_composition",
    "verifier",
    "max_rollout_length",
    "total_compute",
    "resume_mode",
)

DEFAULT_COMPUTE_PROVENANCE_ARTIFACTS: tuple[str, ...] = (
    "base_checkpoint",
    "training_data",
    "training_log",
    "output_checkpoint",
)


def _nested_get(mapping: Mapping[str, Any], dotted_key: str) -> tuple[bool, Any]:
    current: Any = mapping
    for component in dotted_key.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return False, None
        current = current[component]
    return True, current


def _manifest_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12))
    return left == right


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _valid_explicit_provenance(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(
        isinstance(value.get(field), str) and bool(str(value[field]).strip())
        for field in ("artifact_id", "immutable_ref")
    )


def _artifact_provenance_check(
    condition: Mapping[str, Any],
    *,
    condition_label: str,
    artifact: str,
) -> dict[str, Any]:
    hash_field = f"{artifact}_sha256"
    provenance_field = f"{artifact}_provenance"
    digest = condition.get(hash_field)
    provenance = condition.get(provenance_field)
    digest_valid = _valid_sha256(digest)
    explicit_valid = _valid_explicit_provenance(provenance)
    return {
        "condition": condition_label,
        "artifact": artifact,
        "hash_field": hash_field,
        "sha256": digest,
        "sha256_valid": digest_valid,
        "provenance_field": provenance_field,
        "explicit_provenance": provenance,
        "explicit_provenance_valid": explicit_valid,
        "passed": bool(digest_valid or explicit_valid),
        "identity": (
            {"kind": "sha256", "value": str(digest).lower()}
            if digest_valid
            else (
                {"kind": "explicit_provenance", "value": provenance}
                if explicit_valid
                else None
            )
        ),
    }


def audit_same_compute_manifest(
    manifest: Mapping[str, Any],
    *,
    plain_label: str = "plain",
    reasoning_label: str = "reasoning",
    condition_container: str = "conditions",
    equal_fields: Sequence[str] = DEFAULT_COMPUTE_EQUAL_FIELDS,
    provenance_artifacts: Sequence[str] = DEFAULT_COMPUTE_PROVENANCE_ARTIFACTS,
) -> dict[str, Any]:
    """Audit that plain and reasoning training runs use identical compute.

    Expected shape::

        {
          "conditions": {
            "plain": {"base_checkpoint": ..., "training_token_count": ...},
            "reasoning": {"base_checkpoint": ..., "training_token_count": ...}
          }
        }

    ``equal_fields`` supports dotted paths for nested manifests.  Missing
    fields fail the audit rather than being treated as equal.  Every condition
    must additionally identify the base checkpoint, training dataset, training
    log, and output checkpoint with either a SHA-256 digest or an explicit
    immutable provenance object containing ``artifact_id`` and
    ``immutable_ref``.  Artifact hashes are not compared for condition-specific
    datasets/logs/checkpoints; only their presence and immutability are audited.
    """

    if not isinstance(manifest, Mapping):
        raise ExpansionStatsError("manifest must be a mapping")
    container: Any = manifest.get(condition_container)
    if not isinstance(container, Mapping):
        raise ExpansionStatsError(
            f"manifest[{condition_container!r}] must be a condition mapping"
        )
    if plain_label not in container or reasoning_label not in container:
        raise ExpansionStatsError(
            f"manifest must contain {plain_label!r} and {reasoning_label!r} conditions"
        )
    plain = container[plain_label]
    reasoning = container[reasoning_label]
    if not isinstance(plain, Mapping) or not isinstance(reasoning, Mapping):
        raise ExpansionStatsError("each condition manifest must be a mapping")
    if not equal_fields:
        raise ExpansionStatsError("equal_fields must be non-empty")
    if not provenance_artifacts:
        raise ExpansionStatsError("provenance_artifacts must be non-empty")

    checks: list[dict[str, Any]] = []
    for field in equal_fields:
        plain_present, plain_value = _nested_get(plain, field)
        reasoning_present, reasoning_value = _nested_get(reasoning, field)
        equal = (
            plain_present
            and reasoning_present
            and _manifest_values_equal(plain_value, reasoning_value)
        )
        checks.append(
            {
                "field": field,
                "plain_present": plain_present,
                "reasoning_present": reasoning_present,
                "plain": plain_value,
                "reasoning": reasoning_value,
                "equal": bool(equal),
            }
        )

    failed = [check["field"] for check in checks if not check["equal"]]
    provenance_checks = [
        _artifact_provenance_check(
            condition,
            condition_label=label,
            artifact=str(artifact),
        )
        for label, condition in (
            (plain_label, plain),
            (reasoning_label, reasoning),
        )
        for artifact in provenance_artifacts
    ]
    failed_provenance = [
        f"{check['condition']}.{check['artifact']}"
        for check in provenance_checks
        if not check["passed"]
    ]
    base_identities = [
        check["identity"]
        for check in provenance_checks
        if check["artifact"] == "base_checkpoint" and check["passed"]
    ]
    base_checkpoint_identity_equal = bool(
        len(base_identities) == 2
        and _manifest_values_equal(base_identities[0], base_identities[1])
    )
    if not base_checkpoint_identity_equal:
        failed_provenance.append("base_checkpoint.identity_mismatch")
    condition_artifact_identity_distinct: dict[str, bool] = {}
    for artifact in ("training_data", "training_log", "output_checkpoint"):
        identities = {
            check["condition"]: check["identity"]
            for check in provenance_checks
            if check["artifact"] == artifact and check["passed"]
        }
        distinct = bool(
            set(identities) == {plain_label, reasoning_label}
            and not _manifest_values_equal(
                identities[plain_label],
                identities[reasoning_label],
            )
        )
        condition_artifact_identity_distinct[artifact] = distinct
        if not distinct:
            failed_provenance.append(f"{artifact}.condition_identity_not_distinct")
    required_value_failures: list[str] = []
    if plain.get("resume_mode") != "disable" or reasoning.get("resume_mode") != "disable":
        required_value_failures.append("resume_mode_must_be_disable")
    return {
        "passed": (
            not failed
            and not failed_provenance
            and not required_value_failures
        ),
        "plain_label": plain_label,
        "reasoning_label": reasoning_label,
        "checks": checks,
        "failed_fields": failed,
        "provenance_checks": provenance_checks,
        "failed_provenance": failed_provenance,
        "artifact_provenance_complete": not failed_provenance,
        "base_checkpoint_identity_equal": base_checkpoint_identity_equal,
        "condition_artifact_identity_distinct": (
            condition_artifact_identity_distinct
        ),
        "required_value_failures": required_value_failures,
    }


__all__ = [
    "DEFAULT_COMPUTE_EQUAL_FIELDS",
    "DEFAULT_COMPUTE_PROVENANCE_ARTIFACTS",
    "ExpansionStatsError",
    "PCASubspace",
    "aggregate_generator_metrics",
    "aligned_orthogonal_norms",
    "analyze_capability_did",
    "audit_same_compute_manifest",
    "controls_adjusted_paired_ridge",
    "cosine_coverage",
    "cosine_knn_novelty",
    "fit_plain_pca_subspace",
    "hierarchical_bootstrap_adjusted_paired_ridge",
    "hierarchical_bootstrap_paired_effect",
    "l2_normalize",
    "leave_one_out_kth_epsilon",
    "paired_condition_differences",
    "resolve_knn_k",
]
