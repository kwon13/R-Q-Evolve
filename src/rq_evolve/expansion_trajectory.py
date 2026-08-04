"""Optional internal-trajectory diagnostics for the expansion experiment.

StALT is a secondary trajectory statistic, never a semantic novelty or
capability metric.  Callers must supply hidden states for *generated response
tokens only* with shape ``[T, L+1, D]``: token, embedding-plus-layers, hidden
dimension.  Prompt positions must be removed before calling these functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class TrajectoryAnalysisError(ValueError):
    pass


def _hidden_tensor(hidden_states: Any) -> np.ndarray:
    values = np.asarray(hidden_states, dtype=np.float64)
    if values.ndim != 3:
        raise TrajectoryAnalysisError(
            "hidden_states must have shape [generated_tokens, layers+1, hidden]"
        )
    if values.shape[0] < 2:
        raise TrajectoryAnalysisError("at least two generated tokens are required")
    if values.shape[1] < 2:
        raise TrajectoryAnalysisError(
            "hidden_states must contain embedding layer plus at least one block"
        )
    if values.shape[2] < 1 or not np.all(np.isfinite(values)):
        raise TrajectoryAnalysisError("hidden_states must be finite and non-empty")
    return values


def spatiotemporal_transitions(
    hidden_states: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the paper's temporal and layer-wise transition grids.

    ``delta_time`` has shape ``[T-1, L+1]`` and ``delta_layer`` has shape
    ``[T, L]``.
    """

    hidden = _hidden_tensor(hidden_states)
    delta_time = np.linalg.norm(hidden[1:] - hidden[:-1], axis=-1)
    delta_layer = np.linalg.norm(hidden[:, 1:] - hidden[:, :-1], axis=-1)
    return delta_time, delta_layer


def compute_stalt(
    hidden_states: Any,
    *,
    tau: float = 1.0,
) -> dict[str, Any]:
    """Compute StALT and transparent component diagnostics.

    This follows Furuya and Tanimura's aggregation order: remove embedding
    temporal changes, align at tokens 2..T, softmax the within-token
    layer-transition amplitudes, then use those weights to average temporal
    changes over layers and finally over time.
    """

    if not np.isfinite(tau) or tau <= 0:
        raise TrajectoryAnalysisError("tau must be a positive finite value")
    hidden = _hidden_tensor(hidden_states)
    delta_time, delta_layer = spatiotemporal_transitions(hidden)
    return compute_stalt_from_transitions(
        delta_time,
        delta_layer[1:],
        tau=tau,
    )


def compute_stalt_from_transitions(
    delta_time: Any,
    aligned_delta_layer: Any,
    *,
    tau: float = 1.0,
) -> dict[str, Any]:
    """Compute StALT from streamed transition norms.

    This avoids retaining a full ``[tokens, layers, hidden]`` tensor for large
    language models. ``delta_time`` must have shape ``[T-1, L+1]`` and contain
    temporal norms for the embedding plus every decoder layer.
    ``aligned_delta_layer`` must have shape ``[T-1, L]`` and contain adjacent
    layer norms at response tokens 2..T.
    """

    if not np.isfinite(tau) or tau <= 0:
        raise TrajectoryAnalysisError("tau must be a positive finite value")
    temporal = np.asarray(delta_time, dtype=np.float64)
    layer = np.asarray(aligned_delta_layer, dtype=np.float64)
    if temporal.ndim != 2 or temporal.shape[0] < 1 or temporal.shape[1] < 2:
        raise TrajectoryAnalysisError(
            "delta_time must have shape [T-1, layers+1]"
        )
    expected = (temporal.shape[0], temporal.shape[1] - 1)
    if layer.shape != expected:
        raise TrajectoryAnalysisError(
            "aligned_delta_layer must have shape "
            f"{expected}, got {layer.shape}"
        )
    if not np.all(np.isfinite(temporal)) or not np.all(np.isfinite(layer)):
        raise TrajectoryAnalysisError("transition grids must be finite")
    aligned_time = temporal[:, 1:]
    aligned_layer = layer
    logits = aligned_layer / float(tau)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    weights = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    token_amplitude = np.sum(weights * aligned_time, axis=1)
    layer_saliency = weights.mean(axis=0)
    layer_concentration = np.sum(np.square(weights), axis=1)
    return {
        "stalt": float(token_amplitude.mean()),
        "tau": float(tau),
        "generated_tokens": int(temporal.shape[0] + 1),
        "num_decoder_layers": int(temporal.shape[1] - 1),
        "token_wise_amplitude": token_amplitude,
        "temporal_path_length": float(token_amplitude.sum()),
        "mean_unweighted_temporal_amplitude": float(aligned_time.mean()),
        "mean_unweighted_layer_amplitude": float(aligned_layer.mean()),
        "layer_saliency": layer_saliency,
        "mean_layer_concentration_hhi": float(layer_concentration.mean()),
        "peak_saliency_layer_zero_based": int(np.argmax(layer_saliency)),
    }


def interpolate_token_series(values: Any, *, bins: int = 100) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or series.size < 1 or not np.all(np.isfinite(series)):
        raise TrajectoryAnalysisError("token series must be one finite 1-D array")
    if bins < 2:
        raise TrajectoryAnalysisError("bins must be >= 2")
    if series.size == 1:
        return np.repeat(series, bins)
    source = np.linspace(0.0, 1.0, num=series.size)
    target = np.linspace(0.0, 1.0, num=bins)
    return np.interp(target, source, series)


def interpolate_hidden_trajectory(
    hidden_states: Any,
    *,
    bins: int = 100,
    include_embedding: bool = False,
) -> np.ndarray:
    """Interpolate a response trajectory to a common relative-token grid.

    The returned tensor has shape ``[bins, layers, hidden]``.  This alignment
    is a descriptive comparison of differently sized responses, not a claim
    that interpolated positions contain identical tokens or semantics.
    """

    hidden = _hidden_tensor(hidden_states)
    if bins < 2:
        raise TrajectoryAnalysisError("bins must be >= 2")
    if not include_embedding:
        hidden = hidden[:, 1:]
    source = np.linspace(0.0, 1.0, num=hidden.shape[0])
    target = np.linspace(0.0, 1.0, num=bins)
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 1, hidden.shape[0] - 1)
    left = right - 1
    denominator = source[right] - source[left]
    fraction = ((target - source[left]) / denominator)[:, None, None]
    return hidden[left] + fraction * (hidden[right] - hidden[left])


def correct_wrong_divergence_onset(
    correct_hidden_states: Any,
    wrong_hidden_states: Any,
    *,
    absolute_threshold: float,
    bins: int = 100,
    tau: float = 1.0,
) -> dict[str, Any]:
    """Locate the first preregistered hidden-vector trajectory divergence.

    ``absolute_threshold`` must be fixed before inspecting the matched pair;
    choosing it from the observed maximum would make onset circular.
    """

    if not np.isfinite(absolute_threshold) or absolute_threshold < 0:
        raise TrajectoryAnalysisError(
            "absolute_threshold must be a non-negative finite value"
    )
    correct = compute_stalt(correct_hidden_states, tau=tau)
    wrong = compute_stalt(wrong_hidden_states, tau=tau)
    correct_amplitude = interpolate_token_series(
        correct["token_wise_amplitude"],
        bins=bins,
    )
    wrong_amplitude = interpolate_token_series(
        wrong["token_wise_amplitude"],
        bins=bins,
    )
    correct_trajectory = interpolate_hidden_trajectory(
        correct_hidden_states,
        bins=bins,
    )
    wrong_trajectory = interpolate_hidden_trajectory(
        wrong_hidden_states,
        bins=bins,
    )
    if correct_trajectory.shape[1:] != wrong_trajectory.shape[1:]:
        raise TrajectoryAnalysisError(
            "correct and wrong trajectories must use the same layers/hidden size"
        )
    per_layer_distance = np.linalg.norm(
        correct_trajectory - wrong_trajectory,
        axis=-1,
    )
    trajectory_distance = np.sqrt(
        np.mean(np.square(per_layer_distance), axis=1)
    )
    hits = np.flatnonzero(trajectory_distance > absolute_threshold)
    onset_index = int(hits[0]) if hits.size else None
    onset_relative = (
        float(onset_index / (bins - 1)) if onset_index is not None else None
    )
    return {
        "correct_stalt": correct["stalt"],
        "wrong_stalt": wrong["stalt"],
        "absolute_threshold": float(absolute_threshold),
        "bins": int(bins),
        "divergence_metric": (
            "rms_l2_hidden_vector_distance_across_decoder_layers"
        ),
        "divergence_onset_bin": onset_index,
        "divergence_onset_relative_position": onset_relative,
        "interpolated_hidden_trajectory_distance": trajectory_distance,
        "per_layer_hidden_trajectory_distance": per_layer_distance,
        "interpolated_stalt_amplitude_difference": np.abs(
            correct_amplitude - wrong_amplitude
        ),
        "alignment_caveat": (
            "relative-token interpolation is descriptive and does not align "
            "token semantics"
        ),
    }


def summarize_trajectory_records(
    records: Sequence[Mapping[str, Any]],
    *,
    hidden_state_key: str = "hidden_states",
    correct_key: str = "correct",
    tau: float = 1.0,
) -> dict[str, Any]:
    """Summarize StALT without treating trajectories as capability evidence."""

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if hidden_state_key not in record or correct_key not in record:
            raise TrajectoryAnalysisError(
                f"trajectory record {index} is missing hidden states/correctness"
            )
        metrics = compute_stalt(record[hidden_state_key], tau=tau)
        rows.append(
            {
                "trajectory_id": record.get("trajectory_id", str(index)),
                "correct": bool(record[correct_key]),
                "stalt": metrics["stalt"],
                "generated_tokens": metrics["generated_tokens"],
                "temporal_path_length": metrics["temporal_path_length"],
                "mean_layer_concentration_hhi": metrics[
                    "mean_layer_concentration_hhi"
                ],
            }
        )
    correct_values = [row["stalt"] for row in rows if row["correct"]]
    wrong_values = [row["stalt"] for row in rows if not row["correct"]]
    return {
        "trajectory_rows": rows,
        "correct_mean_stalt": (
            float(np.mean(correct_values)) if correct_values else None
        ),
        "wrong_mean_stalt": (
            float(np.mean(wrong_values)) if wrong_values else None
        ),
        "correct_wrong_stalt_difference": (
            float(np.mean(correct_values) - np.mean(wrong_values))
            if correct_values and wrong_values
            else None
        ),
        "interpretation": (
            "auxiliary internal-dynamics diagnostic; not semantic novelty or "
            "post-training capability evidence"
        ),
    }


__all__ = [
    "TrajectoryAnalysisError",
    "compute_stalt",
    "compute_stalt_from_transitions",
    "correct_wrong_divergence_onset",
    "interpolate_hidden_trajectory",
    "interpolate_token_series",
    "spatiotemporal_transitions",
    "summarize_trajectory_records",
]
