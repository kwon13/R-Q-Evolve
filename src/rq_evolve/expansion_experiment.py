"""Data contracts for the reasoning-informed expansion experiment.

This module deliberately keeps *generation units* separate from the multiple
problem instances and Solver rollouts emitted by one generator.  The latter are
nested observations, not independent experimental replicates.

The adapter understands the artifacts written by
``scripts/compare_mutation_methods_vllm.py``.  It does not silently turn missing
or invalid generators into zero-displacement observations.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .program import ProblemProgram
from .prompts import SOLVER_SYSTEM_PROMPT

SCHEMA_VERSION = 1
MAX_CONFIRMATORY_CODE_INPUT_TOKEN_DELTA = 64
RQ_PROXY_KIND = "mean_token_negative_logprob_proxy"
RQ_SCALAR_OBJECTIVE_KIND = "actor_logit_entropy_scalar_objective"
CONDITIONS = {"legacy": "plain", "metacognitive": "reasoning"}
TRANSFER_LEVELS = {
    "in_family",
    "structural",
    "cross_domain",
    "archive",
    "benchmark",
}


def stable_id(*parts: Any, length: int = 16) -> str:
    """Return a deterministic identifier without exposing problem contents."""

    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


def _normalize_optional_sha256(value: Any, *, field: str) -> str | None:
    """Validate an externally supplied immutable digest without reading it."""

    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value.strip()) is None:
        raise ValueError(f"{field} must be exactly 64 hexadecimal characters")
    return value.strip().lower()


def numeric_summary(problem_text: str) -> dict[str, float | int | None]:
    """Extract transparent surface-number controls from a problem statement."""

    values = [float(match.group(0)) for match in _NUMBER_RE.finditer(problem_text)]
    if not values:
        return {
            "numeric_count": 0,
            "numeric_min": None,
            "numeric_max": None,
            "numeric_span": 0.0,
            "numeric_max_abs": 0.0,
            "numeric_range_source": "text_regex",
        }
    return {
        "numeric_count": len(values),
        "numeric_min": min(values),
        "numeric_max": max(values),
        "numeric_span": max(values) - min(values),
        "numeric_max_abs": max(abs(value) for value in values),
        "numeric_range_source": "text_regex",
    }


def _load_parent(
    comparison_root: Path,
    parent_program: str | Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores_path = comparison_root / "parent_scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"missing comparison artifact: {scores_path}")
    parent_scores = read_json(scores_path)
    seed_rows = parent_scores.get("per_seed") or []
    if not seed_rows:
        raise ValueError(f"{scores_path} has no per_seed parent instances")

    if parent_program is not None:
        program = ProblemProgram.from_file(parent_program)
        parent_id = program.program_id
        concept_group = program.get_concept_group()
        concept_type = program.get_concept_type()
        source_path = str(Path(parent_program).resolve())
    else:
        fingerprint = [
            (row.get("seed"), row.get("problem"), row.get("answer"))
            for row in seed_rows
        ]
        parent_id = stable_id("inferred-parent", json.dumps(fingerprint))
        concept_group = None
        concept_type = None
        source_path = None

    metadata = {
        "parent_program_id": parent_id,
        "parent_concept_group": concept_group,
        "parent_concept_type": concept_type,
        "parent_source_path": source_path,
        "parent_status": parent_scores.get("status"),
        "parent_p_hat": parent_scores.get("p_hat"),
        "parent_rq_proxy": parent_scores.get("rq_proxy"),
    }
    return metadata, seed_rows


def operator_contract_valid(
    operator: str,
    *,
    parent_group: str | None,
    parent_type: str | None,
    child_group: str | None,
    child_type: str | None,
) -> bool:
    """Apply the same concept contract to both mutation methods."""

    if not all((parent_group, parent_type, child_group, child_type)):
        return False
    if operator == "in_depth":
        return child_group == parent_group and child_type == parent_type
    if operator == "in_breadth":
        return child_group != parent_group
    return False


def _seed_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        seed = int(row["seed"])
        if seed in result:
            raise ValueError(f"duplicate seed {seed} in comparison artifact")
        result[seed] = row
    return result


def _source_value(
    source_manifest: Mapping[str, Any],
    source_sampling: Mapping[str, Any],
    *names: str,
) -> Any:
    """Read an optional provenance field without manufacturing a default."""

    for name in names:
        if name in source_sampling:
            return source_sampling[name]
        if name in source_manifest:
            return source_manifest[name]
    return None


def _normalized_seed_range(values: Sequence[Any]) -> list[int] | None:
    seeds = sorted({int(value) for value in values if value is not None})
    if not seeds:
        return None
    return [seeds[0], seeds[-1]]


def _score_rq(score: Mapping[str, Any]) -> tuple[Any, str | None]:
    """Keep the true scalar objective distinct from a standalone NLL proxy."""

    if score.get("rq") is not None:
        return score.get("rq"), str(
            score.get("rq_kind") or RQ_SCALAR_OBJECTIVE_KIND
        )
    if score.get("rq_proxy") is not None:
        return score.get("rq_proxy"), RQ_PROXY_KIND
    return None, None


def _rollout_rows(
    comparison_root: Path,
    *,
    instance_ids_by_seed: Mapping[int, str],
    relative_path: str,
    condition: str,
    generator_unit_id: str,
    solver_checkpoint_id: str | None,
    source_run_manifest_sha256: str | None,
) -> list[dict[str, Any]]:
    artifact_path = comparison_root / relative_path
    if not artifact_path.exists():
        return []
    artifact = read_json(artifact_path)
    result: list[dict[str, Any]] = []
    for seed_row in artifact.get("per_seed") or []:
        seed = int(seed_row["seed"])
        instance_id = instance_ids_by_seed.get(seed)
        if instance_id is None:
            continue
        for rollout in seed_row.get("rollouts") or []:
            result.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "instance_id": instance_id,
                    "generator_unit_id": generator_unit_id,
                    "condition": condition,
                    "instance_seed": seed,
                    "rollout_idx": int(rollout.get("rollout_idx", 0)),
                    "solver_sampling_seed": rollout.get("sampling_seed"),
                    "response": rollout.get("text", ""),
                    "cleaned_response": rollout.get(
                        "cleaned_text", rollout.get("text", "")
                    ),
                    "predicted_answer": rollout.get("predicted_answer"),
                    "correct": bool(rollout.get("correct", False)),
                    "cumulative_logprob": rollout.get("cumulative_logprob"),
                    "mean_negative_logprob": rollout.get(
                        "mean_negative_logprob"
                    ),
                    "solver_checkpoint_id": solver_checkpoint_id,
                    "source_run_manifest_sha256": source_run_manifest_sha256,
                    "source_path": str(artifact_path.resolve()),
                }
            )
    return result


def prepare_comparison_manifests(
    comparison_root: str | Path,
    *,
    output_dir: str | Path,
    parent_program: str | Path | None = None,
    run_id: str | None = None,
    evolution_iteration: int = 0,
    generator_draw_idx: int | None = None,
    solver_checkpoint_id: str | None = None,
    solver_checkpoint_sha256: str | None = None,
    generator_checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Normalize one legacy-vs-metacognitive comparison directory.

    The current notebook artifact does not record an explicit child-to-parent
    instance lineage.  We preserve the useful same-seed mapping, but mark it as
    a post-hoc assumption in every child row.
    """

    comparison_root = Path(comparison_root).resolve()
    output_dir = Path(output_dir).resolve()
    summaries_path = comparison_root / "summaries.json"
    if not summaries_path.exists():
        raise FileNotFoundError(f"missing comparison artifact: {summaries_path}")
    summaries = read_json(summaries_path)
    if not isinstance(summaries, list):
        raise ValueError(f"{summaries_path} must contain a JSON list")
    source_manifest_path = comparison_root / "manifest.json"
    source_manifest = (
        read_json(source_manifest_path) if source_manifest_path.exists() else {}
    )
    source_sampling = (
        source_manifest.get("sampling", {})
        if isinstance(source_manifest, dict)
        else {}
    )
    source_model_id = (
        source_manifest.get("model") if isinstance(source_manifest, dict) else None
    )
    if solver_checkpoint_id is None and isinstance(source_manifest, dict):
        solver_checkpoint_id = source_model_id
    if solver_checkpoint_sha256 is None and isinstance(source_manifest, dict):
        solver_checkpoint_sha256 = source_manifest.get(
            "solver_checkpoint_sha256"
        )
    solver_checkpoint_sha256 = _normalize_optional_sha256(
        solver_checkpoint_sha256,
        field="solver_checkpoint_sha256",
    )
    if generator_checkpoint_id is None and isinstance(source_manifest, dict):
        generator_checkpoint_id = source_model_id
    if generator_draw_idx is None:
        generator_draw_idx = int(source_manifest.get("llm_seed", 0))

    run_id = run_id or comparison_root.name
    parent_meta, parent_seed_rows = _load_parent(comparison_root, parent_program)
    parent_id = parent_meta["parent_program_id"]
    parent_group = parent_meta["parent_concept_group"]
    parent_type = parent_meta["parent_concept_type"]
    recorded_parent_id = (
        source_manifest.get("parent_program_id")
        if isinstance(source_manifest, dict)
        else None
    )
    if recorded_parent_id and str(recorded_parent_id) != str(parent_id):
        raise ValueError(
            "parent program does not match the comparison manifest: "
            f"{parent_id} != {recorded_parent_id}"
        )

    generator_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    representation_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    source_manifest_sha256 = (
        hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
        if source_manifest_path.exists()
        else None
    )
    source_evaluation_seeds = list(
        _source_value(
            source_manifest,
            source_sampling,
            "evaluation_seeds",
        )
        or []
    )
    source_llm_seed = (
        source_manifest.get("llm_seed")
        if isinstance(source_manifest, dict)
        else None
    )
    sampling_seed_policy = (
        source_manifest.get("sampling_seed_policy")
        if isinstance(source_manifest, dict)
        else None
    )
    if sampling_seed_policy is None and source_llm_seed is not None:
        sampling_seed_policy = "recorded_vllm_engine_seed_per_run"
    generator_sampling_seed_range = (
        source_manifest.get("llm_seed_range")
        if isinstance(source_manifest, dict)
        else None
    )
    if generator_sampling_seed_range is None and source_llm_seed is not None:
        generator_sampling_seed_range = [
            int(source_llm_seed),
            int(source_llm_seed),
        ]
    instance_seed_policy = (
        source_manifest.get("instance_seed_policy")
        if isinstance(source_manifest, dict)
        else None
    )
    if instance_seed_policy is None and source_evaluation_seeds:
        instance_seed_policy = "explicit_evaluation_seed_list"
    source_instance_seed_range = _normalized_seed_range(source_evaluation_seeds)
    source_acceptance_budget = _source_value(
        source_manifest,
        source_sampling,
        "acceptance_budget",
        "max_acceptance_attempts",
    )
    source_child_rollout_budget = _source_value(
        source_manifest,
        source_sampling,
        "child_rollouts",
    )
    source_plan_max_tokens = _source_value(
        source_manifest,
        source_sampling,
        "plan_max_tokens",
    )
    source_code_max_tokens = _source_value(
        source_manifest,
        source_sampling,
        "mutation_max_tokens",
        "code_max_tokens",
    )

    if (
        source_model_id is not None
        and solver_checkpoint_id is not None
        and str(source_model_id) != str(solver_checkpoint_id)
    ):
        warnings.append(
            "source_solver_checkpoint_mismatch: source comparison model "
            f"{source_model_id!r} differs from normalized solver checkpoint "
            f"{solver_checkpoint_id!r}."
        )
    if (
        source_model_id is not None
        and generator_checkpoint_id is not None
        and str(source_model_id) != str(generator_checkpoint_id)
    ):
        warnings.append(
            "source_generator_checkpoint_mismatch: source comparison model "
            f"{source_model_id!r} differs from normalized generator checkpoint "
            f"{generator_checkpoint_id!r}."
        )

    parent_generator_id = stable_id(run_id, parent_id, "parent")
    generator_rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": output_dir.name,
            "independent_run_id": run_id,
            "evolution_iteration": int(evolution_iteration),
            "parent_program_id": parent_id,
            "parent_concept_group": parent_group,
            "parent_concept_type": parent_type,
            "conditioning_parent_seed": None,
            "operator": "parent",
            "condition": "parent",
            "method": "parent",
            "generator_draw_idx": 0,
            "generator_pair_id": None,
            "generator_unit_id": parent_generator_id,
            "plan_id": None,
            "plan_valid": None,
            "plan_invalid_reason": None,
            "program_id": parent_id,
            "code_valid": True,
            "code_invalid_reason": None,
            "operator_contract_valid": True,
            "evaluator_passed": None,
            "evaluator_total": None,
            "overall_status": parent_meta["parent_status"],
            "solver_checkpoint_id": solver_checkpoint_id,
            "solver_checkpoint_sha256": solver_checkpoint_sha256,
            "generator_checkpoint_id": generator_checkpoint_id,
            "source_model_id": source_model_id,
            "source_run_manifest_sha256": source_manifest_sha256,
            "generator_call_count": 0,
            "generator_candidate_draw_count": 0,
            "llm_generation_call_count": 0,
            "total_generator_max_token_budget": 0,
            "llm_seed": source_llm_seed,
            "sampling_seed_policy": sampling_seed_policy,
            "generator_sampling_seed_range": generator_sampling_seed_range,
            "instance_seed_policy": instance_seed_policy,
            "instance_seed_range": source_instance_seed_range,
            "plan_temperature": None,
            "plan_top_p": None,
            "code_temperature": None,
            "code_top_p": None,
            "plan_max_tokens": None,
            "code_max_tokens": None,
            "plan_stage_max_tokens": 0,
            "code_stage_max_tokens": 0,
            "evaluation_seeds": sorted(
                int(row["seed"]) for row in parent_seed_rows
            ),
            "requested_instance_budget": len(parent_seed_rows),
            "acceptance_budget": source_acceptance_budget,
            "child_rollout_budget": source_child_rollout_budget,
            "artifact_dir": str(comparison_root),
        }
    )

    parent_instance_ids: dict[int, str] = {}
    for row in parent_seed_rows:
        seed = int(row["seed"])
        instance_id = stable_id(run_id, parent_id, "parent", seed)
        parent_instance_ids[seed] = instance_id
        problem = str(row["problem"])
        instance = {
            "schema_version": SCHEMA_VERSION,
            "item_id": instance_id,
            "instance_id": instance_id,
            "role": "parent",
            "independent_run_id": run_id,
            "parent_program_id": parent_id,
            "generator_pair_id": None,
            "generator_unit_id": parent_generator_id,
            "generator_draw_idx": 0,
            "instance_pair_id": None,
            "parent_instance_id": instance_id,
            "instance_seed": seed,
            "parent_seed": seed,
            "evolution_iteration": int(evolution_iteration),
            "operator": "parent",
            "condition": "parent",
            "method": "parent",
            "generation_status": parent_meta["parent_status"],
            "code_valid": True,
            "operator_contract_valid": True,
            "evaluator_valid": None,
            "evaluator_reason": None,
            "valid_for_confirmatory": True,
            "problem_text": problem,
            "common_io_spec": SOLVER_SYSTEM_PROMPT,
            "answer": str(row["answer"]),
            "concept_group": parent_group,
            "concept_type": parent_type,
            "target_reasoning_move": None,
            "num_rollouts": row.get("num_rollouts"),
            "num_correct": row.get("num_correct"),
            "p_hat": row.get("p_hat"),
            "uncertainty_proxy": row.get("uncertainty_proxy"),
            "rq": row.get("rq_proxy"),
            "rq_kind": RQ_PROXY_KIND,
            "lineage_source": "observed_parent_instance",
            "source_path": str(
                (comparison_root / "parent_scores.json").resolve()
            ),
            **numeric_summary(problem),
        }
        instance_rows.append(instance)
        representation_rows.append(representation_input_row(instance))

    rollout_rows.extend(
        _rollout_rows(
            comparison_root,
            instance_ids_by_seed=parent_instance_ids,
            relative_path="parent_rollouts.json",
            condition="parent",
            generator_unit_id=parent_generator_id,
            solver_checkpoint_id=solver_checkpoint_id,
            source_run_manifest_sha256=source_manifest_sha256,
        )
    )

    for summary in summaries:
        operator = str(summary.get("operator", ""))
        method = str(summary.get("method", ""))
        condition = CONDITIONS.get(method)
        if condition is None:
            warnings.append(
                f"Skipped unknown method {method!r} for operator {operator!r}."
            )
            continue

        pair_id = stable_id(
            run_id,
            parent_id,
            operator,
            int(generator_draw_idx),
        )
        program_id = summary.get("program_id")
        unit_id = stable_id(pair_id, condition, program_id or "missing-program")
        status = str(summary.get("status", "unknown"))
        generated = summary.get("generated_instances") or []
        code_valid = bool(generated) and status not in {
            "invalid_plan",
            "invalid_code",
            "generation_failed",
        }
        child_group = summary.get("concept_group")
        child_type = summary.get("concept_type")
        contract_valid = (
            operator_contract_valid(
                operator,
                parent_group=parent_group,
                parent_type=parent_type,
                child_group=child_group,
                child_type=child_type,
            )
            if code_valid
            else False
        )
        evaluator_rows = _seed_lookup(summary.get("evaluator_by_seed") or [])
        evaluator_passed = summary.get("evaluator_passed")
        evaluator_total = summary.get("evaluator_total")
        generator_evaluator_valid = bool(summary.get("evaluator_valid", False))
        plan = summary.get("plan") if isinstance(summary.get("plan"), dict) else None
        configured_calls = summary.get(
            "configured_llm_generation_call_count"
        )
        if configured_calls is None:
            configured_calls = 1 if method == "legacy" else 2
        plan_expected = (
            int(configured_calls) >= 2
            or int(summary.get("plan_stage_max_tokens") or 0) > 0
        )
        plan_valid = plan is not None if plan_expected else None
        reason = summary.get("reason")
        evaluation_seeds = list(
            summary.get("evaluation_seeds") or source_evaluation_seeds
        )
        evaluation_seeds = sorted(int(value) for value in evaluation_seeds)
        # New comparison artifacts record the actual design. Older artifacts
        # retain the historical one-stage legacy vs two-stage metacognitive
        # fallback so their compute mismatch remains visible.
        llm_generation_call_count = int(configured_calls)
        plan_stage_max_tokens = summary.get(
            "plan_stage_max_tokens",
            0 if method == "legacy" else source_plan_max_tokens,
        )
        code_stage_max_tokens = summary.get(
            "code_stage_max_tokens",
            source_code_max_tokens,
        )
        total_generator_max_token_budget = summary.get(
            "total_generator_max_token_budget"
        )
        if (
            total_generator_max_token_budget is None
            and code_stage_max_tokens is not None
            and plan_stage_max_tokens is not None
        ):
            total_generator_max_token_budget = int(code_stage_max_tokens) + int(
                plan_stage_max_tokens
            )
        conditioning_parent_seed = None
        if method == "metacognitive":
            evidence_path = comparison_root / operator / "planning_evidence.json"
            if evidence_path.exists():
                evidence = read_json(evidence_path)
                evidence_seeds = {
                    int(item["seed"])
                    for item in evidence
                    if isinstance(item, dict) and item.get("seed") is not None
                }
                if len(evidence_seeds) == 1:
                    conditioning_parent_seed = next(iter(evidence_seeds))

        generator_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": output_dir.name,
                "independent_run_id": run_id,
                "evolution_iteration": int(evolution_iteration),
                "parent_program_id": parent_id,
                "parent_concept_group": parent_group,
                "parent_concept_type": parent_type,
                "conditioning_parent_seed": conditioning_parent_seed,
                "operator": operator,
                "condition": condition,
                "method": method,
                "generator_draw_idx": int(generator_draw_idx),
                "generator_pair_id": pair_id,
                "generator_unit_id": unit_id,
                "plan_id": summary.get("plan_id"),
                "plan_valid": plan_valid,
                "plan_invalid_reason": reason if status == "invalid_plan" else None,
                "mutation_code_backend": summary.get(
                    "mutation_code_backend"
                ),
                "generation_path": summary.get("generation_path"),
                "generator_family": summary.get("generator_family"),
                "family_config": summary.get("family_config"),
                "compiler_registry_version": summary.get(
                    "compiler_registry_version"
                ),
                "compiler_status": summary.get("compiler_status"),
                "compiler_source_hash": summary.get(
                    "compiler_source_hash"
                ),
                "compiler_reasons": summary.get("compiler_reasons") or [],
                "quarantined": bool(summary.get("quarantined", False)),
                "program_id": program_id,
                "code_valid": code_valid,
                "code_invalid_reason": (
                    reason if not code_valid and status != "invalid_plan" else None
                ),
                "operator_contract_valid": contract_valid,
                "evaluator_passed": evaluator_passed,
                "evaluator_total": evaluator_total,
                "overall_status": status,
                "solver_checkpoint_id": solver_checkpoint_id,
                "solver_checkpoint_sha256": solver_checkpoint_sha256,
                "generator_checkpoint_id": generator_checkpoint_id,
                "source_model_id": source_model_id,
                "source_run_manifest_sha256": source_manifest_sha256,
                "generator_call_count": 1,
                "generator_candidate_draw_count": 1,
                "llm_generation_call_count": llm_generation_call_count,
                "actual_llm_generation_call_count": summary.get(
                    "llm_generation_call_count"
                ),
                "plan_input_token_count": summary.get(
                    "plan_input_token_count"
                ),
                "plan_input_token_target": summary.get(
                    "plan_input_token_target"
                ),
                "plan_input_token_delta": summary.get(
                    "plan_input_token_delta"
                ),
                "plan_neutral_padding_units": summary.get(
                    "plan_neutral_padding_units"
                ),
                "code_input_token_count": summary.get(
                    "code_input_token_count"
                ),
                "total_generator_input_token_count": summary.get(
                    "total_generator_input_token_count"
                ),
                "total_generator_max_token_budget": (
                    total_generator_max_token_budget
                ),
                "llm_seed": source_llm_seed,
                "paired_request_seed_policy": summary.get(
                    "paired_request_seed_policy"
                ),
                "plan_sampling_seed": summary.get("plan_sampling_seed"),
                "code_sampling_seed": summary.get("code_sampling_seed"),
                "evaluator_sampling_seed_policy": summary.get(
                    "evaluator_sampling_seed_policy"
                ),
                "evaluator_sampling_seed_sha256": summary.get(
                    "evaluator_sampling_seed_sha256"
                ),
                "evaluator_sampling_seed_range": summary.get(
                    "evaluator_sampling_seed_range"
                ),
                "child_solver_sampling_seed_policy": summary.get(
                    "child_solver_sampling_seed_policy"
                ),
                "child_solver_sampling_seed_sha256": summary.get(
                    "child_solver_sampling_seed_sha256"
                ),
                "child_solver_sampling_seed_range": summary.get(
                    "child_solver_sampling_seed_range"
                ),
                "sampling_seed_policy": sampling_seed_policy,
                "generator_sampling_seed_range": (
                    generator_sampling_seed_range
                ),
                "instance_seed_policy": instance_seed_policy,
                "instance_seed_range": _normalized_seed_range(evaluation_seeds),
                "plan_temperature": source_sampling.get("plan_temperature"),
                "plan_top_p": source_sampling.get("plan_top_p"),
                "code_temperature": source_sampling.get("code_temperature"),
                "code_top_p": source_sampling.get("code_top_p"),
                "plan_max_tokens": source_plan_max_tokens,
                "code_max_tokens": source_code_max_tokens,
                "plan_stage_max_tokens": plan_stage_max_tokens,
                "code_stage_max_tokens": code_stage_max_tokens,
                "evaluation_seeds": evaluation_seeds,
                "requested_instance_budget": (
                    len(evaluation_seeds) if evaluation_seeds else None
                ),
                "acceptance_budget": source_acceptance_budget,
                "child_rollout_budget": source_child_rollout_budget,
                "artifact_dir": str(
                    (comparison_root / str(summary.get("method_dir", ""))).resolve()
                ),
            }
        )

        if code_valid and not contract_valid:
            warnings.append(
                f"{operator}/{method} violates the operator concept contract "
                f"({parent_group}/{parent_type} -> "
                f"{child_group}/{child_type})."
            )
        if status == "evaluator_rejected":
            warnings.append(
                f"{operator}/{method} was evaluator_rejected; diagnostic "
                "code-valid analysis is selection-biased and non-confirmatory."
            )

        seed_scores = _seed_lookup(summary.get("seed_scores") or [])
        child_instance_ids: dict[int, str] = {}
        for generated_row in generated:
            seed = int(generated_row["seed"])
            parent_instance_id = parent_instance_ids.get(seed)
            if parent_instance_id is None:
                warnings.append(
                    f"{operator}/{method} seed={seed} has no same-seed parent."
                )
                continue
            instance_pair_id = stable_id(pair_id, seed)
            item_id = stable_id(unit_id, seed)
            child_instance_ids[seed] = item_id
            score = seed_scores.get(seed, {})
            rq_value, rq_kind = _score_rq(score)
            evaluator = evaluator_rows.get(seed, {})
            evaluator_valid = (
                bool(evaluator.get("valid")) if evaluator else None
            )
            problem = str(generated_row["problem"])
            valid_for_confirmatory = (
                status == "ok"
                and code_valid
                and contract_valid
                and generator_evaluator_valid
                and evaluator_valid is True
            )
            instance = {
                "schema_version": SCHEMA_VERSION,
                "item_id": item_id,
                "instance_id": item_id,
                "role": "child",
                "independent_run_id": run_id,
                "parent_program_id": parent_id,
                "generator_pair_id": pair_id,
                "generator_unit_id": unit_id,
                "generator_draw_idx": int(generator_draw_idx),
                "instance_pair_id": instance_pair_id,
                "parent_instance_id": parent_instance_id,
                "instance_seed": seed,
                "parent_seed": seed,
                "evolution_iteration": int(evolution_iteration),
                "operator": operator,
                "condition": condition,
                "method": method,
                "generation_status": status,
                "code_valid": code_valid,
                "operator_contract_valid": contract_valid,
                "evaluator_valid": evaluator_valid,
                "evaluator_reason": evaluator.get("reason") if evaluator else None,
                "valid_for_confirmatory": valid_for_confirmatory,
                "problem_text": problem,
                "common_io_spec": SOLVER_SYSTEM_PROMPT,
                "answer": str(generated_row["answer"]),
                "concept_group": child_group,
                "concept_type": child_type,
                "target_reasoning_move": (
                    plan.get("target_reasoning_move") if plan else None
                ),
                "generation_path": summary.get("generation_path"),
                "generator_family": summary.get("generator_family"),
                "quarantined": bool(summary.get("quarantined", False)),
                "num_rollouts": score.get("num_rollouts"),
                "num_correct": score.get("num_correct"),
                "p_hat": score.get("p_hat"),
                "uncertainty_proxy": score.get("uncertainty_proxy"),
                "rq": rq_value,
                "rq_kind": rq_kind,
                "lineage_source": "same_seed_posthoc_assumption",
                "source_path": str(
                    (
                        comparison_root
                        / str(summary.get("method_dir", ""))
                        / "child.py"
                    ).resolve()
                ),
                **numeric_summary(problem),
            }
            instance_rows.append(instance)
            representation_rows.append(representation_input_row(instance))

        rollout_rows.extend(
            _rollout_rows(
                comparison_root,
                instance_ids_by_seed=child_instance_ids,
                relative_path=(
                    f"{summary.get('method_dir', '')}/child_rollouts.json"
                ),
                condition=condition,
                generator_unit_id=unit_id,
                solver_checkpoint_id=solver_checkpoint_id,
                source_run_manifest_sha256=source_manifest_sha256,
            )
        )

    sufficiency = audit_generation_sufficiency(
        generator_rows,
        instance_rows,
    )
    warnings.extend(sufficiency["warnings"])
    warnings = list(dict.fromkeys(warnings))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": output_dir.name,
        "source_comparison_root": str(comparison_root),
        "source_run_manifest_path": (
            str(source_manifest_path.resolve())
            if source_manifest_path.exists()
            else None
        ),
        "source_run_manifest_sha256": (
            source_manifest_sha256
        ),
        "source_comparison_design": (
            source_manifest.get("comparison_design")
            if isinstance(source_manifest, dict)
            else None
        ),
        "source_evidence_gate_valid": (
            source_manifest.get("evidence_gate_valid")
            if isinstance(source_manifest, dict)
            else None
        ),
        "source_evidence_gate_issues": (
            source_manifest.get("evidence_gate_issues")
            if isinstance(source_manifest, dict)
            else None
        ),
        "source_parent_evidence_sampling_seed_policy": (
            source_manifest.get("parent_evidence_sampling_seed_policy")
            if isinstance(source_manifest, dict)
            else None
        ),
        "source_parent_evidence_sampling_seed_sha256": (
            source_manifest.get("parent_evidence_sampling_seed_sha256")
            if isinstance(source_manifest, dict)
            else None
        ),
        "source_parent_evidence_sampling_seed_range": (
            source_manifest.get("parent_evidence_sampling_seed_range")
            if isinstance(source_manifest, dict)
            else None
        ),
        "source_evidence_gate_version": (
            source_manifest.get("evidence_gate_version")
            if isinstance(source_manifest, dict)
            else None
        ),
        "source_prompt_files": (
            source_manifest.get("prompt_files")
            if isinstance(source_manifest, dict)
            else None
        ),
        "source_sampling": dict(source_sampling),
        "source_sampling_seed_policy": sampling_seed_policy,
        "source_paired_request_seed_policy": (
            source_manifest.get("paired_request_seed_policy")
            if isinstance(source_manifest, dict)
            else None
        ),
        "source_paired_request_seeds": (
            source_manifest.get("paired_request_seeds")
            if isinstance(source_manifest, dict)
            else None
        ),
        "source_instance_seed_policy": instance_seed_policy,
        "source_llm_seed": source_llm_seed,
        "source_llm_seed_range": generator_sampling_seed_range,
        "source_code_contract_version": (
            source_manifest.get("code_contract_version")
            if isinstance(source_manifest, dict)
            else None
        ),
        "source_code_contract": (
            source_manifest.get("code_contract")
            if isinstance(source_manifest, dict)
            else None
        ),
        "source_mutation_code_backend": (
            source_manifest.get("mutation_code_backend")
            if isinstance(source_manifest, dict)
            else None
        ),
        "source_plan_schema_version": (
            source_manifest.get("plan_schema_version")
            if isinstance(source_manifest, dict)
            else None
        ),
        "source_compiler_registry_version": (
            source_manifest.get("compiler_registry_version")
            if isinstance(source_manifest, dict)
            else None
        ),
        "source_paired_route_valid": (
            source_manifest.get("paired_route_valid")
            if isinstance(source_manifest, dict)
            else None
        ),
        "independent_run_id": run_id,
        "evolution_iteration": int(evolution_iteration),
        "solver_checkpoint_id": solver_checkpoint_id,
        "solver_checkpoint_sha256": solver_checkpoint_sha256,
        "solver_checkpoint_sha256_source": (
            "normalization_argument_or_source_manifest"
            if solver_checkpoint_sha256 is not None
            else None
        ),
        "generator_checkpoint_id": generator_checkpoint_id,
        "parent": parent_meta,
        "lineage_note": (
            "The source artifact did not record explicit child-instance lineage; "
            "same child/parent seed is retained as a post-hoc pairing assumption."
        ),
        "rq_note": (
            "rq values are mean-token negative-logprobability proxies, not the "
            "actor-logit entropy scalar objective used during training."
        ),
        "generator_compute_note": (
            "generator_call_count is a historical candidate-draw count. "
            "llm_generation_call_count and total_generator_max_token_budget "
            "record configured plan/code generation stages; "
            "actual_llm_generation_call_count records completed calls."
        ),
        "counts": {
            "generators": len(generator_rows),
            "instances": len(instance_rows),
            "rollouts": len(rollout_rows),
            "representation_inputs": len(representation_rows),
        },
        "sufficiency": sufficiency,
        "warnings": warnings,
    }
    if not source_manifest_path.exists():
        manifest["warnings"].append(
            "source_manifest_missing: generator temperatures, token budgets, "
            "and model provenance could not be recovered from this older run."
        )
    write_json(output_dir / "experiment_manifest.json", manifest)
    write_jsonl(output_dir / "generator_manifest.jsonl", generator_rows)
    write_jsonl(output_dir / "instance_manifest.jsonl", instance_rows)
    write_jsonl(output_dir / "rollout_manifest.jsonl", rollout_rows)
    write_jsonl(output_dir / "representation_inputs.jsonl", representation_rows)
    return manifest


def representation_input_row(instance: Mapping[str, Any]) -> dict[str, Any]:
    """Create a leakage-safe representation row.

    This row intentionally has no answer, mutation plan, reasoning trace,
    generator source, evaluator rationale, RQ score, or other analysis metadata.
    Identifiers are retained for joining the resulting vectors after inference.
    """

    required = (
        "item_id",
        "role",
        "problem_text",
        "common_io_spec",
        "independent_run_id",
        "parent_program_id",
        "generator_unit_id",
        "instance_seed",
    )
    missing = [name for name in required if name not in instance]
    if missing:
        raise ValueError(f"representation row is missing fields: {missing}")
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": instance["item_id"],
        "role": instance["role"],
        "independent_run_id": instance["independent_run_id"],
        "parent_program_id": instance["parent_program_id"],
        "generator_unit_id": instance["generator_unit_id"],
        "instance_seed": instance["instance_seed"],
        "common_io_spec": instance["common_io_spec"],
        "problem_text": instance["problem_text"],
    }


def select_child_instances(
    rows: Sequence[Mapping[str, Any]],
    *,
    validity_policy: str = "strict",
) -> list[dict[str, Any]]:
    """Select observed child locations without imputing invalid generators.

    ``code_valid`` exists only for explicitly labelled descriptive diagnostics.
    Confirmatory analysis must use ``strict``.
    """

    if validity_policy not in {"strict", "code_valid"}:
        raise ValueError("validity_policy must be 'strict' or 'code_valid'")
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("role") != "child":
            continue
        if validity_policy == "strict":
            keep = row.get("valid_for_confirmatory") is True
        else:
            keep = row.get("code_valid") is True
        if keep:
            selected.append(dict(row))
    return selected


def audit_generator_pair_design(
    generator_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit paired generation provenance and requested-compute parity.

    Outcome-dependent fields such as the number of accepted instances are not
    required to match here.  The *requested* design must match, including the
    actual number of plan/code LLM calls and their total maximum-token budget.
    Optional acceptance budgets are compared whenever either condition records
    one.
    """

    child_rows = [
        dict(row)
        for row in generator_rows
        if row.get("condition") in {"plain", "reasoning"}
    ]
    by_pair: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in child_rows:
        pair_id = row.get("generator_pair_id")
        if pair_id is None:
            continue
        by_pair[str(pair_id)][str(row["condition"])].append(row)

    required_equal_fields = (
        "solver_checkpoint_id",
        "solver_checkpoint_sha256",
        "generator_checkpoint_id",
        "source_model_id",
        "source_run_manifest_sha256",
        "generator_candidate_draw_count",
        "llm_generation_call_count",
        "total_generator_max_token_budget",
        "plan_stage_max_tokens",
        "code_stage_max_tokens",
        "plan_temperature",
        "plan_top_p",
        "code_temperature",
        "code_top_p",
        "sampling_seed_policy",
        "paired_request_seed_policy",
        "plan_sampling_seed",
        "code_sampling_seed",
        "evaluator_sampling_seed_policy",
        "evaluator_sampling_seed_sha256",
        "evaluator_sampling_seed_range",
        "child_solver_sampling_seed_policy",
        "child_solver_sampling_seed_sha256",
        "plan_input_token_count",
        "plan_input_token_target",
        "plan_input_token_delta",
        "generator_sampling_seed_range",
        "llm_seed",
        "generator_draw_idx",
        "instance_seed_policy",
        "instance_seed_range",
        "evaluation_seeds",
        "requested_instance_budget",
        "child_rollout_budget",
    )
    optional_equal_fields = (
        "acceptance_budget",
        "acceptance_retry_budget",
        "actual_llm_generation_call_count",
        "child_solver_sampling_seed_range",
        "mutation_code_backend",
        "generation_path",
        "generator_family",
        "compiler_registry_version",
    )

    def canonical(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    pair_reports: list[dict[str, Any]] = []
    all_failures: list[str] = []
    for pair_id in sorted(by_pair):
        conditions = by_pair[pair_id]
        failures: list[str] = []
        for condition in ("plain", "reasoning"):
            count = len(conditions.get(condition, []))
            if count != 1:
                failures.append(
                    f"{condition} generator row count is {count}, expected 1"
                )
        if not failures:
            plain = conditions["plain"][0]
            reasoning = conditions["reasoning"][0]
            for field in required_equal_fields:
                plain_value = plain.get(field)
                reasoning_value = reasoning.get(field)
                if plain_value is None or reasoning_value is None:
                    failures.append(f"required provenance field {field} is missing")
                elif canonical(plain_value) != canonical(reasoning_value):
                    failures.append(
                        f"paired field {field} differs: "
                        f"plain={plain_value!r}, reasoning={reasoning_value!r}"
                    )
            for condition, row in (
                ("plain", plain),
                ("reasoning", reasoning),
            ):
                digest = row.get("solver_checkpoint_sha256")
                if (
                    not isinstance(digest, str)
                    or _SHA256_RE.fullmatch(digest) is None
                ):
                    failures.append(
                        f"{condition} solver_checkpoint_sha256 must be "
                        "exactly 64 hexadecimal characters"
                    )
            for field in optional_equal_fields:
                plain_value = plain.get(field)
                reasoning_value = reasoning.get(field)
                if plain_value is None and reasoning_value is None:
                    continue
                if (
                    plain_value is None
                    or reasoning_value is None
                    or canonical(plain_value) != canonical(reasoning_value)
                ):
                    failures.append(
                        f"optional paired field {field} is asymmetric or differs: "
                        f"plain={plain_value!r}, reasoning={reasoning_value!r}"
                    )
            plain_code_tokens = plain.get("code_input_token_count")
            reasoning_code_tokens = reasoning.get("code_input_token_count")
            code_input_delta = None
            compiled_pair = (
                plain.get("generation_path") == "registered_compiled"
                and reasoning.get("generation_path") == "registered_compiled"
            )
            if (
                compiled_pair
                and plain_code_tokens is None
                and reasoning_code_tokens is None
            ):
                code_input_delta = 0
            elif plain_code_tokens is None or reasoning_code_tokens is None:
                failures.append(
                    "required provenance field code_input_token_count is missing"
                )
            else:
                code_input_delta = int(reasoning_code_tokens) - int(
                    plain_code_tokens
                )
                if abs(code_input_delta) > MAX_CONFIRMATORY_CODE_INPUT_TOKEN_DELTA:
                    failures.append(
                        "paired code input token delta exceeds confirmatory "
                        f"tolerance {MAX_CONFIRMATORY_CODE_INPUT_TOKEN_DELTA}: "
                        f"delta={code_input_delta}"
                    )
        else:
            code_input_delta = None
        pair_reports.append(
            {
                "generator_pair_id": pair_id,
                "passed": not failures,
                "failures": failures,
                "code_input_token_delta_reasoning_minus_plain": code_input_delta,
            }
        )
        all_failures.extend(f"pair {pair_id}: {failure}" for failure in failures)

    unpaired_rows = [
        str(row.get("generator_unit_id"))
        for row in child_rows
        if row.get("generator_pair_id") is None
    ]
    if unpaired_rows:
        all_failures.append(
            "child generators without generator_pair_id: "
            + ", ".join(unpaired_rows[:10])
        )
    if not by_pair:
        all_failures.append("no paired child generator rows were recorded")

    return {
        "passed": bool(by_pair) and not all_failures,
        "n_child_generator_rows": len(child_rows),
        "n_generator_pairs": len(by_pair),
        "n_passed_pairs": sum(report["passed"] for report in pair_reports),
        "required_equal_fields": list(required_equal_fields),
        "optional_equal_fields": list(optional_equal_fields),
        "max_confirmatory_code_input_token_delta": (
            MAX_CONFIRMATORY_CODE_INPUT_TOKEN_DELTA
        ),
        "pair_reports": pair_reports,
        "failures": all_failures,
    }


def audit_generation_sufficiency(
    generator_rows: Sequence[Mapping[str, Any]],
    instance_rows: Sequence[Mapping[str, Any]],
    *,
    min_confirmatory_runs: int = 3,
    min_confirmatory_parents: int = 20,
    min_paired_generators_per_parent: int = 3,
) -> dict[str, Any]:
    """Audit independent units and matched strict retention."""

    child_generators = [
        row for row in generator_rows if row.get("condition") in {"plain", "reasoning"}
    ]
    strict_generators = [
        row
        for row in child_generators
        if row.get("overall_status") == "ok"
        and row.get("code_valid") is True
        and row.get("operator_contract_valid") is True
        and row.get("evaluator_passed") == row.get("evaluator_total")
        and (row.get("evaluator_total") or 0) > 0
    ]
    by_pair: dict[str, set[str]] = defaultdict(set)
    pair_to_parent: dict[str, str] = {}
    pair_to_run: dict[str, str] = {}
    for row in strict_generators:
        pair_id = str(row["generator_pair_id"])
        by_pair[pair_id].add(str(row["condition"]))
        pair_to_parent[pair_id] = str(row["parent_program_id"])
        pair_to_run[pair_id] = str(row["independent_run_id"])
    matched_pairs = [
        pair_id
        for pair_id, conditions in by_pair.items()
        if conditions == {"plain", "reasoning"}
    ]
    runs = {pair_to_run[pair_id] for pair_id in matched_pairs}
    parents = {pair_to_parent[pair_id] for pair_id in matched_pairs}
    pairs_per_parent = Counter(pair_to_parent[pair_id] for pair_id in matched_pairs)
    min_pairs_observed = min(pairs_per_parent.values(), default=0)
    confirmatory_ready = (
        len(runs) >= min_confirmatory_runs
        and len(parents) >= min_confirmatory_parents
        and min_pairs_observed >= min_paired_generators_per_parent
    )
    warnings: list[str] = []
    if not confirmatory_ready:
        warnings.append(
            "insufficient_independent_units: generation-space inference is "
            "disabled; instance seeds and rollouts are nested observations."
        )
    if not matched_pairs:
        warnings.append(
            "no_strict_matched_generator_pairs: conditional representation "
            "effects cannot be estimated under the confirmatory validity rule."
        )
    code_valid_children = [
        row
        for row in instance_rows
        if row.get("role") == "child" and row.get("code_valid") is True
    ]
    return {
        "confirmatory_ready": confirmatory_ready,
        "descriptive_only": not confirmatory_ready,
        "strict_matched_generator_pairs": len(matched_pairs),
        "independent_runs_with_strict_pairs": len(runs),
        "parents_with_strict_pairs": len(parents),
        "minimum_strict_pairs_per_parent": min_pairs_observed,
        "child_generator_draws": len(child_generators),
        "strict_generator_draws": len(strict_generators),
        "code_valid_child_instances": len(code_valid_children),
        "gates": {
            "min_confirmatory_runs": min_confirmatory_runs,
            "min_confirmatory_parents": min_confirmatory_parents,
            "min_paired_generators_per_parent": min_paired_generators_per_parent,
        },
        "warnings": warnings,
    }


def prepare_training_jsonl(
    instance_rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    validity_policy: str = "strict",
) -> dict[str, Any]:
    """Write fixed plain/reasoning rows without silently balancing them.

    The separate compute audit must pass before these files are used.  Keeping
    the source rows unmodified makes acceptance-yield differences visible.
    """

    output_dir = Path(output_dir)
    selected = select_child_instances(instance_rows, validity_policy=validity_policy)
    counts: dict[str, int] = {}
    for condition in ("plain", "reasoning"):
        rows = [
            {
                "problem": row["problem_text"],
                "answer": row["answer"],
                "program_id": row["generator_unit_id"],
                "seed": row["instance_seed"],
                "parent_program_id": row["parent_program_id"],
                "generator_pair_id": row["generator_pair_id"],
                "target_reasoning_move": row.get("target_reasoning_move"),
            }
            for row in selected
            if row["condition"] == condition
        ]
        counts[condition] = len(rows)
        write_jsonl(output_dir / f"training_{condition}.jsonl", rows)
    return {
        "validity_policy": validity_policy,
        "counts": counts,
        "equal_instance_count": counts.get("plain") == counts.get("reasoning"),
        "credible_static_training": (
            validity_policy == "strict"
            and counts.get("plain", 0) == counts.get("reasoning", 0)
            and counts.get("plain", 0) > 0
        ),
    }


def validate_heldout_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    required = {
        "problem_id",
        "problem_text",
        "answer",
        "target_reasoning_move",
        "transfer_level",
        "family_id",
        "independent_run_id",
    }
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"heldout row {index} is missing fields: {missing}")
        level = str(row["transfer_level"])
        if level not in TRANSFER_LEVELS:
            raise ValueError(
                f"heldout row {index}: unsupported transfer_level {level!r}"
            )
        key = (str(row["independent_run_id"]), str(row["problem_id"]))
        if key in seen:
            raise ValueError(f"duplicate heldout problem key: {key}")
        seen.add(key)
