"""Held-out capability-evaluation helpers for expansion experiments.

This module deliberately does not train or mutate a Solver.  It provides:

* a small, explicit JSONL contract for independently constructed held-out sets;
* lazy vLLM evaluation of one frozen checkpoint using the production Solver
  prompt and measurement grader; and
* a static audit that checks whether paired plain/reasoning training JSONL
  files have equal instance and pre-computed token budgets.

There are no import-time vLLM, torch, transformers, or math-evaluation imports.
Consequently, schema and budget helpers remain usable in lightweight analysis
environments.  The vLLM and grading dependencies are loaded only when
``evaluate_checkpoint_vllm`` is called.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from .prompts import SOLVER_SYSTEM_PROMPT
from .vllm_runtime import configure_vllm_sampler_backend


HELDOUT_SCHEMA_VERSION = 2
EVALUATION_SCHEMA_VERSION = 2

REQUIRED_HELDOUT_FIELDS: tuple[str, ...] = (
    "problem_id",
    "problem_text",
    "answer",
    "target_reasoning_move",
    "transfer_level",
    "family_id",
    "independent_run_id",
    "construction_seed",
    "split_frozen_before_training",
    "heldout_provenance",
)

REQUIRED_CONFIRMATORY_PROVENANCE_FIELDS: tuple[str, ...] = (
    "provenance_id",
    "construction_method",
    "frozen_at_utc",
    "freeze_manifest_sha256",
)

TRANSFER_LEVELS: frozenset[str] = frozenset(
    {
        "in_family",
        "structural",
        "cross_domain",
        "archive",
        "benchmark",
    }
)

EVALUATION_CONDITIONS: frozenset[str] = frozenset(
    {"base", "plain", "reasoning"}
)

_SURFACE_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
)


class HeldoutSchemaError(ValueError):
    """Raised when one or more held-out JSONL rows violate the contract."""


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file and require every non-empty line to be an object."""

    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(
                    line,
                    parse_constant=_reject_nonstandard_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"{source}: invalid JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{source}: line {line_number} must be a JSON object"
                )
            rows.append(value)
    return rows


def write_jsonl(
    path: str | os.PathLike[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Atomically write mapping rows as UTF-8 JSONL and return the output path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            for row_index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    raise TypeError(
                        f"JSONL row {row_index} must be a mapping, "
                        f"got {type(row).__name__}"
                    )
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                    )
                )
                handle.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _canonical_identifier(
    value: Any,
    *,
    field: str,
    row_index: int,
    errors: list[str],
) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        errors.append(
            f"row {row_index}: {field} must be a non-empty string or integer"
        )
        return None
    canonical = str(value).strip()
    if not canonical:
        errors.append(f"row {row_index}: {field} must not be empty")
        return None
    return canonical


def _canonical_answer(
    value: Any,
    *,
    row_index: int,
    errors: list[str],
) -> str | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (str, Integral, Real))
        or (
            isinstance(value, Real)
            and not isinstance(value, Integral)
            and not math.isfinite(float(value))
        )
    ):
        errors.append(
            f"row {row_index}: answer must be a non-empty string or number"
        )
        return None
    canonical = str(value).strip()
    if not canonical:
        errors.append(f"row {row_index}: answer must not be empty")
        return None
    return canonical


def validate_heldout_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    require_unique_problem_per_run: bool = True,
    mode: str = "pilot",
) -> list[dict[str, Any]]:
    """Validate and canonicalize held-out capability rows.

    Extra metadata fields are preserved.  Identifier-like values are converted
    to strings so that integer and string IDs cannot form accidental duplicate
    groups.  Uniqueness, when requested, is checked on
    ``(independent_run_id, problem_id)`` rather than ``problem_id`` alone:
    independent experiment runs may intentionally reuse a fixed blinded set.
    """

    if mode not in {"pilot", "confirmatory"}:
        raise ValueError("mode must be 'pilot' or 'confirmatory'")

    validated: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()

    for row_index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            errors.append(
                f"row {row_index}: expected a mapping, "
                f"got {type(raw_row).__name__}"
            )
            continue

        row = dict(raw_row)
        missing = [field for field in REQUIRED_HELDOUT_FIELDS if field not in row]
        if missing:
            errors.append(
                f"row {row_index}: missing required fields: {', '.join(missing)}"
            )
            continue

        problem_id = _canonical_identifier(
            row["problem_id"],
            field="problem_id",
            row_index=row_index,
            errors=errors,
        )
        family_id = _canonical_identifier(
            row["family_id"],
            field="family_id",
            row_index=row_index,
            errors=errors,
        )
        independent_run_id = _canonical_identifier(
            row["independent_run_id"],
            field="independent_run_id",
            row_index=row_index,
            errors=errors,
        )
        construction_seed = _canonical_identifier(
            row["construction_seed"],
            field="construction_seed",
            row_index=row_index,
            errors=errors,
        )

        problem_text = row["problem_text"]
        if not isinstance(problem_text, str) or not problem_text.strip():
            errors.append(f"row {row_index}: problem_text must be non-empty text")

        target_move = row["target_reasoning_move"]
        if not isinstance(target_move, str) or not target_move.strip():
            errors.append(
                f"row {row_index}: target_reasoning_move must be non-empty text"
            )

        transfer_level = row["transfer_level"]
        if (
            not isinstance(transfer_level, str)
            or transfer_level not in TRANSFER_LEVELS
        ):
            allowed = ", ".join(sorted(TRANSFER_LEVELS))
            errors.append(
                f"row {row_index}: transfer_level must be one of: {allowed}"
            )

        answer = _canonical_answer(
            row["answer"],
            row_index=row_index,
            errors=errors,
        )
        split_frozen = row["split_frozen_before_training"]
        if not isinstance(split_frozen, bool):
            errors.append(
                f"row {row_index}: split_frozen_before_training must be boolean"
            )
        elif mode == "confirmatory" and not split_frozen:
            errors.append(
                f"row {row_index}: confirmatory held-out data must be frozen "
                "before training"
            )

        provenance = row["heldout_provenance"]
        if not isinstance(provenance, Mapping):
            errors.append(f"row {row_index}: heldout_provenance must be an object")
        elif mode == "confirmatory":
            missing_provenance = [
                field
                for field in REQUIRED_CONFIRMATORY_PROVENANCE_FIELDS
                if not isinstance(provenance.get(field), str)
                or not str(provenance[field]).strip()
            ]
            if missing_provenance:
                errors.append(
                    f"row {row_index}: confirmatory heldout_provenance is missing "
                    f"non-empty fields: {', '.join(missing_provenance)}"
                )
            elif not _is_sha256(provenance["freeze_manifest_sha256"]):
                errors.append(
                    f"row {row_index}: freeze_manifest_sha256 must be 64 hex characters"
                )

        if (
            require_unique_problem_per_run
            and problem_id is not None
            and independent_run_id is not None
        ):
            key = (independent_run_id, problem_id)
            if key in seen:
                errors.append(
                    "row "
                    f"{row_index}: duplicate (independent_run_id, problem_id)={key}"
                )
            seen.add(key)

        if (
            problem_id is not None
            and family_id is not None
            and independent_run_id is not None
            and construction_seed is not None
            and answer is not None
            and isinstance(problem_text, str)
            and problem_text.strip()
            and isinstance(target_move, str)
            and target_move.strip()
            and transfer_level in TRANSFER_LEVELS
            and isinstance(split_frozen, bool)
            and isinstance(provenance, Mapping)
        ):
            row.update(
                {
                    "problem_id": problem_id,
                    "family_id": family_id,
                    "independent_run_id": independent_run_id,
                    "construction_seed": construction_seed,
                    "answer": answer,
                    "transfer_level": str(transfer_level),
                    "heldout_provenance": dict(provenance),
                }
            )
            validated.append(row)

    if errors:
        shown = errors[:25]
        suffix = (
            f"\n... and {len(errors) - len(shown)} additional schema errors"
            if len(errors) > len(shown)
            else ""
        )
        raise HeldoutSchemaError("\n".join(shown) + suffix)
    return validated


def load_heldout_jsonl(
    path: str | os.PathLike[str],
    *,
    require_unique_problem_per_run: bool = True,
    mode: str = "confirmatory",
) -> list[dict[str, Any]]:
    """Read and validate a held-out capability JSONL file."""

    return validate_heldout_rows(
        read_jsonl(path),
        require_unique_problem_per_run=require_unique_problem_per_run,
        mode=mode,
    )


def _validate_vllm_evaluation_options(
    *,
    condition: str,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> None:
    if not isinstance(condition, str) or condition not in EVALUATION_CONDITIONS:
        allowed = ", ".join(sorted(EVALUATION_CONDITIONS))
        raise ValueError(f"condition must be one of: {allowed}")
    if isinstance(n, bool) or not isinstance(n, Integral) or int(n) < 1:
        raise ValueError("n must be a positive integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, Real)
        or not math.isfinite(float(temperature))
        or float(temperature) < 0.0
    ):
        raise ValueError("temperature must be a non-negative number")
    if (
        isinstance(top_p, bool)
        or not isinstance(top_p, Real)
        or not math.isfinite(float(top_p))
        or not 0.0 < float(top_p) <= 1.0
    ):
        raise ValueError("top_p must be in (0, 1]")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, Integral)
        or int(max_tokens) < 1
    ):
        raise ValueError("max_tokens must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")


def evaluate_checkpoint_vllm(
    heldout_rows: Iterable[Mapping[str, Any]],
    *,
    checkpoint: str | os.PathLike[str],
    condition: str,
    checkpoint_id: str | None = None,
    checkpoint_run_id: str | None = None,
    checkpoint_provenance: str | None = None,
    tokenizer: str | os.PathLike[str] | None = None,
    tokenizer_id: str | None = None,
    heldout_input_sha256: str | None = None,
    evaluation_mode: str = "confirmatory",
    n: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    seed: int = 0,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    trust_remote_code: bool = False,
    sampler_backend: str = "pytorch",
    chat_template: str | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    use_tqdm: bool = True,
    llm_kwargs: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one Solver checkpoint and return one row per rollout.

    The same production ``SOLVER_SYSTEM_PROMPT`` and ``grade_eval`` measurement
    grader used elsewhere in the repository are applied to every condition.
    ``seed`` is the vLLM request seed; completion identity is additionally
    recorded by ``rollout_index``.  Passing the same sampling options to base,
    plain, and reasoning checkpoints makes their evaluation contract paired.

    This function performs inference only.  It neither loads training data nor
    changes model weights.
    """

    rows = validate_heldout_rows(heldout_rows, mode=evaluation_mode)
    _validate_vllm_evaluation_options(
        condition=condition,
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )
    if not rows:
        return []
    heldout_runs = sorted({str(row["independent_run_id"]) for row in rows})
    resolved_checkpoint_run_id = (
        str(checkpoint_run_id).strip()
        if checkpoint_run_id is not None
        else (heldout_runs[0] if len(heldout_runs) == 1 else "")
    )
    if not resolved_checkpoint_run_id:
        raise ValueError(
            "checkpoint_run_id is required when held-out rows contain more than "
            "one independent_run_id; evaluate one run/checkpoint mapping at a time"
        )
    if heldout_runs != [resolved_checkpoint_run_id]:
        raise ValueError(
            "checkpoint_run_id must match every held-out independent_run_id: "
            f"checkpoint_run_id={resolved_checkpoint_run_id!r}, rows={heldout_runs}"
        )

    normalized_sampler_backend = str(sampler_backend).strip().lower()
    effective_flashinfer_sampler_env = configure_vllm_sampler_backend(
        normalized_sampler_backend
    )
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vllm is required only for checkpoint evaluation; install the "
            "project's training dependencies in the evaluation environment"
        ) from exc

    # math_eval has heavyweight optional benchmark dependencies, so import its
    # production measurement grader only on the actual evaluation path.
    from .math_eval import grade_eval
    from .solver_trace import SOLVER_CHAT_BOUNDARY_STOPS
    from .reward import extract_boxed

    checkpoint_path = os.fspath(checkpoint)
    if not checkpoint_path.strip():
        raise ValueError("checkpoint must not be empty")
    tokenizer_path = (
        os.fspath(tokenizer) if tokenizer is not None else checkpoint_path
    )
    if not tokenizer_path.strip():
        raise ValueError("tokenizer must not be empty")
    resolved_checkpoint_id = (
        str(checkpoint_id).strip()
        if checkpoint_id is not None
        else checkpoint_path
    )
    if not resolved_checkpoint_id:
        raise ValueError("checkpoint_id must not be empty")
    resolved_tokenizer_id = (
        str(tokenizer_id).strip()
        if tokenizer_id is not None
        else tokenizer_path
    )
    if not resolved_tokenizer_id:
        raise ValueError("tokenizer_id must not be empty")
    if heldout_input_sha256 is None:
        resolved_heldout_input_sha256 = _canonical_json_sha256(rows)
        heldout_hash_kind = "canonical_validated_rows"
    else:
        if not _is_sha256(heldout_input_sha256):
            raise ValueError("heldout_input_sha256 must be 64 hex characters")
        resolved_heldout_input_sha256 = heldout_input_sha256.lower()
        heldout_hash_kind = "input_file_bytes"
    heldout_content_sha256 = _canonical_json_sha256(rows)

    if (
        isinstance(tensor_parallel_size, bool)
        or not isinstance(tensor_parallel_size, Integral)
        or int(tensor_parallel_size) < 1
    ):
        raise ValueError("tensor_parallel_size must be a positive integer")
    if (
        isinstance(gpu_memory_utilization, bool)
        or not isinstance(gpu_memory_utilization, Real)
        or not math.isfinite(float(gpu_memory_utilization))
        or not 0.0 < float(gpu_memory_utilization) <= 1.0
    ):
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError("dtype must be a non-empty string")

    engine_args: dict[str, Any] = {
        "model": checkpoint_path,
        "tokenizer": tokenizer_path,
        "tensor_parallel_size": int(tensor_parallel_size),
        "dtype": dtype,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "trust_remote_code": bool(trust_remote_code),
    }
    if max_model_len is not None:
        if (
            isinstance(max_model_len, bool)
            or not isinstance(max_model_len, Integral)
            or int(max_model_len) < 1
        ):
            raise ValueError("max_model_len must be a positive integer")
        engine_args["max_model_len"] = int(max_model_len)

    extra_engine_args = dict(llm_kwargs or {})
    collisions = sorted(set(engine_args).intersection(extra_engine_args))
    if collisions:
        raise ValueError(
            "llm_kwargs must not override explicit engine options: "
            + ", ".join(collisions)
        )
    engine_args.update(extra_engine_args)

    llm = LLM(**engine_args)
    sampling_params = SamplingParams(
        n=int(n),
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        seed=int(seed),
        stop=list(SOLVER_CHAT_BOUNDARY_STOPS),
        include_stop_str_in_output=False,
    )
    decoding_parameters = {
        "n": int(n),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "seed": int(seed),
        "stop": list(SOLVER_CHAT_BOUNDARY_STOPS),
        "include_stop_str_in_output": False,
        "add_generation_prompt": True,
        "chat_template": chat_template,
        "chat_template_kwargs": dict(chat_template_kwargs or {}),
        "vllm_sampler_backend": normalized_sampler_backend,
        "vllm_use_flashinfer_sampler": effective_flashinfer_sampler_env,
    }
    evaluation_contract_sha256 = _canonical_json_sha256(
        {
            "decoding_parameters": decoding_parameters,
            "tokenizer_id": resolved_tokenizer_id,
            "solver_system_prompt_sha256": _text_sha256(SOLVER_SYSTEM_PROMPT),
        }
    )
    conversations = [
        [
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": str(row["problem_text"])},
        ]
        for row in rows
    ]
    request_outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        chat_template=chat_template,
        chat_template_kwargs=dict(chat_template_kwargs or {}),
        add_generation_prompt=True,
        use_tqdm=use_tqdm,
    )
    if len(request_outputs) != len(rows):
        raise RuntimeError(
            "vLLM returned a different number of requests than supplied: "
            f"expected={len(rows)}, actual={len(request_outputs)}"
        )

    evaluation_rows: list[dict[str, Any]] = []
    for problem, request_output in zip(rows, request_outputs):
        completions = list(getattr(request_output, "outputs", ()) or ())
        if len(completions) != int(n):
            raise RuntimeError(
                "vLLM returned a different number of rollouts than requested "
                f"for problem_id={problem['problem_id']!r}: "
                f"expected={n}, actual={len(completions)}"
            )
        prompt_token_ids = list(
            getattr(request_output, "prompt_token_ids", None) or []
        )
        for rollout_index, completion in enumerate(completions):
            response = str(getattr(completion, "text", "")).strip()
            token_ids = list(getattr(completion, "token_ids", None) or [])
            result = dict(problem)
            problem_text_sha256 = _text_sha256(str(problem["problem_text"]))
            answer_sha256 = _text_sha256(str(problem["answer"]))
            problem_contract_sha256 = _canonical_json_sha256(
                {
                    "problem_text_sha256": problem_text_sha256,
                    "answer_sha256": answer_sha256,
                    "target_reasoning_move": problem["target_reasoning_move"],
                    "transfer_level": problem["transfer_level"],
                    "family_id": problem["family_id"],
                    "construction_seed": problem["construction_seed"],
                }
            )
            result.update(
                {
                    "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                    "condition": condition,
                    "checkpoint_id": resolved_checkpoint_id,
                    "checkpoint_run_id": resolved_checkpoint_run_id,
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_provenance": checkpoint_provenance,
                    "tokenizer_id": resolved_tokenizer_id,
                    "heldout_input_sha256": resolved_heldout_input_sha256,
                    "heldout_input_hash_kind": heldout_hash_kind,
                    "heldout_content_sha256": heldout_content_sha256,
                    "problem_text_sha256": problem_text_sha256,
                    "answer_sha256": answer_sha256,
                    "problem_contract_sha256": problem_contract_sha256,
                    "heldout_provenance_sha256": _canonical_json_sha256(
                        problem["heldout_provenance"]
                    ),
                    "solver_system_prompt_sha256": _text_sha256(
                        SOLVER_SYSTEM_PROMPT
                    ),
                    "evaluation_mode": evaluation_mode,
                    "decoding_parameters": decoding_parameters,
                    "evaluation_contract_sha256": evaluation_contract_sha256,
                    "rollout_index": rollout_index,
                    "sampling_seed": int(seed),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "max_tokens": int(max_tokens),
                    "prompt_token_count": len(prompt_token_ids),
                    "response_token_count": len(token_ids),
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "response": response,
                    "predicted_answer": extract_boxed(response),
                    "correct": bool(grade_eval(response, problem["answer"])),
                }
            )
            evaluation_rows.append(result)
    return evaluation_rows


EVALUATION_MATCH_FIELDS: tuple[str, ...] = (
    "problem_text_sha256",
    "answer_sha256",
    "problem_contract_sha256",
    "target_reasoning_move",
    "family_id",
    "transfer_level",
    "construction_seed",
    "split_frozen_before_training",
    "heldout_provenance",
    "heldout_input_sha256",
    "heldout_content_sha256",
    "heldout_provenance_sha256",
    "evaluation_contract_sha256",
    "decoding_parameters",
    "tokenizer_id",
    "solver_system_prompt_sha256",
    "evaluation_mode",
)


def audit_paired_evaluation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str = "confirmatory",
    expected_conditions: Sequence[str] = ("base", "plain", "reasoning"),
) -> dict[str, Any]:
    """Audit held-out identity, decoding, and checkpoint/run pairing.

    The audit operates on ``(independent_run_id, problem_id)`` units.  It
    refuses to treat conditions as paired when problem/answer hashes, target
    move, family, transfer level, construction seed, frozen held-out source,
    tokenizer, Solver prompt, or decoding options differ.  Multiple rollout
    rows are allowed only when all provenance fields agree.
    """

    if mode not in {"pilot", "confirmatory"}:
        raise ValueError("mode must be 'pilot' or 'confirmatory'")
    conditions = tuple(str(value) for value in expected_conditions)
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("expected_conditions must contain unique labels")

    required = (
        "evaluation_schema_version",
        "condition",
        "checkpoint_id",
        "checkpoint_run_id",
        "independent_run_id",
        "problem_id",
        "problem_text",
        "answer",
        *EVALUATION_MATCH_FIELDS,
    )
    grouped: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = {}
    row_issues: list[dict[str, Any]] = []
    checkpoint_mapping: dict[tuple[str, str], set[str]] = {}

    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            row_issues.append(
                {"row_index": row_index, "issue": "row is not an object"}
            )
            continue
        missing = [field for field in required if field not in row]
        if missing:
            row_issues.append(
                {
                    "row_index": row_index,
                    "issue": "missing evaluation provenance fields",
                    "fields": missing,
                }
            )
            continue
        condition = str(row["condition"])
        if condition not in conditions:
            row_issues.append(
                {
                    "row_index": row_index,
                    "issue": "unrecognized condition",
                    "condition": condition,
                }
            )
            continue
        run_id = str(row["independent_run_id"])
        problem_id = str(row["problem_id"])
        if str(row["checkpoint_run_id"]) != run_id:
            row_issues.append(
                {
                    "row_index": row_index,
                    "issue": "checkpoint_run_id does not match independent_run_id",
                    "checkpoint_run_id": row["checkpoint_run_id"],
                    "independent_run_id": run_id,
                }
            )
        if mode == "confirmatory":
            if row["evaluation_schema_version"] != EVALUATION_SCHEMA_VERSION:
                row_issues.append(
                    {
                        "row_index": row_index,
                        "issue": "confirmatory evaluation schema version mismatch",
                        "expected": EVALUATION_SCHEMA_VERSION,
                        "actual": row["evaluation_schema_version"],
                    }
                )
            if row.get("evaluation_mode") != "confirmatory":
                row_issues.append(
                    {
                        "row_index": row_index,
                        "issue": "row was not generated in confirmatory mode",
                    }
                )
        computed_text_hash = _text_sha256(str(row["problem_text"]))
        computed_answer_hash = _text_sha256(str(row["answer"]))
        computed_problem_contract_hash = _canonical_json_sha256(
            {
                "problem_text_sha256": computed_text_hash,
                "answer_sha256": computed_answer_hash,
                "target_reasoning_move": row["target_reasoning_move"],
                "transfer_level": row["transfer_level"],
                "family_id": row["family_id"],
                "construction_seed": row["construction_seed"],
            }
        )
        if row["problem_text_sha256"] != computed_text_hash:
            row_issues.append(
                {"row_index": row_index, "issue": "problem_text_sha256 mismatch"}
            )
        if row["answer_sha256"] != computed_answer_hash:
            row_issues.append(
                {"row_index": row_index, "issue": "answer_sha256 mismatch"}
            )
        if row["problem_contract_sha256"] != computed_problem_contract_hash:
            row_issues.append(
                {
                    "row_index": row_index,
                    "issue": "problem_contract_sha256 mismatch",
                }
            )
        if isinstance(row["heldout_provenance"], Mapping):
            if row["heldout_provenance_sha256"] != _canonical_json_sha256(
                row["heldout_provenance"]
            ):
                row_issues.append(
                    {
                        "row_index": row_index,
                        "issue": "heldout_provenance_sha256 mismatch",
                    }
                )
        if mode == "confirmatory" and row["split_frozen_before_training"] is not True:
            row_issues.append(
                {
                    "row_index": row_index,
                    "issue": "held-out split was not frozen before training",
                }
            )
        for hash_field in (
            "problem_text_sha256",
            "answer_sha256",
            "problem_contract_sha256",
            "heldout_input_sha256",
            "heldout_content_sha256",
            "heldout_provenance_sha256",
            "evaluation_contract_sha256",
            "solver_system_prompt_sha256",
        ):
            if not _is_sha256(row[hash_field]):
                row_issues.append(
                    {
                        "row_index": row_index,
                        "issue": f"{hash_field} is not a SHA-256 digest",
                    }
                )
        if not isinstance(row["decoding_parameters"], Mapping):
            row_issues.append(
                {
                    "row_index": row_index,
                    "issue": "decoding_parameters must be an object",
                }
            )

        grouped.setdefault((run_id, problem_id), {}).setdefault(
            condition, []
        ).append(row)
        mapping_key = (condition, run_id)
        checkpoint_mapping.setdefault(mapping_key, set()).add(
            str(row["checkpoint_id"])
        )

    incomplete_units: list[dict[str, Any]] = []
    mismatched_units: list[dict[str, Any]] = []
    complete_units = 0
    for (run_id, problem_id), by_condition in sorted(grouped.items()):
        missing_conditions = [
            condition for condition in conditions if condition not in by_condition
        ]
        if missing_conditions:
            incomplete_units.append(
                {
                    "independent_run_id": run_id,
                    "problem_id": problem_id,
                    "missing_conditions": missing_conditions,
                }
            )
            continue
        field_mismatches: dict[str, list[Any]] = {}
        for field in EVALUATION_MATCH_FIELDS:
            canonical_values: dict[str, Any] = {}
            for condition in conditions:
                for row in by_condition[condition]:
                    canonical = json.dumps(
                        row[field],
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    canonical_values.setdefault(canonical, row[field])
            if len(canonical_values) != 1:
                field_mismatches[field] = list(canonical_values.values())
        for condition in conditions:
            checkpoint_ids = {
                str(row["checkpoint_id"]) for row in by_condition[condition]
            }
            if len(checkpoint_ids) != 1:
                field_mismatches[f"{condition}.checkpoint_id"] = sorted(
                    checkpoint_ids
                )
        if field_mismatches:
            mismatched_units.append(
                {
                    "independent_run_id": run_id,
                    "problem_id": problem_id,
                    "field_mismatches": field_mismatches,
                }
            )
        else:
            complete_units += 1

    ambiguous_checkpoint_mappings = [
        {
            "condition": condition,
            "independent_run_id": run_id,
            "checkpoint_ids": sorted(checkpoint_ids),
        }
        for (condition, run_id), checkpoint_ids in sorted(
            checkpoint_mapping.items()
        )
        if len(checkpoint_ids) != 1
    ]
    mapping_rows = [
        {
            "condition": condition,
            "independent_run_id": run_id,
            "checkpoint_id": next(iter(checkpoint_ids)),
        }
        for (condition, run_id), checkpoint_ids in sorted(
            checkpoint_mapping.items()
        )
        if len(checkpoint_ids) == 1
    ]
    passed = bool(
        complete_units > 0
        and not row_issues
        and not incomplete_units
        and not mismatched_units
        and not ambiguous_checkpoint_mappings
    )
    return {
        "schema_version": 1,
        "mode": mode,
        "passed": passed,
        "rows": len(rows),
        "observed_units": len(grouped),
        "complete_units": complete_units,
        "row_issues": row_issues,
        "incomplete_units": incomplete_units,
        "mismatched_units": mismatched_units,
        "checkpoint_run_mapping": mapping_rows,
        "ambiguous_checkpoint_mappings": ambiguous_checkpoint_mappings,
    }


def audit_training_heldout_disjointness(
    evaluation_rows: Sequence[Mapping[str, Any]],
    training_rows_by_condition: Mapping[
        str,
        Sequence[Mapping[str, Any]],
    ],
) -> dict[str, Any]:
    """Check directly observable train/held-out leakage constraints.

    This verifies stable IDs, normalized surface text, construction/sampling
    seeds, and complete numeric-literal signatures.  It cannot by itself prove
    blinded annotation or generator independence, which remain part of the
    frozen held-out provenance contract.
    """

    heldout: dict[tuple[str, str], Mapping[str, Any]] = {}
    issues: list[dict[str, Any]] = []
    for row_index, row in enumerate(evaluation_rows):
        if not isinstance(row, Mapping):
            issues.append(
                {"source": "heldout", "row_index": row_index, "issue": "not an object"}
            )
            continue
        run_id = str(row.get("independent_run_id", "")).strip()
        problem_id = str(row.get("problem_id", "")).strip()
        problem_text = row.get("problem_text")
        construction_seed = str(row.get("construction_seed", "")).strip()
        if not run_id or not problem_id or not isinstance(problem_text, str):
            issues.append(
                {
                    "source": "heldout",
                    "row_index": row_index,
                    "issue": "missing run/problem/text identity",
                }
            )
            continue
        key = (run_id, problem_id)
        previous = heldout.get(key)
        if previous is not None and (
            str(previous.get("problem_text")) != problem_text
            or str(previous.get("construction_seed")) != construction_seed
        ):
            issues.append(
                {
                    "source": "heldout",
                    "independent_run_id": run_id,
                    "problem_id": problem_id,
                    "issue": "evaluation conditions disagree on held-out identity",
                }
            )
            continue
        heldout[key] = row

    condition_reports: list[dict[str, Any]] = []
    for condition in ("plain", "reasoning"):
        source_rows = training_rows_by_condition.get(condition)
        condition_issues: list[dict[str, Any]] = []
        if source_rows is None:
            condition_issues.append({"issue": "training rows were not supplied"})
            source_rows = []

        by_run: dict[str, dict[str, set[Any]]] = {}
        for row_index, row in enumerate(source_rows):
            if not isinstance(row, Mapping):
                condition_issues.append(
                    {"row_index": row_index, "issue": "training row is not an object"}
                )
                continue
            run_id = str(
                row.get("independent_run_id", row.get("run_id", ""))
            ).strip()
            problem_text_value = row.get("problem", row.get("problem_text"))
            if not run_id or not isinstance(problem_text_value, str):
                condition_issues.append(
                    {
                        "row_index": row_index,
                        "issue": "training row lacks run ID or problem text",
                    }
                )
                continue
            problem_text = " ".join(problem_text_value.split())
            sample_id = str(
                row.get(
                    "sample_id",
                    row.get("problem_id", row.get("instance_id", "")),
                )
            ).strip()
            seed = str(
                row.get(
                    "construction_seed",
                    row.get("seed", row.get("instance_seed", "")),
                )
            ).strip()
            run = by_run.setdefault(
                run_id,
                {
                    "sample_ids": set(),
                    "problem_texts": set(),
                    "seeds": set(),
                    "numeric_signatures": set(),
                },
            )
            if sample_id:
                run["sample_ids"].add(sample_id)
            if seed:
                run["seeds"].add(seed)
            run["problem_texts"].add(problem_text)
            numeric_signature = tuple(
                match.group(0)
                for match in _SURFACE_NUMBER_RE.finditer(problem_text)
            )
            if numeric_signature:
                run["numeric_signatures"].add(numeric_signature)

        for (run_id, problem_id), row in sorted(heldout.items()):
            training = by_run.get(run_id)
            if training is None:
                condition_issues.append(
                    {
                        "independent_run_id": run_id,
                        "problem_id": problem_id,
                        "issue": "no training rows for held-out run",
                    }
                )
                continue
            heldout_text = " ".join(str(row["problem_text"]).split())
            heldout_seed = str(row.get("construction_seed", "")).strip()
            numeric_signature = tuple(
                match.group(0)
                for match in _SURFACE_NUMBER_RE.finditer(heldout_text)
            )
            if problem_id in training["sample_ids"]:
                condition_issues.append(
                    {
                        "independent_run_id": run_id,
                        "problem_id": problem_id,
                        "issue": "stable problem/sample ID overlaps training",
                    }
                )
            if heldout_text in training["problem_texts"]:
                condition_issues.append(
                    {
                        "independent_run_id": run_id,
                        "problem_id": problem_id,
                        "issue": "normalized surface problem text overlaps training",
                    }
                )
            if heldout_seed and heldout_seed in training["seeds"]:
                condition_issues.append(
                    {
                        "independent_run_id": run_id,
                        "problem_id": problem_id,
                        "issue": "construction/sampling seed overlaps training",
                    }
                )
            if (
                numeric_signature
                and numeric_signature in training["numeric_signatures"]
            ):
                condition_issues.append(
                    {
                        "independent_run_id": run_id,
                        "problem_id": problem_id,
                        "issue": "complete numeric-literal signature overlaps training",
                        "numeric_signature": list(numeric_signature),
                    }
                )

        condition_reports.append(
            {
                "condition": condition,
                "training_rows": len(source_rows),
                "runs": sorted(by_run),
                "passed": not condition_issues,
                "issues": condition_issues,
            }
        )
        issues.extend(
            {"condition": condition, **issue}
            for issue in condition_issues
        )

    return {
        "schema_version": 1,
        "passed": bool(heldout) and not issues,
        "heldout_run_problem_units": len(heldout),
        "condition_reports": condition_reports,
        "issues": issues,
        "scope_note": (
            "Checks IDs, normalized text, seeds, and complete numeric signatures; "
            "blinding/generator independence is enforced through heldout provenance."
        ),
    }


def _summarize_training_budget(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    token_count_fields: tuple[str, ...],
    run_id_field: str,
) -> dict[str, Any]:
    missing_token_rows: list[int] = []
    invalid_token_rows: list[int] = []
    missing_run_rows: list[int] = []
    observed_tokens = 0
    per_run: dict[str, dict[str, Any]] = {}

    for row_index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise TypeError(
                f"{condition} training row {row_index} must be a mapping"
            )
        row = dict(raw_row)
        row_token_total = 0
        token_complete = True
        for field in token_count_fields:
            if field not in row:
                missing_token_rows.append(row_index)
                token_complete = False
                break
            value = row[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < 0
            ):
                invalid_token_rows.append(row_index)
                token_complete = False
                break
            row_token_total += int(value)

        run_value = row.get(run_id_field)
        if isinstance(run_value, bool) or not isinstance(
            run_value, (str, Integral)
        ):
            missing_run_rows.append(row_index)
            run_id = "__missing__"
        else:
            run_id = str(run_value).strip()
            if not run_id:
                missing_run_rows.append(row_index)
                run_id = "__missing__"

        run_summary = per_run.setdefault(
            run_id,
            {
                "instances": 0,
                "observed_tokens": 0,
                "token_count_complete": True,
            },
        )
        run_summary["instances"] += 1
        if token_complete:
            observed_tokens += row_token_total
            run_summary["observed_tokens"] += row_token_total
        else:
            run_summary["token_count_complete"] = False

    token_count_complete = not missing_token_rows and not invalid_token_rows
    for summary in per_run.values():
        summary["total_tokens"] = (
            summary["observed_tokens"]
            if summary["token_count_complete"]
            else None
        )

    return {
        "condition": condition,
        "instances": len(rows),
        "token_count_fields": list(token_count_fields),
        "observed_tokens": observed_tokens,
        "total_tokens": observed_tokens if token_count_complete else None,
        "token_count_complete": token_count_complete,
        "missing_token_rows": sorted(set(missing_token_rows)),
        "invalid_token_rows": sorted(set(invalid_token_rows)),
        "run_id_field": run_id_field,
        "run_id_complete": not missing_run_rows,
        "missing_run_rows": sorted(set(missing_run_rows)),
        "per_run": dict(sorted(per_run.items())),
    }


def audit_paired_training_budgets(
    plain_rows: Sequence[Mapping[str, Any]],
    reasoning_rows: Sequence[Mapping[str, Any]],
    *,
    token_count_fields: Sequence[str] = ("token_count",),
    run_id_field: str = "independent_run_id",
    require_per_run_match: bool = True,
) -> dict[str, Any]:
    """Statically audit equal instance/token budgets without changing datasets.

    Token counts must already have been computed with the actual training
    tokenizer.  When several ``token_count_fields`` are supplied, their values
    are summed per row (for example prompt and response token counts).
    """

    supplied_fields: Sequence[str] = (
        (token_count_fields,)
        if isinstance(token_count_fields, str)
        else token_count_fields
    )
    fields = tuple(str(field).strip() for field in supplied_fields)
    if not fields or any(not field for field in fields):
        raise ValueError("token_count_fields must contain non-empty field names")
    if len(set(fields)) != len(fields):
        raise ValueError("token_count_fields must not contain duplicates")
    if not isinstance(run_id_field, str) or not run_id_field.strip():
        raise ValueError("run_id_field must be a non-empty string")

    plain = _summarize_training_budget(
        plain_rows,
        condition="plain",
        token_count_fields=fields,
        run_id_field=run_id_field,
    )
    reasoning = _summarize_training_budget(
        reasoning_rows,
        condition="reasoning",
        token_count_fields=fields,
        run_id_field=run_id_field,
    )

    instance_difference = reasoning["instances"] - plain["instances"]
    token_difference = (
        reasoning["total_tokens"] - plain["total_tokens"]
        if plain["total_tokens"] is not None
        and reasoning["total_tokens"] is not None
        else None
    )
    common_runs = sorted(set(plain["per_run"]).intersection(reasoning["per_run"]))
    same_run_ids = set(plain["per_run"]) == set(reasoning["per_run"])
    per_run_equal: bool | None
    if not plain["run_id_complete"] or not reasoning["run_id_complete"]:
        per_run_equal = None
    elif not same_run_ids:
        per_run_equal = False
    else:
        per_run_equal = all(
            plain["per_run"][run_id]["instances"]
            == reasoning["per_run"][run_id]["instances"]
            and plain["per_run"][run_id]["total_tokens"] is not None
            and reasoning["per_run"][run_id]["total_tokens"] is not None
            and plain["per_run"][run_id]["total_tokens"]
            == reasoning["per_run"][run_id]["total_tokens"]
            for run_id in common_runs
        )

    counts_comparable = (
        plain["token_count_complete"] and reasoning["token_count_complete"]
    )
    nonempty = plain["instances"] > 0 and reasoning["instances"] > 0
    overall_equal = (
        nonempty
        and instance_difference == 0
        and token_difference == 0
        and (per_run_equal is True or not require_per_run_match)
    )
    issues: list[str] = []
    if not nonempty:
        issues.append("both conditions must contain at least one training instance")
    if instance_difference != 0:
        issues.append("training instance counts differ")
    if not counts_comparable:
        issues.append("one or both token budgets are incomplete")
    elif token_difference != 0:
        issues.append("training token totals differ")
    if require_per_run_match and per_run_equal is not True:
        issues.append("per-run instance/token budgets are not exactly matched")

    return {
        "schema_version": 1,
        "plain": plain,
        "reasoning": reasoning,
        "instance_difference_reasoning_minus_plain": instance_difference,
        "token_difference_reasoning_minus_plain": token_difference,
        "same_run_ids": same_run_ids,
        "per_run_equal": per_run_equal,
        "require_per_run_match": bool(require_per_run_match),
        "budget_equal": overall_equal,
        "issues": issues,
    }


def audit_paired_training_jsonl(
    plain_path: str | os.PathLike[str],
    reasoning_path: str | os.PathLike[str],
    *,
    token_count_fields: Sequence[str] = ("token_count",),
    run_id_field: str = "independent_run_id",
    require_per_run_match: bool = True,
) -> dict[str, Any]:
    """Read two training JSONLs and return their static paired-budget audit."""

    return audit_paired_training_budgets(
        read_jsonl(plain_path),
        read_jsonl(reasoning_path),
        token_count_fields=token_count_fields,
        run_id_field=run_id_field,
        require_per_run_match=require_per_run_match,
    )


__all__ = [
    "EVALUATION_CONDITIONS",
    "EVALUATION_SCHEMA_VERSION",
    "HELDOUT_SCHEMA_VERSION",
    "HeldoutSchemaError",
    "REQUIRED_HELDOUT_FIELDS",
    "TRANSFER_LEVELS",
    "audit_paired_evaluation_rows",
    "audit_training_heldout_disjointness",
    "audit_paired_training_budgets",
    "audit_paired_training_jsonl",
    "evaluate_checkpoint_vllm",
    "load_heldout_jsonl",
    "read_jsonl",
    "validate_heldout_rows",
    "write_jsonl",
]
