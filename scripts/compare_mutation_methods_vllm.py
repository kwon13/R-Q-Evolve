#!/usr/bin/env python3
"""Compare legacy and metacognitive mutations with standalone vLLM inference.

The script deliberately uses the repository's production prompt builders.
Those builders read ``prompt_templates/*.txt`` and ``prompt_templates/shots/*.txt``;
no mutation prompt is duplicated here.

For one seed program the script:

1. executes every configured evaluation seed;
2. obtains N Solver rollouts per parent seed and selects one same-seed
   correct/confident-wrong pair for planning;
3. runs the legacy one-call mutation;
4. runs metacognitive planning followed by plan-conditioned code generation;
5. validates both children with the same multi-seed checks as ``RQEvolver``;
6. evaluates every valid child seed with the evaluator and optional Solver
   rollouts, then recomputes pooled scores; and
7. saves every rendered prompt and raw output for inspection.

Standalone vLLM does not expose the actor-logit entropy used by the training
pipeline.  With ``logprobs=1`` this sample therefore uses mean token negative
log-probability as a confidence proxy when choosing a confident wrong trace.
This difference is recorded in the output manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if TYPE_CHECKING:
    from rq_evolve.backends import RolloutRecord
    from rq_evolve.program import ProblemInstance, ProblemProgram

Message: TypeAlias = dict[str, str]
Conversation: TypeAlias = list[Message]
ConversationBatch: TypeAlias = list[Conversation]


@dataclass(slots=True)
class GeneratedText:
    text: str
    token_ids: list[int]
    cumulative_logprob: float | None
    mean_negative_logprob: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VLLMChatInference:
    """Small chat-template vLLM wrapper, extended with optional log-prob metadata."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        tokenizer_name_or_path: str | None = None,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.5,
        max_model_len: int | None = None,
        trust_remote_code: bool = False,
        chat_template: str | None = None,
        default_sampling_params: dict[str, Any] | None = None,
        **llm_kwargs: Any,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "vllm is required to run this sample. Install the project's "
                "`train` extras or run it in the existing vLLM environment."
            ) from exc

        engine_args: dict[str, Any] = {
            "model": model_name_or_path,
            "tokenizer": tokenizer_name_or_path or model_name_or_path,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            **llm_kwargs,
        }
        if max_model_len is not None:
            engine_args["max_model_len"] = max_model_len

        self.llm = LLM(**engine_args)
        self.chat_template = chat_template
        self.default_sampling_params: dict[str, Any] = {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 4096,
        }
        if default_sampling_params is not None:
            self.default_sampling_params.update(default_sampling_params)

    @property
    def tokenizer(self):
        return self.llm.get_tokenizer()

    def generate(
        self,
        messages: Conversation | ConversationBatch,
        *,
        sampling_params: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        use_tqdm: bool = False,
    ) -> str | list[str]:
        details = self.generate_detailed(
            messages,
            sampling_params=sampling_params,
            chat_template_kwargs=chat_template_kwargs,
            use_tqdm=use_tqdm,
        )
        if isinstance(details, GeneratedText):
            return details.text
        return [item.text for item in details]

    def generate_detailed(
        self,
        messages: Conversation | ConversationBatch,
        *,
        sampling_params: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        use_tqdm: bool = False,
    ) -> GeneratedText | list[GeneratedText]:
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise RuntimeError("vllm is required to generate responses") from exc

        conversations, is_batch = self._normalize_messages(messages)
        params = SamplingParams(
            **{
                **self.default_sampling_params,
                **(sampling_params or {}),
            }
        )
        outputs = self.llm.chat(
            conversations,
            sampling_params=params,
            chat_template=self.chat_template,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=True,
            use_tqdm=use_tqdm,
        )

        generated: list[GeneratedText] = []
        for request_output in outputs:
            completion = request_output.outputs[0]
            token_ids = list(getattr(completion, "token_ids", None) or [])
            cumulative = getattr(completion, "cumulative_logprob", None)
            try:
                cumulative_value = (
                    float(cumulative) if cumulative is not None else None
                )
            except (TypeError, ValueError):
                cumulative_value = None
            mean_nll = (
                -cumulative_value / len(token_ids)
                if cumulative_value is not None and token_ids
                else None
            )
            generated.append(
                GeneratedText(
                    text=str(completion.text).strip(),
                    token_ids=token_ids,
                    cumulative_logprob=cumulative_value,
                    mean_negative_logprob=mean_nll,
                )
            )

        return generated if is_batch else generated[0]

    def __call__(
        self,
        messages: Conversation | ConversationBatch,
        **kwargs: Any,
    ) -> str | list[str]:
        return self.generate(messages, **kwargs)

    @staticmethod
    def _normalize_messages(
        messages: Conversation | ConversationBatch,
    ) -> tuple[ConversationBatch, bool]:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        first_item = messages[0]
        if isinstance(first_item, dict):
            conversations = [messages]  # type: ignore[list-item]
            is_batch = False
        elif isinstance(first_item, list):
            conversations = messages  # type: ignore[assignment]
            is_batch = True
        else:
            raise TypeError("messages must be Conversation or ConversationBatch")

        for conversation_index, conversation in enumerate(conversations):
            if not conversation:
                raise ValueError(f"conversation {conversation_index} is empty")
            for message_index, message in enumerate(conversation):
                if not isinstance(message, dict):
                    raise TypeError(
                        f"conversation {conversation_index}, message "
                        f"{message_index} is not a dict"
                    )
                if "role" not in message or "content" not in message:
                    raise ValueError(
                        "each message must include both `role` and `content`"
                    )
        return conversations, is_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply legacy and metacognitive mutation prompts to one seed program "
            "using standalone vLLM chat inference."
        )
    )
    parser.add_argument(
        "--seed-program",
        type=Path,
        default=ROOT / "seed_programs" / "04_sequence.py",
    )
    parser.add_argument(
        "--operator",
        choices=("in_depth", "in_breadth", "both"),
        default="both",
    )
    parser.add_argument(
        "--instance-seed",
        type=int,
        default=0,
        help="Parent seed used to choose the same-problem planning evidence pair.",
    )
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Seeds scored after generation. Defaults to every seed from 0 "
            "through --verify-seeds minus one."
        ),
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=10,
        help="Parent Solver rollouts per evaluation seed.",
    )
    parser.add_argument(
        "--child-rollouts",
        type=int,
        default=10,
        help=(
            "Solver rollouts per seed for each valid child; set 0 to skip "
            "child scoring."
        ),
    )
    parser.add_argument("--verify-seeds", type=int, default=5)
    parser.add_argument(
        "--model",
        default="/data1/yhoon113/qwen3-8b-base",
    )
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Compatibility override: when set, use one temperature for every "
            "stage. Prefer the stage-specific flags below."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help=(
            "Compatibility override: when set, use one top_p for every stage."
        ),
    )
    parser.add_argument("--solver-temperature", type=float, default=1.0)
    parser.add_argument("--solver-top-p", type=float, default=0.95)
    parser.add_argument("--plan-temperature", type=float, default=0.7)
    parser.add_argument("--plan-top-p", type=float, default=0.95)
    parser.add_argument("--code-temperature", type=float, default=0.2)
    parser.add_argument("--code-top-p", type=float, default=0.95)
    parser.add_argument("--evaluator-temperature", type=float, default=0.0)
    parser.add_argument("--evaluator-top-p", type=float, default=1.0)
    parser.add_argument("--child-temperature", type=float, default=1.0)
    parser.add_argument("--child-top-p", type=float, default=0.95)
    parser.add_argument("--solver-max-tokens", type=int, default=4096)
    parser.add_argument("--mutation-max-tokens", type=int, default=5000)
    parser.add_argument("--plan-max-tokens", type=int, default=1024)
    parser.add_argument("--evaluator-max-tokens", type=int, default=1024)
    parser.add_argument("--trace-storage-max-tokens", type=int, default=4096)
    parser.add_argument("--monitoring-total-trace-tokens", type=int, default=4096)
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=ROOT / "prompt_templates",
    )
    parser.add_argument(
        "--shot-dir",
        type=Path,
        default=ROOT / "prompt_templates" / "shots",
    )
    parser.add_argument(
        "--evidence-json",
        type=Path,
        default=None,
        help=(
            "Optional production reasoning_evidence JSON list. When supplied, "
            "parent Solver rollouts are skipped."
        ),
    )
    parser.add_argument(
        "--meta-progress-json",
        type=Path,
        default=None,
        help="Optional PRE_UPDATE_META_PROGRESS JSON object.",
    )
    parser.add_argument(
        "--chat-template-kwargs-json",
        default="{}",
        help='For example: \'{"enable_thinking": false}\'.',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "rq_output" / "mutation_method_comparison",
    )
    args = parser.parse_args()
    if args.temperature is not None:
        for name in (
            "solver_temperature",
            "plan_temperature",
            "code_temperature",
            "evaluator_temperature",
            "child_temperature",
        ):
            setattr(args, name, args.temperature)
    if args.top_p is not None:
        for name in (
            "solver_top_p",
            "plan_top_p",
            "code_top_p",
            "evaluator_top_p",
            "child_top_p",
        ):
            setattr(args, name, args.top_p)
    return args


def _messages_for_task(task) -> Conversation:
    if task.messages:
        return [dict(message) for message in task.messages]
    # This is how the production verl backend handles a legacy flat prompt.
    return [{"role": "user", "content": task.prompt}]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text), encoding="utf-8")


def _reset_output_dir(path: Path) -> Path:
    """Start one comparison run with an empty, explicitly scoped directory."""
    output_dir = path.expanduser().resolve()
    protected = {Path("/").resolve(), Path.home().resolve(), ROOT.resolve()}
    if output_dir in protected or output_dir in ROOT.resolve().parents:
        raise ValueError(f"refusing to clear unsafe output directory: {output_dir}")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output path is not a directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt_manifest(prompt_dir: Path, shot_dir: Path) -> list[dict[str, str]]:
    names = (
        prompt_dir / "in_depth.txt",
        prompt_dir / "in_breadth.txt",
        prompt_dir / "metacognitive_plan.txt",
        prompt_dir / "planned_in_depth.txt",
        prompt_dir / "planned_in_breadth.txt",
        shot_dir / "in_depth.txt",
        shot_dir / "in_breadth.txt",
        shot_dir / "metacognitive_in_depth.txt",
        shot_dir / "metacognitive_in_breadth.txt",
        shot_dir / "planned_in_depth.txt",
        shot_dir / "planned_in_breadth.txt",
        shot_dir / "evaluator.txt",
    )
    result: list[dict[str, str]] = []
    for path in names:
        if path.exists():
            result.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _file_digest(path),
                }
            )
    return result


def _chat_kwargs(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("--chat-template-kwargs-json must decode to an object")
    return value


def _resolve_evaluation_seeds(args: argparse.Namespace) -> list[int]:
    configured = getattr(args, "evaluation_seeds", None)
    seeds = (
        list(range(int(args.verify_seeds)))
        if configured is None
        else [int(seed) for seed in configured]
    )
    if not seeds:
        raise ValueError("evaluation_seeds must contain at least one seed")
    if any(seed < 0 for seed in seeds):
        raise ValueError("evaluation seeds must be non-negative")
    if any(seed >= int(args.verify_seeds) for seed in seeds):
        raise ValueError(
            "evaluation seeds must be smaller than verify_seeds"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must not contain duplicates")
    return seeds


def _run_solver_rollouts_many(
    inference: VLLMChatInference,
    instances: list[ProblemInstance],
    *,
    count: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    chat_template_kwargs: dict[str, Any],
    show_progress: bool,
) -> list[dict[str, Any]]:
    """Run all instance/rollout combinations in one vLLM batch."""
    if not instances:
        return []
    from rq_evolve.metacognition import SOLVER_CHAT_BOUNDARY_STOPS
    from rq_evolve.prompts import SOLVER_SYSTEM_PROMPT

    batch: ConversationBatch = []
    for instance in instances:
        conversation: Conversation = [
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": instance.problem},
        ]
        batch.extend(
            [
                [dict(message) for message in conversation]
                for _ in range(count)
            ]
        )
    outputs = inference.generate_detailed(
        batch,
        sampling_params={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            # CompletionOutput.cumulative_logprob is used as a confidence proxy.
            "logprobs": 1,
            "stop": list(SOLVER_CHAT_BOUNDARY_STOPS),
            "include_stop_str_in_output": False,
        },
        chat_template_kwargs=chat_template_kwargs,
        use_tqdm=show_progress,
    )
    assert isinstance(outputs, list)
    expected_outputs = len(instances) * count
    if len(outputs) != expected_outputs:
        raise RuntimeError(
            "Solver returned a different number of outputs than "
            f"requested: expected={expected_outputs}, actual={len(outputs)}"
        )

    grouped: list[dict[str, Any]] = []
    cursor = 0
    for instance in instances:
        instance_outputs = outputs[cursor : cursor + count]
        cursor += count
        records, serialized = _records_from_solver_outputs(
            inference,
            instance,
            instance_outputs,
        )
        grouped.append(
            {
                "instance": instance,
                "records": records,
                "rollouts": serialized,
            }
        )
    return grouped


def _records_from_solver_outputs(
    inference: VLLMChatInference,
    instance: ProblemInstance,
    outputs: list[GeneratedText],
) -> tuple[list[RolloutRecord], list[dict[str, Any]]]:
    from rq_evolve.backends import RolloutRecord
    from rq_evolve.metacognition import clean_and_grade_solver_rollout
    from rq_evolve.reward import answers_match, extract_boxed

    records: list[RolloutRecord] = []
    serialized: list[dict[str, Any]] = []
    for index, output in enumerate(outputs):
        raw_predicted = extract_boxed(output.text)
        raw_correct = (
            raw_predicted is not None
            and answers_match(raw_predicted, instance.answer)
        )
        # Lower mean NLL means a more confident output, matching the ordering
        # expected by select_reasoning_evidence's "lowest entropy wrong" rule.
        confidence_proxy = (
            output.mean_negative_logprob
            if output.mean_negative_logprob is not None
            else 1_000_000.0
        )
        record = RolloutRecord(
            response=output.text,
            predicted_answer=raw_predicted,
            correct=raw_correct,
            entropy=float(confidence_proxy),
            response_tokens=len(output.token_ids),
        )
        cleaned_text, predicted, correct = clean_and_grade_solver_rollout(
            record,
            instance,
        )
        record.response = cleaned_text
        record.predicted_answer = predicted
        record.correct = correct
        if cleaned_text != output.text.strip():
            try:
                record.response_tokens = len(
                    inference.tokenizer.encode(
                        cleaned_text,
                        add_special_tokens=False,
                    )
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                record.response_tokens = len(cleaned_text.split())
        records.append(record)
        serialized.append(
            {
                "rollout_idx": index,
                **output.to_dict(),
                "cleaned_text": cleaned_text,
                "raw_predicted_answer": raw_predicted,
                "raw_correct": raw_correct,
                "predicted_answer": predicted,
                "correct": correct,
                "confidence_measure": "mean_token_negative_logprob",
                "confidence_proxy": confidence_proxy,
            }
        )
    return records, serialized


def _empty_meta_progress() -> dict[str, Any]:
    empty = {
        "count": 0,
        "pre_mean_p": 0.0,
        "post_mean_p": 0.0,
        "delta_p": 0.0,
    }
    return {
        "global": dict(empty),
        "concept_group": dict(empty),
        "concept_type": dict(empty),
        "by_operator": {},
    }


def _validate_generated_program(
    raw_output: str,
    *,
    parent: ProblemProgram,
    op: str,
    mutation_plan: dict[str, Any] | None,
    plan_id: str | None,
    verify_seeds: int,
) -> tuple[ProblemProgram | None, dict[str, Any]]:
    from rq_evolve.archive import MAPElitesArchive
    from rq_evolve.code_utils import extract_generator_code
    from rq_evolve.config import EvolutionConfig, MetacognitionConfig
    from rq_evolve.evolution import RQEvolver
    from rq_evolve.program import ProblemProgram

    code = extract_generator_code(raw_output)
    result: dict[str, Any] = {
        "code_extracted": code is not None,
        "valid": False,
        "reason": None,
        "instances": [],
    }
    if code is None:
        result["reason"] = "no parseable Python generator found"
        return None, result

    metadata: dict[str, Any] = {"op": op}
    if mutation_plan is not None:
        metadata.update(
            {
                "mutation_plan": mutation_plan,
                "plan_id": plan_id,
                "plan_status": "planned",
            }
        )
    child = ProblemProgram(
        source_code=code,
        parent_id=parent.program_id,
        generation=parent.generation + 1,
        metadata=metadata,
    )
    verifier = RQEvolver(
        archive=MAPElitesArchive(),
        backend=object(),  # verify_program does not call the backend.
        evolution_config=EvolutionConfig(verify_seeds=verify_seeds),
        metacognition_config=MetacognitionConfig(
            enabled=True,
            reject_unbounded_sampling=True,
        ),
    )
    first, reason = verifier.verify_program(child, n_seeds=verify_seeds)
    if first is None:
        result["reason"] = reason
        result["source_code"] = code
        return child, result

    instances: list[dict[str, Any]] = []
    for seed in range(verify_seeds):
        instance = child.execute(seed)
        if instance is not None:
            instances.append(
                {
                    "seed": seed,
                    "problem": instance.problem,
                    "answer": instance.answer,
                }
            )
    result.update(
        {
            "valid": True,
            "reason": None,
            "program_id": child.program_id,
            "concept_group": child.get_concept_group(),
            "concept_type": child.get_concept_type(),
            "instances": instances,
            "source_code": code,
        }
    )
    return child, result


def _evaluate_program_seeds(
    inference: VLLMChatInference,
    program: ProblemProgram,
    *,
    seeds: list[int],
    mutation_plan: dict[str, Any] | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    chat_template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate every generated instance in one batched evaluator call."""
    from rq_evolve.prompts import (
        build_evaluator_messages,
        parse_evaluator_verdict,
    )

    pending: list[tuple[int, ProblemInstance, Conversation]] = []
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        instance = program.execute(seed)
        if instance is None:
            per_seed.append(
                {
                    "seed": seed,
                    "valid": False,
                    "reason": "program execution failed",
                }
            )
            continue
        messages = build_evaluator_messages(
            instance.problem,
            mutation_plan,
            answer_text=instance.answer,
            program_source=program.source_code,
        )
        pending.append((seed, instance, messages))

    if pending:
        raw_outputs = inference.generate(
            [messages for _, _, messages in pending],
            sampling_params={
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            },
            chat_template_kwargs=chat_template_kwargs,
        )
        assert isinstance(raw_outputs, list)
        if len(raw_outputs) != len(pending):
            raise RuntimeError(
                "evaluator returned a different number of outputs than seeds"
            )
        for (seed, instance, messages), raw in zip(pending, raw_outputs):
            valid, reason = parse_evaluator_verdict(raw)
            per_seed.append(
                {
                    "seed": seed,
                    "problem": instance.problem,
                    "answer": instance.answer,
                    "messages": messages,
                    "raw_output": raw,
                    "valid": valid,
                    "reason": reason,
                }
            )
    per_seed.sort(key=lambda item: int(item["seed"]))
    num_valid = sum(item.get("valid") is True for item in per_seed)
    return {
        "valid": len(per_seed) == len(seeds) and num_valid == len(seeds),
        "num_valid": num_valid,
        "num_seeds": len(seeds),
        "per_seed": per_seed,
    }


def _score_program_seeds(
    inference: VLLMChatInference,
    program: ProblemProgram,
    *,
    seeds: list[int],
    count: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    chat_template_kwargs: dict[str, Any],
    show_progress: bool = False,
) -> dict[str, Any]:
    """Score every seed and recompute one pooled RQ score over all rollouts."""
    instances: list[ProblemInstance] = []
    failed_seeds: list[int] = []
    for seed in seeds:
        instance = program.execute(seed)
        if instance is None:
            failed_seeds.append(seed)
        else:
            instances.append(instance)

    grouped = _run_solver_rollouts_many(
        inference,
        instances,
        count=count,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        chat_template_kwargs=chat_template_kwargs,
        show_progress=show_progress,
    )
    return _summarize_rollout_groups(
        grouped,
        seeds=seeds,
        failed_seeds=failed_seeds,
    )


def _summarize_rollout_groups(
    grouped: list[dict[str, Any]],
    *,
    seeds: list[int],
    failed_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Build seed-level and pooled scores from already generated rollouts."""
    from rq_evolve.scoring import compute_rq_full

    failed_seeds = list(failed_seeds or [])
    per_seed: list[dict[str, Any]] = []
    pooled_correct: list[bool] = []
    pooled_uncertainty: list[float] = []
    for group in grouped:
        instance = group["instance"]
        records = group["records"]
        serialized = group["rollouts"]
        uncertainty_proxy = (
            sum(float(record.entropy) for record in records) / len(records)
            if records
            else 0.0
        )
        rq = compute_rq_full(
            [bool(record.correct) for record in records],
            uncertainty_proxy,
        )
        for rollout in serialized:
            rollout["seed"] = int(instance.seed)
        per_seed.append(
            {
                "status": "ok",
                "seed": int(instance.seed),
                "problem": instance.problem,
                "answer": instance.answer,
                "num_rollouts": rq.num_rollouts,
                "num_correct": rq.num_correct,
                "p_hat": rq.p_hat,
                "uncertainty_proxy": rq.uncertainty,
                "rq_proxy": rq.rq_score,
                "rollouts": serialized,
            }
        )
        pooled_correct.extend(bool(record.correct) for record in records)
        pooled_uncertainty.extend(float(record.entropy) for record in records)

    for seed in failed_seeds:
        per_seed.append({"status": "execute_failed", "seed": seed})
    per_seed.sort(key=lambda item: int(item["seed"]))
    pooled_uncertainty_proxy = (
        sum(pooled_uncertainty) / len(pooled_uncertainty)
        if pooled_uncertainty
        else 0.0
    )
    pooled = compute_rq_full(pooled_correct, pooled_uncertainty_proxy)
    return {
        "status": "ok" if not failed_seeds else "partial",
        "evaluation_seeds": list(seeds),
        "failed_seeds": failed_seeds,
        "num_seeds": len(seeds),
        "num_scored_seeds": len(seeds) - len(failed_seeds),
        "num_rollouts": pooled.num_rollouts,
        "num_correct": pooled.num_correct,
        "p_hat": pooled.p_hat,
        "uncertainty_proxy": pooled.uncertainty,
        "rq_proxy": pooled.rq_score,
        "uncertainty_note": (
            "Pooled mean token negative log-probability across every seed; "
            "not the actor-logit entropy used in training."
        ),
        "per_seed": per_seed,
    }


def _save_task(path: Path, task) -> None:
    _write_json(
        path,
        {
            "stage": task.stage,
            "operator": task.op,
            "plan_id": task.plan_id,
            "plan_status": task.plan_status,
            "max_output_tokens": task.max_output_tokens,
            "temperature": task.temperature,
            "top_p": task.top_p,
            "messages": _messages_for_task(task),
        },
    )


def _run_method(
    inference: VLLMChatInference,
    *,
    method: str,
    op: str,
    parent: ProblemProgram,
    output_dir: Path,
    args: argparse.Namespace,
    chat_template_kwargs: dict[str, Any],
    planning_evidence: list[dict[str, Any]] | None = None,
    meta_progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from rq_evolve.prompts import (
        build_metacognitive_plan_task,
        build_mutation_task,
        build_planned_mutation_task,
        parse_mutation_plan,
    )

    method_dir = output_dir / op / method
    method_dir.mkdir(parents=True, exist_ok=True)
    evaluation_seeds = _resolve_evaluation_seeds(args)
    plan: dict[str, Any] | None = None
    plan_id: str | None = None
    raw_plan: str | None = None
    raw_code: str | None = None
    extracted_source: str | None = None

    def _artifacts() -> dict[str, Any]:
        """생성이 중간에 실패해도 모델 원본 출력을 summary에 남긴다."""
        return {
            "method_dir": str(method_dir.resolve()),
            "evaluation_seeds": list(evaluation_seeds),
            "raw_plan": raw_plan,
            "plan": plan,
            "raw_code": raw_code,
            "extracted_source": extracted_source,
        }

    if method == "legacy":
        code_task = build_mutation_task(
            op,
            parent,
            temperature=args.code_temperature,
            top_p=args.code_top_p,
        )
    elif method == "metacognitive":
        if len(planning_evidence or []) < 2:
            return {
                "method": method,
                "operator": op,
                "status": "insufficient_evidence",
                "reason": (
                    "metacognitive mutation requires one clean correct/wrong "
                    "contrast; provide --evidence-json to compare temperatures "
                    "with fixed evidence"
                ),
                "plan_id": None,
                **_artifacts(),
            }
        plan_task = build_metacognitive_plan_task(
            op,
            parent,
            evidence=planning_evidence or [],
            meta_progress=meta_progress or _empty_meta_progress(),
            max_output_tokens=args.plan_max_tokens,
            temperature=args.plan_temperature,
            top_p=args.plan_top_p,
        )
        _save_task(method_dir / "01_plan_prompt.json", plan_task)
        raw_plan = inference.generate(
            _messages_for_task(plan_task),
            sampling_params={
                "temperature": args.plan_temperature,
                "top_p": args.plan_top_p,
                "max_tokens": args.plan_max_tokens,
            },
            chat_template_kwargs=chat_template_kwargs,
        )
        assert isinstance(raw_plan, str)
        _write_text(method_dir / "02_plan_raw.txt", raw_plan)
        plan, plan_reason = parse_mutation_plan(raw_plan, op)
        _write_json(
            method_dir / "03_plan_validation.json",
            {
                "valid": plan is not None,
                "reason": plan_reason or None,
                "plan": plan,
            },
        )
        if plan is None:
            return {
                "method": method,
                "operator": op,
                "status": "invalid_plan",
                "reason": plan_reason,
                "plan_id": None,
                **_artifacts(),
            }
        code_task = build_planned_mutation_task(
            op,
            parent,
            plan,
            temperature=args.code_temperature,
            top_p=args.code_top_p,
        )
        plan_id = code_task.plan_id
    else:
        raise ValueError(f"unknown method: {method}")

    prompt_index = "01" if method == "legacy" else "04"
    raw_index = "02" if method == "legacy" else "05"
    validation_index = "03" if method == "legacy" else "06"
    _save_task(method_dir / f"{prompt_index}_code_prompt.json", code_task)
    raw_code = inference.generate(
        _messages_for_task(code_task),
        sampling_params={
            "temperature": args.code_temperature,
            "top_p": args.code_top_p,
            "max_tokens": args.mutation_max_tokens,
        },
        chat_template_kwargs=chat_template_kwargs,
    )
    assert isinstance(raw_code, str)
    _write_text(method_dir / f"{raw_index}_code_raw.txt", raw_code)

    child, validation = _validate_generated_program(
        raw_code,
        parent=parent,
        op=op,
        mutation_plan=plan,
        plan_id=plan_id,
        verify_seeds=args.verify_seeds,
    )
    source = validation.pop("source_code", None)
    extracted_source = source
    if source:
        _write_text(method_dir / "child.py", source)
    _write_json(
        method_dir / f"{validation_index}_code_validation.json",
        validation,
    )
    if child is None or not validation["valid"]:
        return {
            "method": method,
            "operator": op,
            "status": "invalid_code",
            "reason": validation["reason"],
            "plan_id": plan_id,
            "code_extracted": validation.get("code_extracted"),
            "program_id": child.program_id if child is not None else None,
            **_artifacts(),
        }

    evaluator = _evaluate_program_seeds(
        inference,
        child,
        seeds=evaluation_seeds,
        mutation_plan=plan,
        max_tokens=args.evaluator_max_tokens,
        temperature=args.evaluator_temperature,
        top_p=args.evaluator_top_p,
        chat_template_kwargs=chat_template_kwargs,
    )
    _write_json(method_dir / "evaluator.json", evaluator)

    child_score: dict[str, Any] | None = None
    if args.child_rollouts > 0:
        child_score = _score_program_seeds(
            inference,
            child,
            seeds=evaluation_seeds,
            count=args.child_rollouts,
            max_tokens=args.solver_max_tokens,
            temperature=args.child_temperature,
            top_p=args.child_top_p,
            chat_template_kwargs=chat_template_kwargs,
        )
        _write_json(method_dir / "child_rollouts.json", child_score)

    evaluator_reason = None
    if not evaluator["valid"]:
        evaluator_reason = "; ".join(
            f"seed={item['seed']}: {item.get('reason') or 'INVALID'}"
            for item in evaluator["per_seed"]
            if item.get("valid") is not True
        )
    seed_scores = (
        [
            {
                key: value
                for key, value in item.items()
                if key != "rollouts"
            }
            for item in child_score["per_seed"]
        ]
        if child_score
        else []
    )
    evaluator_by_seed = [
        {
            key: value
            for key, value in item.items()
            if key not in {"messages", "raw_output"}
        }
        for item in evaluator["per_seed"]
    ]
    return {
        "method": method,
        "operator": op,
        "status": "ok" if evaluator["valid"] else "evaluator_rejected",
        "program_id": child.program_id,
        "concept_group": child.get_concept_group(),
        "concept_type": child.get_concept_type(),
        "plan_id": plan_id,
        "evaluator_valid": evaluator["valid"],
        "evaluator_reason": evaluator_reason,
        "evaluator_passed": evaluator["num_valid"],
        "evaluator_total": evaluator["num_seeds"],
        "evaluator_by_seed": evaluator_by_seed,
        "evaluation_seeds": evaluation_seeds,
        "p_hat": child_score.get("p_hat") if child_score else None,
        "rq_proxy": child_score.get("rq_proxy") if child_score else None,
        "num_correct": child_score.get("num_correct") if child_score else None,
        "num_rollouts": child_score.get("num_rollouts") if child_score else None,
        "seed_scores": seed_scores,
        "generated_instances": validation.get("instances", []),
        "child_file": str((method_dir / "child.py").resolve()),
        **_artifacts(),
    }


def _report_markdown(
    parent_instance: ProblemInstance,
    summaries: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    parent_scores: dict[str, Any] | None = None,
) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "# Legacy vs. metacognitive mutation",
        "",
        "## Planning evidence instance",
        "",
        f"- Seed: `{parent_instance.seed}`",
        f"- Problem: {cell(parent_instance.problem)}",
        f"- Answer: `{cell(parent_instance.answer)}`",
        f"- Selected evidence traces: {len(evidence)}",
        "",
        "## Pooled results across all evaluation seeds",
        "",
        "| operator | method | status | concept type | evaluator | correct/rollouts | p_hat | rq proxy |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        p_hat = row.get("p_hat")
        rq_proxy = row.get("rq_proxy")
        evaluator = (
            f"{row.get('evaluator_passed', 0)}/{row.get('evaluator_total', 0)}"
            if "evaluator_valid" in row
            else "—"
        )
        correct_rollouts = (
            f"{row['num_correct']}/{row['num_rollouts']}"
            if isinstance(row.get("num_correct"), int)
            and isinstance(row.get("num_rollouts"), int)
            else "—"
        )
        lines.append(
            "| {operator} | {method} | {status} | {concept_type} | {evaluator} "
            "| {correct_rollouts} | {p_hat} | {rq_proxy} |".format(
                operator=cell(row.get("operator", "")),
                method=cell(row.get("method", "")),
                status=cell(row.get("status", "")),
                concept_type=cell(row.get("concept_type") or "—"),
                evaluator=evaluator,
                correct_rollouts=correct_rollouts,
                p_hat=f"{p_hat:.3f}" if isinstance(p_hat, (int, float)) else "—",
                rq_proxy=(
                    f"{rq_proxy:.6f}"
                    if isinstance(rq_proxy, (int, float))
                    else "—"
                ),
            )
        )

    lines.extend(["", "## Seed-by-seed comparison", ""])
    lines.extend(
        [
            "| seed | candidate | evaluator | correct/rollouts | p_hat | rq proxy | problem | answer |",
            "|---:|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    if parent_scores:
        for seed_score in parent_scores.get("per_seed", []):
            correct_rollouts = (
                f"{seed_score.get('num_correct', 0)}/"
                f"{seed_score.get('num_rollouts', 0)}"
            )
            lines.append(
                "| {seed} | parent | — | {correct_rollouts} | {p_hat:.3f} | "
                "{rq_proxy:.6f} | {problem} | `{answer}` |".format(
                    seed=seed_score.get("seed", ""),
                    correct_rollouts=correct_rollouts,
                    p_hat=float(seed_score.get("p_hat", 0.0)),
                    rq_proxy=float(seed_score.get("rq_proxy", 0.0)),
                    problem=cell(seed_score.get("problem", "")),
                    answer=cell(seed_score.get("answer", "")),
                )
            )
    for row in summaries:
        evaluator_by_seed = {
            int(item["seed"]): item
            for item in row.get("evaluator_by_seed", [])
        }
        candidate = f"{row.get('operator', '')}/{row.get('method', '')}"
        for seed_score in row.get("seed_scores", []):
            seed = int(seed_score["seed"])
            evaluator_item = evaluator_by_seed.get(seed, {})
            evaluator = (
                "VALID"
                if evaluator_item.get("valid") is True
                else "INVALID"
            )
            correct_rollouts = (
                f"{seed_score.get('num_correct', 0)}/"
                f"{seed_score.get('num_rollouts', 0)}"
            )
            lines.append(
                "| {seed} | {candidate} | {evaluator} | {correct_rollouts} | "
                "{p_hat:.3f} | {rq_proxy:.6f} | {problem} | `{answer}` |".format(
                    seed=seed,
                    candidate=cell(candidate),
                    evaluator=evaluator,
                    correct_rollouts=correct_rollouts,
                    p_hat=float(seed_score.get("p_hat", 0.0)),
                    rq_proxy=float(seed_score.get("rq_proxy", 0.0)),
                    problem=cell(seed_score.get("problem", "")),
                    answer=cell(seed_score.get("answer", "")),
                )
            )
    lines.extend(
        [
            "",
            "> Pooled `p_hat` and `rq proxy` are recomputed from every rollout "
            "across every evaluation seed. The uncertainty proxy is mean token "
            "negative log-probability, not actor-logit entropy.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.num_rollouts < 1:
        raise ValueError("--num-rollouts must be >= 1")
    if args.child_rollouts < 0:
        raise ValueError("--child-rollouts must be >= 0")
    if args.verify_seeds < 1:
        raise ValueError("--verify-seeds must be >= 1")
    evaluation_seeds = _resolve_evaluation_seeds(args)
    if args.instance_seed not in evaluation_seeds:
        raise ValueError(
            "--instance-seed must be included in --evaluation-seeds"
        )

    prompt_dir = args.prompt_dir.expanduser().resolve()
    shot_dir = args.shot_dir.expanduser().resolve()
    os.environ["RQ_EVOLVE_PROMPT_DIR"] = str(prompt_dir)
    os.environ["RQ_EVOLVE_SHOT_DIR"] = str(shot_dir)
    # Import after the environment is set: prompts.py resolves these directories
    # at module import time, exactly like a production process startup.
    import rq_evolve.prompts as prompt_module
    from rq_evolve.metacognition import (
        collect_planning_evidence,
        select_reasoning_evidence,
    )
    from rq_evolve.program import ProblemProgram

    if prompt_module.PROMPT_TEMPLATE_DIR.resolve() != prompt_dir:
        raise RuntimeError(
            "rq_evolve.prompts was imported before --prompt-dir was applied; "
            "start this script in a fresh Python process"
        )

    provided_evidence = (
        json.loads(args.evidence_json.read_text(encoding="utf-8"))
        if args.evidence_json is not None
        else None
    )
    provided_meta_progress = (
        json.loads(args.meta_progress_json.read_text(encoding="utf-8"))
        if args.meta_progress_json is not None
        else None
    )
    if provided_evidence is not None and not isinstance(provided_evidence, list):
        raise ValueError("--evidence-json must contain a JSON list")
    if provided_meta_progress is not None and not isinstance(
        provided_meta_progress,
        dict,
    ):
        raise ValueError("--meta-progress-json must contain a JSON object")
    output_dir = _reset_output_dir(args.output_dir)
    seed_path = args.seed_program.expanduser().resolve()
    parent = ProblemProgram.from_file(seed_path, generation=0)
    parent_instances = [parent.execute(seed) for seed in evaluation_seeds]
    failed_parent_seeds = [
        seed
        for seed, instance in zip(evaluation_seeds, parent_instances)
        if instance is None
    ]
    if failed_parent_seeds:
        raise RuntimeError(
            f"seed program failed at seeds {failed_parent_seeds}: {seed_path}"
        )
    parent_instances = [
        instance for instance in parent_instances if instance is not None
    ]
    parent_instance = next(
        instance
        for instance in parent_instances
        if int(instance.seed) == args.instance_seed
    )

    chat_template_kwargs = _chat_kwargs(args.chat_template_kwargs_json)
    inference = VLLMChatInference(
        model_name_or_path=args.model,
        tokenizer_name_or_path=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )

    parent_scores: dict[str, Any] | None = None
    if provided_evidence is not None:
        evidence = provided_evidence
        parent.metadata["reasoning_evidence"] = evidence
        evidence_source = "provided_json"
    else:
        parent_groups = _run_solver_rollouts_many(
            inference,
            parent_instances,
            count=args.num_rollouts,
            max_tokens=args.solver_max_tokens,
            temperature=args.solver_temperature,
            top_p=args.solver_top_p,
            chat_template_kwargs=chat_template_kwargs,
            show_progress=True,
        )
        parent_scores = _summarize_rollout_groups(
            parent_groups,
            seeds=evaluation_seeds,
        )
        parent.p_hat = float(parent_scores["p_hat"])
        parent.h_score = float(parent_scores["uncertainty_proxy"])
        parent.rq_score = float(parent_scores["rq_proxy"])
        parent.fitness = parent.rq_score

        evidence_candidates: list[tuple[int, list[Any]]] = []
        for group in parent_groups:
            instance = group["instance"]
            selected = select_reasoning_evidence(
                group["records"],
                program=parent,
                instance=instance,
                iteration=0,
                max_tokens=args.trace_storage_max_tokens,
                tokenizer=inference.tokenizer,
            )
            if selected:
                evidence_candidates.append((int(instance.seed), selected))
        selected = next(
            (
                candidate
                for seed, candidate in evidence_candidates
                if seed == args.instance_seed
            ),
            evidence_candidates[0][1] if evidence_candidates else [],
        )
        evidence = [asdict(item) for item in selected]
        if evidence:
            parent.metadata["reasoning_evidence"] = evidence
        evidence_source = "standalone_vllm_rollouts"
        _write_json(output_dir / "parent_rollouts.json", parent_scores)
        _write_json(
            output_dir / "parent_scores.json",
            {
                **{
                    key: value
                    for key, value in parent_scores.items()
                    if key != "per_seed"
                },
                "per_seed": [
                    {
                        key: value
                        for key, value in item.items()
                        if key != "rollouts"
                    }
                    for item in parent_scores["per_seed"]
                ],
            },
        )

    _write_json(output_dir / "selected_reasoning_evidence.json", evidence)
    try:
        selected_evidence_seed = int(
            evidence[0].get("seed", args.instance_seed)
            if evidence
            else args.instance_seed
        )
    except (TypeError, ValueError):
        selected_evidence_seed = args.instance_seed
    if selected_evidence_seed in evaluation_seeds:
        parent_instance = next(
            instance
            for instance in parent_instances
            if int(instance.seed) == selected_evidence_seed
        )
    meta_progress = (
        provided_meta_progress
        if provided_meta_progress is not None
        else _empty_meta_progress()
    )
    if not isinstance(meta_progress, dict):
        raise ValueError("--meta-progress-json must contain a JSON object")

    operators = (
        ("in_depth", "in_breadth")
        if args.operator == "both"
        else (args.operator,)
    )
    summaries: list[dict[str, Any]] = []
    for op in operators:
        planning_evidence = collect_planning_evidence(
            parent,
            op,
            [parent],
            total_tokens=args.monitoring_total_trace_tokens,
            tokenizer=inference.tokenizer,
        )
        _write_json(
            output_dir / op / "planning_evidence.json",
            planning_evidence,
        )
        for method in ("legacy", "metacognitive"):
            print(f"[compare] operator={op} method={method}")
            summaries.append(
                _run_method(
                    inference,
                    method=method,
                    op=op,
                    parent=parent,
                    output_dir=output_dir,
                    args=args,
                    chat_template_kwargs=chat_template_kwargs,
                    planning_evidence=planning_evidence,
                    meta_progress=meta_progress,
                )
            )

    manifest = {
        "model": args.model,
        "seed_program": str(seed_path),
        "seed_program_sha256": _file_digest(seed_path),
        "parent_program_id": parent.program_id,
        "requested_planning_seed": args.instance_seed,
        "planning_evidence_seed": selected_evidence_seed,
        "evaluation_seeds": evaluation_seeds,
        "parent_problem": parent_instance.problem,
        "parent_answer": parent_instance.answer,
        "parent_score": {
            "p_hat": parent.p_hat,
            "uncertainty_proxy": parent.h_score,
            "rq_proxy": parent.rq_score,
            "num_correct": (
                parent_scores.get("num_correct")
                if parent_scores is not None
                else None
            ),
            "num_rollouts": (
                parent_scores.get("num_rollouts")
                if parent_scores is not None
                else None
            ),
        },
        "evidence_source": evidence_source,
        "evidence_count": len(evidence),
        "confidence_measure": (
            "mean_token_negative_logprob (standalone proxy, not actor entropy)"
        ),
        "prompt_files": _prompt_manifest(prompt_dir, shot_dir),
        "sampling": {
            "solver_temperature": args.solver_temperature,
            "solver_top_p": args.solver_top_p,
            "plan_temperature": args.plan_temperature,
            "plan_top_p": args.plan_top_p,
            "code_temperature": args.code_temperature,
            "code_top_p": args.code_top_p,
            "evaluator_temperature": args.evaluator_temperature,
            "evaluator_top_p": args.evaluator_top_p,
            "child_temperature": args.child_temperature,
            "child_top_p": args.child_top_p,
            "solver_max_tokens": args.solver_max_tokens,
            "mutation_max_tokens": args.mutation_max_tokens,
            "plan_max_tokens": args.plan_max_tokens,
            "num_rollouts": args.num_rollouts,
            "child_rollouts": args.child_rollouts,
            "evaluation_seeds": evaluation_seeds,
        },
        "results": summaries,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_text(
        output_dir / "REPORT.md",
        _report_markdown(
            parent_instance,
            summaries,
            evidence,
            parent_scores=parent_scores,
        ),
    )
    print(f"[compare] saved results to {output_dir}")


if __name__ == "__main__":
    main()
