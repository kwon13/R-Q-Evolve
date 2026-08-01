#!/usr/bin/env python3
"""Compare plain and reasoning-informed mutations with standalone vLLM.

The script deliberately uses the repository's production prompt builders.
Those builders read ``prompt_templates/*.txt`` and ``prompt_templates/shots/*.txt``;
no mutation prompt is duplicated here.

For one seed program the script:

1. executes every configured evaluation seed;
2. obtains N Solver rollouts per parent seed and selects one same-seed
   correct/confident-wrong pair for planning;
3. runs parent-only planning followed by plan-conditioned code generation;
4. runs reasoning-informed planning followed by the identical code stage;
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
import re
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, TypeAlias


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.vllm_runtime import (  # noqa: E402
    VLLM_SAMPLER_BACKENDS,
    configure_vllm_sampler_backend as _configure_vllm_sampler_backend,
)

if TYPE_CHECKING:
    from rq_evolve.backends import RolloutRecord
    from rq_evolve.program import ProblemInstance, ProblemProgram

Message: TypeAlias = dict[str, str]
Conversation: TypeAlias = list[Message]
ConversationBatch: TypeAlias = list[Conversation]
PAIRED_REQUEST_SEED_POLICY = "sha256_llm_seed_operator_stage_v1"


PAIRED_EVALUATOR_SEED_POLICY = (
    "sha256_llm_seed_operator_instance_seed_v1"
)
PAIRED_CHILD_SOLVER_SEED_POLICY = (
    "sha256_llm_seed_operator_instance_seed_rollout_idx_v1"
)
PARENT_EVIDENCE_SEED_POLICY = (
    "sha256_llm_seed_parent_instance_seed_rollout_idx_v1"
)


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
        sampler_backend: str = "pytorch",
        default_sampling_params: dict[str, Any] | None = None,
        **llm_kwargs: Any,
    ) -> None:
        self.sampler_backend = str(sampler_backend).strip().lower()
        self.effective_flashinfer_sampler_env = (
            _configure_vllm_sampler_backend(self.sampler_backend)
        )
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
        sampling_params: (
            dict[str, Any] | list[dict[str, Any]] | None
        ) = None,
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
        sampling_params: (
            dict[str, Any] | list[dict[str, Any]] | None
        ) = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        use_tqdm: bool = False,
    ) -> GeneratedText | list[GeneratedText]:
        try:
            from vllm import SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
        except ImportError as exc:
            raise RuntimeError("vllm is required to generate responses") from exc

        def normalize_params(values: Mapping[str, Any]) -> dict[str, Any]:
            normalized = dict(values)
            structured = normalized.get("structured_outputs")
            if isinstance(structured, dict):
                normalized["structured_outputs"] = StructuredOutputsParams(
                    **structured
                )
            return normalized

        conversations, is_batch = self._normalize_messages(messages)
        if isinstance(sampling_params, list):
            if len(sampling_params) != len(conversations):
                raise ValueError(
                    "per-request sampling_params length must match conversations: "
                    f"{len(sampling_params)} != {len(conversations)}"
                )
            params: SamplingParams | list[SamplingParams] = [
                SamplingParams(
                    **normalize_params(
                        {
                            **self.default_sampling_params,
                            **request_params,
                        }
                    )
                )
                for request_params in sampling_params
            ]
        else:
            params = SamplingParams(
                **normalize_params(
                    {
                        **self.default_sampling_params,
                        **(sampling_params or {}),
                    }
                )
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
            "Apply paired plain and reasoning-informed mutation planning to one "
            "seed program using standalone vLLM chat inference."
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
        "--plain-baseline",
        choices=("two_stage", "one_stage_legacy"),
        default="two_stage",
        help=(
            "Use a parent-only planning call before plain code generation so "
            "plain and reasoning conditions have equal generation stages. "
            "one_stage_legacy is retained only to reproduce older artifacts."
        ),
    )
    parser.add_argument(
        "--mutation-code-backend",
        choices=("freeform", "hybrid", "compiler_only"),
        default="hybrid",
        help=(
            "Compile registered schema-v5 mutation families in trusted Python. "
            "hybrid retains unsupported free-form ideas as quarantined "
            "diagnostics; compiler_only skips their code-model call."
        ),
    )
    parser.add_argument(
        "--planning-contract",
        choices=("registered_v5", "belief_v6", "direct_v7"),
        default="registered_v5",
        help=(
            "registered_v5 keeps the previous 22-field planning experiment. "
            "belief_v6 asks only for an attributed belief/desire and a "
            "verbatim quote, then derives and validates the probe in Python. "
            "direct_v7 drops planning entirely: the correct/wrong solver traces "
            "go straight into the code-writing call, and only the produced "
            "problem is verified."
        ),
    )
    parser.add_argument(
        "--candidates-per-condition",
        type=int,
        default=1,
        help=(
            "Independent planning candidates drawn for each operator/condition. "
            "In belief_v6, eligible candidates compete by R_Q within their "
            "attribution niche."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Run belief attribution, compilation, semantic validation, and "
            "diagnosticity without child Solver rollouts. Only applies to "
            "belief_v6."
        ),
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
        "--vllm-sampler-backend",
        choices=VLLM_SAMPLER_BACKENDS,
        default="pytorch",
        help=(
            "Top-k/top-p sampler backend. Defaults to native PyTorch because "
            "FlashInfer 0.6.x JIT requires a CUDA 12+ nvcc toolkit."
        ),
    )
    parser.add_argument(
        "--llm-seed",
        type=int,
        default=0,
        help=(
            "vLLM engine seed. Vary this across independent generator draws "
            "and record each draw as a distinct experimental unit."
        ),
    )
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
            "Optional fixed reasoning_evidence JSON list used only for planning. "
            "Use --parent-rollouts-json as well to reuse parent evaluation."
        ),
    )
    parser.add_argument(
        "--parent-rollouts-json",
        type=Path,
        default=None,
        help=(
            "Optional cached parent_rollouts.json. The cache is validated "
            "against every requested seed/problem and may be truncated to the "
            "requested --num-rollouts, avoiding repeated parent inference."
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


def _stable_request_seed(
    policy: str,
    llm_seed: int,
    *parts: Any,
) -> int:
    payload = "\x1f".join(
        [str(policy), str(int(llm_seed)), *(str(part) for part in parts)]
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:4],
        "big",
    ) & 0x7FFF_FFFF


def _paired_request_seed(llm_seed: int, operator: str, stage: str) -> int:
    """Derive one stable per-request seed shared by both conditions."""

    return _stable_request_seed(
        PAIRED_REQUEST_SEED_POLICY,
        llm_seed,
        operator,
        stage,
    )


def _evaluator_request_seed(
    llm_seed: int,
    operator: str,
    instance_seed: int,
) -> int:
    return _stable_request_seed(
        PAIRED_EVALUATOR_SEED_POLICY,
        llm_seed,
        operator,
        int(instance_seed),
    )


def _child_solver_request_seed(
    llm_seed: int,
    operator: str,
    instance_seed: int,
    rollout_idx: int,
) -> int:
    return _stable_request_seed(
        PAIRED_CHILD_SOLVER_SEED_POLICY,
        llm_seed,
        operator,
        int(instance_seed),
        int(rollout_idx),
    )


def _parent_evidence_request_seed(
    llm_seed: int,
    instance_seed: int,
    rollout_idx: int,
) -> int:
    return _stable_request_seed(
        PARENT_EVIDENCE_SEED_POLICY,
        llm_seed,
        int(instance_seed),
        int(rollout_idx),
    )


def _seed_schedule_digest(rows: list[dict[str, int]]) -> str:
    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _seed_schedule_range(rows: list[dict[str, int]]) -> list[int] | None:
    seeds = [int(row["sampling_seed"]) for row in rows]
    return [min(seeds), max(seeds)] if seeds else None


def _prompt_manifest(prompt_dir: Path, shot_dir: Path) -> list[dict[str, str]]:
    names = (
        prompt_dir / "belief_plan.txt",
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


def _load_cached_parent_scores(
    path: Path,
    *,
    instances: list[ProblemInstance],
    seeds: list[int],
    rollouts_per_seed: int,
) -> dict[str, Any]:
    """Validate and, when needed, truncate a cached parent rollout batch."""

    from rq_evolve.scoring import compute_rq_full

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("per_seed"),
        list,
    ):
        raise ValueError("--parent-rollouts-json must contain parent_rollouts.json")
    expected = {int(instance.seed): instance for instance in instances}
    rows = {
        int(row["seed"]): row
        for row in payload["per_seed"]
        if isinstance(row, dict) and "seed" in row
    }
    if not set(seeds).issubset(rows):
        raise ValueError(
            "cached parent rollouts do not cover evaluation seeds: "
            f"cached={sorted(rows)}, requested={sorted(seeds)}"
        )

    per_seed: list[dict[str, Any]] = []
    pooled_correct: list[bool] = []
    pooled_uncertainty: list[float] = []
    for seed in seeds:
        row = rows[seed]
        instance = expected[seed]
        if (
            " ".join(str(row.get("problem", "")).split())
            != " ".join(instance.problem.split())
            or str(row.get("answer", "")).strip()
            != str(instance.answer).strip()
        ):
            raise ValueError(
                f"cached parent rollout problem/answer mismatch at seed={seed}"
            )
        cached_rollouts = row.get("rollouts")
        if not isinstance(cached_rollouts, list) or len(cached_rollouts) < int(
            rollouts_per_seed
        ):
            raise ValueError(
                f"cached parent seed={seed} has fewer than "
                f"{rollouts_per_seed} rollouts"
            )
        selected = [dict(item) for item in cached_rollouts[:rollouts_per_seed]]
        correct = [bool(item.get("correct")) for item in selected]
        uncertainty = [
            float(
                item.get(
                    "confidence_proxy",
                    item.get("mean_negative_logprob", 0.0),
                )
            )
            for item in selected
        ]
        mean_uncertainty = (
            sum(uncertainty) / len(uncertainty) if uncertainty else 0.0
        )
        score = compute_rq_full(correct, mean_uncertainty)
        per_seed.append(
            {
                "status": "ok",
                "seed": seed,
                "problem": instance.problem,
                "answer": instance.answer,
                "num_rollouts": score.num_rollouts,
                "num_correct": score.num_correct,
                "p_hat": score.p_hat,
                "uncertainty_proxy": score.uncertainty,
                "rq_proxy": score.rq_score,
                "rollouts": selected,
            }
        )
        pooled_correct.extend(correct)
        pooled_uncertainty.extend(uncertainty)

    pooled_mean = (
        sum(pooled_uncertainty) / len(pooled_uncertainty)
        if pooled_uncertainty
        else 0.0
    )
    pooled = compute_rq_full(pooled_correct, pooled_mean)
    return {
        "status": "cached",
        "evaluation_seeds": list(seeds),
        "failed_seeds": [],
        "num_seeds": len(seeds),
        "num_scored_seeds": len(seeds),
        "num_rollouts": pooled.num_rollouts,
        "num_correct": pooled.num_correct,
        "p_hat": pooled.p_hat,
        "uncertainty_proxy": pooled.uncertainty,
        "rq_proxy": pooled.rq_score,
        "uncertainty_note": (
            "Recomputed from the first requested cached rollouts per seed; "
            "mean token negative log-probability, not actor entropy."
        ),
        "cache_source": str(path.resolve()),
        "per_seed": per_seed,
    }


def _select_same_instance_contrast(
    evidence: list[dict[str, Any]],
    *,
    preferred_seed: int,
    expected_problems: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Select one correct/wrong pair from the same seed and exact problem.

    Fixed evidence is often reused across temperature runs. Rejecting cross-seed
    or stale-problem pairs here keeps the planner's contrast tied to one concrete
    parent instance instead of silently combining unrelated traces.
    """

    from rq_evolve.metacognition import validate_reasoning_contrast

    def normalized_problem(value: Any) -> str:
        return " ".join(str(value or "").split())

    expected = {
        int(seed): normalized_problem(problem)
        for seed, problem in (expected_problems or {}).items()
    }
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for raw in evidence:
        if not isinstance(raw, dict):
            continue
        try:
            seed = int(raw["seed"])
        except (KeyError, TypeError, ValueError):
            continue
        problem = normalized_problem(raw.get("problem"))
        if not problem:
            continue
        if expected and (
            seed not in expected or expected[seed] != problem
        ):
            continue
        grouped.setdefault((seed, problem), []).append(dict(raw))

    candidates: list[tuple[int, str, dict[str, Any], dict[str, Any]]] = []
    for (seed, problem), items in grouped.items():
        success = next(
            (
                item
                for item in items
                if str(item.get("role", "")).lower() == "success"
                and item.get("correct") is not False
            ),
            None,
        )
        failure = next(
            (
                item
                for item in items
                if str(item.get("role", "")).lower() == "failure"
                and item.get("correct") is not True
            ),
            None,
        )
        if success is not None and failure is not None:
            pair = [success, failure]
            if not validate_reasoning_contrast(pair):
                candidates.append((seed, problem, success, failure))

    if not candidates:
        return []
    candidates.sort(
        key=lambda item: (
            item[0] != int(preferred_seed),
            item[0],
            item[1],
        )
    )
    _, _, success, failure = candidates[0]
    return [success, failure]


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
    request_seed_factory: (
        Callable[[ProblemInstance, int], int] | None
    ) = None,
) -> list[dict[str, Any]]:
    """Run all instance/rollout combinations in one vLLM batch."""
    if not instances:
        return []
    from rq_evolve.metacognition import SOLVER_CHAT_BOUNDARY_STOPS
    from rq_evolve.prompts import SOLVER_SYSTEM_PROMPT

    batch: ConversationBatch = []
    request_seeds: list[int] = []
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
        if request_seed_factory is not None:
            request_seeds.extend(
                request_seed_factory(instance, rollout_idx)
                for rollout_idx in range(count)
            )
    common_sampling_params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        # CompletionOutput.cumulative_logprob is used as a confidence proxy.
        "logprobs": 1,
        "stop": list(SOLVER_CHAT_BOUNDARY_STOPS),
        "include_stop_str_in_output": False,
    }
    per_request_sampling_params = (
        [
            {**common_sampling_params, "seed": request_seed}
            for request_seed in request_seeds
        ]
        if request_seed_factory is not None
        else common_sampling_params
    )
    outputs = inference.generate_detailed(
        batch,
        sampling_params=per_request_sampling_params,
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
    seed_cursor = 0
    for instance in instances:
        instance_outputs = outputs[cursor : cursor + count]
        cursor += count
        instance_sampling_seeds = (
            request_seeds[seed_cursor : seed_cursor + count]
            if request_seed_factory is not None
            else None
        )
        seed_cursor += count
        records, serialized = _records_from_solver_outputs(
            inference,
            instance,
            instance_outputs,
            sampling_seeds=instance_sampling_seeds,
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
    *,
    sampling_seeds: list[int] | None = None,
) -> tuple[list[RolloutRecord], list[dict[str, Any]]]:
    from rq_evolve.backends import RolloutRecord
    from rq_evolve.metacognition import (
        clean_and_grade_solver_rollout,
        reasoning_trace_quality_issues,
    )
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
                "sampling_seed": (
                    int(sampling_seeds[index])
                    if sampling_seeds is not None
                    else None
                ),
                **output.to_dict(),
                "cleaned_text": cleaned_text,
                "raw_predicted_answer": raw_predicted,
                "raw_correct": raw_correct,
                "predicted_answer": predicted,
                "correct": correct,
                "confidence_measure": "mean_token_negative_logprob",
                "confidence_proxy": confidence_proxy,
                "evidence_quality_issues": reasoning_trace_quality_issues(
                    cleaned_text,
                    predicted,
                    response_tokens=int(record.response_tokens or 0),
                ),
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
    free_form_direct: bool = False,
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
    if free_form_direct:
        # Opts out of the compiler-shaped instance_data / retry-loop contract.
        # Termination is still enforced at runtime by the sandbox timeout, and
        # correctness by multi-seed execution plus the instance lint below.
        metadata["free_form_direct"] = True
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
            # The bounded-retry requirement presumes rejection sampling. A
            # direct generator that samples once has no such loop, so the check
            # would reject it on shape; the sandbox timeout still bounds it.
            reject_unbounded_sampling=not free_form_direct,
        ),
    )
    first, reason = verifier.verify_program(child, n_seeds=verify_seeds)
    if first is None:
        result["reason"] = reason
        result["source_code"] = code
        return child, result

    # Fair comparison contract: enforce the requested operator for BOTH
    # legacy and metacognitive generators.  The production Evolver historically
    # enforced this only when a mutation plan existed, which allowed a
    # metacognitive "in_breadth" child to remain in the parent's domain in this
    # standalone notebook path.
    parent_group = parent.get_concept_group()
    parent_type = parent.get_concept_type()
    child_group = child.get_concept_group()
    child_type = child.get_concept_type()
    contract_reason = None
    if op == "in_depth" and (
        child_group != parent_group or child_type != parent_type
    ):
        contract_reason = (
            "in-depth mutation must preserve CONCEPT_GROUP and CONCEPT_TYPE: "
            f"{parent_group}/{parent_type} -> {child_group}/{child_type}"
        )
    elif op == "in_breadth" and child_group == parent_group:
        contract_reason = (
            "in-breadth mutation must change CONCEPT_GROUP: "
            f"{parent_group!r} -> {child_group!r}"
        )
    if contract_reason is None and mutation_plan is not None:
        target_group = str(mutation_plan.get("target_concept_group", ""))
        target_type = str(mutation_plan.get("target_concept_type", ""))
        if child_group != target_group:
            contract_reason = (
                "generated mutation must implement target_concept_group: "
                f"{target_group!r} -> {child_group!r}"
            )
        elif child_type != target_type:
            contract_reason = (
                "generated mutation must implement target_concept_type: "
                f"{target_type!r} -> {child_type!r}"
            )
    if contract_reason is not None:
        result["reason"] = contract_reason
        result["source_code"] = code
        result["operator_contract_valid"] = False
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
    if free_form_direct:
        # The static `str(sympy.Integer(answer))` contract was dropped for this
        # path, so the same guarantee is enforced on the executed output: every
        # verified instance must actually carry a plain integer answer.
        bad = [
            item
            for item in instances
            if not re.fullmatch(r"-?\d+", str(item["answer"]).strip())
        ]
        if bad:
            result["reason"] = (
                "free-form generator produced non-integer answers on "
                f"{len(bad)}/{len(instances)} seeds, e.g. "
                f"{bad[0]['answer']!r} at seed={bad[0]['seed']}"
            )
            result["source_code"] = code
            return child, result

    result.update(
        {
            "valid": True,
            "reason": None,
            "program_id": child.program_id,
            "concept_group": child.get_concept_group(),
            "concept_type": child.get_concept_type(),
            "operator_contract_valid": True,
            "distinct_problem_count": len(
                {" ".join(item["problem"].split()) for item in instances}
            ),
            "distinct_answer_count": len(
                {str(item["answer"]).strip() for item in instances}
            ),
            "constant_answer_family": len(
                {str(item["answer"]).strip() for item in instances}
            )
            == 1,
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
    family_contract: Mapping[str, Any] | None = None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    chat_template_kwargs: dict[str, Any],
    operator: str,
    llm_seed: int,
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
            family_contract=family_contract,
        )
        pending.append((seed, instance, messages))

    if pending:
        common_sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        request_sampling_params = [
            {
                **common_sampling_params,
                "seed": _evaluator_request_seed(llm_seed, operator, seed),
            }
            for seed, _, _ in pending
        ]
        raw_outputs = inference.generate(
            [messages for _, _, messages in pending],
            sampling_params=request_sampling_params,
            chat_template_kwargs=chat_template_kwargs,
        )
        assert isinstance(raw_outputs, list)
        if len(raw_outputs) != len(pending):
            raise RuntimeError(
                "evaluator returned a different number of outputs than seeds"
            )
        for (seed, instance, messages), raw in zip(pending, raw_outputs):
            valid, reason = parse_evaluator_verdict(
                raw,
                require_target_move=mutation_plan is not None,
            )
            per_seed.append(
                {
                    "seed": seed,
                    "problem": instance.problem,
                    "answer": instance.answer,
                    "messages": messages,
                    "raw_output": raw,
                    "sampling_seed": _evaluator_request_seed(
                        llm_seed,
                        operator,
                        seed,
                    ),
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
    operator: str,
    llm_seed: int,
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
        request_seed_factory=lambda instance, rollout_idx: (
            _child_solver_request_seed(
                llm_seed,
                operator,
                int(instance.seed),
                rollout_idx,
            )
        ),
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


def _message_token_count(
    tokenizer: Any,
    messages: Conversation,
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> int | None:
    if tokenizer is None:
        return None
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
        return len(token_ids)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        try:
            text = "\n".join(
                f"{message['role']}: {message['content']}"
                for message in messages
            )
            return len(tokenizer.encode(text, add_special_tokens=True))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None


def _save_task(
    path: Path,
    task,
    *,
    tokenizer: Any = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> int | None:
    messages = _messages_for_task(task)
    input_token_count = _message_token_count(
        tokenizer,
        messages,
        chat_template_kwargs=chat_template_kwargs,
    )
    _write_json(
        path,
        {
            "stage": task.stage,
            "operator": task.op,
            "plan_id": task.plan_id,
            "plan_status": task.plan_status,
            "generation_path": getattr(task, "generation_path", "freeform"),
            "generator_family": getattr(task, "generator_family", None),
            "compiler_version": getattr(task, "compiler_version", None),
            "compiler_diagnostics": getattr(
                task,
                "compiler_diagnostics",
                [],
            ),
            "quarantined": bool(getattr(task, "quarantined", False)),
            "max_output_tokens": task.max_output_tokens,
            "temperature": task.temperature,
            "top_p": task.top_p,
            "input_token_count": input_token_count,
            "messages": messages,
        },
    )
    return input_token_count


def _length_match_task(
    task,
    *,
    tokenizer: Any,
    target_tokens: int | None,
    chat_template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Append semantically neutral whitespace/period padding toward a token target."""

    messages = _messages_for_task(task)
    base = _message_token_count(
        tokenizer,
        messages,
        chat_template_kwargs=chat_template_kwargs,
    )
    if target_tokens is None or base is None or base >= target_tokens:
        return {
            "target": target_tokens,
            "actual": base,
            "delta": (
                base - target_tokens
                if base is not None and target_tokens is not None
                else None
            ),
            "padding_units": 0,
        }

    original = str(messages[-1]["content"])
    prefix = (
        "\n\nNEUTRAL_LENGTH_PADDING (no information; ignore completely):\n"
    )

    def candidate(units: int) -> tuple[int | None, Conversation]:
        padded = [dict(message) for message in messages]
        padded[-1]["content"] = original + prefix + (" ." * units)
        return (
            _message_token_count(
                tokenizer,
                padded,
                chat_template_kwargs=chat_template_kwargs,
            ),
            padded,
        )

    high = max(32, (target_tokens - base) * 2)
    while True:
        count, _ = candidate(high)
        if count is None or count >= target_tokens or high >= 131_072:
            break
        high *= 2
    low = 0
    while low < high:
        mid = (low + high) // 2
        count, _ = candidate(mid)
        if count is None or count >= target_tokens:
            high = mid
        else:
            low = mid + 1
    choices = range(max(0, low - 16), low + 17)
    scored = []
    for units in choices:
        count, padded = candidate(units)
        if count is not None:
            scored.append((abs(count - target_tokens), count, units, padded))
    if not scored:
        return {
            "target": target_tokens,
            "actual": base,
            "delta": base - target_tokens,
            "padding_units": 0,
        }
    _, actual, units, padded = min(scored, key=lambda item: (item[0], item[1]))
    task.messages = padded
    task.prompt = "\n\n".join(message["content"] for message in padded)
    return {
        "target": target_tokens,
        "actual": actual,
        "delta": actual - target_tokens,
        "padding_units": units,
    }


def _extract_json_mapping(raw_output: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model completion.

    vLLM base models sometimes retain a thinking prefix or a Markdown fence.
    Scanning with ``JSONDecoder.raw_decode`` accepts either without weakening
    the schema validator that runs immediately afterwards.
    """

    decoder = json.JSONDecoder()
    fallback: dict[str, Any] | None = None
    schema_candidates: list[dict[str, Any]] = []
    for index, character in enumerate(str(raw_output)):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(str(raw_output)[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        candidate = dict(value)
        fallback = candidate
        if "schema_version" in candidate and "operator" in candidate:
            schema_candidates.append(candidate)
    # Base checkpoints sometimes echo the entire prompt, including its example
    # JSON, before emitting their answer. The completion's final schema object
    # is the answer; choosing the first one mistakenly validates the echoed
    # placeholder instead.
    return schema_candidates[-1] if schema_candidates else fallback


def _wrong_trace_from_evidence(
    evidence: list[dict[str, Any]] | None,
) -> str | None:
    for item in evidence or []:
        if (
            str(item.get("role", "")).strip().lower() == "failure"
            and item.get("correct") is not True
        ):
            response = str(item.get("response", "")).strip()
            if response:
                return response
    return None


def _belief_structured_output_schema(
    *,
    generator_family: str,
    operator: str,
    wrong_trace: str | None,
) -> dict[str, Any]:
    """Constrain v6 output to catalog IDs and extractive evidence lines."""

    from rq_evolve.belief_probe import BELIEF_SCHEMA_VERSION, hypotheses_for

    hypothesis_ids = [
        hypothesis.hypothesis_id
        for hypothesis in hypotheses_for(generator_family)
    ]
    quote_options: list[str] = []
    if wrong_trace:
        for raw_line in wrong_trace.splitlines():
            line = raw_line.strip()
            if line and line not in quote_options:
                quote_options.append(line)
    evidence_schema: dict[str, Any] = (
        {"type": "string", "enum": quote_options}
        if quote_options
        else {"type": "null"}
    )
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "integer", "const": BELIEF_SCHEMA_VERSION},
            "operator": {"type": "string", "const": operator},
            "attributed_hypothesis": {
                "type": "string",
                "enum": hypothesis_ids,
            },
            "evidence_quote": evidence_schema,
        },
        "required": [
            "schema_version",
            "operator",
            "attributed_hypothesis",
            "evidence_quote",
        ],
        "additionalProperties": False,
    }


def _belief_candidate_is_eligible(row: Mapping[str, Any]) -> bool:
    eligibility = row.get("eligibility")
    return (
        isinstance(eligibility, Mapping)
        and eligibility.get("eligible") is True
        and row.get("status")
        in {"eligible_scored", "eligible_unscored", "plan_only_eligible"}
    )


def _select_belief_candidates(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Retain one R_Q champion per attribution niche.

    Attribution determines the archive cell; it is never blended into fitness.
    R_Q alone ranks eligible candidates inside a cell. The overall selected row
    is only a convenience for the paired report and one-shot diagnostic; every
    niche champion remains recorded in the selection artifact.
    """

    eligible = [row for row in candidates if _belief_candidate_is_eligible(row)]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        grouped.setdefault(str(row.get("attribution_niche")), []).append(row)

    niche_champions: list[dict[str, Any]] = []
    for niche in sorted(grouped):
        rows = grouped[niche]
        scored = [
            row for row in rows if isinstance(row.get("rq_proxy"), (int, float))
        ]
        champion = (
            max(
                scored,
                key=lambda row: (
                    float(row["rq_proxy"]),
                    -int(row.get("candidate_index", 0)),
                ),
            )
            if scored
            else min(rows, key=lambda row: int(row.get("candidate_index", 0)))
        )
        niche_champions.append(
            {
                "attribution_niche": niche,
                "candidate_index": champion.get("candidate_index"),
                "attributed_hypothesis": champion.get(
                    "attributed_hypothesis"
                ),
                "rq_proxy": champion.get("rq_proxy"),
                "p_hat": champion.get("p_hat"),
                "compiler_source_hash": champion.get("compiler_source_hash"),
                "method_dir": champion.get("method_dir"),
            }
        )

    scored_champions = [
        row
        for row in niche_champions
        if isinstance(row.get("rq_proxy"), (int, float))
    ]
    selected = (
        max(
            scored_champions,
            key=lambda row: (
                float(row["rq_proxy"]),
                -int(row.get("candidate_index", 0)),
            ),
        )
        if scored_champions
        else niche_champions[0]
        if niche_champions
        else None
    )
    return {
        "selection_rule": (
            "eligible gate, then max R_Q within attributed-hypothesis niche; "
            "overall row is max R_Q among niche champions"
        ),
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "niche_champion_count": len(niche_champions),
        "niche_champions": niche_champions,
        "selected_candidate_index": (
            selected.get("candidate_index") if selected else None
        ),
        "selected_attribution_niche": (
            selected.get("attribution_niche") if selected else None
        ),
    }


def _run_belief_candidate(
    inference: VLLMChatInference,
    *,
    method: str,
    op: str,
    parent: ProblemProgram,
    output_dir: Path,
    args: argparse.Namespace,
    chat_template_kwargs: dict[str, Any],
    planning_evidence: list[dict[str, Any]] | None,
    candidate_index: int,
) -> dict[str, Any]:
    """Run one schema-v6 attribution → verified probe → R_Q candidate."""

    from rq_evolve.belief_probe import (
        BELIEF_SCHEMA_VERSION,
        attribution_niche_key,
        build_belief_plan_prompt,
        check_probe_diagnosticity,
        evaluate_eligibility,
        get_hypothesis,
        score_belief_attribution,
        validate_belief_plan,
    )
    from rq_evolve.mutation_compiler import (
        MUTATION_FAMILY_REGISTRY_VERSION,
        compile_belief_probe,
        compiled_family_instances,
        registered_family_descriptor,
        validate_compiled_family_semantics,
    )

    condition = "plain" if method == "legacy" else "reasoning"
    reasoning_informed = method == "metacognitive"
    candidate_dir = (
        output_dir / op / method / f"candidate_{candidate_index:02d}"
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    evaluation_seeds = _resolve_evaluation_seeds(args)
    base = {
        "method": method,
        "operator": op,
        "condition": condition,
        "candidate_index": int(candidate_index),
        "planning_contract": "belief_v6",
        "plan_schema_version": BELIEF_SCHEMA_VERSION,
        "generation_path": "belief_probe_compiled",
        "configured_llm_generation_call_count": 1,
        "llm_generation_call_count": 0,
        "method_dir": str(candidate_dir.resolve()),
        "evaluation_seeds": evaluation_seeds,
        "evaluator_valid": None,
        "evaluator_reason": (
            "not used as a gate in belief_v6; Python semantics and "
            "diagnosticity checks are authoritative"
        ),
        "evaluator_passed": None,
        "evaluator_total": None,
        "evaluator_by_seed": [],
    }

    descriptor = registered_family_descriptor(parent, op)
    if descriptor is None:
        return {
            **base,
            "status": "unsupported_family",
            "reason": "no unique registered family for parent/operator",
            "eligibility": {
                "eligible": False,
                "valid": False,
                "diagnostic": False,
                "attribution_supported": False,
                "reasons": ["no unique registered family"],
            },
        }
    generator_family = descriptor.generator_family
    base.update(
        {
            "generator_family": generator_family,
            "concept_group": descriptor.concept_group,
            "concept_type": descriptor.concept_type,
            "compiler_registry_version": MUTATION_FAMILY_REGISTRY_VERSION,
        }
    )

    wrong_trace = (
        _wrong_trace_from_evidence(planning_evidence)
        if reasoning_informed
        else None
    )
    if reasoning_informed and wrong_trace is None:
        return {
            **base,
            "status": "insufficient_evidence",
            "reason": "belief_v6 reasoning condition needs one clean wrong trace",
            "eligibility": {
                "eligible": False,
                "valid": False,
                "diagnostic": False,
                "attribution_supported": False,
                "reasons": ["clean wrong trace unavailable"],
            },
        }

    prompt = build_belief_plan_prompt(
        parent_problem=parent.execute(args.instance_seed).problem,
        generator_family=generator_family,
        operator=op,
        wrong_trace=wrong_trace,
        # The manipulation is attribution from the wrong trace. A correct trace
        # would reintroduce outcome contrast analysis into the planner.
        correct_trace=None,
        template_dir=args.prompt_dir,
    )
    messages: Conversation = [{"role": "user", "content": prompt}]
    input_token_count = _message_token_count(
        getattr(inference, "tokenizer", None),
        messages,
        chat_template_kwargs=chat_template_kwargs,
    )
    plan_sampling_seed = _paired_request_seed(
        args.llm_seed,
        op,
        f"belief_plan:{candidate_index}",
    )
    structured_output_schema = _belief_structured_output_schema(
        generator_family=generator_family,
        operator=op,
        wrong_trace=wrong_trace,
    )
    _write_json(
        candidate_dir / "01_attribution_prompt.json",
        {
            "stage": "belief_attribution",
            "operator": op,
            "condition": condition,
            "candidate_index": candidate_index,
            "generator_family": generator_family,
            "input_token_count": input_token_count,
            "temperature": args.plan_temperature,
            "top_p": args.plan_top_p,
            "max_output_tokens": args.plan_max_tokens,
            "sampling_seed": plan_sampling_seed,
            "structured_output_schema": structured_output_schema,
            "messages": messages,
        },
    )
    raw_plan = inference.generate(
        messages,
        sampling_params={
            "temperature": args.plan_temperature,
            "top_p": args.plan_top_p,
            "max_tokens": args.plan_max_tokens,
            "seed": plan_sampling_seed,
            "structured_outputs": {"json": structured_output_schema},
        },
        chat_template_kwargs=chat_template_kwargs,
    )
    assert isinstance(raw_plan, str)
    base["llm_generation_call_count"] = 1
    _write_text(candidate_dir / "02_attribution_raw.txt", raw_plan)
    plan = _extract_json_mapping(raw_plan)
    plan_errors = (
        ["no JSON object found in attribution output"]
        if plan is None
        else validate_belief_plan(
            plan,
            operator=op,
            generator_family=generator_family,
            wrong_trace=wrong_trace,
        )
    )
    _write_json(
        candidate_dir / "03_attribution_validation.json",
        {
            "valid": not plan_errors,
            "errors": plan_errors,
            "plan": plan,
        },
    )
    if plan is None or plan_errors:
        eligibility = evaluate_eligibility(
            family_semantics_valid=False,
            diagnosticity=None,
            plan_errors=plan_errors,
        ).to_payload()
        _write_json(candidate_dir / "07_eligibility.json", eligibility)
        return {
            **base,
            "status": "invalid_attribution",
            "reason": "; ".join(plan_errors),
            "eligibility": eligibility,
            "plan_sampling_seed": plan_sampling_seed,
        }

    plan_id = hashlib.sha256(
        json.dumps(
            plan,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    hypothesis_id = str(plan["attributed_hypothesis"])
    hypothesis = get_hypothesis(generator_family, hypothesis_id)
    assert hypothesis is not None
    compilation = compile_belief_probe(plan, parent, op)
    _write_json(
        candidate_dir / "04_compiler_validation.json",
        {
            "registry_version": MUTATION_FAMILY_REGISTRY_VERSION,
            "status": compilation.status.value,
            "generator_family": compilation.generator_family,
            "family_variant": compilation.family_variant,
            "variant_selected_by_plan": False,
            "variant_selected_by_hypothesis": True,
            "variant_targets_failure_mode": (
                compilation.variant_targets_failure_mode
            ),
            "family_config": dict(compilation.family_config),
            "source_hash": compilation.source_hash,
            "reasons": list(compilation.reasons),
            "attributed_hypothesis": hypothesis_id,
            "hypothesis_kind": hypothesis.kind,
            "probe_rationale": hypothesis.probe_rationale,
        },
    )
    if not compilation.compiled or compilation.source_code is None:
        eligibility = evaluate_eligibility(
            family_semantics_valid=False,
            family_semantics_reasons=compilation.reasons,
            diagnosticity=None,
            plan_errors=(),
        ).to_payload()
        _write_json(candidate_dir / "07_eligibility.json", eligibility)
        return {
            **base,
            "status": compilation.status.value,
            "reason": "; ".join(compilation.reasons),
            "plan_id": plan_id,
            "attributed_hypothesis": hypothesis_id,
            "hypothesis_kind": hypothesis.kind,
            "eligibility": eligibility,
            "plan_sampling_seed": plan_sampling_seed,
        }

    semantics = validate_compiled_family_semantics(
        compilation,
        evaluation_seeds,
    )
    _write_json(
        candidate_dir / "05_family_semantics.json",
        dict(semantics.to_payload()),
    )
    structured_instances = compiled_family_instances(compilation)(
        evaluation_seeds
    )
    diagnosticity = check_probe_diagnosticity(
        hypothesis,
        structured_instances,
    )
    _write_json(
        candidate_dir / "06_probe_diagnosticity.json",
        diagnosticity.to_payload(),
    )
    eligibility_object = evaluate_eligibility(
        family_semantics_valid=semantics.valid,
        family_semantics_reasons=semantics.reasons,
        diagnosticity=diagnosticity,
        plan_errors=(),
    )
    eligibility = eligibility_object.to_payload()
    _write_json(candidate_dir / "07_eligibility.json", eligibility)

    common = {
        **base,
        "plan_id": plan_id,
        "plan_sampling_seed": plan_sampling_seed,
        "attributed_hypothesis": hypothesis_id,
        "hypothesis_kind": hypothesis.kind,
        "evidence_quote": plan.get("evidence_quote"),
        "attribution_niche": attribution_niche_key(
            descriptor.concept_type,
            hypothesis_id,
        ),
        "family_variant": compilation.family_variant,
        "variant_selected_by_plan": False,
        "variant_selected_by_hypothesis": True,
        "variant_targets_failure_mode": (
            compilation.variant_targets_failure_mode
        ),
        "compiler_source_hash": compilation.source_hash,
        "family_semantics_valid": semantics.valid,
        "probe_diagnosticity_valid": diagnosticity.valid,
        "eligibility": eligibility,
        "generated_instances": [
            {
                "seed": int(entry["seed"]),
                "problem": entry.get("problem"),
                "answer": entry.get("answer"),
            }
            for entry in semantics.per_seed
        ],
    }
    if not eligibility_object.eligible:
        return {
            **common,
            "status": "probe_ineligible",
            "reason": "; ".join(eligibility_object.reasons),
        }
    if getattr(args, "plan_only", False):
        return {
            **common,
            "status": "plan_only_eligible",
            "reason": None,
        }

    raw_code = f"```python\n{compilation.source_code}\n```"
    child, validation = _validate_generated_program(
        raw_code,
        parent=parent,
        op=op,
        mutation_plan=None,
        plan_id=plan_id,
        verify_seeds=args.verify_seeds,
    )
    source = validation.pop("source_code", None)
    if source:
        _write_text(candidate_dir / "child.py", source)
    _write_json(candidate_dir / "08_code_validation.json", validation)
    if child is None or not validation["valid"]:
        failed_eligibility = {
            **eligibility,
            "eligible": False,
            "valid": False,
            "reasons": [
                *eligibility.get("reasons", []),
                f"mechanical code validation: {validation.get('reason')}",
            ],
        }
        _write_json(candidate_dir / "07_eligibility.json", failed_eligibility)
        return {
            **common,
            "status": "invalid_code",
            "reason": validation.get("reason"),
            "eligibility": failed_eligibility,
        }

    child.metadata.update(
        {
            "mutation_plan": plan,
            "plan_id": plan_id,
            "plan_status": "belief_v6",
            "generator_family": generator_family,
            "family_variant": compilation.family_variant,
            "compiler_source_hash": compilation.source_hash,
            "attributed_hypothesis": hypothesis_id,
            "attribution_niche": common["attribution_niche"],
        }
    )
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
            operator=op,
            llm_seed=args.llm_seed,
        )
        _write_json(candidate_dir / "09_child_rollouts.json", child_score)

    attribution_score = None
    seed_scores: list[dict[str, Any]] = []
    if child_score is not None:
        rollout_map: dict[int, list[dict[str, Any]]] = {}
        for seed_row in child_score.get("per_seed", []):
            if seed_row.get("status") != "ok":
                continue
            seed = int(seed_row["seed"])
            rollout_map[seed] = [
                {
                    "predicted_answer": rollout.get("predicted_answer"),
                    "correct": rollout.get("correct"),
                    "response": rollout.get("cleaned_text", ""),
                    "mean_negative_logprob": rollout.get(
                        "mean_negative_logprob"
                    ),
                }
                for rollout in seed_row.get("rollouts", [])
            ]
            seed_scores.append(
                {
                    key: value
                    for key, value in seed_row.items()
                    if key != "rollouts"
                }
            )
        attribution_score = score_belief_attribution(
            hypothesis,
            structured_instances,
            rollout_map,
        )
        _write_json(
            candidate_dir / "10_attribution_score.json",
            attribution_score,
        )

    return {
        **common,
        "status": (
            "eligible_scored"
            if child_score is not None
            else "eligible_unscored"
        ),
        "reason": None,
        "program_id": child.program_id,
        "p_hat": child_score.get("p_hat") if child_score else None,
        "rq_proxy": child_score.get("rq_proxy") if child_score else None,
        "num_correct": child_score.get("num_correct") if child_score else None,
        "num_rollouts": (
            child_score.get("num_rollouts") if child_score else None
        ),
        "seed_scores": seed_scores,
        "attribution_score": attribution_score,
        "child_file": str((candidate_dir / "child.py").resolve()),
    }


def _run_belief_condition(
    inference: VLLMChatInference,
    *,
    method: str,
    op: str,
    parent: ProblemProgram,
    output_dir: Path,
    args: argparse.Namespace,
    chat_template_kwargs: dict[str, Any],
    planning_evidence: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    import traceback

    method_dir = output_dir / op / method
    method_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for index in range(int(args.candidates_per_condition)):
        try:
            candidate = _run_belief_candidate(
                inference,
                method=method,
                op=op,
                parent=parent,
                output_dir=output_dir,
                args=args,
                chat_template_kwargs=chat_template_kwargs,
                planning_evidence=planning_evidence,
                candidate_index=index,
            )
        except Exception as exc:
            candidate_dir = method_dir / f"candidate_{index:02d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate = {
                "method": method,
                "operator": op,
                "condition": (
                    "plain" if method == "legacy" else "reasoning"
                ),
                "candidate_index": index,
                "planning_contract": "belief_v6",
                "status": "exception",
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "method_dir": str(candidate_dir.resolve()),
                "eligibility": {
                    "eligible": False,
                    "valid": False,
                    "diagnostic": False,
                    "attribution_supported": False,
                    "reasons": [f"candidate exception: {exc}"],
                },
            }
        candidates.append(candidate)
    selection = _select_belief_candidates(candidates)
    _write_json(method_dir / "candidates.json", candidates)
    _write_json(method_dir / "rq_selection.json", selection)
    selected_index = selection["selected_candidate_index"]
    if selected_index is None:
        return {
            "method": method,
            "operator": op,
            "condition": "plain" if method == "legacy" else "reasoning",
            "planning_contract": "belief_v6",
            "status": "no_eligible_candidate",
            "reason": "no candidate passed the Python eligibility gate",
            "method_dir": str(method_dir.resolve()),
            **selection,
        }
    selected = next(
        row
        for row in candidates
        if int(row["candidate_index"]) == int(selected_index)
    )
    return {
        **selected,
        **selection,
        "selected_candidate_dir": selected["method_dir"],
    }


def _run_direct_candidate(
    inference: VLLMChatInference,
    *,
    method: str,
    op: str,
    parent: ProblemProgram,
    output_dir: Path,
    args: argparse.Namespace,
    chat_template_kwargs: dict[str, Any],
    planning_evidence: list[dict[str, Any]] | None,
    candidate_index: int,
) -> dict[str, Any]:
    """One direct mutation: read the traces, write the generator, verify, score.

    No plan, no schema, no hypothesis catalog. The whole experimental
    manipulation is whether the correct/wrong solver pair appears in the prompt,
    so both conditions make exactly one generation call against the same
    template with the same sampling. Verification stays on the produced problem
    -- lint, multi-seed execution, seed variation -- which does not care who
    wrote the generator.
    """
    from rq_evolve.prompts import build_mutation_task

    condition = "plain" if method == "legacy" else "reasoning"
    candidate_dir = output_dir / op / method / f"candidate_{candidate_index:02d}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    evaluation_seeds = _resolve_evaluation_seeds(args)
    # Only the reasoning condition is shown the traces; everything else matches.
    evidence = planning_evidence if condition == "reasoning" else None

    task = build_mutation_task(
        op,
        parent,
        reasoning_evidence=evidence,
        temperature=args.code_temperature,
        top_p=args.code_top_p,
    )
    request_seed = _paired_request_seed(
        args.llm_seed,
        op,
        f"direct_code:{candidate_index}",
    )
    input_token_count = _save_task(
        candidate_dir / "01_mutation_prompt.json",
        task,
        tokenizer=inference.tokenizer,
        chat_template_kwargs=chat_template_kwargs,
    )
    raw = inference.generate(
        [task.messages],
        sampling_params=[
            {
                "temperature": args.code_temperature,
                "top_p": args.code_top_p,
                "max_tokens": args.mutation_max_tokens,
                "seed": request_seed,
            }
        ],
        chat_template_kwargs=chat_template_kwargs,
    )[0]
    _write_text(candidate_dir / "02_mutation_raw.txt", raw)

    common = {
        "method": method,
        "operator": op,
        "condition": condition,
        "candidate_index": candidate_index,
        "planning_contract": "direct_v7",
        "generation_path": "direct_freeform",
        "sees_reasoning_traces": evidence is not None,
        "evidence_trace_count": len(evidence or []),
        "llm_generation_call_count": 1,
        "method_dir": str(candidate_dir.resolve()),
        "mutation_request_seed": request_seed,
        "mutation_input_tokens": input_token_count,
    }

    child, validation = _validate_generated_program(
        raw,
        parent=parent,
        op=op,
        mutation_plan=None,
        plan_id=None,
        verify_seeds=args.verify_seeds,
        free_form_direct=True,
    )
    source = validation.pop("source_code", None)
    if source:
        _write_text(candidate_dir / "child.py", source)
    _write_json(candidate_dir / "03_code_validation.json", validation)
    if child is None or not validation["valid"]:
        return {
            **common,
            "status": "invalid_code",
            "reason": validation["reason"],
            "program_id": child.program_id if child is not None else None,
        }

    # Coherence gate. Relaxing the notation lint let through a generator whose
    # problem never stated n while its answer used n = 6 -- unanswerable, and
    # the solver scored 0/250 on it. Static checks cannot see that; the
    # evaluator's original job is exactly "does the answer solve the visible
    # problem", and with no plan text in play it is only asked that.
    evaluator = _evaluate_program_seeds(
        inference,
        child,
        seeds=evaluation_seeds,
        mutation_plan=None,
        family_contract=None,
        max_tokens=args.evaluator_max_tokens,
        temperature=args.evaluator_temperature,
        top_p=args.evaluator_top_p,
        chat_template_kwargs=chat_template_kwargs,
        operator=op,
        llm_seed=args.llm_seed,
    )
    _write_json(candidate_dir / "04_evaluator.json", evaluator)
    if not evaluator["valid"]:
        return {
            **common,
            "status": "evaluator_rejected",
            "reason": "; ".join(
                f"seed={row['seed']}: {row.get('reason') or 'INVALID'}"
                for row in evaluator["per_seed"]
                if row.get("valid") is not True
            )[:400],
            "program_id": child.program_id,
            "evaluator_passed": evaluator["num_valid"],
            "evaluator_total": evaluator["num_seeds"],
        }

    child_score = _score_program_seeds(
        inference,
        child,
        seeds=evaluation_seeds,
        count=args.child_rollouts,
        max_tokens=args.solver_max_tokens,
        temperature=args.child_temperature,
        top_p=args.child_top_p,
        chat_template_kwargs=chat_template_kwargs,
        operator=op,
        llm_seed=args.llm_seed,
    )
    _write_json(candidate_dir / "05_child_rollouts.json", child_score)
    return {
        **common,
        "status": "scored",
        "reason": None,
        "program_id": child.program_id,
        "evaluator_passed": evaluator["num_valid"],
        "evaluator_total": evaluator["num_seeds"],
        "concept_group": child.get_concept_group(),
        "concept_type": child.get_concept_type(),
        "evaluation_seeds": evaluation_seeds,
        "p_hat": child_score.get("p_hat"),
        "rq_proxy": child_score.get("rq_proxy"),
        "num_correct": child_score.get("num_correct"),
        "num_rollouts": child_score.get("num_rollouts"),
        "generated_instances": validation.get("instances", []),
        "child_file": str((candidate_dir / "child.py").resolve()),
        "seed_scores": [
            {key: value for key, value in item.items() if key != "rollouts"}
            for item in child_score.get("per_seed", [])
        ],
    }


def _run_direct_condition(
    inference: VLLMChatInference,
    *,
    method: str,
    op: str,
    parent: ProblemProgram,
    output_dir: Path,
    args: argparse.Namespace,
    chat_template_kwargs: dict[str, Any],
    planning_evidence: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Draw N direct mutations and keep the highest R_Q among the valid ones."""
    import traceback

    method_dir = output_dir / op / method
    method_dir.mkdir(parents=True, exist_ok=True)
    condition = "plain" if method == "legacy" else "reasoning"
    candidates: list[dict[str, Any]] = []
    for index in range(int(args.candidates_per_condition)):
        try:
            candidate = _run_direct_candidate(
                inference,
                method=method,
                op=op,
                parent=parent,
                output_dir=output_dir,
                args=args,
                chat_template_kwargs=chat_template_kwargs,
                planning_evidence=planning_evidence,
                candidate_index=index,
            )
        except Exception as exc:
            candidate = {
                "method": method,
                "operator": op,
                "condition": condition,
                "candidate_index": index,
                "planning_contract": "direct_v7",
                "status": "exception",
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "method_dir": str(
                    (method_dir / f"candidate_{index:02d}").resolve()
                ),
            }
        candidates.append(candidate)

    scored = [
        row
        for row in candidates
        if row.get("status") == "scored"
        and isinstance(row.get("rq_proxy"), (int, float))
    ]
    selection = {
        "candidate_count": len(candidates),
        "valid_candidate_count": len(scored),
        "selection_rule": (
            "lint + multi-seed verification gate, then max R_Q among valid "
            "candidates"
        ),
        "selected_candidate_index": (
            max(scored, key=lambda row: float(row["rq_proxy"]))["candidate_index"]
            if scored
            else None
        ),
    }
    _write_json(method_dir / "candidates.json", candidates)
    _write_json(method_dir / "rq_selection.json", selection)
    if selection["selected_candidate_index"] is None:
        invalid = [row.get("reason") for row in candidates if row.get("reason")]
        return {
            "method": method,
            "operator": op,
            "condition": condition,
            "planning_contract": "direct_v7",
            "status": "no_valid_candidate",
            "reason": "; ".join(str(item) for item in invalid[:3]) or "unknown",
            "method_dir": str(method_dir.resolve()),
            **selection,
        }
    selected = next(
        row
        for row in candidates
        if int(row["candidate_index"])
        == int(selection["selected_candidate_index"])
    )
    return {
        **selected,
        **selection,
        "selected_candidate_dir": selected["method_dir"],
    }


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
        build_plain_plan_task,
        build_planned_mutation_task,
        parse_mutation_plan,
    )
    from rq_evolve.metacognition import mutation_plan_id
    from rq_evolve.mutation_compiler import (
        MUTATION_FAMILY_REGISTRY_VERSION,
        CompilationStatus,
        compile_mutation_plan,
        family_contract_payload,
        validate_compiled_family_semantics,
    )

    method_dir = output_dir / op / method
    method_dir.mkdir(parents=True, exist_ok=True)
    evaluation_seeds = _resolve_evaluation_seeds(args)
    plan: dict[str, Any] | None = None
    plan_id: str | None = None
    raw_plan: str | None = None
    raw_code: str | None = None
    extracted_source: str | None = None
    code_task = None
    compiled_source: str | None = None
    compiler_status: str | None = None
    compiler_source_hash: str | None = None
    compiler_reasons: list[str] = []
    generation_path = "freeform"
    generator_family: str | None = None
    family_config: dict[str, Any] | None = None
    family_contract: Mapping[str, Any] | None = None
    family_semantics = None
    family_variant: str | None = None
    variant_selected_by_plan: bool | None = None
    variant_targets_failure_mode: str | None = None
    quarantined = False
    condition = "plain" if method == "legacy" else "reasoning"
    reasoning_informed = method == "metacognitive"
    uses_plan = reasoning_informed or (
        method == "legacy"
        and getattr(args, "plain_baseline", "two_stage") == "two_stage"
    )
    plan_calls_completed = 0
    code_calls_completed = 0
    configured_code_call_count = 1
    plan_input_token_count: int | None = None
    code_input_token_count: int | None = None
    plan_input_token_target: int | None = None
    plan_input_token_delta: int | None = None
    plan_neutral_padding_units = 0
    base_llm_seed = int(getattr(args, "llm_seed", 0))
    mutation_code_backend = str(
        getattr(args, "mutation_code_backend", "freeform")
    )
    plan_sampling_seed = (
        _paired_request_seed(base_llm_seed, op, "plan")
        if uses_plan
        else None
    )
    code_sampling_seed = _paired_request_seed(
        base_llm_seed,
        op,
        "code",
    )
    evaluator_seed_schedule = [
        {
            "instance_seed": int(seed),
            "sampling_seed": _evaluator_request_seed(
                base_llm_seed,
                op,
                int(seed),
            ),
        }
        for seed in evaluation_seeds
    ]
    child_solver_seed_schedule = [
        {
            "instance_seed": int(seed),
            "rollout_idx": int(rollout_idx),
            "sampling_seed": _child_solver_request_seed(
                base_llm_seed,
                op,
                int(seed),
                int(rollout_idx),
            ),
        }
        for seed in evaluation_seeds
        for rollout_idx in range(
            max(0, int(getattr(args, "child_rollouts", 0)))
        )
    ]

    def _artifacts() -> dict[str, Any]:
        """생성이 중간에 실패해도 모델 원본 출력을 summary에 남긴다."""
        return {
            "method_dir": str(method_dir.resolve()),
            "evaluation_seeds": list(evaluation_seeds),
            "raw_plan": raw_plan,
            "plan": plan,
            "raw_code": raw_code,
            "extracted_source": extracted_source,
            "mutation_code_backend": mutation_code_backend,
            "generation_path": generation_path,
            "generator_family": generator_family,
            "family_config": family_config,
            "compiler_registry_version": (
                MUTATION_FAMILY_REGISTRY_VERSION
                if compiler_status is not None
                else None
            ),
            "compiler_status": compiler_status,
            "compiler_source_hash": compiler_source_hash,
            "compiler_reasons": list(compiler_reasons),
            "quarantined": quarantined,
            "condition": condition,
            "conditioning": (
                "solver_reasoning_contrast"
                if reasoning_informed
                else "parent_only"
            ),
            "comparison_design": (
                "hybrid_compiler_plain_vs_reasoning_v2"
                if uses_plan and mutation_code_backend != "freeform"
                else "two_stage_plain_vs_reasoning_v1"
                if uses_plan
                else "legacy_one_stage_vs_reasoning"
            ),
            "configured_llm_generation_call_count": (
                (1 if uses_plan else 0) + configured_code_call_count
            ),
            "llm_generation_call_count": (
                plan_calls_completed + code_calls_completed
            ),
            "plan_stage_max_tokens": (
                int(args.plan_max_tokens) if uses_plan else 0
            ),
            "code_stage_max_tokens": (
                int(args.mutation_max_tokens)
                if configured_code_call_count
                else 0
            ),
            "total_generator_max_token_budget": (
                (int(args.plan_max_tokens) if uses_plan else 0)
                + (
                    int(args.mutation_max_tokens)
                    if configured_code_call_count
                    else 0
                )
            ),
            "plan_input_token_count": plan_input_token_count,
            "plan_input_token_target": plan_input_token_target,
            "plan_input_token_delta": plan_input_token_delta,
            "plan_neutral_padding_units": plan_neutral_padding_units,
            "code_input_token_count": code_input_token_count,
            "total_generator_input_token_count": (
                plan_input_token_count + code_input_token_count
                if plan_input_token_count is not None
                and code_input_token_count is not None
                else plan_input_token_count
                if plan_input_token_count is not None
                and configured_code_call_count == 0
                else code_input_token_count
                if code_input_token_count is not None
                and not uses_plan
                else None
            ),
            "paired_request_seed_policy": PAIRED_REQUEST_SEED_POLICY,
            "plan_sampling_seed": plan_sampling_seed,
            "code_sampling_seed": code_sampling_seed,
            "evaluator_sampling_seed_policy": (
                PAIRED_EVALUATOR_SEED_POLICY
            ),
            "evaluator_sampling_seed_sha256": _seed_schedule_digest(
                evaluator_seed_schedule
            ),
            "evaluator_sampling_seed_range": _seed_schedule_range(
                evaluator_seed_schedule
            ),
            "child_solver_sampling_seed_policy": (
                PAIRED_CHILD_SOLVER_SEED_POLICY
            ),
            "child_solver_sampling_seed_sha256": _seed_schedule_digest(
                child_solver_seed_schedule
            ),
            "child_solver_sampling_seed_range": _seed_schedule_range(
                child_solver_seed_schedule
            ),
        }

    if method not in {"legacy", "metacognitive"}:
        raise ValueError(f"unknown method: {method}")

    if not uses_plan:
        code_task = build_mutation_task(
            op,
            parent,
            temperature=args.code_temperature,
            top_p=args.code_top_p,
        )
    else:
        if reasoning_informed and len(planning_evidence or []) < 2:
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
        plan_kwargs = {
            "meta_progress": meta_progress or _empty_meta_progress(),
            "max_output_tokens": args.plan_max_tokens,
            "temperature": args.plan_temperature,
            "top_p": args.plan_top_p,
        }
        plan_task = (
            build_metacognitive_plan_task(
                op,
                parent,
                evidence=planning_evidence or [],
                reasoning_informed=True,
                **plan_kwargs,
            )
            if reasoning_informed
            else build_plain_plan_task(
                op,
                parent,
                **plan_kwargs,
            )
        )
        if reasoning_informed:
            plan_input_token_target = _message_token_count(
                getattr(inference, "tokenizer", None),
                _messages_for_task(plan_task),
                chat_template_kwargs=chat_template_kwargs,
            )
        else:
            reasoning_reference = build_metacognitive_plan_task(
                op,
                parent,
                evidence=planning_evidence or [],
                reasoning_informed=True,
                **plan_kwargs,
            )
            plan_input_token_target = _message_token_count(
                getattr(inference, "tokenizer", None),
                _messages_for_task(reasoning_reference),
                chat_template_kwargs=chat_template_kwargs,
            )
            padding = _length_match_task(
                plan_task,
                tokenizer=getattr(inference, "tokenizer", None),
                target_tokens=plan_input_token_target,
                chat_template_kwargs=chat_template_kwargs,
            )
            plan_input_token_delta = padding["delta"]
            plan_neutral_padding_units = padding["padding_units"]
        plan_input_token_count = _save_task(
            method_dir / "01_plan_prompt.json",
            plan_task,
            tokenizer=getattr(inference, "tokenizer", None),
            chat_template_kwargs=chat_template_kwargs,
        )
        if plan_input_token_delta is None and plan_input_token_count is not None:
            plan_input_token_delta = (
                plan_input_token_count - plan_input_token_target
                if plan_input_token_target is not None
                else None
            )
        raw_plan = inference.generate(
            _messages_for_task(plan_task),
            sampling_params={
                "temperature": args.plan_temperature,
                "top_p": args.plan_top_p,
                "max_tokens": args.plan_max_tokens,
                "seed": plan_sampling_seed,
            },
            chat_template_kwargs=chat_template_kwargs,
        )
        assert isinstance(raw_plan, str)
        plan_calls_completed += 1
        _write_text(method_dir / "02_plan_raw.txt", raw_plan)
        plan, plan_reason = parse_mutation_plan(
            raw_plan,
            op,
            reasoning_informed=reasoning_informed,
            parent=parent,
        )
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
        plan_id = mutation_plan_id(plan)
        if mutation_code_backend in {"hybrid", "compiler_only"}:
            compilation = compile_mutation_plan(plan, parent, op)
            compiler_status = compilation.status.value
            generator_family = compilation.generator_family
            family_config = dict(compilation.family_config)
            compiler_source_hash = compilation.source_hash
            compiler_reasons = list(compilation.reasons)
            family_variant = compilation.family_variant
            variant_selected_by_plan = compilation.variant_selected_by_plan
            variant_targets_failure_mode = (
                compilation.variant_targets_failure_mode
            )
            _write_json(
                method_dir / "04_compiler_validation.json",
                {
                    "registry_version": MUTATION_FAMILY_REGISTRY_VERSION,
                    "status": compiler_status,
                    "generator_family": generator_family,
                    "family_variant": compilation.family_variant,
                    "variant_selected_by_plan": (
                        compilation.variant_selected_by_plan
                    ),
                    "variant_targets_failure_mode": (
                        compilation.variant_targets_failure_mode
                    ),
                    "variant_overridden_plan_keys": list(
                        compilation.variant_overridden_plan_keys
                    ),
                    "family_config": family_config,
                    "source_hash": compiler_source_hash,
                    "reasons": compiler_reasons,
                },
            )
            if compilation.status is CompilationStatus.COMPILED:
                generation_path = "registered_compiled"
                configured_code_call_count = 0
                compiled_source = compilation.source_code
                # Deterministic family-semantic gate: prove correctness AND the
                # contract's necessity on every evaluation seed before spending
                # an LLM evaluator call on the candidate.
                family_semantics = validate_compiled_family_semantics(
                    compilation,
                    evaluation_seeds,
                )
                _write_json(
                    method_dir / "05_family_semantics.json",
                    dict(family_semantics.to_payload()),
                )
                family_contract = family_contract_payload(
                    compilation,
                    family_semantics,
                )
                if not family_semantics.valid:
                    return {
                        "method": method,
                        "operator": op,
                        "status": "family_semantics_rejected",
                        "reason": "; ".join(family_semantics.reasons),
                        "plan_id": plan_id,
                        **_artifacts(),
                    }
            elif compilation.status in {
                CompilationStatus.INVALID_SPEC,
                CompilationStatus.COMPILER_ERROR,
            }:
                configured_code_call_count = 0
                return {
                    "method": method,
                    "operator": op,
                    "status": compilation.status.value,
                    "reason": "; ".join(compilation.reasons),
                    "plan_id": plan_id,
                    **_artifacts(),
                }
            else:
                generation_path = "free_form_quarantine"
                quarantined = True
                if mutation_code_backend == "compiler_only":
                    configured_code_call_count = 0
                    return {
                        "method": method,
                        "operator": op,
                        "status": "quarantined_unsupported_family",
                        "reason": "; ".join(compilation.reasons),
                        "plan_id": plan_id,
                        **_artifacts(),
                    }

        if compiled_source is None:
            code_task = build_planned_mutation_task(
                op,
                parent,
                plan,
                temperature=args.code_temperature,
                top_p=args.code_top_p,
            )
            code_task.plan_status = (
                "reasoning_planned"
                if reasoning_informed
                else "plain_planned"
            )
            code_task.generation_path = generation_path
            code_task.generator_family = generator_family
            code_task.compiler_version = (
                str(MUTATION_FAMILY_REGISTRY_VERSION)
                if compiler_status is not None
                else None
            )
            code_task.compiler_diagnostics = {
                "status": compiler_status,
                "reasons": list(compiler_reasons),
                "source_hash": compiler_source_hash,
                "family_config": family_config or {},
            }
            code_task.quarantined = quarantined

    prompt_index = "04" if uses_plan else "01"
    raw_index = "05" if uses_plan else "02"
    validation_index = "06" if uses_plan else "03"
    if compiled_source is not None:
        raw_code_for_validation = f"```python\n{compiled_source}\n```"
    else:
        assert code_task is not None
        code_input_token_count = _save_task(
            method_dir / f"{prompt_index}_code_prompt.json",
            code_task,
            tokenizer=getattr(inference, "tokenizer", None),
            chat_template_kwargs=chat_template_kwargs,
        )
        raw_code = inference.generate(
            _messages_for_task(code_task),
            sampling_params={
                "temperature": args.code_temperature,
                "top_p": args.code_top_p,
                "max_tokens": args.mutation_max_tokens,
                "seed": code_sampling_seed,
            },
            chat_template_kwargs=chat_template_kwargs,
        )
        assert isinstance(raw_code, str)
        code_calls_completed += 1
        _write_text(method_dir / f"{raw_index}_code_raw.txt", raw_code)
        raw_code_for_validation = raw_code

    child, validation = _validate_generated_program(
        raw_code_for_validation,
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
        family_contract=family_contract,
        max_tokens=args.evaluator_max_tokens,
        temperature=args.evaluator_temperature,
        top_p=args.evaluator_top_p,
        chat_template_kwargs=chat_template_kwargs,
        operator=op,
        llm_seed=base_llm_seed,
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
            operator=op,
            llm_seed=base_llm_seed,
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
        "status": (
            "quarantined_freeform"
            if quarantined
            else "ok"
            if evaluator["valid"]
            else "evaluator_rejected"
        ),
        "reason": (
            "unsupported free-form mutation retained for diagnostics only; "
            "it is not eligible for archive insertion or training"
            if quarantined
            else evaluator_reason
        ),
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
        "family_semantics_valid": (
            family_semantics.valid if family_semantics is not None else None
        ),
        "family_target_reasoning_move": (
            family_contract.get("target_reasoning_move")
            if family_contract
            else None
        ),
        "family_variant": family_variant,
        "variant_selected_by_plan": variant_selected_by_plan,
        "variant_targets_failure_mode": variant_targets_failure_mode,
        "compiler_source_hash": compiler_source_hash,
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
    *,
    planning_contract: str = "registered_v5",
    candidates_per_condition: int = 1,
    plan_only: bool = False,
) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    if planning_contract == "direct_v7":
        design_lines = [
            (
                "- Contract: `direct_v7`; no plan, no schema, no hypothesis "
                "catalog -- the model reads the traces and writes the generator"
            ),
            (
                "- Both conditions make one generation call against the same "
                "template; only the reasoning prompt contains the trace pair"
            ),
            (
                "- Verification stays on the produced problem: lint, "
                "multi-seed execution, seed variation, then R_Q ranks the "
                "valid candidates"
            ),
            (
                f"- Candidates per operator/condition: "
                f"`{candidates_per_condition}`"
            ),
        ]
    elif planning_contract == "belief_v6":
        design_lines = [
            (
                "- Contract: schema `v6 belief/desire attribution`; the model "
                "chooses one closed-set hypothesis and grounds it with a quote"
            ),
            (
                "- Python derives the probe, verifies family semantics and "
                "diagnosticity, then R_Q alone ranks eligible candidates"
            ),
            (
                "- The LLM evaluator is not an eligibility gate in v6; "
                "attribution hit rate is reported separately from R_Q"
            ),
            (
                f"- Candidates per operator/condition: "
                f"`{candidates_per_condition}`; plan-only: `{plan_only}`"
            ),
        ]
    else:
        design_lines = [
            (
                "- Contract: plan schema `v5 registered_family`; "
                "registered-family compiler plus canonical validation"
            ),
            (
                "- Both conditions use one plan call; registered families are "
                "compiled without a code-model call"
            ),
            (
                "- Unsupported free-form ideas may be evaluated diagnostically "
                "in hybrid mode, but remain quarantined"
            ),
        ]

    lines = [
        "# Plain vs. reasoning-informed mutation",
        "",
        *design_lines,
        (
            "- Plain sees the parent only; reasoning additionally sees one "
            "correct and one wrong solver trace on the same parent instance"
        ),
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
        "| operator | condition | selected | hypothesis | variant | source hash | status | eligible | correct/rollouts | p_hat | rq proxy | hit rate |",
        "|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        p_hat = row.get("p_hat")
        rq_proxy = row.get("rq_proxy")
        evaluator = (
            f"{row.get('evaluator_passed', 0)}/{row.get('evaluator_total', 0)}"
            if isinstance(row.get("evaluator_valid"), bool)
            else "—"
        )
        correct_rollouts = (
            f"{row['num_correct']}/{row['num_rollouts']}"
            if isinstance(row.get("num_correct"), int)
            and isinstance(row.get("num_rollouts"), int)
            else "—"
        )
        variant = row.get("family_variant") or "—"
        if (
            planning_contract != "belief_v6"
            and row.get("variant_selected_by_plan") is False
        ):
            variant = f"{variant} (defaulted)"
        eligibility = row.get("eligibility")
        eligible = (
            eligibility.get("eligible")
            if isinstance(eligibility, Mapping)
            else "—"
        )
        attribution = row.get("attribution_score")
        hit_rate = (
            attribution.get("attribution_hit_rate")
            if isinstance(attribution, Mapping)
            else None
        )
        lines.append(
            "| {operator} | {condition} | {selected} | {hypothesis} | "
            "{variant} | {source_hash} | {status} | {eligible} | "
            "{correct_rollouts} | {p_hat} | {rq_proxy} | {hit_rate} |".format(
                operator=cell(row.get("operator", "")),
                condition=cell(row.get("condition", "")),
                selected=cell(
                    row.get("selected_candidate_index", "—")
                ),
                hypothesis=cell(
                    row.get("attributed_hypothesis") or "—"
                ),
                variant=cell(variant),
                source_hash=cell(row.get("compiler_source_hash") or "—"),
                status=cell(row.get("status", "")),
                eligible=cell(eligible),
                correct_rollouts=correct_rollouts,
                p_hat=f"{p_hat:.3f}" if isinstance(p_hat, (int, float)) else "—",
                rq_proxy=(
                    f"{rq_proxy:.6f}"
                    if isinstance(rq_proxy, (int, float))
                    else "—"
                ),
                hit_rate=(
                    f"{hit_rate:.3f}"
                    if isinstance(hit_rate, (int, float))
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
        candidate = f"{row.get('operator', '')}/{row.get('condition', row.get('method', ''))}"
        for seed_score in row.get("seed_scores", []):
            seed = int(seed_score["seed"])
            evaluator_item = evaluator_by_seed.get(seed, {})
            evaluator = (
                "VALID"
                if evaluator_item.get("valid") is True
                else "INVALID"
                if evaluator_item
                else "—"
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
    if args.candidates_per_condition < 1:
        raise ValueError("--candidates-per-condition must be >= 1")
    if args.plan_only and args.planning_contract != "belief_v6":
        raise ValueError("--plan-only is supported only with --planning-contract belief_v6")
    evaluation_seeds = _resolve_evaluation_seeds(args)
    if args.instance_seed not in evaluation_seeds:
        raise ValueError(
            "--instance-seed must be included in --evaluation-seeds"
        )
    # The monitoring budget covers the correct+wrong trace pair together, while
    # trace-storage-max-tokens caps each trace on its own. If the combined
    # budget cannot hold two maximal traces, a single truncated failure trace
    # silently consumes it and the whole pair is dropped -- which left the
    # reasoning condition with zero evidence and no fallback to another seed.
    if args.monitoring_total_trace_tokens < 2 * args.trace_storage_max_tokens:
        raise ValueError(
            "--monitoring-total-trace-tokens must be at least twice "
            "--trace-storage-max-tokens because it budgets a pair of traces: "
            f"got {args.monitoring_total_trace_tokens} for two traces of up to "
            f"{args.trace_storage_max_tokens} each"
        )

    prompt_dir = args.prompt_dir.expanduser().resolve()
    shot_dir = args.shot_dir.expanduser().resolve()
    os.environ["RQ_EVOLVE_PROMPT_DIR"] = str(prompt_dir)
    os.environ["RQ_EVOLVE_SHOT_DIR"] = str(shot_dir)
    # Import after the environment is set: prompts.py resolves these directories
    # at module import time, exactly like a production process startup.
    import rq_evolve.prompts as prompt_module
    from rq_evolve.metacognition import (
        EVIDENCE_QUALITY_VERSION,
        collect_planning_evidence,
        select_reasoning_evidence,
        validate_reasoning_contrast,
    )
    from rq_evolve.mutation_compiler import (
        MUTATION_FAMILY_REGISTRY_VERSION,
        registered_family_catalog,
        registered_family_descriptor,
    )
    from rq_evolve.belief_probe import (
        BELIEF_SCHEMA_VERSION,
        catalog_payload as belief_catalog_payload,
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
        sampler_backend=args.vllm_sampler_backend,
        seed=args.llm_seed,
    )

    parent_evidence_seed_schedule = [
        {
            "instance_seed": int(instance.seed),
            "rollout_idx": int(rollout_idx),
            "sampling_seed": _parent_evidence_request_seed(
                args.llm_seed,
                int(instance.seed),
                int(rollout_idx),
            ),
        }
        for instance in parent_instances
        for rollout_idx in range(args.num_rollouts)
    ]
    planning_parent_score = {
        "p_hat": float(parent.p_hat),
        "uncertainty_proxy": float(parent.h_score),
        "rq_proxy": float(parent.rq_score),
    }
    if args.parent_rollouts_json is not None:
        parent_groups: list[dict[str, Any]] = []
        parent_scores = _load_cached_parent_scores(
            args.parent_rollouts_json.expanduser().resolve(),
            instances=parent_instances,
            seeds=evaluation_seeds,
            rollouts_per_seed=args.num_rollouts,
        )
        parent_rollout_source = "provided_json"
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
            request_seed_factory=lambda instance, rollout_idx: (
                _parent_evidence_request_seed(
                    args.llm_seed,
                    int(instance.seed),
                    rollout_idx,
                )
            ),
        )
        parent_scores = _summarize_rollout_groups(
            parent_groups,
            seeds=evaluation_seeds,
        )
        parent_rollout_source = "standalone_vllm_rollouts"
    if provided_evidence is None:
        if not parent_groups:
            raise ValueError(
                "--parent-rollouts-json requires --evidence-json because "
                "cached score rows do not reconstruct planning records"
            )
        parent.p_hat = float(parent_scores["p_hat"])
        parent.h_score = float(parent_scores["uncertainty_proxy"])
        parent.rq_score = float(parent_scores["rq_proxy"])
        parent.fitness = parent.rq_score
        planning_parent_score = {
            "p_hat": float(parent.p_hat),
            "uncertainty_proxy": float(parent.h_score),
            "rq_proxy": float(parent.rq_score),
        }

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
    else:
        evidence = _select_same_instance_contrast(
            provided_evidence,
            preferred_seed=args.instance_seed,
            expected_problems={
                int(instance.seed): instance.problem
                for instance in parent_instances
            },
        )
        if evidence:
            parent.metadata["reasoning_evidence"] = evidence
        evidence_source = "provided_json"

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

    # Gate and persist the exact live-parent-regraded, budget-fitted object that
    # the planners will receive, rather than the raw supplied labels.
    evidence = collect_planning_evidence(
        parent,
        "in_depth",
        [parent],
        total_tokens=args.monitoring_total_trace_tokens,
        tokenizer=inference.tokenizer,
    )
    evidence_gate_version = (
        f"clean_same_instance_v2:{EVIDENCE_QUALITY_VERSION}"
    )
    evidence_gate_issues = validate_reasoning_contrast(evidence)
    if not evidence and provided_evidence is not None:
        evidence_gate_issues = [
            "no clean same-seed, same-problem success/failure pair remained "
            "after evidence quality filtering"
        ]
    _write_json(
        output_dir / "evidence_gate.json",
        {
            "gate_version": evidence_gate_version,
            "valid": not evidence_gate_issues,
            "issues": evidence_gate_issues,
            "source_count": (
                len(provided_evidence)
                if provided_evidence is not None
                else sum(len(group["records"]) for group in parent_groups)
            ),
            "selected_count": len(evidence),
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
            try:
                result = (
                    _run_direct_condition(
                        inference,
                        method=method,
                        op=op,
                        parent=parent,
                        output_dir=output_dir,
                        args=args,
                        chat_template_kwargs=chat_template_kwargs,
                        planning_evidence=planning_evidence,
                    )
                    if args.planning_contract == "direct_v7"
                    else _run_belief_condition(
                        inference,
                        method=method,
                        op=op,
                        parent=parent,
                        output_dir=output_dir,
                        args=args,
                        chat_template_kwargs=chat_template_kwargs,
                        planning_evidence=planning_evidence,
                    )
                    if args.planning_contract == "belief_v6"
                    else _run_method(
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
            except Exception as exc:
                import traceback

                result = {
                    "method": method,
                    "operator": op,
                    "condition": (
                        "plain" if method == "legacy" else "reasoning"
                    ),
                    "status": "exception",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "method_dir": str(
                        (output_dir / op / method).resolve()
                    ),
                }
            summaries.append(result)
            print(
                f"  -> status={result['status']} "
                f"reason={str(result.get('reason'))[:200]}"
            )

    paired_route_checks: list[dict[str, Any]] = []
    for operator in operators:
        rows = [row for row in summaries if row.get("operator") == operator]
        paths = {row.get("generation_path") for row in rows}
        families = {row.get("generator_family") for row in rows}
        call_counts = {
            row.get("configured_llm_generation_call_count") for row in rows
        }
        valid = (
            len(rows) == 2
            and all("generation_path" in row for row in rows)
            and all(
                "configured_llm_generation_call_count" in row
                for row in rows
            )
            and len(paths) == 1
            and len(families) == 1
            and len(call_counts) == 1
        )
        paired_route_checks.append(
            {
                "operator": operator,
                "valid": valid,
                "generation_paths": sorted(str(value) for value in paths),
                "generator_families": sorted(
                    str(value) for value in families
                ),
                "configured_llm_generation_call_counts": sorted(
                    int(value)
                    for value in call_counts
                    if isinstance(value, int)
                ),
            }
        )

    manifest = {
        "model": args.model,
        "vllm_sampler_backend": args.vllm_sampler_backend,
        "vllm_use_flashinfer_sampler": os.environ.get(
            "VLLM_USE_FLASHINFER_SAMPLER"
        ),
        "seed_program": str(seed_path),
        "seed_program_sha256": _file_digest(seed_path),
        "parent_program_id": parent.program_id,
        "requested_planning_seed": args.instance_seed,
        "llm_seed": args.llm_seed,
        "llm_seed_range": [args.llm_seed, args.llm_seed],
        "sampling_seed_policy": "recorded_vllm_engine_seed_per_run",
        "paired_request_seed_policy": PAIRED_REQUEST_SEED_POLICY,
        "paired_evaluator_sampling_seed_policy": (
            PAIRED_EVALUATOR_SEED_POLICY
        ),
        "paired_child_solver_sampling_seed_policy": (
            PAIRED_CHILD_SOLVER_SEED_POLICY
        ),
        "paired_request_seeds": {
            operator: (
                {
                    f"belief_plan:{index}": _paired_request_seed(
                        args.llm_seed,
                        operator,
                        f"belief_plan:{index}",
                    )
                    for index in range(args.candidates_per_condition)
                }
                if args.planning_contract == "belief_v6"
                else {
                    stage: _paired_request_seed(
                        args.llm_seed,
                        operator,
                        stage,
                    )
                    for stage in ("plan", "code")
                }
            )
            for operator in operators
        },
        "parent_evidence_sampling_seed_policy": (
            PARENT_EVIDENCE_SEED_POLICY
        ),
        "parent_evidence_sampling_seed_sha256": _seed_schedule_digest(
            parent_evidence_seed_schedule
        ),
        "parent_evidence_sampling_seed_range": _seed_schedule_range(
            parent_evidence_seed_schedule
        ),
        "instance_seed_policy": "explicit_evaluation_seed_list",
        "acceptance_budget": (
            "one R_Q champion per attribution niche"
            if args.planning_contract == "belief_v6"
            else 1
        ),
        "generator_candidate_draw_count": args.candidates_per_condition,
        "candidates_per_condition": args.candidates_per_condition,
        "plan_only": args.plan_only,
        "planning_evidence_seed": selected_evidence_seed,
        "evaluation_seeds": evaluation_seeds,
        "parent_problem": parent_instance.problem,
        "parent_answer": parent_instance.answer,
        "comparison_design": (
            "belief_v6_gate_then_rq_selection_v1"
            if args.planning_contract == "belief_v6"
            else "hybrid_compiler_plain_vs_reasoning_v2"
            if args.plain_baseline == "two_stage"
            and args.mutation_code_backend != "freeform"
            else "two_stage_plain_vs_reasoning_v1"
            if args.plain_baseline == "two_stage"
            else "legacy_one_stage_vs_reasoning"
        ),
        "planning_contract": args.planning_contract,
        "plain_baseline": args.plain_baseline,
        "mutation_code_backend": args.mutation_code_backend,
        "plan_schema_version": (
            BELIEF_SCHEMA_VERSION
            if args.planning_contract == "belief_v6"
            else 5
        ),
        "compiler_registry_version": MUTATION_FAMILY_REGISTRY_VERSION,
        "registered_family_catalog_by_operator": {
            operator: json.loads(
                registered_family_catalog(parent, operator)
            )
            for operator in operators
        },
        "belief_hypothesis_catalog_by_operator": {
            operator: belief_catalog_payload(
                registered_family_descriptor(
                    parent,
                    operator,
                ).generator_family
            )
            for operator in operators
            if registered_family_descriptor(parent, operator) is not None
        },
        "paired_route_checks": paired_route_checks,
        "paired_route_valid": all(
            check["valid"] for check in paired_route_checks
        ),
        "code_contract_version": 6,
        "code_contract": (
            "belief_v6_registry_probe+python_semantics+diagnosticity"
            if args.planning_contract == "belief_v6"
            else "registered_family_compiler_v1+canonical_freeform_v5"
        ),
        "planning_parent_score": planning_parent_score,
        "parent_evaluation_score": {
            "p_hat": parent_scores["p_hat"],
            "uncertainty_proxy": parent_scores["uncertainty_proxy"],
            "rq_proxy": parent_scores["rq_proxy"],
            "num_correct": parent_scores["num_correct"],
            "num_rollouts": parent_scores["num_rollouts"],
        },
        "evidence_source": evidence_source,
        "evidence_json_sha256": (
            _file_digest(args.evidence_json.expanduser().resolve())
            if args.evidence_json is not None
            else None
        ),
        "parent_rollout_source": parent_rollout_source,
        "parent_rollouts_json": (
            str(args.parent_rollouts_json.expanduser().resolve())
            if args.parent_rollouts_json is not None
            else None
        ),
        "parent_rollouts_json_sha256": (
            _file_digest(args.parent_rollouts_json.expanduser().resolve())
            if args.parent_rollouts_json is not None
            else None
        ),
        "evidence_count": len(evidence),
        "evidence_gate_valid": not evidence_gate_issues,
        "evidence_gate_issues": evidence_gate_issues,
        "evidence_gate_version": evidence_gate_version,
        "confidence_measure": (
            "mean_token_negative_logprob (standalone proxy, not actor entropy)"
        ),
        "prompt_files": _prompt_manifest(prompt_dir, shot_dir),
        "sampling": {
            "vllm_sampler_backend": args.vllm_sampler_backend,
            "vllm_use_flashinfer_sampler": os.environ.get(
                "VLLM_USE_FLASHINFER_SAMPLER"
            ),
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
    _write_json(output_dir / "summaries.json", summaries)
    _write_json(output_dir / "manifest.json", manifest)
    _write_text(
        output_dir / "REPORT.md",
        _report_markdown(
            parent_instance,
            summaries,
            evidence,
            parent_scores=parent_scores,
            planning_contract=args.planning_contract,
            candidates_per_condition=args.candidates_per_condition,
            plan_only=args.plan_only,
        ),
    )
    print(f"[compare] saved results to {output_dir}")


if __name__ == "__main__":
    main()
