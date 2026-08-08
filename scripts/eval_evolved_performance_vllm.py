#!/usr/bin/env python3
"""Evaluate one model on a fixed Evolved Performance benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.evolved_performance import (  # noqa: E402
    load_benchmark,
    summarize_scored_rows,
)
from rq_evolve.prompts import SOLVER_SYSTEM_PROMPT  # noqa: E402
from rq_evolve.reward import answers_match, extract_boxed  # noqa: E402
from rq_evolve.vllm_runtime import (  # noqa: E402
    VLLM_SAMPLER_BACKENDS,
    configure_vllm_sampler_backend,
)

logger = logging.getLogger("eval_evolved_performance")


def _build_prompt(tokenizer, problem: str) -> str:
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            add_special_tokens=True,
        )
    return (
        f"System: {SOLVER_SYSTEM_PROMPT}\n"
        f"User: {problem}\nAssistant:"
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    effective_sampler = configure_vllm_sampler_backend(args.vllm_sampler_backend)
    from vllm import LLM, SamplingParams

    rows, manifest = load_benchmark(args.benchmark, args.manifest)
    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": tokenizer_name,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    sampling_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "n": 1,
        "seed": 0,
    }
    if tokenizer.eos_token_id is not None:
        sampling_kwargs["stop_token_ids"] = [int(tokenizer.eos_token_id)]
    sampling_params = SamplingParams(**sampling_kwargs)
    prompts = [_build_prompt(tokenizer, row["problem"]) for row in rows]

    started = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=not args.no_tqdm)
    scored_rows: list[dict[str, Any]] = []
    for row, output in zip(rows, outputs):
        response = output.outputs[0].text
        predicted = extract_boxed(response)
        correct = predicted is not None and answers_match(predicted, row["answer"])
        scored_rows.append(
            {
                **row,
                "response": response,
                "predicted_answer": predicted,
                "correct": bool(correct),
                "score": 1 if correct else 0,
            }
        )

    aggregate = summarize_scored_rows(scored_rows)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "details.jsonl"
    details_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored_rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark": manifest["benchmark"],
        "benchmark_sha256": manifest["benchmark_sha256"],
        "model": str(args.model),
        "tokenizer": str(tokenizer_name),
        "global_step": args.global_step,
        "grader": "last-boxed + rq_evolve.reward.answers_match",
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "seed": 0,
            "max_tokens": args.max_tokens,
            "vllm_sampler_backend": args.vllm_sampler_backend,
            "vllm_use_flashinfer_sampler": effective_sampler,
        },
        "elapsed_sec": round(time.time() - started, 2),
        **aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "EPS step=%s: %.2f%% (%d/%d), macro over %d seed programs",
        args.global_step,
        summary["score_percent"],
        summary["correct"],
        summary["num_examples"],
        len(summary["per_program"]),
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-step", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--vllm-sampler-backend",
        choices=VLLM_SAMPLER_BACKENDS,
        default="pytorch",
    )
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    evaluate(args)


if __name__ == "__main__":
    main()
