"""Greedy-decode many instances per family variant to test for error structure.

Why
---
Every error analysis so far ran at ``CHILD_TEMPERATURE = 1.0``. Under that
sampling the wrong answers within one instance were indistinguishable from
uniform guessing (observed mode share 0.50 vs a chance baseline of 0.49), but
that result cannot separate "the solver has no stable disposition" from
"temperature washed the disposition out". Greedy decoding removes the second
explanation: one instance yields exactly one answer, so any structure has to
show up *across* instances instead of within them.

Design
------
The routes under test were specified from the temperature-1.0 data, so scoring
them here is a **confirmatory** test on fresh instances, not another search. The
analysis script keeps that arm separate from any exploratory mining.

The solver prompt, answer extraction and grading are imported from the same
modules the comparison pipeline uses, so a greedy record is directly comparable
to a sampled one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.vllm_runtime import (  # noqa: E402
    VLLM_SAMPLER_BACKENDS,
    configure_vllm_sampler_backend,
)

# (operator, generator_family, variant) triples worth probing. These are the
# constructions the recorded rollouts actually came from.
DEFAULT_TARGETS = (
    "in_depth:linear_system_aggregate:asymmetric_combination",
    "in_depth:linear_system_aggregate:balanced",
    "in_depth:linear_system_aggregate:heavy_division",
    "in_breadth:modular_linear_system_aggregate:hard_inverse",
    "in_breadth:modular_linear_system_aggregate:balanced",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Greedy-decode N instances per registered family variant and record "
            "the answers, so error routes can be tested without temperature "
            "scatter."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help="operator:generator_family:variant triples",
    )
    parser.add_argument("--num-instances", type=int, default=200)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--vllm-sampler-backend",
        choices=sorted(VLLM_SAMPLER_BACKENDS),
        default="pytorch",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _compile_variant(operator: str, family: str, variant: str):
    from rq_evolve.mutation_compiler import (
        CompilationStatus,
        MutationSpec,
        compile_mutation_spec,
    )

    result = compile_mutation_spec(
        MutationSpec(
            generator_family=family,
            operator=operator,
            family_variant=variant,
        )
    )
    if result.status is not CompilationStatus.COMPILED:
        raise RuntimeError(
            f"{family}/{variant} did not compile: {result.reasons}"
        )
    return result


def main() -> None:
    args = parse_args()
    configure_vllm_sampler_backend(args.vllm_sampler_backend)

    from vllm import LLM, SamplingParams

    from rq_evolve.prompts import SOLVER_SYSTEM_PROMPT
    from rq_evolve.reward import answers_match, extract_boxed

    seeds = list(range(args.start_seed, args.start_seed + args.num_instances))

    # Build every prompt first so one vLLM batch covers all variants.
    pending: list[dict[str, Any]] = []
    for target in args.targets:
        operator, family, variant = target.split(":")
        compiled = _compile_variant(operator, family, variant)
        namespace: dict[str, Any] = {}
        exec(compile(compiled.source_code, "<compiled>", "exec"), namespace)
        for seed in seeds:
            try:
                data = namespace["build_instance_data"](seed)
                problem, answer = namespace["generate"](seed)
            except Exception as exc:  # a variant may fail to sample some seeds
                print(f"[greedy-probe] {target} seed={seed} skipped: {exc}")
                continue
            pending.append(
                {
                    "operator": operator,
                    "generator_family": family,
                    "family_variant": variant,
                    "source_hash": compiled.source_hash,
                    "seed": seed,
                    "problem": problem,
                    "answer": answer,
                    "instance_data": {
                        key: value
                        for key, value in data.items()
                        if key != "witnesses"
                    },
                }
            )
    print(f"[greedy-probe] {len(pending)} prompts across {len(args.targets)} variants")

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.tokenizer:
        llm_kwargs["tokenizer"] = args.tokenizer
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)

    conversations = [
        [
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": row["problem"]},
        ]
        for row in pending
    ]
    # Greedy: one deterministic answer per instance, so any structure must be
    # visible across instances rather than within them.
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, n=1)
    outputs = llm.chat(conversations, sampling_params=sampling)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    correct_by_variant: dict[str, list[int]] = {}
    with records_path.open("w", encoding="utf-8") as handle:
        for row, output in zip(pending, outputs):
            text = output.outputs[0].text
            predicted = extract_boxed(text)
            correct = predicted is not None and answers_match(
                predicted, row["answer"]
            )
            key = f"{row['operator']}/{row['family_variant']}"
            correct_by_variant.setdefault(key, []).append(int(bool(correct)))
            handle.write(
                json.dumps(
                    {
                        **row,
                        "response": text,
                        "predicted_answer": predicted,
                        "correct": bool(correct),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "schema_version": 1,
        "diagnostic": "greedy_route_probe",
        "purpose": (
            "test error-route structure without temperature-1.0 scatter; the "
            "routes were pre-specified from the sampled data, so scoring them "
            "here is confirmatory rather than exploratory"
        ),
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "n": 1, "greedy": True},
        "targets": list(args.targets),
        "seeds": [seeds[0], seeds[-1]],
        "num_instances_requested": args.num_instances,
        "num_records": len(pending),
        "accuracy_by_variant": {
            key: sum(values) / len(values)
            for key, values in correct_by_variant.items()
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[greedy-probe] wrote {records_path}")
    for key, values in sorted(correct_by_variant.items()):
        print(
            f"  {key:<44} accuracy={sum(values)/len(values):.3f} "
            f"({sum(values)}/{len(values)})"
        )


if __name__ == "__main__":
    main()
