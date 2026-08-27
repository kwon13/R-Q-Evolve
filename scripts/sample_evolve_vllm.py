#!/usr/bin/env python
"""One evolve phase against a real policy, without the verl/Ray/FSDP stack.

Boots vLLM directly, runs the pipeline that actually changed -- parent
selection, the single free-form mutation operator, verification, the taxonomy
judge, solver rollouts, R_Q, archive placement -- and reports where the children
landed. Nothing is trained; weights are frozen, so this measures the prompts and
the grid, not the RL loop.

    python scripts/sample_evolve_vllm.py --steps 32 --batch 8 --rollouts 4

Entropy for R_Q comes from vLLM's own sampled-token logprobs, not an actor
forward pass, which is the one place this diverges from the training backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.archive import MAPElitesArchive  # noqa: E402
from rq_evolve.backends import PendingRollouts, RolloutRecord  # noqa: E402
from rq_evolve.concepts import DOMAINS, PROBLEM_TYPES  # noqa: E402
from rq_evolve.config import load_config  # noqa: E402
from rq_evolve.evolution import RQEvolver  # noqa: E402
from rq_evolve.openai_evaluator import load_project_dotenv  # noqa: E402
from rq_evolve.prompts import build_solver_messages  # noqa: E402
from rq_evolve.reward import answers_match, extract_boxed  # noqa: E402
from rq_evolve.solver_trace import SOLVER_CHAT_BOUNDARY_STOPS  # noqa: E402
from rq_evolve.vllm_runtime import (  # noqa: E402
    VLLM_SAMPLER_BACKENDS,
    configure_vllm_sampler_backend,
)


class VLLMBackend:
    """EvolutionBackend over a resident vLLM engine with frozen weights."""

    def __init__(self, model: str, args) -> None:
        # Must happen before vLLM builds its engine workers. FlashInfer 0.6.x
        # JIT-compiles CUDA 12-only sampling kernels, and this host pairs a
        # CUDA 12 torch wheel with a CUDA 11.8 system nvcc, so the default
        # backend fails engine startup outright. See docs/PIPELINE.md.
        effective = configure_vllm_sampler_backend(args.vllm_sampler_backend)
        print(f"[sample] vLLM sampler backend: {args.vllm_sampler_backend} "
              f"(VLLM_USE_FLASHINFER_SAMPLER={effective})", flush=True)

        from vllm import LLM

        self.llm = LLM(
            model=model,
            tokenizer=model,
            dtype="bfloat16",
            trust_remote_code=True,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_util,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.args = args
        # Read by the evaluator gate to drop over-budget candidates.
        self.max_model_len = args.max_model_len
        self.current_iteration = 0
        self.n_mutate_calls = 0
        self.n_generated = 0
        self.transcript: list[dict] = []
        self.last_mutation_logprobs: list[tuple[int, float] | None] = []

    # --- session lifecycle: a resident engine has nothing to wake or push ---
    def sync_weights(self) -> None:
        pass

    def begin_session(self) -> None:
        pass

    def end_session(self) -> None:
        pass

    def _chat(
        self,
        messages,
        *,
        temperature,
        top_p,
        max_tokens,
        stop=None,
        logprobs=0,
        allowed_token_ids=None,
    ):
        from vllm import SamplingParams

        prompts = [
            self.tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages
        ]
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=list(stop) if stop else None,
            logprobs=logprobs,
            allowed_token_ids=allowed_token_ids,
        )
        return self.llm.generate(prompts, params)

    def mutate(self, tasks) -> list[str | None]:
        if not tasks:
            return []
        self.n_mutate_calls += 1
        stage = tasks[0].stage
        conversations = [
            task.messages
            or [{"role": "user", "content": task.prompt}]
            for task in tasks
        ]
        logprob_values = {task.logprobs for task in tasks}
        allowed_values = {
            tuple(task.allowed_token_ids) if task.allowed_token_ids is not None else None
            for task in tasks
        }
        if len(logprob_values) != 1 or len(allowed_values) != 1:
            raise ValueError(
                "one standalone vLLM mutation batch must share logprobs and "
                "allowed_token_ids"
            )
        # Stage-2 writes a core, while a DOMAIN label arm writes one token.
        max_tokens = max(
            (
                int(task.max_output_tokens)
                if task.max_output_tokens is not None
                else (
                    self.args.judge_tokens
                    if task.stage == "judge"
                    else self.args.mutate_tokens
                )
            )
            for task in tasks
        )
        requested_logprobs = next(iter(logprob_values))
        allowed_token_ids = next(iter(allowed_values))
        outputs = self._chat(
            conversations,
            temperature=tasks[0].temperature if tasks[0].temperature is not None else 0.0,
            top_p=tasks[0].top_p if tasks[0].top_p is not None else 1.0,
            max_tokens=max_tokens,
            logprobs=requested_logprobs or 0,
            allowed_token_ids=(
                list(allowed_token_ids) if allowed_token_ids is not None else None
            ),
        )
        self.n_generated += len(outputs)
        texts = [o.outputs[0].text for o in outputs]
        self.last_mutation_logprobs = []
        for output in outputs:
            sample = output.outputs[0]
            pair = None
            try:
                token_id = int(sample.token_ids[0])
                token_logprob = sample.logprobs[0][token_id]
                pair = (token_id, float(token_logprob.logprob))
            except (IndexError, KeyError, TypeError, ValueError, AttributeError):
                pair = None
            self.last_mutation_logprobs.append(pair)
        # Keep every raw generation. A CandidateReport carries only a reason
        # string, so without this the source behind "execute failed at seed=0"
        # is unrecoverable and the failure cannot be diagnosed after the fact.
        for task, out, text in zip(tasks, outputs, texts):
            self.transcript.append(
                {
                    "stage": stage,
                    "op": task.op,
                    "parent_id": getattr(task.parent, "program_id", ""),
                    "parent_domain": task.parent.get_domain(),
                    "parent_problem_type": task.parent.get_problem_type(),
                    "finish_reason": out.outputs[0].finish_reason,
                    "output_tokens": len(out.outputs[0].token_ids),
                    "text": text,
                }
            )
        return texts

    def generate_rollouts(self, instances, n_rollouts) -> PendingRollouts:
        pending = PendingRollouts(instances=list(instances), n_rollouts=n_rollouts)
        if not instances:
            pending.grouped = []
            return pending
        messages = [
            build_solver_messages(inst.problem)
            for inst in instances
            for _ in range(n_rollouts)
        ]
        outputs = self._chat(
            messages,
            temperature=self.args.solver_temperature,
            top_p=self.args.solver_top_p,
            max_tokens=self.args.solver_tokens,
            stop=SOLVER_CHAT_BOUNDARY_STOPS,
        )
        grouped: list[list[RolloutRecord]] = []
        cursor = 0
        for inst in instances:
            records = []
            for _ in range(n_rollouts):
                out = outputs[cursor].outputs[0]
                cursor += 1
                text = out.text
                predicted = extract_boxed(text)
                # TOTAL entropy over the trajectory, matching the training
                # backend: H is the exploration cost paid, not a per-token
                # rate. vLLM's sampled logprobs stand in for the actor forward
                # the trainer uses.
                entropies = []
                for step in out.logprobs or []:
                    for lp in step.values():
                        entropies.append(-lp.logprob)
                        break
                records.append(
                    RolloutRecord(
                        response=text,
                        predicted_answer=predicted,
                        correct=bool(
                            predicted is not None
                            and answers_match(predicted, inst.answer)
                        ),
                        entropy=float(sum(entropies)),
                        response_tokens=len(out.token_ids),
                    )
                )
            grouped.append(records)
        pending.grouped = grouped
        return pending

    def finalize_rollouts(self, pending) -> list[list[RolloutRecord]]:
        return pending.grouped or []

    def rollout(self, instances, n_rollouts):
        return self.finalize_rollouts(self.generate_rollouts(instances, n_rollouts))


def render_grid(archive: MAPElitesArchive, title: str) -> str:
    occupied = {
        archive.program_to_cell(c): c for c in archive.champions()
    }
    lines = [
        title,
        " " * 16 + "".join(f"{kind[:7]:>9}" for kind in PROBLEM_TYPES),
    ]
    for domain_index, domain in enumerate(DOMAINS):
        row = "".join(
            "        O" if (domain_index, type_index) in occupied else "        ."
            for type_index in range(len(PROBLEM_TYPES))
        )
        lines.append(f"{domain[:15]:<16}{row}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq_evolve_base.yaml")
    ap.add_argument("--model", default="/data1/yhoon113/qwen3-8b-base")
    ap.add_argument("--steps", type=int, default=32,
                    help="total mutation candidates this run")
    ap.add_argument("--batch", type=int, default=8,
                    help="mutations per batch; parents are resampled each batch")
    ap.add_argument("--rollouts", type=int, default=4, help="solver rollouts per child")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=12000)
    ap.add_argument("--mutate-tokens", type=int, default=3000)
    ap.add_argument("--judge-tokens", type=int, default=900)
    ap.add_argument("--code-temperature", type=float, default=None,
                    help="override evolution.code_temperature (the swept knob)")
    ap.add_argument("--solver-tokens", type=int, default=2000)
    ap.add_argument("--solver-temperature", type=float, default=1.0)
    ap.add_argument("--solver-top-p", type=float, default=0.95)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument(
        "--vllm-sampler-backend",
        default="pytorch",
        choices=VLLM_SAMPLER_BACKENDS,
        help="pytorch avoids FlashInfer's CUDA-12-only JIT (repo default)",
    )
    ap.add_argument("--out", default="rq_output/sample_evolve")
    args = ap.parse_args()

    # The judge may be the OpenAI provider, whose key lives in R-Q-Evolve/.env.
    load_project_dotenv(ROOT)
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[sample] booting vLLM on {args.model} (tp={args.tp})", flush=True)
    t0 = time.monotonic()
    backend = VLLMBackend(args.model, args)
    print(f"[sample] engine ready in {time.monotonic() - t0:.0f}s", flush=True)

    archive = MAPElitesArchive(**asdict(cfg.archive))
    evolution = cfg.evolution
    evolution.inner_iterations = args.steps
    evolution.inner_iteration_batch_size = min(args.batch, args.steps)
    evolution.num_rollouts = args.rollouts
    if args.code_temperature is not None:
        evolution.code_temperature = args.code_temperature
    print(
        f"[sample] mutation sampling: temperature={evolution.code_temperature} "
        f"top_p={evolution.code_top_p} | judge: "
        f"temperature={evolution.judge_temperature} top_p={evolution.judge_top_p}",
        flush=True,
    )
    evolver = RQEvolver(
        archive=archive,
        backend=backend,
        evolution_config=evolution,
        training_config=cfg.training_data,
    )

    print("[sample] bootstrapping seeds (real R_Q rollouts)", flush=True)
    t0 = time.monotonic()
    inserted = evolver.initialize_archive(evolution.seed_programs_dir)
    print(
        f"[sample] {inserted} seeds placed in {time.monotonic() - t0:.0f}s",
        flush=True,
    )
    print(render_grid(archive, "\n[sample] grid after seeding"), flush=True)

    before = {archive.program_to_cell(c) for c in archive.champions()}
    print(
        f"\n[sample] running {args.steps} mutations "
        f"in batches of {evolution.inner_iteration_batch_size}",
        flush=True,
    )
    t0 = time.monotonic()
    metrics = evolver.run_outer_iteration(0)
    elapsed = time.monotonic() - t0

    reports = evolver.last_reports
    status = Counter(r.status for r in reports)
    ops = Counter(r.op for r in reports)
    after = {archive.program_to_cell(c) for c in archive.champions()}

    print(f"\n[sample] evolve phase done in {elapsed:.0f}s")
    print(f"[sample] metrics: {json.dumps(metrics, ensure_ascii=False)}")
    print(f"[sample] operators: {dict(ops)}")
    print("[sample] candidate outcomes:")
    for name, count in status.most_common():
        print(f"    {name:<28} {count}")
    reasons = [r.reason for r in reports if r.reason]
    if reasons:
        print("[sample] rejection reasons (first 8):")
        for reason in reasons[:8]:
            print(f"    - {str(reason)[:140]}")

    # --- judge agreement: the measurement this pipeline exists to produce ---
    judged = [r for r in reports if r.status == "judge_rejected"]
    mismatch = [r for r in judged if "label mismatch" in (r.reason or "")]
    closed = [r for r in judged if "failed closed" in (r.reason or "")]
    reached_judge = [
        r for r in reports
        if r.status not in {"mutation_failed", "no_code", "verify_failed", "no_parent"}
    ]
    agreed = len(reached_judge) - len(judged)
    judge_summary = {
        "reached_judge": len(reached_judge),
        "agreed": agreed,
        "label_mismatch": len(mismatch),
        "failed_closed": len(closed),
        "agreement_rate": round(agreed / len(reached_judge), 3) if reached_judge else None,
    }
    print("\n[sample] judge agreement:")
    for k, v in judge_summary.items():
        print(f"    {k:<20} {v}")
    axis = Counter()
    for r in mismatch:
        for token in ("GROUP", "SKILL"):
            if f"{token} declared=" in (r.reason or ""):
                axis[token] += 1
    if axis:
        print(f"    mismatching axis     {dict(axis)}")
    for r in judged[:6]:
        print(f"    - {str(r.reason)[:150]}")

    print(render_grid(archive, "\n[sample] grid after evolve"))
    new_cells = sorted(after - before)
    print(f"\n[sample] cells opened this phase: {len(new_cells)}")
    for cell in new_cells:
        print(f"    + {archive.cell_labels(cell)}")

    evolver.save_state(out, iteration=0)
    evolver.append_evolution_log(out, 0, metrics)
    (out / "sample_summary.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "steps": args.steps,
                "batch": evolution.inner_iteration_batch_size,
                "code_temperature": evolution.code_temperature,
                "code_top_p": evolution.code_top_p,
                "rollouts": args.rollouts,
                "seeds_placed": inserted,
                "metrics": metrics,
                "status_counts": dict(status),
                "judge": judge_summary,
                "operator_counts": dict(ops),
                "cells_before": len(before),
                "cells_after": len(after),
                "cells_opened": [list(c) for c in new_cells],
                "elapsed_s": round(elapsed, 1),
                "mutate_calls": backend.n_mutate_calls,
                "generations": backend.n_generated,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out / "generations.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in backend.transcript),
        encoding="utf-8",
    )
    print(f"\n[sample] artifacts written to {out}/ "
          f"({len(backend.transcript)} generations captured)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
