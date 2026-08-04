#!/usr/bin/env python
"""One evolve phase against a real policy, without the verl/Ray/FSDP stack.

Boots vLLM directly, runs the pipeline that actually changed -- parent
selection, the two GROUP x SKILL mutation operators, verification, the operator
contract, the coherence evaluator, solver rollouts, R_Q, archive placement --
and reports where the children landed. Nothing is trained; weights are frozen,
so this measures the prompts and the grid, not the RL loop.

    python scripts/sample_evolve_vllm.py --batch 8 --rollouts 4

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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.archive import MAPElitesArchive  # noqa: E402
from rq_evolve.backends import PendingRollouts, RolloutRecord  # noqa: E402
from rq_evolve.concepts import GROUPS, SKILLS  # noqa: E402
from rq_evolve.config import load_config  # noqa: E402
from rq_evolve.evolution import RQEvolver  # noqa: E402
from rq_evolve.prompts import build_solver_prompt  # noqa: E402
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

    # --- session lifecycle: a resident engine has nothing to wake or push ---
    def sync_weights(self) -> None:
        pass

    def begin_session(self) -> None:
        pass

    def end_session(self) -> None:
        pass

    def _chat(self, messages, *, temperature, top_p, max_tokens, stop=None):
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
            logprobs=0,
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
        # A mutation writes a whole generator; an evaluator verdict is 3 lines.
        max_tokens = max(
            (
                self.args.eval_tokens
                if task.stage == "evaluate"
                else self.args.mutate_tokens
            )
            for task in tasks
        )
        outputs = self._chat(
            conversations,
            temperature=tasks[0].temperature if tasks[0].temperature is not None else 0.0,
            top_p=tasks[0].top_p if tasks[0].top_p is not None else 1.0,
            max_tokens=max_tokens,
        )
        self.n_generated += len(outputs)
        texts = [o.outputs[0].text for o in outputs]
        # Keep every raw generation. A CandidateReport carries only a reason
        # string, so without this the source behind "execute failed at seed=0"
        # is unrecoverable and the failure cannot be diagnosed after the fact.
        for task, out, text in zip(tasks, outputs, texts):
            self.transcript.append(
                {
                    "stage": stage,
                    "op": task.op,
                    "parent_id": getattr(task.parent, "program_id", ""),
                    "parent_group": getattr(task.parent, "get_group", lambda: None)(),
                    "parent_skill": getattr(task.parent, "get_skill", lambda: None)(),
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
            [{"role": "user", "content": build_solver_prompt(inst.problem)}]
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
    lines = [title, "     " + "".join(f"{s[:7]:>9}" for s in SKILLS)]
    for gi, group in enumerate(GROUPS):
        row = "".join(
            "        O" if (gi, si) in occupied else "        ."
            for si in range(len(SKILLS))
        )
        lines.append(f"{group[:13]:<14}{row}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq_evolve_base.yaml")
    ap.add_argument("--model", default="/data1/yhoon113/qwen3-8b-base")
    ap.add_argument("--batch", type=int, default=8, help="mutations this phase")
    ap.add_argument("--rollouts", type=int, default=4, help="solver rollouts per child")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=12000)
    ap.add_argument("--mutate-tokens", type=int, default=3000)
    ap.add_argument("--eval-tokens", type=int, default=256)
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

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[sample] booting vLLM on {args.model} (tp={args.tp})", flush=True)
    t0 = time.monotonic()
    backend = VLLMBackend(args.model, args)
    print(f"[sample] engine ready in {time.monotonic() - t0:.0f}s", flush=True)

    archive = MAPElitesArchive(
        epsilon=cfg.archive.epsilon,
        ucb_c=cfg.archive.ucb_c,
        selection_strategy=cfg.archive.selection_strategy,
    )
    evolution = cfg.evolution
    evolution.inner_iterations = args.batch
    evolution.inner_iteration_batch_size = args.batch
    evolution.num_rollouts = args.rollouts
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
    print(f"\n[sample] running one evolve phase: {args.batch} mutations", flush=True)
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
                "batch": args.batch,
                "rollouts": args.rollouts,
                "seeds_placed": inserted,
                "metrics": metrics,
                "status_counts": dict(status),
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
