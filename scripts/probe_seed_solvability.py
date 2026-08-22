#!/usr/bin/env python
"""Per-seed pass rate of the current policy over the seed corpus.

The archive only admits a program with R_Q = p(1-p)H > 0, so a seed the policy
never solves (p=0) and one it always solves (p=1) are both dropped and cannot
become mutation parents. This measures that directly, before a sweep spends an
hour discovering it.

    python scripts/probe_seed_solvability.py --rollouts 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.program import ProblemProgram  # noqa: E402
from rq_evolve.prompts import build_solver_messages  # noqa: E402
from rq_evolve.reward import answers_match, extract_boxed  # noqa: E402
from rq_evolve.solver_trace import SOLVER_CHAT_BOUNDARY_STOPS  # noqa: E402
from rq_evolve.vllm_runtime import configure_vllm_sampler_backend  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data1/yhoon113/qwen3-8b-base")
    ap.add_argument("--seed-dir", default="seed_programs")
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gpu-util", type=float, default=0.4)
    ap.add_argument("--max-model-len", type=int, default=12000)
    ap.add_argument("--out", default="rq_output/seed_solvability.json")
    args = ap.parse_args()

    configure_vllm_sampler_backend("pytorch")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    tok = llm.get_tokenizer()

    programs = sorted(Path(args.seed_dir).glob("*.py"))
    instances = []
    for path in programs:
        program = ProblemProgram.from_file(path)
        inst = program.execute(seed=0)
        if inst is None:
            print(f"{path.name}: EXECUTE FAILED")
            continue
        instances.append((path.name, inst))

    prompts = [
        tok.apply_chat_template(
            build_solver_messages(inst.problem), tokenize=False,
            add_generation_prompt=True,
        )
        for _, inst in instances
        for _ in range(args.rollouts)
    ]
    params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.tokens, stop=list(SOLVER_CHAT_BOUNDARY_STOPS),
    )
    outputs = llm.generate(prompts, params)

    rows = []
    print(f"\n{'seed program':<56}{'answer':>12}{'p_hat':>8}  admits?")
    print("-" * 92)
    for i, (name, inst) in enumerate(instances):
        chunk = outputs[i * args.rollouts : (i + 1) * args.rollouts]
        correct = sum(
            1
            for o in chunk
            if (pred := extract_boxed(o.outputs[0].text)) is not None
            and answers_match(pred, inst.answer)
        )
        p_hat = correct / args.rollouts
        admits = 0.0 < p_hat < 1.0
        rows.append({"program": name, "answer": inst.answer, "p_hat": p_hat,
                     "correct": correct, "n": args.rollouts, "admits": admits})
        print(f"{name:<56}{inst.answer[:12]:>12}{p_hat:>8.2f}  {'YES' if admits else 'no'}")

    n_admit = sum(1 for r in rows if r["admits"])
    print(f"\n{n_admit} of {len(rows)} seeds can enter the archive "
          f"(R_Q = p(1-p)H > 0 needs 0 < p < 1)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "rollouts": args.rollouts,
                               "rows": rows}, indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
