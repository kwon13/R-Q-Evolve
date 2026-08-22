#!/usr/bin/env python
"""Phase 1 of the signed reward-entropy covariance experiment: rollout generation.

Generates G rollouts per archived champion at its seed=0 representative instance
using the run's final solver checkpoint, and stores every rollout RAW (text,
response token ids, grade) so the covariance analysis never has to regenerate.

Entropy is NOT computed here -- see cov_sign_entropy.py, which runs the actor
forward pass over these exact token ids. Splitting the phases keeps vLLM and the
HF actor off the same GPU at the same time and makes the expensive half
restartable.

Sampling is matched to the training rollout config (configs/rq_evolve_base.yaml,
actor_rollout_ref.rollout): temperature 1.0, top_p 0.95, top_k -1, no stop
strings, max_tokens = data.max_response_length. verl's agent loop applies the
chat template to [system, user] and passes token ids straight to vLLM, which is
reproduced here exactly.

    python scripts/cov_sign_generate.py --g 32
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# RQ_COV_SIGN_SRC lets the run import rq_evolve from somewhere other than the
# working tree. Needed when the working copy of src/ is mid-edit and does not
# parse: the measurement must not silently pick up a half-written prompt module.
sys.path.insert(0, os.environ.get("RQ_COV_SIGN_SRC", str(ROOT / "src")))

from rq_evolve.prompts import build_solver_messages  # noqa: E402
from rq_evolve.program import ProblemProgram  # noqa: E402
from rq_evolve.reward import answers_match, extract_boxed  # noqa: E402
from rq_evolve.solver_trace import sanitize_solver_trace  # noqa: E402
from rq_evolve.vllm_runtime import (  # noqa: E402
    VLLM_SAMPLER_BACKENDS,
    configure_vllm_sampler_backend,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--archive",
        default=str(ROOT / "rq_output/rq_evolve_base_8b/rq_archive/archive.json"),
    )
    p.add_argument(
        "--archive-glob",
        default="",
        help=(
            "instead of one archive.json, take the UNION of every champion that "
            "ever occupied a cell, deduped by program_id across snapshots "
            "(e.g. 'rq_output/rq_evolve_base_8b/rq_archive/archive_iter*.json'). "
            "Raises the problem count for the sign-distribution statistics."
        ),
    )
    p.add_argument(
        "--instances-from",
        default="",
        help=(
            "reuse the problem set of a completed run verbatim (its instances.json), "
            "including the stored prompt token ids. Use for a PAIRED comparison across "
            "checkpoints: the programs are not re-executed and the prompts are not "
            "re-rendered, so the two runs differ only in the policy."
        ),
    )
    p.add_argument(
        "--checkpoint",
        default=str(ROOT / "rq_output/rq_evolve_base_8b/global_step_256/hf_merged"),
        help="solver checkpoint the rollouts are drawn from (recorded in meta.json)",
    )
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "analysis/rq_evolve_base_8b/cov_sign"),
    )
    p.add_argument("--g", type=int, default=32, help="rollouts per problem")
    p.add_argument("--instance-seed", type=int, default=0)
    # Training rollout config (configs/rq_evolve_base.yaml).
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=5000)
    p.add_argument("--max-prompt-length", type=int, default=6000)
    p.add_argument("--max-model-len", type=int, default=12000)
    p.add_argument("--gpu-util", type=float, default=0.85)
    p.add_argument("--tp", type=int, default=1, help="TP>1 faults on sm_120")
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    p.add_argument("--vllm-sampler-backend", choices=VLLM_SAMPLER_BACKENDS, default="pytorch")
    p.add_argument("--seed", type=int, default=1234, help="vLLM engine seed")
    p.add_argument("--limit", type=int, default=0, help="debug: first N champions only")
    # Sharding splits the problem list across GPUs; each shard writes a complete,
    # self-contained output dir that cov_sign_merge_shards.py concatenates.
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reused: list[dict] | None = None
    if args.instances_from:
        reused = json.loads(Path(args.instances_from).read_text(encoding="utf-8"))
        source = f"{args.instances_from} (reused instances)"
        n_total = len(reused)
        if args.limit:
            reused = reused[: args.limit]
        if args.num_shards > 1:
            reused = reused[args.shard :: args.num_shards]
            source += f" [shard {args.shard}/{args.num_shards}]"
        print(f"[gen] {len(reused)}/{n_total} instances reused from {source}", flush=True)
    elif args.archive_glob:
        import glob as _glob
        import re as _re

        def _iter_no(path: str) -> int:
            m = _re.search(r"archive_iter(\d+)\.json$", path)
            return int(m.group(1)) if m else -1

        snapshots = sorted(_glob.glob(args.archive_glob), key=_iter_no)
        by_id: dict[str, dict] = {}
        for snap in snapshots:
            for c in json.loads(Path(snap).read_text(encoding="utf-8"))["champions"]:
                by_id[c["program_id"]] = c  # latest snapshot wins
        champions = list(by_id.values())
        source = f"{args.archive_glob} ({len(snapshots)} snapshots)"
    else:
        champions = json.loads(Path(args.archive).read_text(encoding="utf-8"))["champions"]
        source = args.archive
    if reused is None:
        if args.limit:
            champions = champions[: args.limit]
        n_total = len(champions)
        if args.num_shards > 1:
            champions = champions[args.shard :: args.num_shards]
            source += f" [shard {args.shard}/{args.num_shards}]"
        print(f"[gen] {len(champions)}/{n_total} champions from {source}", flush=True)
    else:
        champions = []

    # --- seed=0 representative instance per champion -----------------------
    programs, instances, failures = [], [], []
    if reused is None:
      for payload_prog in champions:
        prog = ProblemProgram.from_dict(payload_prog)
        inst = prog.execute(seed=args.instance_seed)
        if inst is None:
            failures.append(
                {"program_id": prog.program_id, "error": prog.last_execution_error}
            )
            print(f"[gen] SKIP {prog.program_id}: {prog.last_execution_error}", flush=True)
            continue
        programs.append(prog)
        instances.append(inst)
      print(f"[gen] {len(instances)} instances built, {len(failures)} execute failures", flush=True)

    configure_vllm_sampler_backend(args.vllm_sampler_backend)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=args.checkpoint,
        tokenizer=args.checkpoint,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()

    # One work list for both sourcing modes: (metadata, problem, answer, prompt ids).
    # verl renders the chat template to text, then encodes with
    # add_special_tokens=False (verl_backend._make_prompt_batch). Same here so
    # the prompt ids are byte-identical to the training path.
    work: list[dict] = []
    if reused is not None:
        # Paired mode: the prompt ids are taken verbatim from the reference run.
        # Re-rendering would let a tokenizer difference between checkpoints move
        # the prompt, and then the two runs would not be paired on the input.
        for rec in reused:
            meta_fields = {k: v for k, v in rec.items() if k != "prompt_token_ids"}
            work.append(
                {
                    "meta": meta_fields,
                    "problem": rec["problem"],
                    "answer": rec["answer"],
                    "prompt_ids": list(rec["prompt_token_ids"]),
                }
            )
        # Sanity: this checkpoint's tokenizer must read those ids the same way.
        drift = sum(
            1 for w in work
            if tokenizer.decode(w["prompt_ids"]) != tokenizer.decode(
                tokenizer.encode(
                    tokenizer.apply_chat_template(
                        build_solver_messages(w["problem"]),
                        add_generation_prompt=True, tokenize=False,
                    ),
                    add_special_tokens=False,
                )
            )
        )
        print(f"[gen] tokenizer drift on reused prompts: {drift}/{len(work)}", flush=True)
    else:
        for prog, inst in zip(programs, instances):
            rendered = tokenizer.apply_chat_template(
                build_solver_messages(inst.problem), add_generation_prompt=True, tokenize=False
            )
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            if len(ids) > args.max_prompt_length:
                ids = ids[-args.max_prompt_length:]  # truncation: left
            work.append(
                {
                    "meta": {
                        "program_id": prog.program_id,
                        "parent_id": prog.parent_id,
                        "generation": prog.generation,
                        "group": prog.get_group(),
                        "skill": prog.get_skill(),
                        "niche_group": prog.niche_group,
                        "niche_skill": prog.niche_skill,
                        "instance_seed": args.instance_seed,
                        "problem": inst.problem,
                        "answer": inst.answer,
                        # archived values, from the G=10 fitness rollouts at insert time
                        "archived_p_hat": prog.s_hat,
                        "archived_h_score": prog.u_score,
                        "archived_rq_score": prog.rq_score,
                        "archived_fitness": prog.rq_score,
                        "last_reeval_step": prog.last_reeval_step,
                    },
                    "problem": inst.problem,
                    "answer": inst.answer,
                    "prompt_ids": ids,
                }
            )
    prompt_ids_all = [w["prompt_ids"] for w in work]

    params = SamplingParams(
        n=args.g,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=-1,
        repetition_penalty=1.0,
        max_tokens=args.max_tokens,
        logprobs=0,  # sampled-token logprob: the vLLM-surprisal cross-check
    )
    t0 = time.time()
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids_all], params
    )
    gen_s = time.time() - t0
    print(f"[gen] generation done in {gen_s/60:.1f} min", flush=True)

    inst_path = out_dir / "instances.json"
    roll_path = out_dir / "rollouts.jsonl"
    n_rows = 0
    inst_records = []
    with roll_path.open("w", encoding="utf-8") as fh:
        for w, out in zip(work, outputs):
            prompt_ids = w["prompt_ids"]
            answer = w["answer"]
            program_id = w["meta"]["program_id"]
            inst_records.append(
                {
                    **w["meta"],
                    "prompt_tokens": len(prompt_ids),
                    "prompt_token_ids": prompt_ids,
                }
            )
            for ri, comp in enumerate(out.outputs):
                text = comp.text
                # Grading follows evolution._score_from_rollouts: the trace is
                # sanitized to one chat turn first, so a base-model completion
                # that opens a second conversation is not graded on it.
                cleaned = sanitize_solver_trace(text)
                pred_raw = extract_boxed(text)
                pred = extract_boxed(cleaned)
                surprisal = 0.0
                for step in comp.logprobs or []:
                    for lp in step.values():
                        surprisal += -float(lp.logprob)
                        break
                fh.write(
                    json.dumps(
                        {
                            "program_id": program_id,
                            "instance_seed": args.instance_seed,
                            "rollout_idx": ri,
                            "response": text,
                            "response_token_ids": list(comp.token_ids),
                            "response_tokens": len(comp.token_ids),
                            "finish_reason": comp.finish_reason,
                            "stop_reason": str(comp.stop_reason),
                            "predicted_answer_raw": pred_raw,
                            "predicted_answer": pred,
                            "correct": bool(pred is not None and answers_match(pred, answer)),
                            "sanitized": cleaned != text.strip(),
                            "vllm_surprisal_sum": surprisal,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_rows += 1
    inst_path.write_text(json.dumps(inst_records, ensure_ascii=False, indent=1), encoding="utf-8")

    meta = {
        "phase": "generate",
        "archive": source,
        "checkpoint": str(args.checkpoint),
        "git_commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "g": args.g,
        "instance_seed": args.instance_seed,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "max_tokens": args.max_tokens,
            "stop": None,
        },
        "max_prompt_length": args.max_prompt_length,
        "max_model_len": args.max_model_len,
        "vllm_seed": args.seed,
        "instances_from": args.instances_from,
        "n_champions": len(work),
        "shard": args.shard,
        "num_shards": args.num_shards,
        "n_problems": len(work),
        "n_rollouts_written": n_rows,
        "execute_failures": failures,
        "generation_seconds": gen_s,
    }
    (out_dir / "meta_generate.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[gen] wrote {n_rows} rollouts -> {roll_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
