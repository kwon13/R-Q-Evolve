#!/usr/bin/env python
"""Phase 2: actor-forward entropy over the rollouts written by cov_sign_generate.

Reproduces verl_backend._response_entropies for a frozen checkpoint without
booting verl/Ray/FSDP. The training path computes, per response token,

    H_t = logsumexp(z_t) - sum(softmax(z_t) * z_t)          (verl_F.entropy_from_logits)

on the ACTOR's logits divided by rollout.temperature (fsdp_workers.compute_log_prob
sets meta_info["temperature"] = config.rollout.temperature), and R_Q uses the
per-trajectory SUM over the response mask. The same quantity is computed here by
running the merged HF actor over prompt_ids + response_ids.

Two deliberate differences from the trainer, both recorded in meta_entropy.json:
  * entropy is accumulated in float32, not the actor's bf16 (a measurement, not
    a training step -- bf16 softmax over a 152k vocab is needlessly noisy);
  * no FSDP sharding / remove-padding kernel, so only low-order digits differ.

Writes H_sum and H_mean per rollout plus the full per-token entropy vector, so
every later re-analysis (prefix means, token-position effects) is offline.

    python scripts/cov_sign_entropy.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(ROOT / "analysis/rq_evolve_base_8b/cov_sign"))
    p.add_argument(
        "--checkpoint",
        default=str(ROOT / "rq_output/rq_evolve_base_8b/global_step_256/hf_merged"),
    )
    p.add_argument("--temperature", type=float, default=1.0, help="rollout.temperature")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--pos-chunk", type=int, default=512, help="positions per entropy chunk")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def token_entropies(logits: torch.Tensor, temperature: float, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position Shannon entropy and log-softmax, computed in float32 chunks.

    Returns (entropy[T], logprobs[T, V] is NOT returned -- only what the caller
    needs: entropy and the full log-softmax row is consumed inline).
    """
    out = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)
    lse = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)
    for start in range(0, logits.shape[0], chunk):
        z = logits[start : start + chunk].float()
        if temperature != 1.0:
            z = z / temperature
        z_lse = torch.logsumexp(z, dim=-1)
        pd = torch.softmax(z, dim=-1)
        out[start : start + chunk] = z_lse - (pd * z).sum(dim=-1)
        lse[start : start + chunk] = z_lse
    return out, lse


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    instances = json.loads((out_dir / "instances.json").read_text(encoding="utf-8"))
    prompt_by_id = {r["program_id"]: r["prompt_token_ids"] for r in instances}

    rows = [
        json.loads(line)
        for line in (out_dir / "rollouts.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[ent] {len(rows)} rollouts, {len(prompt_by_id)} problems", flush=True)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
        trust_remote_code=True,
    ).eval().cuda()

    records = []
    per_token: dict[str, np.ndarray] = {}
    t0 = time.time()
    for i, row in enumerate(rows):
        prompt_ids = prompt_by_id[row["program_id"]]
        resp_ids = row["response_token_ids"]
        r_len = len(resp_ids)
        key = f"{row['program_id']}:{row['rollout_idx']}"
        if r_len == 0:
            records.append(
                {
                    "program_id": row["program_id"],
                    "rollout_idx": row["rollout_idx"],
                    "response_tokens": 0,
                    "h_sum": 0.0,
                    "h_mean": 0.0,
                    "hf_surprisal_sum": 0.0,
                }
            )
            per_token[key] = np.zeros(0, dtype=np.float32)
            continue
        ids = torch.tensor([prompt_ids + resp_ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            # logits_to_keep=r_len+1 keeps only the positions that predict the
            # response (plus the last, dropped below): full-sequence logits over a
            # 152k vocab would be several GB for no gain.
            out = model(input_ids=ids, use_cache=False, logits_to_keep=r_len + 1)
        logits = out.logits[0, :-1, :]  # positions p-1 .. p+R-2 -> predict resp[0..R-1]
        ent, lse = token_entropies(logits, args.temperature, args.pos_chunk)
        tgt = torch.tensor(resp_ids, dtype=torch.long, device=logits.device)
        # sampled-token surprisal under the actor (alignment cross-check vs vLLM)
        chosen = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).float()
        if args.temperature != 1.0:
            chosen = chosen / args.temperature
        surprisal = float((lse - chosen).sum().item())
        ent_cpu = ent.float().cpu().numpy()
        per_token[key] = ent_cpu.astype(np.float32)
        h_sum = float(ent_cpu.sum())
        records.append(
            {
                "program_id": row["program_id"],
                "rollout_idx": row["rollout_idx"],
                "response_tokens": r_len,
                "h_sum": h_sum,
                "h_mean": h_sum / r_len,
                "hf_surprisal_sum": surprisal,
            }
        )
        del out, logits, ent, lse
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[ent] {i+1}/{len(rows)}  {el/60:.1f} min  "
                  f"eta {(el/(i+1)*(len(rows)-i-1))/60:.1f} min", flush=True)

    ent_path = out_dir / "entropy.jsonl"
    with ent_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    np.savez_compressed(out_dir / "per_token_entropy.npz", **per_token)

    # Alignment check: the actor's sampled-token surprisal must track vLLM's own
    # logprobs. A broken prompt/response offset shows up here first.
    vs = {(r["program_id"], r["rollout_idx"]): r["vllm_surprisal_sum"] for r in rows}
    a = np.array([r["hf_surprisal_sum"] for r in records])
    b = np.array([vs[(r["program_id"], r["rollout_idx"])] for r in records])
    ok = (a > 0) & (b > 0)
    corr = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    rel = float(np.median(np.abs(a[ok] - b[ok]) / np.maximum(b[ok], 1e-9))) if ok.sum() else float("nan")
    print(f"[ent] actor-vs-vLLM surprisal: pearson={corr:.5f} median|rel diff|={rel:.4f}", flush=True)

    (out_dir / "meta_entropy.json").write_text(
        json.dumps(
            {
                "phase": "entropy",
                "checkpoint": str(args.checkpoint),
                "temperature": args.temperature,
                "attn_implementation": args.attn,
                "entropy_accumulation_dtype": "float32",
                "model_dtype": "bfloat16",
                "n_rollouts": len(records),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "seconds": time.time() - t0,
                "alignment_check": {
                    "pearson_actor_vs_vllm_surprisal": corr,
                    "median_abs_rel_diff": rel,
                },
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"[ent] wrote {ent_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
