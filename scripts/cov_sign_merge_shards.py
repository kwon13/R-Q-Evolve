#!/usr/bin/env python
"""Concatenate per-GPU shard outputs into one covariance dataset.

Each shard is a complete cov_sign output dir (its own instances.json,
rollouts.jsonl, entropy.jsonl, per_token_entropy.npz, meta_*.json) over a
disjoint slice of the problem list, so merging is concatenation plus a
consistency check that every shard used the same checkpoint and sampling.

    python scripts/cov_sign_merge_shards.py --out-dir <union dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-glob", default="shard_*")
    args = ap.parse_args()

    out = Path(args.out_dir)
    shards = sorted(out.glob(args.shard_glob), key=lambda p: int(p.name.split("_")[-1]))
    if not shards:
        raise SystemExit(f"no shard dirs under {out}")

    instances: list[dict] = []
    rollouts: list[str] = []
    entropy: list[str] = []
    per_token: dict[str, np.ndarray] = {}
    metas_g, metas_e = [], []
    for sd in shards:
        mg = json.loads((sd / "meta_generate.json").read_text())
        me = json.loads((sd / "meta_entropy.json").read_text())
        metas_g.append(mg)
        metas_e.append(me)
        instances.extend(json.loads((sd / "instances.json").read_text()))
        rollouts.extend(
            l for l in (sd / "rollouts.jsonl").read_text().split("\n") if l.strip()
        )
        entropy.extend(
            l for l in (sd / "entropy.jsonl").read_text().split("\n") if l.strip()
        )
        with np.load(sd / "per_token_entropy.npz") as z:
            for k in z.files:
                per_token[k] = z[k]

    # Every shard must have run the same measurement.
    for key in ("checkpoint", "g", "instance_seed", "sampling", "max_prompt_length"):
        vals = {json.dumps(m[key], sort_keys=True) for m in metas_g}
        if len(vals) != 1:
            raise SystemExit(f"shards disagree on {key}: {vals}")
    for key in ("checkpoint", "temperature", "entropy_accumulation_dtype"):
        vals = {json.dumps(m[key], sort_keys=True) for m in metas_e}
        if len(vals) != 1:
            raise SystemExit(f"shards disagree on entropy {key}: {vals}")

    ids = [r["program_id"] for r in instances]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate program_id across shards")

    (out / "instances.json").write_text(
        json.dumps(instances, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out / "rollouts.jsonl").write_text("\n".join(rollouts) + "\n", encoding="utf-8")
    (out / "entropy.jsonl").write_text("\n".join(entropy) + "\n", encoding="utf-8")
    np.savez_compressed(out / "per_token_entropy.npz", **per_token)

    mg = dict(metas_g[0])
    mg.update(
        {
            "phase": "generate (merged)",
            "shards": len(shards),
            "shard": None,
            "num_shards": len(shards),
            "n_problems": len(instances),
            "n_rollouts_written": len(rollouts),
            "n_champions": sum(m["n_champions"] for m in metas_g),
            "execute_failures": [f for m in metas_g for f in m["execute_failures"]],
            "generation_seconds": max(m["generation_seconds"] for m in metas_g),
            "generation_seconds_per_shard": [m["generation_seconds"] for m in metas_g],
            "cuda_visible_devices": [m["cuda_visible_devices"] for m in metas_g],
        }
    )
    (out / "meta_generate.json").write_text(json.dumps(mg, ensure_ascii=False, indent=1), encoding="utf-8")

    me = dict(metas_e[0])
    checks = [m["alignment_check"] for m in metas_e]
    me.update(
        {
            "phase": "entropy (merged)",
            "shards": len(shards),
            "n_rollouts": len(entropy),
            "seconds": max(m["seconds"] for m in metas_e),
            "alignment_check": {
                # worst shard, so the merged report never overstates the check
                "pearson_actor_vs_vllm_surprisal": min(
                    c["pearson_actor_vs_vllm_surprisal"] for c in checks
                ),
                "median_abs_rel_diff": max(c["median_abs_rel_diff"] for c in checks),
            },
            "alignment_check_per_shard": checks,
        }
    )
    (out / "meta_entropy.json").write_text(json.dumps(me, indent=1), encoding="utf-8")

    print(f"[merge] {len(shards)} shards -> {len(instances)} problems, {len(rollouts)} rollouts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
