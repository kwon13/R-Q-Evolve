"""Merge a VERL FSDP actor checkpoint into a HuggingFace model directory.

Usage:
    python scripts/merge_fsdp_to_hf.py \
        --ckpt_dir rq_output/verl_ckpt_grpo_h/global_step_256/actor \
        --out_dir  rq_output/verl_ckpt_grpo_h/global_step_256/hf_merged
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor, Shard, Replicate
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def find_world_size(ckpt_dir: Path) -> int:
    sizes = set()
    for p in ckpt_dir.glob("model_world_size_*_rank_*.pt"):
        sizes.add(int(p.stem.split("_")[3]))
    if len(sizes) != 1:
        raise RuntimeError(f"Expected one world_size in {ckpt_dir}, found {sizes}")
    return sizes.pop()


def merge_param(name: str, shards: list) -> torch.Tensor:
    """Reassemble a full tensor from per-rank DTensor shards."""
    first = shards[0]
    if not isinstance(first, DTensor):
        return first.detach().cpu()

    placements = first.placements
    assert len(placements) == 1, f"{name}: only 1D mesh supported, got {placements}"
    placement = placements[0]
    locals_ = [s._local_tensor.detach().cpu() for s in shards]

    if isinstance(placement, Replicate):
        return locals_[0]
    if isinstance(placement, Shard):
        full = torch.cat(locals_, dim=placement.dim)
        target = tuple(first.shape)
        # FSDP pads the sharded dim so each rank gets equal size; trim back.
        if full.shape != target:
            slices = [slice(0, s) for s in target]
            full = full[tuple(slices)]
        return full.contiguous()
    raise NotImplementedError(f"{name}: unsupported placement {placement}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="path to .../global_step_X/actor")
    ap.add_argument("--out_dir", required=True, help="output HF model dir")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--safe_serialization", action="store_true", default=True)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    hf_src = ckpt_dir / "huggingface"
    if not hf_src.is_dir():
        raise FileNotFoundError(f"Missing {hf_src} (config/tokenizer source).")

    world_size = find_world_size(ckpt_dir)
    print(f"[merge] world_size = {world_size}")

    print("[merge] loading shards...")
    shard_sds = []
    for rank in range(world_size):
        path = ckpt_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        print(f"  - rank {rank}: {path}")
        shard_sds.append(torch.load(path, weights_only=False, map_location="cpu"))

    keys = list(shard_sds[0].keys())
    for r, sd in enumerate(shard_sds[1:], 1):
        if list(sd.keys()) != keys:
            raise RuntimeError(f"key mismatch between rank 0 and rank {r}")

    print(f"[merge] merging {len(keys)} parameters...")
    full_sd = {}
    target_dtype = getattr(torch, args.dtype)
    for i, k in enumerate(keys):
        t = merge_param(k, [sd[k] for sd in shard_sds])
        if t.is_floating_point():
            t = t.to(target_dtype)
        full_sd[k] = t
        if (i + 1) % 50 == 0 or i == len(keys) - 1:
            print(f"  merged {i + 1}/{len(keys)}")

    # free shard memory before building model
    del shard_sds

    print("[merge] building HF model skeleton...")
    config = AutoConfig.from_pretrained(hf_src)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=target_dtype)
    model.to_empty(device="cpu")

    missing, unexpected = model.load_state_dict(full_sd, strict=False, assign=True)
    if unexpected:
        print(f"[merge] WARN unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    real_missing = [m for m in missing if "lm_head" not in m or not config.tie_word_embeddings]
    if real_missing:
        print(f"[merge] WARN missing keys: {real_missing[:8]}{' ...' if len(real_missing) > 8 else ''}")
    if config.tie_word_embeddings:
        model.tie_weights()

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving model to {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=args.safe_serialization)

    print("[merge] saving tokenizer / generation config / chat template")
    AutoTokenizer.from_pretrained(hf_src).save_pretrained(out_dir)
    gen_cfg_path = hf_src / "generation_config.json"
    if gen_cfg_path.exists():
        GenerationConfig.from_pretrained(hf_src).save_pretrained(out_dir)
    chat_tmpl = hf_src / "chat_template.jinja"
    if chat_tmpl.exists():
        shutil.copy2(chat_tmpl, out_dir / "chat_template.jinja")

    print(f"[merge] done -> {out_dir}")


if __name__ == "__main__":
    main()
