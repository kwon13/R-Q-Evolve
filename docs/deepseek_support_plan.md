# DeepSeek v4 Flash support plan (deferred work)

Status **2026-07**: DeepSeek v4 Flash CANNOT run in this environment. This
file records exactly why, what was prepared, and the gated steps to enable it.
The pipeline itself is already model-agnostic — when the blockers below clear,
DeepSeek slots in via `configs/rq_evolve_deepseek_template.yaml` + preflight.

## What blocks it today (azr-bw-blackwell)

1. **No checkpoint.** Nothing DeepSeek-like on disk except a weightless
   `/data1/hub/deepseek-v3.2` config (arch `DeepseekV32ForCausalLM`). `/data1`
   has ~529 GB free — check the checkpoint size FIRST (`preflight_check.py`
   computes bf16 bytes from the config; a V3-sized 671B MoE needs ~1.3 TB in
   bf16 and does not fit this disk, let alone 8x96 GB for training).
2. **transformers 4.57.6** knows `deepseek_v2/v3` only — no `deepseek_v32`, no
   v4-era arch. `AutoModel` cannot build the model (unless the checkpoint ships
   its own `trust_remote_code` modeling files that work on 4.57.6).
3. **vllm 0.10.1.1** serves up to `DeepseekV3ForCausalLM` (no V3.2 DSA indexer,
   no v4-era arch), and its DeepSeek MoE classes have **no `SupportsLoRA`** —
   runtime LoRA adapter reload is impossible for DeepSeek in this vLLM.
4. FSDP training side is fine in principle: verl builds any HF architecture via
   `AutoModelForCausalLM` and FSDP-shards MoE experts like dense params (no
   expert parallelism in training; rollout-side EP exists:
   `rollout.expert_parallel_size`). Megatron has a native DeepseekV3 path but
   is marked "not tested" — not our route.

Everything is verified mechanically by:

```bash
python scripts/preflight_check.py \
    --model-path <checkpoint> --config configs/rq_evolve_deepseek_template.yaml
```

which checks: transformers loadability (meta device), vLLM arch registry,
vLLM LoRA support, LoRA target-module matching, TP/EP divisibility, disk/VRAM
estimates, weight-sync bucket size, and forbidden config combos.

## Gated enablement steps (in order)

### Step 1 — obtain a runnable checkpoint + arch support
Either (a) the checkpoint's architecture maps to `DeepseekV3ForCausalLM`
(then today's stack serves it), or (b) build a NEW conda env (clone of
azr-bw-blackwell) with a vllm/transformers pair that supports the v4
architecture on sm_120/cu128. Do NOT upgrade azr-bw-blackwell in place — the
Blackwell stack (torch 2.7.1+cu128 / vllm 0.10.1.1 / flash_attn 2.8.3 / verl
0.7.1-patched) is known-good for the current experiments.
Gate: `preflight_check.py` checks 1–2 pass.

### Step 2 — LoRA target modules
DeepSeek MLA does not have `q_proj/k_proj/v_proj`; use
`q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj, gate_proj,
up_proj, down_proj` (the template default). Consider excluding router/gate
(`*.mlp.gate`) — it is a Linear named `gate`, not matched by `gate_proj`, so
the default list is already safe.
Gate: preflight `lora_target_modules` check passes (zero-match patterns fail).

### Step 3 — weight sync for a vLLM without DeepSeek LoRA support
Preferred order:

1. **If the newer vLLM in the new env supports LoRA for the class** (check
   `supports_lora`): use the existing `native_adapter` path unchanged.
2. **merge_push worker (to implement)** — `RQLoRAMergeWorker`, an
   `AsyncActorRolloutRefWorker` subclass registered in
   `verl_adapter._select_worker_classes` when `lora.sync_mode == "merge_push"`:
   - override `rollout_mode()`: under FSDP
     `summon_full_params(writeback=False)` iterate peft `LoraLayer`s and
     compute out-of-place `W' = W + (alpha/r) * B @ A`; pass through non-target
     weights unchanged;
   - push with `peft_config=None` so the worker-extension receiver takes the
     plain `model.load_weights` branch (never `add_lora`);
   - config unit: engine must boot with LoRA OFF while the trainer still holds
     a peft model — verified as one combo by preflight (the naive
     `model.lora.merge=true` flag alone CRASHES under the legacy fsdp_workers
     path: only verl's new engine workers honor it; preflight fails it).
   - risks: memory spike for a large MoE during merge (bucketed per-tensor
     merge mitigates), slower per-step sync (full weights every step). Every
     merge-push is logged with duration by the PolicyVersionTracker.
   - Also plumb `lora_dropout` while touching verl: the 0.7.1 legacy worker
     builds its peft config without dropout, so **today every LoRA run trains
     with effective dropout 0.0** (startup prints a warning when
     `lora.dropout > 0`). The fix is a one-line addition to the `lora_config`
     dict in `verl/workers/fsdp_workers.py` (~line 528): `"lora_dropout":
     <value>`. This patch is NOT applied yet; patching in place is acceptable
     practice here only because this verl install already carries other local
     modifications.
3. **engine-restart fallback (works today, slow, always correct)**: train N
   steps → save → merge the adapter into an HF checkpoint with
   `scripts/merge_fsdp_to_hf.py` → relaunch the run with
   `actor_rollout_ref.model.path` pointed at the merged checkpoint (fresh vLLM
   loads merged weights) → continue. Every restart is explicit in the logs.

Gate: a smoke run (`--smoke`) on a SMALL MoE with the same code path (e.g.
Qwen3-30B-A3B if available, or forced-merge on Qwen3-4B) before DeepSeek.

### Step 4 — memory/parallelism tuning
Template starting points: rollout TP=8 (+EP=8 for MoE), `gpu_memory_utilization
0.75`, training `strategy: fsdp2` + param/optimizer offload + LoRA r32 (full-FT
optimizer state for a large MoE does not fit 8x96 GB), grad checkpointing on
(verl passes `use_cache=False` in every training forward). Sleep mode stays
OFF on this machine (cumem allocator crash), so budget vLLM + FSDP coexistence
via `gpu_memory_utilization` — preflight's `vram_rollout` / `vram_training`
heuristics flag obvious misfits, activations are on you.
Gate: `dry_run_async.py --mode live`, then `--smoke`, then the real run.

## Explicitly rejected alternatives

- Upgrading vllm/transformers inside azr-bw-blackwell (would destabilize the
  running Qwen3 experiment series).
- Megatron backend for DeepSeek (verl marks its DeepseekV3 path "not tested";
  FSDP path is the known quantity here).
- Silent fallbacks: `merge_push` fails fast at startup until implemented;
  preflight fails `model.lora.merge=true`; nothing degrades quietly.
