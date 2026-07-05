"""Model/config preflight checks: fail early, with actionable messages.

Shared by ``scripts/preflight_check.py`` (CLI gate before a run) and
``verl_adapter._apply_lora_config`` (startup fail-fast). Every check is
CPU-only: models are instantiated on the meta device (no weights, no GPU), so
this runs even while the GPUs are occupied.

Check functions return :class:`CheckResult`; ``run_all`` composes the full
gate for a (model, rq_config, verl_config) triple.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    message: str
    details: dict = field(default_factory=dict)
    # False for advisory results that should not fail the gate.
    fatal: bool = True

    def render(self) -> str:
        mark = "PASS" if self.ok else ("FAIL" if self.fatal else "WARN")
        return f"[{mark}] {self.name}: {self.message}"


GIB = 1024**3


# ---------------------------------------------------------------------------
# 1. transformers loadability
# ---------------------------------------------------------------------------

def load_hf_config(model_path: str, trust_remote_code: bool = True):
    """AutoConfig or raise with the transformers error attached."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)


def check_transformers_loadable(model_path: str, trust_remote_code: bool = True) -> CheckResult:
    import transformers

    try:
        config = load_hf_config(model_path, trust_remote_code)
    except Exception as exc:
        return CheckResult(
            "transformers_config",
            False,
            f"AutoConfig failed for {model_path}: {exc}. "
            f"transformers {transformers.__version__} supports up to deepseek_v3 "
            f"in the DeepSeek family -- a deepseek_v32/v4-era checkpoint needs a "
            f"transformers upgrade (see docs/deepseek_support_plan.md) or "
            f"trust_remote_code modeling files shipped with the checkpoint.",
        )
    archs = list(getattr(config, "architectures", None) or [])
    try:
        model = build_meta_model(config, trust_remote_code)
        n_params = sum(p.numel() for p in model.parameters())
    except Exception as exc:
        return CheckResult(
            "transformers_model",
            False,
            f"config loads (model_type={config.model_type}, architectures={archs}) "
            f"but AutoModelForCausalLM cannot build it: {exc}. This architecture "
            f"is not runnable under installed transformers "
            f"{transformers.__version__}.",
            details={"model_type": config.model_type, "architectures": archs},
        )
    return CheckResult(
        "transformers_model",
        True,
        f"model_type={config.model_type}, architectures={archs}, "
        f"params={n_params / 1e9:.2f}B (meta-device instantiation OK)",
        details={
            "model_type": config.model_type,
            "architectures": archs,
            "n_params": int(n_params),
        },
    )


def build_meta_model(hf_config, trust_remote_code: bool = True):
    """Instantiate the architecture on the meta device (no weights, no GPU)."""
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    with init_empty_weights():
        return AutoModelForCausalLM.from_config(
            hf_config, trust_remote_code=trust_remote_code
        )


# ---------------------------------------------------------------------------
# 2/3. vLLM architecture + LoRA support
# ---------------------------------------------------------------------------

def resolve_vllm_model_cls(architectures: list[str]):
    """The vLLM model class serving these architectures, or raise.

    Uses ``_try_load_model_cls`` (arch-name only): the public
    ``resolve_model_cls`` in vllm 0.10.x requires a full ModelConfig, which
    would load the checkpoint config through vLLM -- overkill for a gate.
    """
    from vllm.model_executor.models.registry import ModelRegistry

    for arch in architectures:
        cls = ModelRegistry._try_load_model_cls(arch)
        if cls is not None:
            return cls
    supported = [a for a in ModelRegistry.get_supported_archs() if "ForCausalLM" in a]
    raise ValueError(
        f"none of {architectures} is in the vLLM model registry "
        f"(closest registered CausalLM archs: "
        f"{[a for a in supported if a[:4] in {arch[:4] for arch in architectures}] or supported[:8]})"
    )


def check_vllm_arch(architectures: list[str]) -> CheckResult:
    import vllm

    try:
        cls = resolve_vllm_model_cls(architectures)
    except Exception as exc:
        return CheckResult(
            "vllm_arch",
            False,
            f"vllm {vllm.__version__} cannot serve architectures={architectures}: "
            f"{exc}. The newest supported DeepSeek class is DeepseekV3ForCausalLM; "
            f"a v3.2/v4-era model needs a vllm upgrade "
            f"(see docs/deepseek_support_plan.md).",
        )
    return CheckResult(
        "vllm_arch", True, f"served by {cls.__name__}", details={"cls": cls.__name__}
    )


def vllm_supports_lora(architectures: list[str]) -> bool:
    """Whether vLLM can hot-load LoRA adapters for this model class."""
    from vllm.model_executor.models.interfaces import supports_lora

    return bool(supports_lora(resolve_vllm_model_cls(architectures)))


def check_vllm_lora(architectures: list[str], sync_mode: str) -> CheckResult:
    try:
        supported = vllm_supports_lora(architectures)
    except Exception as exc:
        return CheckResult(
            "vllm_lora", False, f"could not resolve vLLM LoRA support: {exc}"
        )
    if supported:
        return CheckResult(
            "vllm_lora",
            True,
            "vLLM supports runtime LoRA for this class -> "
            "lora.sync_mode=native_adapter (verl TensorLoRARequest/add_lora push)",
            details={"supports_lora": True},
        )
    return CheckResult(
        "vllm_lora",
        sync_mode not in ("auto", "native_adapter"),
        "this model class has NO runtime-LoRA support in the installed vLLM "
        "(no SupportsLoRA interface -- true for all DeepSeek MoE classes in "
        "0.10.1.1). native_adapter sync is impossible; options: "
        "(a) engine-restart fallback: merge the adapter with "
        "scripts/merge_fsdp_to_hf.py and relaunch on the merged checkpoint, "
        "(b) the planned merge_push worker (NOT implemented -- see "
        "docs/deepseek_support_plan.md).",
        details={"supports_lora": False},
    )


# ---------------------------------------------------------------------------
# 4. TP/EP divisibility and length budget
# ---------------------------------------------------------------------------

def check_parallelism(hf_config, verl_config) -> list[CheckResult]:
    results: list[CheckResult] = []
    rollout = verl_config.actor_rollout_ref.rollout
    tp = int(rollout.get("tensor_model_parallel_size", 1) or 1)
    ep = int(rollout.get("expert_parallel_size", 1) or 1)
    world = int(verl_config.trainer.get("n_gpus_per_node", 1)) * max(
        1, int(verl_config.trainer.get("nnodes", 1))
    )

    heads = int(getattr(hf_config, "num_attention_heads", 0) or 0)
    kv_heads = int(getattr(hf_config, "num_key_value_heads", heads) or heads)
    if heads and heads % tp != 0:
        results.append(
            CheckResult(
                "tp_heads", False,
                f"num_attention_heads={heads} not divisible by rollout TP={tp}",
            )
        )
    elif kv_heads and not (kv_heads % tp == 0 or tp % kv_heads == 0):
        results.append(
            CheckResult(
                "tp_kv_heads", False,
                f"num_key_value_heads={kv_heads} vs TP={tp}: needs kv%tp==0 or "
                f"tp%kv==0 (vLLM KV replication rule)",
            )
        )
    else:
        results.append(
            CheckResult("tp_heads", True, f"heads={heads}, kv_heads={kv_heads}, TP={tp} OK")
        )

    if world % tp != 0:
        results.append(
            CheckResult(
                "tp_world", False, f"world_size={world} not divisible by rollout TP={tp}"
            )
        )
    else:
        results.append(CheckResult("tp_world", True, f"world_size={world}, TP={tp} OK"))

    n_experts = int(getattr(hf_config, "n_routed_experts", 0) or getattr(hf_config, "num_experts", 0) or 0)
    if n_experts:
        if ep > 1 and n_experts % ep != 0:
            results.append(
                CheckResult(
                    "ep_experts", False,
                    f"n_routed_experts={n_experts} not divisible by expert_parallel_size={ep}",
                )
            )
        else:
            results.append(
                CheckResult("ep_experts", True, f"experts={n_experts}, EP={ep} OK")
            )

    max_model_len = int(rollout.get("max_model_len", 0) or 0)
    max_prompt = int(verl_config.data.get("max_prompt_length", 0) or 0)
    max_response = int(verl_config.data.get("max_response_length", 0) or 0)
    if max_model_len and max_model_len < max_prompt + max_response:
        results.append(
            CheckResult(
                "length_budget", False,
                f"rollout.max_model_len={max_model_len} < max_prompt_length"
                f"({max_prompt}) + max_response_length({max_response})",
            )
        )
    else:
        results.append(
            CheckResult(
                "length_budget", True,
                f"max_model_len={max_model_len} >= {max_prompt}+{max_response}",
            )
        )
    return results


# ---------------------------------------------------------------------------
# 5. disk / VRAM estimates (heuristic, clearly labeled)
# ---------------------------------------------------------------------------

def check_memory(
    n_params: int,
    verl_config,
    *,
    model_path: str,
    lora_enabled: bool,
    data_root: str = "/data1",
) -> list[CheckResult]:
    results: list[CheckResult] = []
    rollout = verl_config.actor_rollout_ref.rollout
    tp = int(rollout.get("tensor_model_parallel_size", 1) or 1)
    gpu_util = float(rollout.get("gpu_memory_utilization", 0.5) or 0.5)
    world = int(verl_config.trainer.get("n_gpus_per_node", 1)) * max(
        1, int(verl_config.trainer.get("nnodes", 1))
    )

    per_gpu_vram = _per_gpu_vram_bytes()
    weights_bytes = n_params * 2  # bf16 serving/training weights

    # disk: only relevant if the checkpoint is not already local
    if not Path(model_path).exists():
        free = shutil.disk_usage(data_root).free
        ok = weights_bytes < free
        results.append(
            CheckResult(
                "disk", ok,
                f"checkpoint not local; bf16 weights ~{weights_bytes / GIB:.0f} GiB vs "
                f"{free / GIB:.0f} GiB free on {data_root}"
                + ("" if ok else " -- DOES NOT FIT"),
            )
        )

    # rollout: TP shards weights; KV cache needs headroom inside gpu_util budget
    rollout_per_gpu = weights_bytes / max(1, tp)
    budget = per_gpu_vram * gpu_util
    ok = rollout_per_gpu < budget * 0.9  # leave >=10% of the vLLM budget for KV
    results.append(
        CheckResult(
            "vram_rollout", ok,
            f"[heuristic] vLLM weights/GPU ~{rollout_per_gpu / GIB:.1f} GiB (TP={tp}) vs "
            f"budget {budget / GIB:.1f} GiB ({gpu_util:.2f} x {per_gpu_vram / GIB:.0f} GiB)"
            + ("" if ok else " -- no room for KV cache; raise TP or gpu_memory_utilization"),
        )
    )

    # training (FSDP shards over world): full-FT ~16 B/param (bf16 weights+grads
    # + fp32 adam m/v + master); LoRA: bf16 base only + adapter optimizer noise
    train_bytes_per_param = 4 if lora_enabled else 16
    train_per_gpu = n_params * train_bytes_per_param / max(1, world)
    headroom = per_gpu_vram * (1.0 - gpu_util)  # vLLM is resident (sleep off)
    ok = train_per_gpu < headroom
    results.append(
        CheckResult(
            "vram_training", ok,
            f"[heuristic] FSDP {'LoRA' if lora_enabled else 'full-FT'} state/GPU "
            f"~{train_per_gpu / GIB:.1f} GiB vs ~{headroom / GIB:.1f} GiB left beside "
            f"resident vLLM"
            + ("" if ok else " -- enable param/optimizer offload or use LoRA"),
            fatal=False,  # activation memory dominates in practice; advisory only
        )
    )
    return results


def _per_gpu_vram_bytes() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    return 96 * GIB  # this box: RTX PRO 6000 Blackwell 96GB (GPUs may be busy)


# ---------------------------------------------------------------------------
# 6. LoRA target-module dry-run
# ---------------------------------------------------------------------------

def check_lora_target_modules(
    hf_config,
    target_modules: list[str] | tuple[str, ...],
    trust_remote_code: bool = True,
    max_examples: int = 3,
) -> CheckResult:
    """Meta-device match of each configured pattern against real module names.

    Mirrors peft's suffix matching (a pattern matches modules whose name ends
    with it as a path component). Fails if ANY pattern matches nothing, and
    lists the linear-module suffixes that DO exist so target_modules can be
    fixed (DeepSeek MLA: q_a_proj/q_b_proj/kv_a_proj_with_mqa/kv_b_proj, not
    q_proj/k_proj/v_proj).
    """
    import torch

    model = build_meta_model(hf_config, trust_remote_code)
    linear_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    available = sorted({name.rsplit(".", 1)[-1] for name in linear_names})

    matched: dict[str, list[str]] = {}
    for pattern in target_modules:
        hits = [
            n for n in linear_names
            if n == pattern or n.endswith("." + pattern) or n.rsplit(".", 1)[-1] == pattern
        ]
        matched[pattern] = hits

    missing = [p for p, hits in matched.items() if not hits]
    lines = []
    for pattern, hits in matched.items():
        example = ", ".join(hits[:max_examples]) + (", ..." if len(hits) > max_examples else "")
        lines.append(f"{pattern}: {len(hits)} matches" + (f" ({example})" if hits else ""))
    summary = "; ".join(lines)
    if missing:
        return CheckResult(
            "lora_target_modules",
            False,
            f"patterns with ZERO matches: {missing}. {summary}. "
            f"Linear module suffixes that exist in this model: {available}",
            details={"missing": missing, "available": available},
        )
    return CheckResult(
        "lora_target_modules",
        True,
        summary,
        details={"counts": {p: len(h) for p, h in matched.items()}},
    )


# ---------------------------------------------------------------------------
# 7. config-combo lint
# ---------------------------------------------------------------------------

def check_config_combos(verl_config, lora_cfg=None) -> list[CheckResult]:
    results: list[CheckResult] = []
    rollout = verl_config.actor_rollout_ref.rollout
    model = verl_config.actor_rollout_ref.model

    # model.lora.merge is honored only by verl's NEW engine workers; under the
    # legacy fsdp_workers path this repo uses, the engine would start LoRA-off
    # while the trainer still pushes adapter tensors -> add_lora crash.
    merge = bool(_get(_get(model, "lora", {}) or {}, "merge", False))
    if merge:
        results.append(
            CheckResult(
                "lora_merge_combo", False,
                "actor_rollout_ref.model.lora.merge=true is only supported by "
                "verl's new engine workers, NOT the legacy fsdp_workers path "
                "this pipeline uses -- remove it (see docs/deepseek_support_plan.md)",
            )
        )

    # sleep/wake must stay off on this machine (cumem allocator crash).
    if bool(rollout.get("free_cache_engine", True)) or bool(
        rollout.get("enable_sleep_mode", True)
    ):
        results.append(
            CheckResult(
                "sleep_mode", False,
                "rollout.free_cache_engine / enable_sleep_mode must both be false "
                "on this machine: the vLLM cumem sleep/wake allocator crashes "
                "after ~3 cycles ('CUDA Error: invalid argument at "
                "cumem_allocator.cpp'). Weight sync works with sleep off.",
            )
        )
    else:
        results.append(CheckResult("sleep_mode", True, "sleep/wake disabled (required here)"))

    strategy = str(verl_config.actor_rollout_ref.actor.get("strategy", "fsdp"))
    if lora_cfg is not None and getattr(lora_cfg, "enabled", False) and strategy not in ("fsdp", "fsdp2"):
        results.append(
            CheckResult(
                "lora_strategy", False,
                f"LoRA is only wired for strategy fsdp/fsdp2, got {strategy!r}",
            )
        )

    grad_ckpt = bool(model.get("enable_gradient_checkpointing", False))
    results.append(
        CheckResult(
            "grad_ckpt_use_cache", True,
            f"gradient checkpointing={'on' if grad_ckpt else 'off'}; verl's "
            f"dp_actor passes use_cache=False on every training/logprob forward "
            f"(dp_actor.py), so KV-cache is disabled during training regardless "
            f"of the HF config default",
            fatal=False,
        )
    )
    return results


def check_weight_sync_bucket(hf_config, verl_config) -> CheckResult:
    """The largest single tensor (embed table, fp32 master) must fit a bucket."""
    rollout = verl_config.actor_rollout_ref.rollout
    bucket_mb = int(
        _get(_get(rollout, "checkpoint_engine", {}) or {}, "update_weights_bucket_megabytes", 2048)
        or 2048
    )
    vocab = int(getattr(hf_config, "vocab_size", 0) or 0)
    hidden = int(getattr(hf_config, "hidden_size", 0) or 0)
    embed_mb = vocab * hidden * 4 / (1024**2)  # fp32 master weight
    ok = embed_mb <= bucket_mb
    return CheckResult(
        "weight_sync_bucket", ok,
        f"embed table {vocab}x{hidden} fp32 ~{embed_mb:.0f} MB vs "
        f"rollout.checkpoint_engine.update_weights_bucket_megabytes={bucket_mb}"
        + ("" if ok else " -- raise the bucket size"),
    )


def _get(node, key, default=None):
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    try:
        value = node.get(key, default)
        return default if value is None else value
    except Exception:
        return getattr(node, key, default)


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

def run_all(
    model_path: str,
    verl_config,
    lora_cfg=None,
    *,
    trust_remote_code: bool = True,
) -> list[CheckResult]:
    """Full preflight for one model+config. Never raises; failures are results."""
    results: list[CheckResult] = []

    tf_result = check_transformers_loadable(model_path, trust_remote_code)
    results.append(tf_result)
    if not tf_result.ok:
        return results  # nothing further is meaningful without a loadable config

    hf_config = load_hf_config(model_path, trust_remote_code)
    archs = tf_result.details.get("architectures") or []
    n_params = int(tf_result.details.get("n_params", 0))

    vllm_result = check_vllm_arch(archs)
    results.append(vllm_result)

    lora_enabled = bool(lora_cfg is not None and getattr(lora_cfg, "enabled", False))
    if lora_enabled:
        # merge_push is designed but NOT implemented: startup will refuse it
        # (verl_adapter._apply_lora_config raises), so the gate must too --
        # a green preflight must never precede a guaranteed startup failure.
        if getattr(lora_cfg, "sync_mode", "auto") == "merge_push":
            results.append(
                CheckResult(
                    "lora_sync_mode",
                    False,
                    "lora.sync_mode=merge_push is not implemented yet -- startup "
                    "will fail. Use sync_mode=auto/native_adapter for vLLM-LoRA-"
                    "capable models, or the engine-restart fallback "
                    "(scripts/merge_fsdp_to_hf.py); see docs/deepseek_support_plan.md.",
                )
            )
        if vllm_result.ok:
            results.append(check_vllm_lora(archs, getattr(lora_cfg, "sync_mode", "auto")))
        try:
            results.append(
                check_lora_target_modules(
                    hf_config, list(lora_cfg.target_modules), trust_remote_code
                )
            )
        except Exception as exc:
            results.append(
                CheckResult("lora_target_modules", False, f"meta-device match failed: {exc}")
            )

    results.extend(check_parallelism(hf_config, verl_config))
    results.extend(
        check_memory(
            n_params, verl_config, model_path=model_path, lora_enabled=lora_enabled
        )
    )
    results.append(check_weight_sync_bucket(hf_config, verl_config))
    results.extend(check_config_combos(verl_config, lora_cfg))
    return results


def has_fatal_failure(results: list[CheckResult]) -> bool:
    return any((not r.ok) and r.fatal for r in results)
