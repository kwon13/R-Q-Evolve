"""Config-schema + fail-fast tests for the LoRA / async_rollout sections."""

import pytest

from rq_evolve.config import (
    AsyncRolloutConfig,
    LoraConfig,
    RQEvolveConfig,
    load_config,
)


def test_defaults():
    cfg = RQEvolveConfig.from_dict({})
    assert cfg.async_rollout.streaming_enabled is True
    assert cfg.async_rollout.chunk_size == 1
    assert cfg.async_rollout.staleness_mode == "bounded"
    assert cfg.async_rollout.max_policy_lag == 1
    assert cfg.lora.enabled is False
    assert cfg.lora.rank == 32
    assert cfg.lora.alpha == 64
    assert cfg.lora.sync_mode == "auto"
    assert "q_proj" in cfg.lora.target_modules


def test_yaml_roundtrip(tmp_path):
    yaml_text = """
async_rollout:
  streaming_enabled: false
  chunk_size: 2
  max_in_flight_chunks: 4
  staleness_mode: strict
  reject_duplicates: true
lora:
  enabled: true
  rank: 16
  alpha: 32
  target_modules: [q_a_proj, kv_b_proj]
  sync_mode: native_adapter
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(path)
    assert cfg.async_rollout.streaming_enabled is False
    assert cfg.async_rollout.chunk_size == 2
    assert cfg.async_rollout.staleness_mode == "strict"
    assert cfg.async_rollout.reject_duplicates is True
    assert cfg.lora.enabled is True and cfg.lora.rank == 16
    assert cfg.lora.target_modules == ("q_a_proj", "kv_b_proj")  # tuple-coerced
    assert cfg.lora.sync_mode == "native_adapter"


def test_invalid_values_fail_fast():
    with pytest.raises(ValueError, match="staleness_mode"):
        AsyncRolloutConfig(staleness_mode="loose")
    with pytest.raises(ValueError, match="entropy_mode"):
        AsyncRolloutConfig(entropy_mode="streamed")
    with pytest.raises(ValueError, match="chunk_size"):
        AsyncRolloutConfig(chunk_size=0)
    with pytest.raises(ValueError, match="max_policy_lag"):
        AsyncRolloutConfig(max_policy_lag=-1)
    # 0 is NOT "unbounded" here -- it would gate all submissions off and hang
    with pytest.raises(ValueError, match="queue_maxsize"):
        AsyncRolloutConfig(queue_maxsize=0)
    with pytest.raises(ValueError, match="request_timeout_s"):
        AsyncRolloutConfig(request_timeout_s=0)
    with pytest.raises(ValueError, match="verify_workers"):
        AsyncRolloutConfig(verify_workers=0)
    with pytest.raises(ValueError, match="sync_mode"):
        LoraConfig(sync_mode="hotswap")
    with pytest.raises(ValueError, match="rank"):
        LoraConfig(enabled=True, rank=0)


def test_shipped_configs_parse():
    for name in (
        "configs/rq_evolve_base.yaml",
        "configs/rq_evolve_smoke_lora.yaml",
        "configs/rq_evolve_deepseek_template.yaml",
    ):
        cfg = load_config(name)
        assert isinstance(cfg.async_rollout, AsyncRolloutConfig)
        assert isinstance(cfg.lora, LoraConfig)
    base = load_config("configs/rq_evolve_base.yaml")
    assert base.lora.enabled is False  # existing full-FT series unchanged
    smoke = load_config("configs/rq_evolve_smoke_lora.yaml")
    assert smoke.lora.enabled is True and smoke.lora.rank == 32
    deepseek = load_config("configs/rq_evolve_deepseek_template.yaml")
    assert deepseek.lora.sync_mode == "merge_push"  # must fail-fast at startup


def test_apply_lora_config_fail_fast():
    """Adapter-level gates that must raise BEFORE ray/worker startup."""
    omegaconf = pytest.importorskip("omegaconf")
    verl_adapter = pytest.importorskip("rq_evolve.verl_adapter")

    def make_adapter(lora_kwargs, strategy="fsdp"):
        rq = RQEvolveConfig.from_dict({"lora": {"enabled": True, **lora_kwargs}})
        adapter = verl_adapter.VerlTrainerAdapter(
            config=verl_adapter.VerlAdapterConfig(inline_config={}),
            rq_config=rq,
            project_root=".",
        )
        verl_cfg = omegaconf.OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {"path": "/nonexistent/model", "trust_remote_code": True},
                    "actor": {"strategy": strategy},
                }
            }
        )
        return adapter, verl_cfg

    # merge_push is deferred work -> NotImplementedError with the plan pointer
    adapter, verl_cfg = make_adapter({"sync_mode": "merge_push"})
    with pytest.raises(NotImplementedError, match="deepseek_support_plan"):
        adapter._apply_lora_config(verl_cfg)

    # non-FSDP strategy is not wired for LoRA
    adapter, verl_cfg = make_adapter({"sync_mode": "auto"}, strategy="megatron")
    with pytest.raises(ValueError, match="fsdp"):
        adapter._apply_lora_config(verl_cfg)

    # unresolvable model path -> actionable RuntimeError, not a worker crash
    adapter, verl_cfg = make_adapter({"sync_mode": "auto"})
    with pytest.raises(RuntimeError, match="preflight"):
        adapter._apply_lora_config(verl_cfg)

    # lora disabled -> no-op even with an invalid model path
    rq = RQEvolveConfig.from_dict({})
    adapter = verl_adapter.VerlTrainerAdapter(
        config=verl_adapter.VerlAdapterConfig(inline_config={}),
        rq_config=rq,
        project_root=".",
    )
    adapter._apply_lora_config(verl_cfg)  # must not raise
