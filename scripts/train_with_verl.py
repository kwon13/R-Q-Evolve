"""Entry point for connecting R_Q-Evolve to pip-installed verl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.config import load_config, load_raw_config
from rq_evolve.prompts import PROMPT_TEMPLATE_DIR
from rq_evolve.run_contract import freeze_evolution_run_contract
from rq_evolve.verl_adapter import (
    VerlAdapterConfig,
    VerlTrainerAdapter,
    describe_verl_runtime,
)


def _require_verl_sampling_patch() -> None:
    """Abort unless verl honours per-call sampling overrides.

    Without the patch, per-call mutation sampling overrides are read from the
    yaml, plumbed onto meta_info, and then dropped by verl's agent loop. A
    missing patch is therefore a startup error, not a warning.
    """
    sys.path.insert(0, str(ROOT / "patches"))
    try:
        import verl_agent_loop_sampling as patch
    except ImportError as exc:  # pragma: no cover - only if the file is deleted
        raise SystemExit(f"patches/verl_agent_loop_sampling.py is missing: {exc}")
    if not patch.is_applied():
        raise SystemExit(
            "verl's agent loop is NOT patched for per-call sampling overrides, "
            "so evolution.code_temperature would be silently ignored.\n"
            "  fix: python patches/verl_agent_loop_sampling.py"
        )


def _require_flash_attn_if_needed(config_path: str) -> None:
    """Ensure flash_attn is properly installed if use_remove_padding is enabled."""
    try:
        raw = load_raw_config(config_path)
        remove_padding = bool(
            raw.get("verl_config", {})
            .get("actor_rollout_ref", {})
            .get("model", {})
            .get("use_remove_padding", False)
        )
        if remove_padding:
            try:
                import flash_attn
                from flash_attn.bert_padding import unpad_input, pad_input
            except Exception as exc:
                raise SystemExit(
                    f"\n[FATAL] Config '{config_path}' specifies `use_remove_padding: true`, "
                    f"but FlashAttention (flash_attn) is not installed or cannot be imported ({exc}).\n"
                    f"  Please install flash_attn (or build from source with your CUDA toolkit) "
                    f"before running training with use_remove_padding.\n"
                )
    except SystemExit:
        raise
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "rq_evolve_base.yaml")
    )
    parser.add_argument(
        "--print-verl-env",
        action="store_true",
        help="print the Python executable and verl package resolved by this environment",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "smoke-test mode: force WANDB_MODE=offline and, after fit() "
            "returns, verify the LoRA adapter checkpoint + rollout JSONL logs "
            "exist. Pair with configs/rq_evolve_smoke_lora.yaml (LoRA r32, "
            "1 training step). Requires free GPUs."
        ),
    )
    parser.add_argument(
        "--audit-static-data",
        action="store_true",
        help=(
            "load static_training_jsonl with the actual training tokenizer, "
            "print exact row/token/schedule accounting, and exit before Ray "
            "or any model workers start"
        ),
    )
    args = parser.parse_args()

    if args.print_verl_env:
        for key, value in describe_verl_runtime().items():
            print(f"{key}: {value}")
        return

    if args.smoke:
        import os

        os.environ.setdefault("WANDB_MODE", "offline")

    _require_verl_sampling_patch()
    _require_flash_attn_if_needed(args.config)
    _warn_if_project_venv_exists()
    # Load ordinary training credentials (for example WANDB_API_KEY) without
    # importing or initializing any classifier/API client.
    load_dotenv(ROOT / ".env", override=False)
    config = load_config(args.config)

    if not config.verl.enabled:
        print(
            "verl.enabled=false. Set it true and either embed a `verl_config:` "
            "block in the same yaml or set verl.config_path to train with verl."
        )
        return

    # The rq_evolve yaml may embed the verl override inline under `verl_config:`
    # (preferred); otherwise verl.config_path must point to a separate file.
    inline_verl_config = _read_inline_verl_config(args.config)
    if inline_verl_config is None and not config.verl.config_path:
        raise ValueError(
            "either embed a `verl_config:` block in the yaml or set "
            "verl.config_path when verl.enabled=true"
        )

    resolved_raw = load_raw_config(args.config)
    contract_name = resolved_raw.get("run_contract")
    if config.evolution.structural_inspiration and not contract_name:
        contract_name = "structural_inspiration_v1"
    if contract_name:
        if inline_verl_config is None:
            raise ValueError(
                "the frozen run contract requires an inline verl_config so its "
                "output directory can be frozen before launch"
            )
        output_dir = inline_verl_config.trainer.get("default_local_dir")
        if not output_dir:
            raise ValueError(
                "verl_config.trainer.default_local_dir is required for the "
                "frozen run contract"
            )
        manifest = freeze_evolution_run_contract(
            contract_name=str(contract_name),
            config_path=args.config,
            resolved_config=resolved_raw,
            output_dir=str(output_dir),
            project_root=ROOT,
            prompt_dir=PROMPT_TEMPLATE_DIR,
        )
        print(f"[RQ-Evolve] frozen run contract: {manifest}")

    adapter = VerlTrainerAdapter(
        config=VerlAdapterConfig(
            config_path=config.verl.config_path,
            reward_function=config.verl.reward_function,
            inline_config=inline_verl_config,
        ),
        rq_config=config,
        project_root=ROOT,
    )
    if args.audit_static_data:
        report = adapter.audit_static_training_data()
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return

    import time as _time

    smoke_started_at = _time.time()
    adapter.fit()

    if args.smoke:
        raise SystemExit(
            _run_smoke_checks(config, inline_verl_config, smoke_started_at)
        )


def _run_smoke_checks(config, inline_verl_config, started_at: float) -> int:
    """Post-fit PASS/FAIL: did the LoRA + async instrumentation actually land?

    Artifacts must be NEWER than this run's start (``started_at``) so a stale
    ./rq_output/smoke from a previous run can never green-light a regression.
    """
    if inline_verl_config is None and config.verl.config_path:
        from omegaconf import OmegaConf

        path = Path(config.verl.config_path)
        if not path.is_absolute():
            path = ROOT / path
        inline_verl_config = OmegaConf.load(path)
    if inline_verl_config is None:
        print("[smoke] cannot resolve the verl config -> no checks run")
        return 1

    local_dir = Path(
        str(inline_verl_config.trainer.get("default_local_dir", "./rq_output/smoke"))
    )
    if not local_dir.is_absolute():
        # verl and the adapter resolve relative dirs against the process cwd;
        # the checks must look in the same place.
        local_dir = Path.cwd() / local_dir

    def fresh(path: Path) -> bool:
        try:
            return path.stat().st_mtime >= started_at
        except OSError:
            return False

    checks: list[tuple[str, bool, str]] = []

    if config.lora.enabled:
        adapters = [
            p
            for p in sorted(
                local_dir.glob(
                    "global_step_*/**/lora_adapter/adapter_model.safetensors"
                )
            )
            if fresh(p)
        ]
        checks.append(
            (
                "lora_adapter_saved",
                bool(adapters),
                (
                    str(adapters[-1])
                    if adapters
                    else f"no fresh lora_adapter under {local_dir}/global_step_*"
                ),
            )
        )

    archive_dir = local_dir / "rq_archive"
    samples = archive_dir / "rollout_samples.jsonl"
    metrics = archive_dir / "rollout_metrics.jsonl"
    if config.async_rollout.streaming_enabled and config.async_rollout.log_samples:
        checks.append(
            (
                "rollout_samples_jsonl",
                samples.exists() and samples.stat().st_size > 0 and fresh(samples),
                str(samples),
            )
        )
    checks.append(
        (
            "rollout_metrics_jsonl",
            metrics.exists() and metrics.stat().st_size > 0 and fresh(metrics),
            str(metrics),
        )
    )

    failed = False
    print("== smoke checks ==")
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failed = failed or not ok
    print("smoke:", "FAILED" if failed else "OK")
    return 1 if failed else 0


def _read_inline_verl_config(yaml_path: str):
    """Return the `verl_config` sub-tree from the rq_evolve yaml, or None.

    The typed RQEvolveConfig dataclass intentionally doesn't model verl's
    schema, so we re-load the yaml as a raw OmegaConf DictConfig just to pull
    out the embedded verl override.
    """
    from omegaconf import OmegaConf

    raw = load_raw_config(yaml_path)
    if not isinstance(raw, type(OmegaConf.create({}))):
        return None
    return raw.get("verl_config", None) if "verl_config" in raw else None


def _warn_if_project_venv_exists() -> None:
    project_python = ROOT / ".venv" / "bin" / "python"
    if (
        project_python.exists()
        and Path(sys.executable).resolve() != project_python.resolve()
    ):
        print(
            "[RQ-Evolve] project .venv detected. "
            f"Use {project_python} to train against that environment's verl."
        )


if __name__ == "__main__":
    main()
