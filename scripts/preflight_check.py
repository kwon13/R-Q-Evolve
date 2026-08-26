#!/usr/bin/env python
"""Fail-early model/config gate. Run BEFORE any training launch.

CPU-only (meta-device instantiation, no weights, no GPU) -- runs even while
the GPUs are busy. Exits non-zero on any fatal failure.

Examples (inside `conda activate azr-bw-blackwell`):

  # expected: all green
  python scripts/preflight_check.py \
      --model-path /data1/yhoon113/qwen3-8b-base \
      --config configs/rq_evolve_base.yaml

  # expected: fails with actionable messages (unsupported architecture)
  python scripts/preflight_check.py \
      --model-path /data1/hub/deepseek-v3.2 \
      --config configs/rq_evolve_deepseek_template.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.config import load_config, load_raw_config  # noqa: E402
from rq_evolve.preflight import has_fatal_failure, run_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=None,
        help="model to check; defaults to actor_rollout_ref.model.path from --config",
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "rq_evolve_base.yaml")
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    rq_config = load_config(args.config)
    raw = load_raw_config(args.config)
    verl_config = raw.get("verl_config")
    if verl_config is None:
        print(f"error: {args.config} has no inline verl_config block", file=sys.stderr)
        return 2

    model_path = args.model_path or str(
        verl_config.actor_rollout_ref.model.get("path", "")
    )
    if not model_path:
        print("error: no --model-path and no model.path in config", file=sys.stderr)
        return 2

    trust_remote_code = bool(
        verl_config.actor_rollout_ref.model.get("trust_remote_code", True)
    )
    results = run_all(
        model_path,
        verl_config,
        rq_config.lora,
        trust_remote_code=trust_remote_code,
    )

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": r.name,
                        "ok": r.ok,
                        "fatal": r.fatal,
                        "message": r.message,
                        "details": r.details,
                    }
                    for r in results
                ],
                indent=2,
                default=str,
            )
        )
    else:
        print(f"== preflight: model={model_path}")
        print(
            f"==            config={args.config} (lora.enabled={rq_config.lora.enabled})"
        )
        for r in results:
            print(r.render())

    if has_fatal_failure(results):
        print("\npreflight: FAILED -- fix the [FAIL] items above before launching")
        return 1
    print("\npreflight: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
