"""Derive a per-run config from a base yaml, overriding evolution.num_rollouts.

The trainer entrypoint only takes --config (no key overrides), and editing the
base yaml in place is what made the streaming A/B unreproducible (no snapshot of
what each run actually used). So each sweep arm gets its own generated config,
and a copy of it is written into the run's output dir.
"""

import argparse
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(ROOT / "configs" / "rq_evolve_4b_base.yaml"))
    parser.add_argument("--num-rollouts", type=int, required=True)
    parser.add_argument("--tag", required=True, help="run tag, e.g. nr20")
    parser.add_argument("--out-dir", default=str(ROOT / "configs" / "generated"))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.base).read_text())

    cfg["evolution"]["num_rollouts"] = args.num_rollouts

    trainer = cfg["verl_config"]["trainer"]
    trainer["experiment_name"] = f"qwen3_4b_rq_evolve_{args.tag}"
    local_dir = f"./rq_output/rq_evolve_4b_{args.tag}"
    trainer["default_local_dir"] = local_dir

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / f"rq_evolve_4b_{args.tag}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))

    # Snapshot alongside the run's artifacts so the arm is reconstructable later.
    run_dir = ROOT / local_dir.lstrip("./")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, run_dir / "config_used.yaml")

    print(cfg_path)


if __name__ == "__main__":
    main()
