"""Entry point for connecting R_Q-Evolve to pip-installed verl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.config import load_config
from rq_evolve.verl_adapter import (
    VerlAdapterConfig,
    VerlTrainerAdapter,
    describe_verl_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "rq_evolve.yaml"))
    parser.add_argument(
        "--print-verl-env",
        action="store_true",
        help="print the Python executable and verl package resolved by this environment",
    )
    args = parser.parse_args()

    if args.print_verl_env:
        for key, value in describe_verl_runtime().items():
            print(f"{key}: {value}")
        return

    _warn_if_project_venv_exists()
    config = load_config(args.config)

    if not config.verl.enabled:
        print("verl.enabled=false. Set it true and set verl.config_path to train with verl.")
        return
    if not config.verl.config_path:
        raise ValueError("verl.config_path must be set when verl.enabled=true")

    adapter = VerlTrainerAdapter(
        config=VerlAdapterConfig(
            config_path=config.verl.config_path,
            reward_function=config.verl.reward_function,
        ),
        rq_config=config,
        project_root=ROOT,
    )
    adapter.fit()


def _warn_if_project_venv_exists() -> None:
    project_python = ROOT / ".venv" / "bin" / "python"
    if project_python.exists() and Path(sys.executable).resolve() != project_python.resolve():
        print(
            "[RQ-Evolve] project .venv detected. "
            f"Use {project_python} to train against that environment's verl."
        )


if __name__ == "__main__":
    main()
