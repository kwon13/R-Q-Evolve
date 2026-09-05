"""GPU-free checks before dispatching the expensive eight-GPU experiment."""

import importlib.util
import os
from pathlib import Path
import json
import shutil
import signal
import subprocess
import sys
import time

from omegaconf import OmegaConf
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_train_domain_type_epsilon_8gpu.sh"
spec = importlib.util.spec_from_file_location("_rq_epsilon_launch_config", ROOT / "src/rq_evolve/config.py")
config_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = config_module
spec.loader.exec_module(config_module)


def launch(*args, env=None):
    process_env = os.environ.copy()
    process_env.update(env or {})
    return subprocess.run(
        ["bash", str(SCRIPT), *args, "--dry-run"],
        cwd=ROOT,
        env=process_env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("child,parent", [
    ("rq_evolve_4b_8gpu_domain_type_epsilon.yaml", "rq_evolve_4b_8gpu_domain_type.yaml"),
    ("rq_evolve_8b_8gpu_domain_type_epsilon_a100.yaml", "rq_evolve_8b_8gpu_domain_type_a100.yaml"),
    ("rq_evolve_8b_8gpu_domain_type_epsilon_rtxpro6000.yaml", "rq_evolve_8b_8gpu_domain_type_rtxpro6000.yaml"),
])
def test_epsilon_profiles_only_change_admission_and_identity(child, parent, monkeypatch):
    monkeypatch.delenv("RQ_ADMISSION_EPSILON", raising=False)
    monkeypatch.delenv("RQ_EPSILON_MODEL_PATH", raising=False)
    load = lambda filename: OmegaConf.to_container(
        config_module.load_raw_config(ROOT / "configs" / filename), resolve=True
    )
    candidate, standard = load(child), load(parent)
    assert candidate.pop("run_contract") == "epsilon_admission_v1"
    assert candidate["archive"].pop("admission_strategy") == "epsilon_greedy"
    assert candidate["archive"].pop("admission_epsilon") == 0.25
    for key in ("experiment_name", "default_local_dir"):
        assert "epsilon_0.25" in candidate["verl_config"]["trainer"][key]
        candidate["verl_config"]["trainer"][key] = standard["verl_config"]["trainer"][key]
    assert candidate == standard
    assert candidate["verl_config"]["trainer"]["total_training_steps"] == 256
    assert candidate["training_data"]["training_budget"] == 32


@pytest.mark.parametrize("args,expected", [
    ([], "4b_domain_type_epsilon_0.25_35cell_8gpu_a100"),
    (["--model-size", "8b", "--epsilon", "0.10"], "8b_domain_type_epsilon_0.1_35cell_8gpu_a100"),
    (["--model-size", "8b", "--profile", "rtxpro6000", "--epsilon", "0.5"], "8b_domain_type_epsilon_0.5_35cell_8gpu_rtxpro6000"),
    (["--epsilon", "0"], "4b_domain_type_epsilon_0.0_35cell_8gpu_a100"),
    (["--epsilon", "1"], "4b_domain_type_epsilon_1.0_35cell_8gpu_a100"),
])
def test_dry_run_model_profiles_and_epsilon_identity(args, expected):
    result = launch(*args, "--gpus", "0,1,2,3,4,5,6,7", "--detach")
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout
    assert "fitness     : standard -- L * U" in result.stdout
    assert "admission   : epsilon_greedy" in result.stdout
    assert "detached    : true" in result.stdout
    assert "256 / every 32" in result.stdout
    assert "no process started" in result.stdout


@pytest.mark.parametrize("args", [
    ["--epsilon", "nan"], ["--epsilon", "inf"], ["--epsilon", "-0.1"],
    ["--epsilon", "1.1"], ["--epsilon", "nope"], ["--epsilon"],
    ["--model-size", "7b"], ["--profile", "rtxpro6000"], ["--resume"],
    ["--gpus", "0,1,2,3,4,5,6,6"], ["--gpus", "0,1,2,3,4,5,6,01"],
    ["--gpus", "0,1"], ["--model-path"],
])
def test_invalid_launch_options_fail_without_starting(args):
    result = launch(*args)
    assert result.returncode != 0
    assert "started detached" not in result.stdout


def test_wrapper_owns_overrides_and_preserves_model_path_argument():
    result = launch("--model-path", "/unused/model path for dry run", env={
        "RQ_DOMAIN_TYPE_CONFIG": "/invalid/old.yaml",
        "RQ_ADMISSION_EPSILON": "nan",
        "RQ_EPSILON_MODEL_PATH": "/stale/old/model",
        "RQ_EXPECTED_RQ_FITNESS_MODE": "no_u",
        "RQ_EXPECTED_ADMISSION_STRATEGY": "random",
        "RQ_EXPECTED_ADMISSION_EPSILON": "0.9",
        "RQ_EXPECTED_RESUME_MODE": "auto",
    })
    assert result.returncode == 0, result.stdout + result.stderr
    assert "epsilon=0.25" in result.stdout
    assert "resume mode : disable" in result.stdout
    assert "/unused/model path for dry run" in result.stdout


def test_detach_preserves_config_and_environment_with_mock_workers(tmp_path):
    # A separate miniature repository contains ONLY mock trainer/merge/patch
    # scripts; this executes the nohup branch without invoking real training.
    project = tmp_path / "mock project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    (project / "src/rq_evolve").mkdir(parents=True)
    shutil.copy(ROOT / "src/rq_evolve/config.py", project / "src/rq_evolve/config.py")
    shutil.copytree(ROOT / "configs", project / "configs")
    for name in ("run_train_domain_type_8gpu.sh", SCRIPT.name):
        shutil.copy(ROOT / "scripts" / name, scripts / name)
    (project / "seed_programs_domain_type").mkdir()
    for index in range(7):
        (project / "seed_programs_domain_type" / f"seed_{index}.py").write_text("# mock\n")
    (project / "patches").mkdir()
    (project / "patches/verl_agent_loop_sampling.py").write_text("print('mock patch')\n")
    (scripts / "auto_merge_checkpoints.py").write_text("import time\ntime.sleep(30)\n")
    (scripts / "train_with_verl.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path('mock_training.json').write_text(json.dumps({"
        "'args': sys.argv[1:], 'epsilon': os.environ['RQ_ADMISSION_EPSILON'],"
        "'gpus': os.environ['CUDA_VISIBLE_DEVICES'],"
        "'model': os.environ['RQ_EPSILON_MODEL_PATH']}))\n"
    )
    model = project / "model path"
    model.mkdir()
    log_dir = project / "logs/rq_evolve_4b_domain_type_epsilon_0.1_35cell_8gpu_a100"
    try:
        result = subprocess.run(
            ["bash", str(scripts / SCRIPT.name), "--epsilon", "0.10", "--detach",
             "--model-path", str(model), "--gpus", "7,6,5,4,3,2,1,0"],
            cwd=project, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        marker = project / "mock_training.json"
        for _ in range(30):
            if marker.exists():
                break
            time.sleep(0.1)
        payload = json.loads(marker.read_text())
        assert payload == {
            "args": ["--config", "configs/rq_evolve_4b_8gpu_domain_type_epsilon.yaml"],
            "epsilon": "0.1", "gpus": "7,6,5,4,3,2,1,0", "model": str(model),
        }
        assert (log_dir / "latest.log").is_symlink()
        assert (log_dir / "train.pid").is_file()
    finally:
        for name in ("train.pid", "auto_merge.pid"):
            pid_file = log_dir / name
            if pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text().strip()), signal.SIGTERM)
                except ProcessLookupError:
                    pass
