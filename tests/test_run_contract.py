import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rq_evolve.config import load_raw_config
from rq_evolve.run_contract import (
    RunContractMismatch,
    freeze_evolution_run_contract,
)


def _project(tmp_path: Path):
    root = tmp_path / "project"
    configs = root / "configs"
    prompts = root / "prompts"
    seeds = root / "seeds"
    configs.mkdir(parents=True)
    prompts.mkdir()
    seeds.mkdir()
    (root / "src" / "rq_evolve").mkdir(parents=True)
    (root / "patches").mkdir()
    (root / "scripts").mkdir()

    base = configs / "base.yaml"
    base.write_text(
        """
evolution:
  seed_programs_dir: seeds
  two_stage_mutation: true
verl_config:
  trainer:
    default_local_dir: ignored
credentials:
  api_key: literal-secret-must-not-be-snapshotted
""".strip()
        + "\n",
        encoding="utf-8",
    )
    child = configs / "child.yaml"
    child.write_text(
        "extends: base.yaml\n"
        "run_contract: test_contract\n"
        "evolution:\n"
        "  structural_inspiration: true\n",
        encoding="utf-8",
    )
    (prompts / "structural.txt").write_text("prompt v1\n", encoding="utf-8")
    (seeds / "seed.py").write_text("def generate(seed): pass\n", encoding="utf-8")
    (root / "src" / "rq_evolve" / "core.py").write_text(
        "VERSION = 1\n", encoding="utf-8"
    )
    return root, child, prompts


def _freeze(root: Path, child: Path, prompts: Path, output: Path):
    return freeze_evolution_run_contract(
        contract_name="test_contract",
        config_path=child,
        resolved_config=load_raw_config(child),
        output_dir=output,
        project_root=root,
        prompt_dir=prompts,
    )


def test_run_contract_freezes_resolved_inputs_and_allows_identical_resume(tmp_path):
    root, child, prompts = _project(tmp_path)
    output = tmp_path / "run"

    manifest_path = _freeze(root, child, prompts, output)
    first = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(first["contract_sha256"]) == 64
    assert first["prompt_files"] == {
        "structural.txt": first["prompt_files"]["structural.txt"]
    }
    assert (manifest_path.parent / "config_resolved.yaml").exists()
    assert (manifest_path.parent / "prompt_templates" / "structural.txt").exists()
    assert (manifest_path.parent / "seed_programs" / "seed.py").exists()

    snapshots = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (manifest_path.parent / "config_sources").glob("*.yaml")
    )
    resolved = (manifest_path.parent / "config_resolved.yaml").read_text(
        encoding="utf-8"
    )
    assert "literal-secret-must-not-be-snapshotted" not in snapshots + resolved
    assert "<redacted>" in snapshots + resolved

    assert _freeze(root, child, prompts, output) == manifest_path
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == first


@pytest.mark.parametrize("changed", ["base", "prompt", "implementation"])
def test_run_contract_rejects_a_changed_resume(tmp_path, changed):
    root, child, prompts = _project(tmp_path)
    output = tmp_path / "run"
    _freeze(root, child, prompts, output)

    if changed == "base":
        with (root / "configs" / "base.yaml").open("a", encoding="utf-8") as f:
            f.write("extra_method_knob: true\n")
    elif changed == "prompt":
        (prompts / "structural.txt").write_text("prompt v2\n", encoding="utf-8")
    else:
        (root / "src" / "rq_evolve" / "core.py").write_text(
            "VERSION = 2\n", encoding="utf-8"
        )

    with pytest.raises(RunContractMismatch, match="contract changed"):
        _freeze(root, child, prompts, output)


def test_training_state_without_a_contract_requires_a_fresh_directory(tmp_path):
    root, child, prompts = _project(tmp_path)
    output = tmp_path / "legacy"
    (output / "global_step_1").mkdir(parents=True)

    with pytest.raises(RunContractMismatch, match="training state"):
        _freeze(root, child, prompts, output)


def test_config_extends_is_relative_and_rejects_a_cycle(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "base.yaml").write_text(
        "evolution:\n  inner_iterations: 7\n", encoding="utf-8"
    )
    child = nested / "child.yaml"
    child.write_text(
        "extends: base.yaml\nevolution:\n  verify_seeds: 3\n",
        encoding="utf-8",
    )
    merged = load_raw_config(child)
    assert merged.evolution.inner_iterations == 7
    assert merged.evolution.verify_seeds == 3

    (nested / "base.yaml").write_text("extends: child.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cyclic config extends chain"):
        load_raw_config(child)
