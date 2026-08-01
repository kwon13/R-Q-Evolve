import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_test_notebook.py"
)
SPEC = importlib.util.spec_from_file_location(
    "rq_build_test_notebook",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_notebook_is_thin_cli_frontend_without_stale_outputs():
    notebook = builder.build_notebook()
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert len(notebook["cells"]) == 6
    runtime_cell = next(
        cell
        for cell in notebook["cells"]
        if cell.get("id") == "runtime-environment"
    )
    runtime_source = "".join(runtime_cell["source"])
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "7"' in runtime_source
    assert (
        'os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"'
        in runtime_source
    )
    assert "compare_mutation_methods_vllm.py" in source
    assert "run_shot_internal_diagnostics.py" in source
    assert '"--plain-baseline", "two_stage"' in source
    assert '"--mutation-code-backend", MUTATION_CODE_BACKEND' in source
    assert 'MUTATION_CODE_BACKEND = "hybrid"' in source
    assert "subprocess.run" in source
    assert "evidence_input_hash" in source
    assert "run_config_hash" in source
    assert "solver_system_prompt_sha256" in source
    assert 'VLLM_SAMPLER_BACKEND = "pytorch"' in source
    assert '"--vllm-sampler-backend", VLLM_SAMPLER_BACKEND' in source
    assert "CODE_CONTRACT_VERSION = 6" in source
    assert "mutation_compiler_source_sha256" in source
    assert "MUTATION_FAMILY_REGISTRY_VERSION" in source
    assert "PLANNING_CONTRACT = \"belief_v6\"" in source
    assert "CANDIDATES_PER_CONDITION = 3" in source
    assert "PLAN_ONLY = False" in source
    assert "RUN_SHOT_INTERNAL_DIAGNOSTIC = False" in source
    assert '"--planning-contract", PLANNING_CONTRACT' in source
    assert '"--candidates-per-condition", str(CANDIDATES_PER_CONDITION)' in source
    assert "belief_probe_source_sha256" in source
    assert "BELIEF_SCHEMA_VERSION" in source
    assert "FIXED_PARENT_ROLLOUTS_JSON" in source
    assert '"--parent-rollouts-json"' in source
    assert "SHOT_TARGET_SEEDS" in source
    assert 'SHOT_PRESENTATION = "user_context"' in source
    assert '"--shot-presentation", SHOT_PRESENTATION' in source
    assert '"--stalt-tau", str(SHOT_STALT_TAU)' in source
    assert '"--primary-layer-fraction", str(SHOT_PRIMARY_LAYER_FRACTION)' in source
    assert "shot_internal" in source
    assert "instance_data" in source
    assert "evidence={evidence_input_hash}" in source
    assert "config={run_config_hash}" in source
    assert "_run_method(" not in source
    assert all(not cell.get("outputs") for cell in notebook["cells"])

    generated = json.loads(builder.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert generated == notebook


def test_notebook_builder_preserves_user_runtime_environment_cell():
    notebook = builder.build_notebook()
    runtime_cell = next(
        cell
        for cell in notebook["cells"]
        if cell.get("id") == "runtime-environment"
    )
    runtime_cell["source"] = [
        "import os\n",
        'os.environ["CUDA_VISIBLE_DEVICES"] = "6"\n',
        'os.environ["CUSTOM_RUNTIME_FLAG"] = "keep-me"',
    ]

    rebuilt = builder.build_notebook(notebook)
    rebuilt_runtime_cell = next(
        cell
        for cell in rebuilt["cells"]
        if cell.get("id") == "runtime-environment"
    )

    assert rebuilt_runtime_cell["source"] == runtime_cell["source"]
