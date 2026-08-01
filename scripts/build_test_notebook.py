"""Generate the thin, reproducible mutation-comparison notebook.

The notebook intentionally delegates all experiment logic to
``compare_mutation_methods_vllm.py``.  Keeping orchestration in one tracked
Python source prevents notebook cells from silently drifting away from the CLI
and its provenance manifest.  The tagged ``runtime-environment`` cell is
user-owned and is preserved when an existing notebook is regenerated.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "scripts" / "test copy 2.ipynb"


def _lines(text: str) -> list[str]:
    lines = text.strip("\n").splitlines()
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else [])


def _code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def _markdown(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": _lines(source),
    }


def _runtime_environment_cell(existing_notebook: dict | None = None) -> dict:
    if existing_notebook is not None:
        for cell in existing_notebook.get("cells", []):
            if (
                cell.get("cell_type") == "code"
                and cell.get("id") == "runtime-environment"
            ):
                return _code(
                    "".join(cell.get("source", [])),
                    "runtime-environment",
                )

    return _code(
        r"""
import os

# 사용자 소유 설정: notebook 재생성 시 이 셀의 내용을 그대로 보존합니다.
# subprocess가 시작되기 전에 물리 GPU 7만 노출합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# CUDA 11.8에서 FlashInfer sampling JIT를 피합니다.
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
print(
    "VLLM_USE_FLASHINFER_SAMPLER:",
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"],
)
""",
        "runtime-environment",
    )


def build_notebook(existing_notebook: dict | None = None) -> dict:
    return {
        "cells": [
            _markdown(
                r"""
# Plain vs. reasoning-informed mutation

이 notebook은 실험 로직을 복제하지 않고
`scripts/compare_mutation_methods_vllm.py`를 실행합니다.

- 두 조건 모두 schema-v6의 동일한 closed belief/desire catalog를 봅니다.
- 모델은 `attributed_hypothesis`와 근거 인용문만 작성합니다.
- reasoning planner는 clean wrong trace를 보고 attribution하며, plain
  planner는 trace 없이 prior만으로 같은 catalog에서 attribution합니다.
- hypothesis가 정해지면 probe family/variant와 수학·진단성 검증은 Python
  registry가 담당하므로 code-generation LLM과 LLM evaluator gate를
  사용하지 않습니다.
- Python eligibility gate를 통과한 후보만 attribution niche별로 R_Q가
  가장 높은 문제를 champion으로 선택합니다. attribution hit rate와
  R_Q를 가중합하지 않습니다.
- 반복·truncation·최종 답 부재 trace는 evidence에서 제외됩니다.
- brute/decoy 및 lint-feedback retry는 사용하지 않습니다.
- in-depth는 target move가 수학적·정보적으로 필수인 construction을,
  in-breadth는 parent와 다른 concept group을 요구합니다.
- compiler 생성 코드는 하나의 `instance_data`에서 답과 문제를 함께 만들고,
  family별 semantic consistency assertion을 통과해야 합니다.
- 생성된 child seed 0과 compiler-verified canonical 풀이를 one-shot으로
  제시한 뒤, 별도 parent seed를 풀 때의 정확도·StALT·prompt representation
  변화를 측정합니다. 이는 학습이 아닌 frozen-Solver in-context 진단입니다.
- CUDA 11.8 `nvcc`와 FlashInfer 0.6.x의 JIT 충돌을 피하기 위해 vLLM의
  native PyTorch sampler를 명시적으로 사용합니다.
- `PLAN_TEMPERATURE`만 development set에서 조절하고, 최종 held-out 실험
  전에는 prompt와 설정을 동결합니다.
""",
                "overview",
            ),
            _runtime_environment_cell(existing_notebook),
            _code(
                r"""
from pathlib import Path
import hashlib
import json
import sys


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "rq_evolve").is_dir():
            return candidate
    raise RuntimeError("R-Q-Evolve repository root를 찾을 수 없습니다.")


ROOT = find_repo_root(Path.cwd().resolve())
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from rq_evolve.metacognition import EVIDENCE_QUALITY_VERSION
from rq_evolve.belief_probe import BELIEF_SCHEMA_VERSION
from rq_evolve.mutation_compiler import MUTATION_FAMILY_REGISTRY_VERSION
from rq_evolve.prompts import SOLVER_SYSTEM_PROMPT

SEED_PROGRAM = ROOT / "seed_programs" / "09_linear_algebra.py"
OPERATOR = "both"  # "in_depth", "in_breadth", "both"
PLANNING_SEED = 0

MODEL = "/data1/yhoon113/qwen3-8b-base"
TOKENIZER = None
TENSOR_PARALLEL_SIZE = 1
DTYPE = "bfloat16"
GPU_MEMORY_UTILIZATION = 0.5
MAX_MODEL_LEN = None
TRUST_REMOTE_CODE = False
VLLM_SAMPLER_BACKEND = "pytorch"  # CUDA 11.8 nvcc에서 FlashInfer JIT를 피합니다.
MUTATION_CODE_BACKEND = "hybrid"
PLANNING_CONTRACT = "belief_v6"
CANDIDATES_PER_CONDITION = 3
# True이면 attribution/compile/Python gate까지만 보고 Solver child rollout과
# R_Q 선택은 생략합니다.
PLAN_ONLY = False

# 각 LLM_SEED는 독립 generator draw입니다. child seed와 혼동하지 않습니다.
LLM_SEED = 0
NUM_ROLLOUTS = 5
CHILD_ROLLOUTS = 5
VERIFY_SEEDS = 3
EVALUATION_SEEDS = list(range(VERIFY_SEEDS))

# 우선 이 값만 development set에서 {0.0, 0.3, 0.7}로 비교합니다.
# O 또는 최종 accuracy가 아니라 plan/operator/necessity 통과율로 선택합니다.
PLAN_TEMPERATURE = 0.3
PLAN_TOP_P = 0.95

# 나머지는 plain/reasoning 양쪽에서 고정합니다.
SOLVER_TEMPERATURE = 1.0
SOLVER_TOP_P = 0.95
CODE_TEMPERATURE = 0.2
CODE_TOP_P = 0.95
EVALUATOR_TEMPERATURE = 0.0
EVALUATOR_TOP_P = 1.0
CHILD_TEMPERATURE = 1.0
CHILD_TOP_P = 0.95

SOLVER_MAX_TOKENS = 4096
MUTATION_MAX_TOKENS = 5000
# v6 출력은 네 필드 JSON뿐이며 structured decoding으로 강제됩니다.
PLAN_MAX_TOKENS = 256
EVALUATOR_MAX_TOKENS = 1024
TRACE_STORAGE_MAX_TOKENS = 4096
MONITORING_TOTAL_TRACE_TOKENS = 4096

# Frozen-Solver one-shot internal diagnostic. Shot seed와 target seed를
# 분리하여 같은 숫자 instance를 그대로 복사하는 효과를 피합니다.
RUN_SHOT_INTERNAL_DIAGNOSTIC = False
SHOT_SEED = 0
SHOT_TARGET_SEEDS = [
    seed for seed in EVALUATION_SEEDS if seed != SHOT_SEED
]
SHOT_MAX_NEW_TOKENS = 1024
SHOT_PRESENTATION = "user_context"
SHOT_STALT_TAU = 1.0
SHOT_PRIMARY_LAYER_FRACTION = 2 / 3
SHOT_DEVICE = "cuda"  # CUDA_VISIBLE_DEVICES 이후의 논리 device 0

PROMPT_DIR = ROOT / "prompt_templates"
SHOT_DIR = PROMPT_DIR / "shots"
CHAT_TEMPLATE_KWARGS = {}
META_PROGRESS_JSON = None
EVIDENCE_GATE_VERSION = (
    f"clean_same_instance_v2:{EVIDENCE_QUALITY_VERSION}"
)
EVIDENCE_PIPELINE_VERSION = "standalone_vllm_clean_pair_v1"
COMPARISON_DESIGN_VERSION = "belief_v6_gate_then_rq_selection_v1"
CODE_CONTRACT_VERSION = 6

# Artifact names bind every input that may change evidence or generation.
seed_hash = hashlib.sha256(SEED_PROGRAM.read_bytes()).hexdigest()[:10]
prompt_hashes = {
    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for directory in (PROMPT_DIR, SHOT_DIR)
    for path in sorted(directory.glob("*.txt"))
}
evidence_inputs = {
    "schema": 1,
    "evidence_pipeline_version": EVIDENCE_PIPELINE_VERSION,
    "metacognition_source_sha256": hashlib.sha256(
        (ROOT / "src" / "rq_evolve" / "metacognition.py").read_bytes()
    ).hexdigest(),
    "seed_program_sha256": hashlib.sha256(SEED_PROGRAM.read_bytes()).hexdigest(),
    "model": MODEL,
    "tokenizer": TOKENIZER or MODEL,
    "vllm_sampler_backend": VLLM_SAMPLER_BACKEND,
    "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
    "solver_system_prompt_sha256": hashlib.sha256(
        SOLVER_SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest(),
    "solver_sampling": {
        "temperature": SOLVER_TEMPERATURE,
        "top_p": SOLVER_TOP_P,
        "max_tokens": SOLVER_MAX_TOKENS,
        "trace_storage_max_tokens": TRACE_STORAGE_MAX_TOKENS,
        "monitoring_total_trace_tokens": MONITORING_TOTAL_TRACE_TOKENS,
    },
    "evidence_gate_version": EVIDENCE_GATE_VERSION,
}
evidence_input_hash = hashlib.sha256(
    json.dumps(
        evidence_inputs, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
).hexdigest()[:16]
run_inputs = {
    "evidence_input_hash": evidence_input_hash,
    "comparison_design": COMPARISON_DESIGN_VERSION,
    "code_contract_version": CODE_CONTRACT_VERSION,
    "comparison_script_sha256": hashlib.sha256(
        (ROOT / "scripts" / "compare_mutation_methods_vllm.py").read_bytes()
    ).hexdigest(),
    "mutation_compiler_source_sha256": hashlib.sha256(
        (ROOT / "src" / "rq_evolve" / "mutation_compiler.py").read_bytes()
    ).hexdigest(),
    "belief_probe_source_sha256": hashlib.sha256(
        (ROOT / "src" / "rq_evolve" / "belief_probe.py").read_bytes()
    ).hexdigest(),
    "belief_schema_version": BELIEF_SCHEMA_VERSION,
    "mutation_family_registry_version": MUTATION_FAMILY_REGISTRY_VERSION,
    "mutation_code_backend": MUTATION_CODE_BACKEND,
    "planning_contract": PLANNING_CONTRACT,
    "candidates_per_condition": CANDIDATES_PER_CONDITION,
    "plan_only": PLAN_ONLY,
    "runtime_environment": {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get(
            "VLLM_USE_FLASHINFER_SAMPLER"
        ),
    },
    "vllm_sampler_backend": VLLM_SAMPLER_BACKEND,
    "prompt_hashes": prompt_hashes,
    "operator": OPERATOR,
    "planning_seed": PLANNING_SEED,
    "llm_seed": LLM_SEED,
    "verify_seeds": VERIFY_SEEDS,
    "child_rollouts": CHILD_ROLLOUTS,
    "plan_sampling": [PLAN_TEMPERATURE, PLAN_TOP_P, PLAN_MAX_TOKENS],
    "code_sampling": [CODE_TEMPERATURE, CODE_TOP_P, MUTATION_MAX_TOKENS],
    "evaluator_sampling": [
        EVALUATOR_TEMPERATURE, EVALUATOR_TOP_P, EVALUATOR_MAX_TOKENS
    ],
    "child_sampling": [
        CHILD_TEMPERATURE, CHILD_TOP_P, SOLVER_MAX_TOKENS
    ],
}
run_config_hash = hashlib.sha256(
    json.dumps(run_inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()[:16]
FIXED_INPUT_DIR = ROOT / "rq_output" / "mutation_comparison_fixed_inputs"
FIXED_EVIDENCE_JSON = (
    FIXED_INPUT_DIR
    / (
        f"{SEED_PROGRAM.stem}_{seed_hash}_seed={PLANNING_SEED}_"
        f"evidence={evidence_input_hash}.json"
    )
)
FIXED_PARENT_ROLLOUTS_JSON = (
    FIXED_INPUT_DIR
    / (
        f"{SEED_PROGRAM.stem}_{seed_hash}_seed={PLANNING_SEED}_"
        f"evidence={evidence_input_hash}_parent_rollouts.json"
    )
)

run_name = (
    f"{SEED_PROGRAM.stem}_llm_seed={LLM_SEED}_"
    f"plan_temp={PLAN_TEMPERATURE:.1f}_config={run_config_hash}"
)
OUTPUT_DIR = (
    ROOT / "rq_output" / "mutation_method_comparison_notebook" / run_name
)

print(f"Repository: {ROOT}")
print(f"Python environment should contain vLLM.")
print(f"Seed program: {SEED_PROGRAM}")
print(f"Fixed evidence: {FIXED_EVIDENCE_JSON}")
print(f"Fixed parent rollouts: {FIXED_PARENT_ROLLOUTS_JSON}")
print(
    "Evidence mode:",
    "reuse fixed JSON" if FIXED_EVIDENCE_JSON.exists()
    else "collect once, then freeze",
)
print(f"Evidence input hash: {evidence_input_hash}")
print(f"Run config hash: {run_config_hash}")
print(f"Output: {OUTPUT_DIR}")
print(f"Shot target seeds: {SHOT_TARGET_SEEDS}")
""",
                "config",
            ),
            _code(
                r"""
import os
import shutil
import subprocess
import sys

script = ROOT / "scripts" / "compare_mutation_methods_vllm.py"
command = [
    sys.executable,
    str(script),
    "--seed-program", str(SEED_PROGRAM),
    "--operator", OPERATOR,
    "--plain-baseline", "two_stage",
    "--mutation-code-backend", MUTATION_CODE_BACKEND,
    "--planning-contract", PLANNING_CONTRACT,
    "--candidates-per-condition", str(CANDIDATES_PER_CONDITION),
    "--instance-seed", str(PLANNING_SEED),
    "--evaluation-seeds", *[str(seed) for seed in EVALUATION_SEEDS],
    "--num-rollouts", str(NUM_ROLLOUTS),
    "--child-rollouts", str(CHILD_ROLLOUTS),
    "--verify-seeds", str(VERIFY_SEEDS),
    "--model", MODEL,
    "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
    "--dtype", DTYPE,
    "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
    "--vllm-sampler-backend", VLLM_SAMPLER_BACKEND,
    "--llm-seed", str(LLM_SEED),
    "--solver-temperature", str(SOLVER_TEMPERATURE),
    "--solver-top-p", str(SOLVER_TOP_P),
    "--plan-temperature", str(PLAN_TEMPERATURE),
    "--plan-top-p", str(PLAN_TOP_P),
    "--code-temperature", str(CODE_TEMPERATURE),
    "--code-top-p", str(CODE_TOP_P),
    "--evaluator-temperature", str(EVALUATOR_TEMPERATURE),
    "--evaluator-top-p", str(EVALUATOR_TOP_P),
    "--child-temperature", str(CHILD_TEMPERATURE),
    "--child-top-p", str(CHILD_TOP_P),
    "--solver-max-tokens", str(SOLVER_MAX_TOKENS),
    "--mutation-max-tokens", str(MUTATION_MAX_TOKENS),
    "--plan-max-tokens", str(PLAN_MAX_TOKENS),
    "--evaluator-max-tokens", str(EVALUATOR_MAX_TOKENS),
    "--trace-storage-max-tokens", str(TRACE_STORAGE_MAX_TOKENS),
    "--monitoring-total-trace-tokens", str(MONITORING_TOTAL_TRACE_TOKENS),
    "--prompt-dir", str(PROMPT_DIR),
    "--shot-dir", str(SHOT_DIR),
    "--chat-template-kwargs-json", json.dumps(CHAT_TEMPLATE_KWARGS),
    "--output-dir", str(OUTPUT_DIR),
]
if PLAN_ONLY:
    command.append("--plan-only")
if TOKENIZER is not None:
    command.extend(["--tokenizer", str(TOKENIZER)])
if MAX_MODEL_LEN is not None:
    command.extend(["--max-model-len", str(MAX_MODEL_LEN)])
if TRUST_REMOTE_CODE:
    command.append("--trust-remote-code")
if META_PROGRESS_JSON is not None:
    command.extend(["--meta-progress-json", str(META_PROGRESS_JSON)])
if FIXED_EVIDENCE_JSON.exists():
    command.extend(["--evidence-json", str(FIXED_EVIDENCE_JSON)])
if FIXED_PARENT_ROLLOUTS_JSON.exists():
    command.extend(
        ["--parent-rollouts-json", str(FIXED_PARENT_ROLLOUTS_JSON)]
    )

print("Running:", " ".join(command))
subprocess.run(
    command,
    check=True,
    cwd=ROOT,
    env=os.environ.copy(),
)

# 첫 run의 clean evidence를 output 정리 범위 밖에 고정합니다.
if not FIXED_EVIDENCE_JSON.exists():
    gate = json.loads(
        (OUTPUT_DIR / "evidence_gate.json").read_text(encoding="utf-8")
    )
    if gate.get("valid"):
        FIXED_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            OUTPUT_DIR / "selected_reasoning_evidence.json",
            FIXED_EVIDENCE_JSON,
        )
        print(
            "Clean evidence를 새로 고정했습니다. 이 comparison은 이미 "
            "generation을 마쳤으므로 다음 PLAN_TEMPERATURE/LLM_SEED "
            "comparison부터 planning input으로 재사용됩니다. 이는 별도 "
            "hidden-state diagnostic의 model-memory 재사용을 뜻하지 않습니다."
        )
    else:
        print("Clean contrast가 없어 evidence를 고정하지 않았습니다:", gate)

if not FIXED_PARENT_ROLLOUTS_JSON.exists():
    parent_rollout_source = OUTPUT_DIR / "parent_rollouts.json"
    if parent_rollout_source.exists():
        FIXED_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent_rollout_source, FIXED_PARENT_ROLLOUTS_JSON)
        print(
            "Parent rollout cache를 고정했습니다. 다음 comparison에서는 "
            "요청한 rollout 수만큼 잘라 재사용합니다."
        )
""",
                "run",
            ),
            _code(
                r"""
import os
import subprocess
import sys

if RUN_SHOT_INTERNAL_DIAGNOSTIC:
    if not SHOT_TARGET_SEEDS:
        raise RuntimeError(
            "SHOT_TARGET_SEEDS가 비어 있습니다. SHOT_SEED와 다른 seed가 "
            "하나 이상 필요합니다."
        )
    shot_script = ROOT / "scripts" / "run_shot_internal_diagnostics.py"
    shot_output_dir = OUTPUT_DIR / "shot_internal"
    shot_command = [
        sys.executable,
        str(shot_script),
        "--comparison-root", str(OUTPUT_DIR),
        "--seed-program", str(SEED_PROGRAM),
        "--model", MODEL,
        "--operator", OPERATOR,
        "--shot-seed", str(SHOT_SEED),
        "--shot-presentation", SHOT_PRESENTATION,
        "--target-seeds", *[str(seed) for seed in SHOT_TARGET_SEEDS],
        "--max-new-tokens", str(SHOT_MAX_NEW_TOKENS),
        "--dtype", DTYPE,
        "--device", SHOT_DEVICE,
        "--primary-layer-fraction", str(SHOT_PRIMARY_LAYER_FRACTION),
        "--stalt-tau", str(SHOT_STALT_TAU),
        "--output-dir", str(shot_output_dir),
    ]
    if TOKENIZER is not None:
        shot_command.extend(["--tokenizer", str(TOKENIZER)])
    if TRUST_REMOTE_CODE:
        shot_command.append("--trust-remote-code")
    print("Running internal diagnostic:", " ".join(shot_command))
    subprocess.run(
        shot_command,
        check=True,
        cwd=ROOT,
        env=os.environ.copy(),
    )
else:
    print("Frozen-Solver one-shot internal diagnostic을 생략했습니다.")
""",
                "shot-internal",
            ),
            _code(
                r"""
from IPython.display import JSON, Markdown, display

manifest = json.loads(
    (OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8")
)
summaries = json.loads(
    (OUTPUT_DIR / "summaries.json").read_text(encoding="utf-8")
)
evidence_gate = json.loads(
    (OUTPUT_DIR / "evidence_gate.json").read_text(encoding="utf-8")
)
report = (OUTPUT_DIR / "REPORT.md").read_text(encoding="utf-8")

display(Markdown(report))
display(Markdown("## Evidence gate"))
display(JSON(evidence_gate, expanded=True))
display(Markdown("## Experiment manifest"))
display(JSON(manifest, expanded=False))

shot_summary_path = OUTPUT_DIR / "shot_internal" / "summary.json"
shot_report_path = OUTPUT_DIR / "shot_internal" / "REPORT.md"
if shot_summary_path.exists() and shot_report_path.exists():
    shot_summary = json.loads(
        shot_summary_path.read_text(encoding="utf-8")
    )
    display(Markdown(shot_report_path.read_text(encoding="utf-8")))
    display(Markdown("## One-shot internal diagnostic data"))
    display(JSON(shot_summary, expanded=True))

print(
    "status:",
    [
        (
            row.get("operator"),
            row.get("condition"),
            row.get("status"),
            row.get("attributed_hypothesis"),
            row.get("selected_candidate_index"),
            row.get("rq_proxy"),
            row.get("generation_path"),
            row.get("generator_family"),
            row.get("llm_generation_call_count"),
        )
        for row in summaries
    ],
)
""",
                "display",
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    destination = DEFAULT_OUTPUT
    existing_notebook = None
    if destination.exists():
        existing_notebook = json.loads(destination.read_text(encoding="utf-8"))
    destination.write_text(
        json.dumps(
            build_notebook(existing_notebook),
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
