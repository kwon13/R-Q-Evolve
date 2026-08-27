# R-Q-Evolve

`evo-sample`의 전체 아이디어를 직접 구현해보기 위한 교육용 추상화 프로젝트입니다. 원본의 `verl/` 폴더는 복제하지 않았고, pip로 설치한 `verl`에 붙일 경계는 `src/rq_evolve/verl_adapter.py` 하나로 분리했습니다.

새 production pipeline은 `GROUP × SKILL` 대신 Omni-MATH top-level
`DOMAIN(7) × PROBLEM_TYPE(5)`의 full 35-cell MAP을 사용합니다. 이전 실험
결과는 분석용으로 보존하지만 새 archive에 resume하지 않습니다. descriptor,
mutation, verifier의 정확한 계약은 [Domain × Problem Type Pipeline](docs/DOMAIN_TYPE_PIPELINE.md)을
참조하세요.

## Pipeline Map

```text
seed_programs_domain_type/*.py
  -> ProblemProgram.execute(seed)
  -> DOMAIN declaration + deterministic PROBLEM_TYPE checks
  -> generic validity / declarative-verifier checks
  -> MAPElitesArchive.try_insert()   # curated seed coordinate
  -> parent selection
  -> stage 1: untargeted child-family design
  -> stage 2: generator implementation + one DOMAIN declaration
  -> generic validity gates over every verification seed
  -> rule-based PROBLEM_TYPE derivation from statement + verifier
  -> backend.rollout(G)
  -> R_Q = p_hat * (1 - p_hat) * uncertainty
  -> novelty gates + full 7 x 5 MAP-Elites update   # no supported-cell mask
  -> DynamicProblemDataset refresh
  -> VerlTrainerAdapter.fit()
```

## What Is Abstracted

- `program.py`: 문제 생성 프로그램의 실행 단위
- `archive.py`: DOMAIN × PROBLEM_TYPE MAP-Elites archive (7×5 = 35셀)
- `scoring.py`: `R_Q = p(1-p)U`
- `prompts.py`: untargeted family design 및 generator/DOMAIN contract
- `concepts.py`: Omni top-level DOMAIN(7)과 computational PROBLEM_TYPE(5)
- `problem_type.py`: statement와 verifier를 함께 보는 보수적 규칙 기반 type 판정
- `verifier.py`: `expression | boolean | one_of | set` 선언형 답안 계약
- `solver_trace.py`: rollout 위생 처리 (chat boundary 절단 후 재채점)
- `prompt_templates/`: two-stage mutation과 optional structural inspiration
- `backends.py`: LLM mutation과 solver rollout 인터페이스
- `evolution.py`: outer iteration, mutation, verification, scoring, dataset refresh
- `dataset.py`: champion에서 학습 문제를 만드는 framework-free dataset
- `reward.py`: `\boxed{}` 기반 verl reward function
- `verl_adapter.py`: pip 설치된 `verl` trainer 연결부

## Quick Start

```bash
cd /data1/yhoon113/R-Q-Evolve

# 의존성 없이 도는 sanity check (pyproject: pythonpath=src, testpaths=tests)
pytest

# import / 설정 preflight 점검 (GPU 불필요)
python scripts/preflight_check.py

# 새 35-cell production config 점검
python scripts/preflight_check.py \
  --config configs/rq_evolve_4b_8gpu_domain_type.yaml

# 실제 학습 스모크 (LoRA r32, 1 step; free GPU 필요)
python scripts/train_with_verl.py --smoke --config configs/rq_evolve_smoke_lora.yaml
```

## Domain × Problem Type Training

새 4B/8-GPU 런은 전용 launcher로 시작합니다.

```bash
bash scripts/run_train_domain_type_8gpu.sh
```

launcher는 `seed_programs_domain_type/`의 7개 bootstrap seed와 빈 output
directory를 확인한 뒤 8-GPU 학습 launcher를 호출합니다.
`configs/rq_evolve_4b_8gpu_domain_type.yaml`은 target을 주지 않는 two-stage
mutation, stage-2의 단일 DOMAIN 선언, statement+verifier 규칙으로 결정하는
PROBLEM_TYPE, 전체 35-cell grid, checkpoint resume 비활성화를 고정합니다. 좌표
판정을 위한 별도 모델·API·credential은 필요하지 않습니다.

## verl Training

현재 production 학습은 Domain × Problem Type 전용 config를 사용합니다. 모델
경로·GPU 수를 바꿀 때는 해당 config의 inline `verl_config`를 수정합니다.

```bash
python scripts/train_with_verl.py --print-verl-env          # 환경이 잡은 verl 버전 확인
python scripts/train_with_verl.py \
  --config configs/rq_evolve_4b_8gpu_domain_type.yaml
```

학습 중에는 현재 Python 환경에서 import되는 `verl`의 `RayPPOTrainer`가
solver update를 맡고, `R-Q-Evolve`의 sampler가 epoch 시작마다 archive
re-evaluation, mutation, R_Q scoring, dataset refresh를 실행합니다.

## Historical 4B Structural-Inspiration Notes

아래 내용은 이전 `GROUP × SKILL` 실행 당시의 기록입니다. 현재 source schema와
호환되는 launch guide가 아닙니다. 기존 result/config는 분석과 재현 근거로만
보존합니다. 새 Domain × Problem Type production run은 structural inspiration을
끈 clean config와 위 전용 launcher만 사용합니다.

## Docs

- `docs/DOMAIN_TYPE_PIPELINE.md`: 새 7×5 descriptor, untargeted mutation, verifier,
  admission 및 fresh-run contract
- `docs/PIPELINE.md`: 원본 코드 ↔ 새 스켈레톤 대응표 + evolution 상태 전이
- `docs/GRADING.md`: grader 3종(학습 reward / trainer val / offline eval) 비교표
- `docs/EVOLVED_PERFORMANCE.md`: Seed-ID 및 Structural-OOD checkpoint benchmark
- `docs/RUNTIME_NOTES.md`: 장비별 운영 노트(OOM, vLLM cumem crash, 메모리 튜닝 근거)
- `docs/async_pipeline.md`, `docs/deepseek_support_plan.md`: async rollout / DeepSeek 지원 계획
- `docs/EVOLVED_PERFORMANCE.md`: 고정 240문제 Seed-ID checkpoint 평가 + inner/outer evolution 그래프

## Important Contracts

생성 프로그램은 아래 형식을 지켜야 합니다.

```python
DOMAIN = "algebra"  # 허용 어휘 중 정확히 하나

def generate(seed):
    ...
    return problem_text, answer_text, {"mode": "expression"}
```

stage 1에는 DOMAIN/PROBLEM_TYPE 어휘나 목표 cell이 주어지지 않습니다. stage 2는
이미 확정된 child family를 코드로 옮기면서 허용 어휘의 DOMAIN 하나만 선언하며,
특정 값으로 바꾸라는 지시는 받지 않습니다. `PROBLEM_TYPE`은 source가 선언하지
않고, 각 verification seed의 visible statement와 verifier에서 코드가 보수적으로
판정합니다. 모든 seed가 같은 high-confidence type을 내지 않으면 candidate를
거절합니다. verifier는 실행 가능한 임의 predicate가 아니라 JSON-safe 답안 계약과
type 교차검증 근거입니다.

`EvolutionBackend`는 두 메서드만 구현하면 됩니다.

```python
backend.mutate(tasks) -> list[str | None]
backend.rollout(instances, n_rollouts) -> list[list[RolloutRecord]]
```

production은 기존 mutation/solver backend만 사용합니다. archive 좌표를 정하기
위한 별도 분류 모델이나 외부 API 호출은 없습니다.
