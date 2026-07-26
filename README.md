# R-Q-Evolve

`evo-sample`의 전체 아이디어를 직접 구현해보기 위한 교육용 추상화 프로젝트입니다. 원본의 `verl/` 폴더는 복제하지 않았고, pip로 설치한 `verl`에 붙일 경계는 `src/rq_evolve/verl_adapter.py` 하나로 분리했습니다.

## Pipeline Map

```text
seed_programs/*.py
  -> ProblemProgram.execute(seed)
  -> verify_program()
  -> MAPElitesArchive.try_insert()
  -> parent selection
  -> reuse correct/wrong traces from existing R_Q rollouts
  -> metacognitive MutationPlan JSON
  -> backend.mutate(plan)
  -> generated ProblemProgram
  -> backend.rollout(G)
  -> R_Q = p_hat * (1 - p_hat) * uncertainty
  -> MAP-Elites update
  -> DynamicProblemDataset refresh
  -> VerlTrainerAdapter.fit()
```

## What Is Abstracted

- `program.py`: 문제 생성 프로그램의 실행 단위
- `archive.py`: H축 × D축 MAP-Elites archive
- `scoring.py`: `R_Q = p(1-p)U`
- `prompts.py`: mutation / solver prompt builder
- `metacognition.py`: rollout evidence, pre-update delta-p, operator controller
- `prompt_templates/`: legacy and metacognitive depth/breadth prompts + shots
- `backends.py`: LLM mutation과 solver rollout 인터페이스
- `evolution.py`: outer iteration, mutation, verification, scoring, dataset refresh
- `dataset.py`: champion에서 학습 문제를 만드는 framework-free dataset
- `reward.py`: `\boxed{}` 기반 verl reward function
- `verl_adapter.py`: pip 설치된 `verl` trainer 연결부

## Quick Start

```bash
cd "/Users/kyhoon13/Desktop/Code/수식 증명/R-Q-Evolve"

# 의존성 없이 도는 sanity check (pyproject: pythonpath=src, testpaths=tests)
pytest

# import / 설정 preflight 점검 (GPU 불필요)
python scripts/preflight_check.py

# 실제 학습 스모크 (LoRA r32, 1 step; free GPU 필요)
python scripts/train_with_verl.py --smoke --config configs/rq_evolve_smoke_lora.yaml
```

## verl Training

`verl` 학습을 켜려면 `configs/rq_evolve_base.yaml`에서 `verl.enabled: true`로
바꾸고, 같은 파일의 inline `verl_config.actor_rollout_ref.model.path`를 실제
Hugging Face 모델 경로로 수정합니다. (별도의 `verl_ppo_rq.yaml`은 없고, verl
PPO 오버라이드가 각 `rq_evolve_*.yaml`의 `verl_config:` 블록에 인라인돼 있습니다.)

```bash
python scripts/train_with_verl.py --print-verl-env          # 환경이 잡은 verl 버전 확인
python scripts/train_with_verl.py --config configs/rq_evolve_base.yaml
```

학습 중에는 현재 Python 환경에서 import되는 `verl`의 `RayPPOTrainer`가
solver update를 맡고, `R-Q-Evolve`의 sampler가 epoch 시작마다 archive
re-evaluation, mutation, R_Q scoring, dataset refresh를 실행합니다.

## Implementation Order

1. `tests/test_program.py`, `tests/test_scoring.py`가 통과하는지 확인
2. `archive.py`의 parent selection과 duplicate rejection을 원하는 방식으로 강화
3. `prompt_templates/`에 few-shot mutation prompt 작성
4. `backends.py`에 실제 mutation backend 추가
5. `backends.py` 또는 `verl_adapter.py`에 solver rollout + entropy 계산 연결
6. `python scripts/train_with_verl.py --print-verl-env`로 실제 `verl` 버전 확인

## Docs

- `docs/PIPELINE.md`: 원본 코드 ↔ 새 스켈레톤 대응표 + evolution 상태 전이
- `docs/GRADING.md`: grader 3종(학습 reward / trainer val / offline eval) 비교표
- `docs/RUNTIME_NOTES.md`: 장비별 운영 노트(OOM, vLLM cumem crash, 메모리 튜닝 근거)
- `docs/async_pipeline.md`, `docs/deepseek_support_plan.md`: async rollout / DeepSeek 지원 계획

## Important Contracts

생성 프로그램은 아래 형식을 지켜야 합니다.

```python
def generate(seed):
    ...
    return problem_text, answer_text

CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.quadratic"
```

`EvolutionBackend`는 두 메서드만 구현하면 됩니다.

```python
backend.mutate(tasks) -> list[str | None]
backend.rollout(instances, n_rollouts) -> list[list[RolloutRecord]]
```

이 두 경계만 맞추면 OpenAI API, vLLM, Ollama, 또는 `verl` 내부 rollout worker로 바꿔 끼울 수 있습니다.
