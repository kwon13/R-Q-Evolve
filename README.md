# R-Q-Evolve

`evo-sample`의 전체 아이디어를 직접 구현해보기 위한 교육용 추상화 프로젝트입니다. 원본의 `verl/` 폴더는 복제하지 않았고, pip로 설치한 `verl`에 붙일 경계는 `src/rq_evolve/verl_adapter.py` 하나로 분리했습니다.

## Pipeline Map

```text
seed_programs/*.py
  -> ProblemProgram.execute(seed)
  -> verify_program()
  -> MAPElitesArchive.try_insert()   # cell = (GROUP, SKILL)
  -> parent selection
  -> stage 1: child-family design (+ optional structural inspiration)
  -> stage 2: generator implementation   # donor is absent here
  -> generated ProblemProgram
  -> blind GROUP/SKILL readback
  -> optional judge(problem, answer) == resolved (GROUP, SKILL)?
  -> backend.rollout(G)
  -> R_Q = p_hat * (1 - p_hat) * uncertainty
  -> MAP-Elites update
  -> DynamicProblemDataset refresh
  -> VerlTrainerAdapter.fit()
```

## What Is Abstracted

- `program.py`: 문제 생성 프로그램의 실행 단위
- `archive.py`: GROUP × SKILL MAP-Elites archive (6×8 = 48셀)
- `scoring.py`: `R_Q = p(1-p)U`
- `prompts.py`: mutation / solver prompt builder
- `concepts.py`: GROUP(6) × SKILL(8) 라벨 vocabulary — 두 축은 독립
- `solver_trace.py`: rollout 위생 처리 (chat boundary 절단 후 재채점)
- `prompt_templates/`: two-stage mutation, structural-inspiration and label prompts
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

## 4B Structural Inspiration

주 실행 설정은 target-cell 지시를 쓰지 않습니다. Primary parent와 다른
lineage에서, 가능하면 GROUP과 SKILL도 모두 다른 champion을 하나 뽑고 그
프로그램의 **문제 문장 skeleton만** stage 1에 제공합니다. Source, 정답/check,
라벨, 점수와 metadata는 숨겨지며 stage 2에는 donor가 들어가지 않습니다.
완성된 child의 두 라벨은 blind readback하고, donor 복제는 rollout 전에
전용 gate로 거절합니다.

```bash
# 이 머신의 verl/vLLM 학습 환경
conda activate azr-bw-blackwell

# GPU 없이 양쪽 설정 점검
python scripts/preflight_check.py \
  --config configs/rq_evolve_4b_4gpu_structural_inspiration.yaml
python scripts/preflight_check.py \
  --config configs/rq_evolve_4b_4gpu_structural_control.yaml

# treatment (4 GPU)
bash launch_4b_train.sh \
  --config configs/rq_evolve_4b_4gpu_structural_inspiration.yaml \
  --gpus 0,1,2,3

# certified donor-v2: manual seed donors + positive-R_Q + Jaccard gate
bash scripts/run_4b_certified_donor.sh --gpus 0,1,2,3

# SSH 종료 후에도 돌리려면
bash scripts/run_4b_certified_donor.sh --gpus 0,1,2,3 --detach
```

Certified donor-v2는 수동 검증 allowlist에 들어간 seed 중 현재 `R_Q > 0`인
항목만 structural donor로 사용합니다. 인증 seed는 별도 registry에 보존되므로
나중에 child가 해당 MAP cell의 champion을 교체해도 donor 자격은 사라지지
않습니다. 생성 child는 AST/실행/32-seed 검증을
통과하면 MAP과 primary-parent 경로에는 참여하지만 structural donor로 승격되지
않습니다. Donor-child statement token Jaccard가 0.45 이상이면 rollout 전에
복제로 거절합니다. 외부 API, 별도 evaluator 모델, local-policy self-judge는 모두
사용하지 않습니다. Runner는 config 불변조건, checkpoint auto-merge, 학습만
실행합니다.

`save_freq: 32`이므로 처음에는 `global_step_32/actor`를 resume용으로 그대로
두고, step 64가 저장되면 step 32를 `hf_merged/`로 변환ㆍ검증한 뒤 step 32의
`actor/`만 삭제합니다. 이후에도 항상 최신 actor 하나는 보존됩니다.
merge 상태는 다음처럼 확인할 수 있습니다.

```bash
tail -f logs/rq_evolve_4b_4gpu_certified_structural_inspiration_v2/auto_merge.log
```

학습이 완전히 끝난 뒤에도 마지막 checkpoint의 actor는 의도적으로 남습니다.
resume 가능성을 버리고 마지막 것까지 HF로 합칠 때만
`scripts/merge_fsdp_to_hf.py`를 수동 실행하세요. 자동화를 끄고 싶을 때는
launcher에 `--no-auto-merge`를 추가할 수 있습니다.

V2는 unrestricted donor-v1 checkpoint를 resume하지 않고 base model에서 새
output으로 시작합니다. 첫 실행 때 effective config, prompt, seed corpus와 구현
hash가 output의 `run_contract/`에 동결되며, 다른 방법으로 같은 디렉터리를
resume하면 Ray가 시작되기 전에 실패합니다.

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
- `docs/EVOLVED_PERFORMANCE.md`: Seed-ID 및 Structural-OOD checkpoint benchmark
- `docs/RUNTIME_NOTES.md`: 장비별 운영 노트(OOM, vLLM cumem crash, 메모리 튜닝 근거)
- `docs/async_pipeline.md`, `docs/deepseek_support_plan.md`: async rollout / DeepSeek 지원 계획
- `docs/EVOLVED_PERFORMANCE.md`: 고정 240문제 Seed-ID checkpoint 평가 + inner/outer evolution 그래프

## Important Contracts

생성 프로그램은 아래 형식을 지켜야 합니다.

```python
def generate(seed):
    ...
    return problem_text, answer_text

GROUP = "algebra"
SKILL = "transformation"
```

`EvolutionBackend`는 두 메서드만 구현하면 됩니다.

```python
backend.mutate(tasks) -> list[str | None]
backend.rollout(instances, n_rollouts) -> list[list[RolloutRecord]]
```

이 두 경계만 맞추면 OpenAI API, vLLM, Ollama, 또는 `verl` 내부 rollout worker로 바꿔 끼울 수 있습니다.
