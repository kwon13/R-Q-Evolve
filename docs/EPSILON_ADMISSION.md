# R_Q 경쟁 + ε-admission 실행

기존 Standard와 동일한 모델별 8-GPU 설정에서, 검증을 통과한 신규 후보의
occupied-cell 교체 규칙만 바꾼다. 더 높은 기존 selection priority를 갖는 후보는
항상 교체하고, 동점 또는 낮은 점수의 후보는 `admission_epsilon` 확률로 교체한다.
빈 셀은 기존처럼 채운다. 유효성/중복/다양성 검증, 챔피언 재평가, `R_Q = L × U`,
부모 셀 균등 선택, Replay 구성은 그대로 유지한다. `archive.epsilon`은 별개의
부모 선택 설정이며 이번 입성 확률이 아니다.

기존에 허용되던 `R_Q=0` 후보도 유효성 검증을 통과하면 입성 경쟁에 참여할 수 있다.
아카이브 입성과 실제 학습 배치 포함은 다르며, 학습에는 기존 frontier 조건이 적용된다.

`0.25`는 이번 실험에서 고정한 후보값이다. 기존 논문의 추천 ε를 재현하는 설정으로
기술하지 않는다. 기존 Standard는 나머지 조건이 일치할 때 확률적 수용을 제거한
대조군으로 쓸 수 있다. 기존 No-U/Random Cell은 각 조건에서 얻은 별도 비교 결과다.

## 서버 실행

서버에 변경한 코드와 config 세 개를 함께 동기화하고, R-Q-Evolve 루트에서 해당
서버의 기존 학습 환경을 활성화한다. 진행 중인 학습이 끝난 뒤 실행한다.

4B, 80-GB A100 8개, ε=0.25:

```bash
bash scripts/run_train_domain_type_epsilon_8gpu.sh --gpus 0,1,2,3,4,5,6,7 --dry-run
bash scripts/run_train_domain_type_epsilon_8gpu.sh --gpus 0,1,2,3,4,5,6,7 --detach
```

8B, 80-GB A100 8개, ε=0.25:

```bash
bash scripts/run_train_domain_type_epsilon_8gpu.sh --model-size 8b --profile a100 --gpus 0,1,2,3,4,5,6,7 --dry-run
bash scripts/run_train_domain_type_epsilon_8gpu.sh --model-size 8b --profile a100 --gpus 0,1,2,3,4,5,6,7 --detach
```

8B, 96-GB RTX PRO 6000 Blackwell Server Edition 8개:

```bash
bash scripts/run_train_domain_type_epsilon_8gpu.sh --model-size 8b --profile rtxpro6000 --gpus 0,1,2,3,4,5,6,7 --dry-run
bash scripts/run_train_domain_type_epsilon_8gpu.sh --model-size 8b --profile rtxpro6000 --gpus 0,1,2,3,4,5,6,7 --detach
```

다른 ε를 쓰려면 `--epsilon 0.1`처럼 지정한다. 유한한 `[0, 1]` 값만 허용한다.
ε=0은 엄격한 점수 경쟁이고, ε=1은 검증을 통과한 신규 후보의 항상 교체다.
ε는 전체 교체율이 아니라 **점수 경쟁에서 이기지 못한 후보의 추가 수용 확률**이다.

A100 모델 기본 경로는 `/data1/yhoon113/qwen3-4b-base`와
`/data1/yhoon113/qwen3-8b-base`, RTX 8B 기본값은 `Qwen/Qwen3-8B-Base`다.
경로가 다르면 동일한 명령에 `--model-path /server/path/to/model`을 추가한다.
`--dry-run`은 모델을 로드하거나 GPU/서버 환경을 검사하지 않으며, Python과 OmegaConf로
설정·GPU 인자·실험 경로를 검사한다. 실제 GPU 메모리 적합성은 서버에서 확인해야 한다.
`--detach --dry-run`도 학습/병합 프로세스를 시작하지 않는다.

## 결과, 로그, 중단

모델 크기, ε, 하드웨어 profile이 output/W&B 이름에 들어간다. 기본 4B 예시는:

```text
rq_output/rq_evolve_4b_domain_type_epsilon_0.25_35cell_8gpu_a100/
logs/rq_evolve_4b_domain_type_epsilon_0.25_35cell_8gpu_a100/
```

기존 Standard 디렉터리를 재사용하지 않는다. 같은 새 실험 디렉터리가 이미 비어 있지
않으면 실행을 거부한다. 모든 profile은 기존과 동일하게 256 step, 32-step 평가/저장을
사용한다. 종료 시점의 동일 step 점수와 학습 곡선을 비교하고 최고점은 보조로 기록한다.
교체 규칙 변경은 adaptive refill 종료 시점에도 영향을 줄 수 있으므로, 생성 후보/토큰
예산도 기록해 실제 연산량 차이를 확인한다.

매 iteration의 evolution 로그/통계에는 다음 admission 진단값을 기록한다:

| 필드 | 의미 |
| --- | --- |
| `epsilon_score_wins` | 점수가 더 높아 교체된 후보 수 |
| `epsilon_nonwinning_candidates` | 동점/낮은 점수로 ε 수용 대상이 된 유효 후보 수 |
| `epsilon_overrides` | ε 수용으로 교체된 후보 수 |
| `epsilon_override_rate` | overrides / nonwinning candidates (대상이 없으면 0) |
| `total_epsilon_replacements` | ε 수용으로 교체된 누적 횟수 |
| `archive_admission_epsilon` | 실행에 사용한 ε |

한 iteration의 비율이 정확히 ε일 필요는 없다. 점수 승리·빈 셀 입성·검증 탈락은
ε 추첨 횟수를 소비하지 않으며, ε=0에서도 추첨하지 않는다.

Detached 실행 로그는 명령이 출력하는 경로를 사용한다. 기본 4B:

```bash
tail -f logs/rq_evolve_4b_domain_type_epsilon_0.25_35cell_8gpu_a100/latest.log
```

필요하면 해당 run의 PID를 확인한 뒤 TERM 신호를 보낸다:

```bash
RUN_LOG_DIR=logs/rq_evolve_4b_domain_type_epsilon_0.25_35cell_8gpu_a100
ps -p "$(cat "$RUN_LOG_DIR/train.pid")" -o pid=,args=
kill -TERM "$(cat "$RUN_LOG_DIR/train.pid")"
ps -p "$(cat "$RUN_LOG_DIR/auto_merge.pid")" -o pid=,args=
kill -TERM "$(cat "$RUN_LOG_DIR/auto_merge.pid")"
```

다음 실행 전 `nvidia-smi`로 해당 학습의 GPU workers도 종료되었는지 확인한다.
Foreground 실행은 Ctrl-C로 중단한다.

## 체크포인트와 재개 범위

실행 스크립트는 **fresh-only**이며 `--resume`을 지원하지 않는다. 기존 Standard
체크포인트에서 ε를 켜고 이어 학습한 결과를 fresh ε 어블레이션으로 쓰지 않는다.
학습은 기존 actor 체크포인트와 아카이브를 저장하며, 새 run은 `run_contract/`에
해결된 설정과 코드/프롬프트/seed 해시를 함께 남긴다. ε 실행 자체를 복구해야 하면
해당 run의 저장된 계약과 같은 ε/코드/설정을 사용하여 별도 resume 설정을 검증해야 한다.
아카이브 snapshot에는 admission 설정과 난수 진행 상태가 보존된다.

## 로컬 검증

아래 핵심 검사는 총 158개가 통과했다. 실행기 검사는 세 모델/profile의 설정 비교,
GPU 없는 dry-run과 격리된 모의 trainer/merge를 사용한 실제 detach 분기를 포함한다.
실제 GPU 학습이나 서버 메모리 사용량을 측정한 결과는 아니다.

```bash
python3 -m pytest -q tests/test_archive.py tests/test_archive_axes.py tests/test_epsilon_integration.py tests/test_run_contract.py tests/test_evolution_guard.py tests/test_replay_and_constancy.py tests/test_degenerate_frontier.py tests/test_epsilon_launch.py
bash -n scripts/run_train_domain_type_epsilon_8gpu.sh scripts/run_train_domain_type_8gpu.sh
```

추가 회귀 검사에서 `test_certified_donor.py` 4개와 `test_lora_config.py` 1개는
기존 설정 파일 누락, 4-GPU training_budget 기본값, Random Cell 128/256-step 기대값,
RTX 모델 경로 대소문자 불일치로 실패했다. 수정 전 HEAD를 별도 디렉터리에 추출해
동일한 5개 실패를 재현했다. 해당 기존 실험 설정은 이번 변경에서 수정하지 않았다.
