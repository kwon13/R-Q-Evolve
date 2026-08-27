# Pipeline Notes

이 문서는 현재 `DOMAIN × PROBLEM_TYPE` implementation의 코드 대응표와 상태
전이를 요약합니다. descriptor 정의와 설계 근거는
[DOMAIN_TYPE_PIPELINE.md](DOMAIN_TYPE_PIPELINE.md)가 기준입니다.

## Source-to-code mapping

| 역할 | 구현 |
|---|---|
| generator 실행 단위와 sandbox 경계 | `src/rq_evolve/program.py`, `src/rq_evolve/_sandbox_worker.py` |
| 두 descriptor vocabulary | `src/rq_evolve/concepts.py` |
| full 7×5 MAP-Elites archive | `src/rq_evolve/archive.py` |
| untargeted stage 1, Stage-2 trusted core, DOMAIN labeler | `src/rq_evolve/prompts.py`, `prompt_templates/` |
| deterministic PROBLEM_TYPE 판정 | `src/rq_evolve/problem_type.py` |
| outer evolution loop | `src/rq_evolve/evolution.py` |
| declarative verifier schema와 reward dispatch | `src/rq_evolve/verifier.py`, `src/rq_evolve/reward.py`, `src/rq_evolve/_grader_worker.py` |
| champion/replay → training rows | `src/rq_evolve/dataset.py` |
| installed `verl` trainer 경계 | `src/rq_evolve/verl_adapter.py`, `src/rq_evolve/verl_backend.py` |
| production config와 launcher | `configs/rq_evolve_4b_8gpu_domain_type.yaml`, `scripts/run_train_domain_type_8gpu.sh` |

repository는 `verl/`를 vendoring하지 않습니다. 현재 Python environment에서
import되는 `verl`이 solver update를 담당합니다.

## Initialization

`RQEvolver.initialize_archive()`는 `seed_programs_domain_type/*.py`를 읽고 각
generator를 여러 seed에서 실행·검증한 뒤 한 번 평가합니다. 사람이 검수한 seed의
단일 `DOMAIN` 선언과 statement/verifier 계약으로부터 bootstrap 좌표를 정합니다.
일곱 seed는 일곱 domain과 다섯 problem type을 모두 한 번 이상 열지만, 허용 가능한
cell을 제한하지 않습니다.

archive는 시작할 때 35개 cell을 모두 생성합니다. 좌표는
`(domain_bin, problem_type_bin)`이며 runtime supported mask나 benchmark 빈도
threshold는 없습니다.

## Outer iteration

```text
sync solver weights into rollout backend
        │
        ├─ re-evaluate current champions on fresh instances
        │    └─ refresh R_Q = s_hat(1-s_hat)U
        │
        ├─ sample parents from occupied cells
        │
        ├─ stage 1: propose a structurally new problem family
        │    └─ no descriptor vocabulary, target cell, held/moved axis
        │
        ├─ stage 2: implement build_instance(rng)
        │    ├─ sees descriptor-free parent program/family as transformation context
        │    └─ trusted assembler owns generate(seed), renderer, and verifier shell
        │
        ├─ static/AST/multi-seed validity checks
        │
        ├─ local descriptor assignment and validation
        │    ├─ 7 restricted YES/NO arms directly assign one DOMAIN
        │    ├─ statement + verifier deterministically derive PROBLEM_TYPE
        │    └─ missing/ambiguous score or type disagreement ──────> reject
        │
        ├─ solver rollouts on fresh instances and R_Q scoring
        │
        ├─ novelty gates and same-cell champion competition
        │
        └─ refresh replay-backed training dataset and run one solver update
```

stage 1은 archive의 빈 cell이나 descriptor 목표를 보지 않습니다. stage 2는 이미
확정된 family를 구현할 뿐 DOMAIN을 선언하지 않습니다. 그 뒤 같은 로컬 policy에
7개 candidate-domain YES/NO 질문을 한 batch로 보내 argmax 하나를 직접 라벨로
사용합니다. PROBLEM_TYPE 판정은 각 rendered problem과 normalized verifier에 대한
고정된 로컬 규칙입니다. 별도 모델이나 외부 API를 추가하지 않습니다.

## Candidate state

`inner_iteration_batch()`의 entry는 별도 enum 대신 현재 가진 key로 상태를
표현합니다.

| entry shape | 의미 |
|---|---|
| `{"task", "child", "inst"}` | generic/descriptor verification을 통과해 rollout 대상으로 살아 있음 |
| `{"_retry": {...}}` | source parse는 됐지만 verification 실패; 현재 two-stage 경로에서는 terminal report로 변환 |
| `{"report": CandidateReport}` | terminal outcome |

production Domain×Type config는 `fix_retry: false`입니다. terminal report에는
가능하면 child source와 AST finding을 함께 남겨, archive에 들어오지 못한 후보도
gate별로 감사할 수 있게 합니다. 대표 결과는 mutation/parse 실패, verify 실패,
DOMAIN labeling 실패, duplicate/copy 거절, rollout 실패, insertion,
non-elite rejection입니다.

## Admission order

generated child는 destination을 받지 않은 채 다음 순서로 확인됩니다.

1. strict MODE/CORE parsing, trusted assembler 조립, deterministic RNG;
2. `answer == check` dataflow/AST contract;
3. sandbox execution과 multi-seed statement/answer consistency;
4. declarative verifier normalization;
5. 모든 verification seed에서 statement+verifier 기반 PROBLEM_TYPE이
   high-confidence이고 하나로 일치하는지 확인;
6. prompt-example 및 optional structural-donor copy rejection;
7. 7개 restricted YES/NO score의 argmax·confidence로 DOMAIN 하나 할당;
8. fresh-instance rollout과 R_Q 계산;
9. seed variation 및 behavior/template/near-template/structural duplicate gates;
10. 해당 cell champion과 strict score competition.

DOMAIN labeler는 stage 1/2에 원하는 cell을 전달하지 않으므로 mutation destination을
지시하는 scheduler가 아닙니다. candidate가 parent와 같은 cell로 돌아와도
구조적으로 새롭고 점수가 더 높으면 정상적인 mutation입니다.

## Answer verification path

`ProblemInstance.verifier`는 dataset, replay, rollout log, verl reward metadata를
통해 보존됩니다. reward는 solver response의 마지막 `\boxed{...}`를 추출한 뒤
mode별로 dispatch합니다.

- `expression`: `math_verify`
- `boolean`: canonical Yes/No equality
- `one_of`: 허용된 완전한 답 중 하나와 symbolic equality
- `set`: 순서 무관, 중복 없는 exact finite-set equality

grader는 별도 hard-killable process에서 실행됩니다. schema에 없는 field, unknown
mode, executable predicate, reference와 모순되는 verifier는 fail closed입니다.
PROBLEM_TYPE은 mode만으로 정하지 않습니다. statement의 출력 요구를 보수적으로
판정한 뒤 verifier mode 및 reference answer 제약과 교차검증합니다. 한 seed라도
모호하거나 서로 다른 type을 만들면 candidate 전체를 거절합니다.

## Archive persistence

snapshot metadata는 schema/version뿐 아니라 다음 값을 정확히 저장합니다.

```text
axes = [domain, problem_type]
domain_labels = seven-value ordered vocabulary
problem_type_labels = five-value ordered vocabulary
binning = grid
domain_authority = source_exact_one_literal
problem_type_authority = deterministic_statement_and_verifier
problem_type_ruleset = computational-output-contract-v1
problem_type_ruleset_sha256 = hash of the local rule implementation
```

각 program에는 source hash와 같은 ruleset identity를 포함한
`descriptor_contract`도 저장합니다. load 시 snapshot 또는 program contract 중
하나라도 다르면 live archive를 바꾸기 전에 `ArchiveSchemaError`가 납니다.
따라서 old `GROUP × SKILL` champion을 새 좌표로 암묵적으로 재해석하지 않습니다.
production launcher도 non-empty output directory를 거절하고 trainer
`resume_mode: disable`을 사용합니다.

## Production entry point

```bash
python scripts/preflight_check.py \
  --config configs/rq_evolve_4b_8gpu_domain_type.yaml

bash scripts/run_train_domain_type_8gpu.sh \
  --gpus 0,1,2,3,4,5,6,7 --detach

# Four selected GPUs; --detach uses nohup and also starts checkpoint auto-merge.
bash scripts/run_train_domain_type_4gpu.sh --gpus 0,1,2,3 --detach
```

launcher의 exact seed-count와 output-directory 검사를 우회해 기존 결과 directory에
새 schema를 섞지 마세요. 좌표 판정에는 별도 credential이 필요하지 않습니다.
