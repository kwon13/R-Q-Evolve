# Reasoning-informed Evolver 확장 가설 실험

이 실험은 다음 두 주장을 의도적으로 분리한다.

1. **Generation-space expansion**: frozen Solver의 representation에서
   reasoning-informed child가 plain expansion subspace로 설명되지 않는
   방향으로 더 이동하는가?
2. **Capability-space expansion**: 같은 checkpoint에서 같은 compute로
   학습한 뒤, reasoning-informed 조건이 독립 held-out reasoning move에 더
   잘 transfer하는가?

첫 번째 결과만으로 두 번째를 주장하지 않는다. 최종 판정은
`generation_report.json`과 `capability_report.json`의 claim gate가 모두
통과했는지로 한다.

## 현재 notebook 데이터의 판정

`rq_output/mutation_method_comparison_notebook`에는 independent evolution
run 1개, parent 1개, operator/method별 generator draw 1개가 있다. 각
generator의 seed 5개와 seed당 Solver rollout 10개는 nested observation이지
독립 replicate가 아니다.

| cell | 현재 상태 | confirmatory 사용 |
|---|---|---|
| in-depth / legacy | invalid code | 불가 |
| in-depth / metacognitive | invalid plan | 불가 |
| in-breadth / legacy | code valid, evaluator 4/5 | 불가 |
| in-breadth / metacognitive | evaluator 5/5, breadth contract 위반 | 불가 |

특히 legacy breadth는 `sequence`로 이동했지만 metacognitive breadth는
parent와 같은 `algebra.linear_system_sum`이다. 이 둘의 거리를 그대로
비교하면 mutation method와 domain change가 confound된다. 그래서 현재
데이터의 strict matched generator pair는 0이고, 코드는 inferential CI와
확장 주장을 자동 차단한다.

현재 데이터로 만들어 둔 실제 pilot artifact는 다음 위치에 있다.

```text
rq_output/reasoning_expansion_pilot/
├── experiment_manifest.json
├── generator_manifest.jsonl
├── instance_manifest.jsonl
├── rollout_manifest.jsonl
├── representation_inputs.jsonl
├── representations.npz
├── representations.json
├── generation_confirmatory/
└── generation_diagnostic/
```

`generation_diagnostic`은 rejected/contract-invalid code를 포함한
**descriptive smoke test**일 뿐이다. 이 결과를 가설의 근거로 사용하면
안 된다.

수정한 uncentered primary geometry로 실제 Qwen3-8B
representation(block 23, masked mean)을 계산하면 diagnostic
generator-mean
\(O_{\text{reasoning}}-O_{\text{plain}}=+0.086289\)이다. block 22–24의
mean/last pooling 6개 설정도 모두 양수다. 반대로 centered-PC/raw-vector
조합에서는 부호가 음수였는데, 그 조합은 plain 평균 이동 방향을
직교 성분에 남기는 문제가 있어 primary에서 제외했다. 이처럼 현재
한 pair의 부호는 geometry 선택에도 민감하다. 더구나 plain PCA를 같은
pilot sample에 fit/evaluate했고, 두 child가 breadth contract 및 생성
compute상 동등하지 않으며 실제 scalar \(R_Q\)도 없다. 따라서 양수라는
사실은 그림의 파란 화살표를 지지하는 증거가 아니라 pipeline
smoke-test 수치일 뿐이다.

## 환경

GPU/model 단계는 repository의 vLLM conda environment를 사용한다.

```bash
cd /data1/yhoon113/R-Q-Evolve
PY=/data1/yhoon113/miniforge3/envs/vllm/bin/python
```

사전등록 설정은 `configs/expansion_hypothesis.json`이다. main layer는
36-block Qwen3-8B에서 zero-based block 23(24번째 block), primary pooling은
masked prompt-token mean이다. block 22/24와 last-prompt-token pooling은
robustness analysis다. Subspace는 plain calibration displacement만
사용하며, raw displacement를 원점을 지나는 \(U_{\mathrm{plain}}\)에
투영한다는 정의에 맞춰 **uncentered SVD/PCA**의 누적 energy 0.95를
primary로 고정한다. Centered PCA는 plain 평균 이동 방향을 \(O\)에
잘못 남길 수 있으므로 sensitivity로만 보고한다.

이 설정은 이미 관찰한 notebook pilot의 사전등록이라고 소급해 부르지
않는다. 현재 pilot은 오직 구현 smoke test이며, 설정을 hash로 고정한 뒤
새로 생성하는 run/parent부터 prospective confirmatory 분석에 포함한다.
설정의 `preregistration_scope`도 기존 notebook artifact를 명시적으로
제외한다.

## 1. 각 independent run/parent의 artifact 정규화

```bash
$PY scripts/run_expansion_hypothesis.py prepare \
  --comparison-root rq_output/mutation_method_comparison_notebook \
  --output-dir rq_output/reasoning_expansion_pilot \
  --parent-program seed_programs/09_linear_algebra.py \
  --run-id notebook_pilot_001 \
  --solver-checkpoint /data1/yhoon113/qwen3-8b-base \
  --solver-checkpoint-sha256 <registry-or-operator-supplied-64-hex-digest>
```

이 단계는 invalid generator도 `generator_manifest.jsonl`에 보존한다.
invalid generator에 displacement 0을 대입하지 않는다. 원래 artifact에
명시적인 child-instance lineage가 없으므로 동일 child/parent seed pairing은
`same_seed_posthoc_assumption`으로 기록된다.
Checkpoint digest는 모델 경로를 식별자로 오인하지 않도록 외부 immutable
registry 또는 운영자가 제공해야 한다. 스크립트는 16GB checkpoint를
자동으로 hash하지 않으며, CLI 값이 없으면
`frozen_solver.checkpoint_sha256` 설정을 사용한다.

여러 run/parent를 준비한 뒤에는 합친다.

```bash
$PY scripts/run_expansion_hypothesis.py merge \
  --inputs \
    rq_output/expansion/run01_parent01 \
    rq_output/expansion/run01_parent02 \
    rq_output/expansion/run02_parent01 \
  --output-dir rq_output/expansion/confirmatory_merged
```

merge는 frozen Solver checkpoint ID/SHA, preregistration hash, prompt와
sampling protocol이 다르면 중단한다. 같은 independent run·parent에서
여러 generator draw를 합칠 때 parent identity/representation은 한 번만
남기고 child lineage는 draw별로 보존한다. Parent의 Solver 점수는 identity
필드가 아니므로 draw별 원값을 `source_manifests.parent_score_telemetry`에
보존한 뒤 merged parent 행에서는 rollout 수에 따라 pooled
`p_hat`/uncertainty/\(R_Q\)를 다시 계산한다. Problem text나 concept처럼
parent 의미가 충돌하거나 child ID가 중복되면 merge는 실패한다.

## 2. Frozen Solver representation 추출

representation input manifest에는 system I/O instruction과 visible
`problem_text`만 남는다. answer, correct/wrong trace, mutation plan,
generator code, evaluator rationale, RQ metadata는 모델에 들어가지 않는다.

```bash
$PY scripts/run_expansion_hypothesis.py extract \
  --representation-inputs \
    rq_output/expansion/confirmatory_merged/representation_inputs.jsonl \
  --model /data1/yhoon113/qwen3-8b-base \
  --device cuda:4 \
  --dtype bfloat16 \
  --batch-size 8 \
  --max-prompt-tokens 2048 \
  --output rq_output/expansion/confirmatory_merged/representations
```

추출기는 `model.eval()`, `torch.inference_mode()`, dropout off,
`use_cache=False`, SDPA를 사용한다. 문제를 max length에 맞춰 조용히
자르지 않고 초과 시 실패한다. 저장 metadata에는 checkpoint/config/
tokenizer hash, prompt/token hash, zero-based layer index, pooling, dtype,
token count가 포함된다.

## 3. 실제 archive representation

현재 notebook 폴더에는 같은 Qwen3-8B 시점의 archive snapshot이 없다.
4B run archive를 대신 쓰거나 parent 5개를 “archive”라고 부르면 안 된다.
실제 \(\pi_t\) archive JSONL을 준비한 뒤 leakage field를 제거하고 같은
checkpoint로 추출한다.

```bash
$PY scripts/run_expansion_hypothesis.py prepare-archive \
  --archive-jsonl /path/to/frozen_pi_t_archive.jsonl \
  --output rq_output/expansion/archive_representation_inputs.jsonl

$PY scripts/run_expansion_hypothesis.py extract \
  --representation-inputs \
    rq_output/expansion/archive_representation_inputs.jsonl \
  --model /data1/yhoon113/qwen3-8b-base \
  --device cuda:4 \
  --output rq_output/expansion/archive_representations
```

Archive source row에는 stable `sample_id`와 `problem_text`가 필요하다.
`archive_generator_id`도 보존하면 generator-group leave-out sensitivity를
추가하기 쉽다.

## 4. Generation-space 분석

Confirmatory 실행:

```bash
$PY scripts/run_expansion_hypothesis.py analyze-generation \
  --instance-manifest \
    rq_output/expansion/confirmatory_merged/instance_manifest.jsonl \
  --representations \
    rq_output/expansion/confirmatory_merged/representations.npz \
  --archive-representations \
    rq_output/expansion/archive_representations.npz \
  --archive-kind frozen_pi_t_archive_snapshot \
  --mode confirmatory \
  --validity-policy strict \
  --run-sensitivity \
  --output-dir rq_output/expansion/generation_result
```

분석 순서는 다음과 같다.

1. parent ID를 calibration/evaluation으로 deterministic split한다.
2. calibration parent의 **plain displacement만** weighted uncentered
   SVD/PCA에 넣는다.
   generator마다 총 weight가 같아서 seed가 많은 generator가 PCA를
   지배하지 않는다.
3. held-out parent에서 \(A=\|U^\top\Delta z\|\)와
   \(O=\|(I-UU^\top)\Delta z\|\)를 계산한다.
4. instance metric을 먼저 generator로 평균한 뒤 condition pair를 만든다.
5. run → parent → generator-pair 순 hierarchical bootstrap을 수행한다.
6. RQ, Solver-token length, 표면 숫자 count/span/max-abs와 concept
   one-hot 차이를 사용한 paired ridge-adjusted robustness를 보고한다.
7. archive kNN median cosine novelty와 leave-one-out k-th distance의 0.95
   분위수로 coverage epsilon을 계산한다.

`--run-sensitivity`는 config에 고정된 PCA cumulative threshold/centering,
k와 coverage quantile의 전체 grid를 primary representation에서 실행하고
`sensitivity_summary.jsonl/csv`에 모두 기록한다. 이 결과는 primary
claim gate를 대신하거나 가장 좋은 설정을 사후 선택하는 데 쓰이지 않는다.

여기서 RQ는 학습 때 사용한 actor-logit entropy 기반 scalar objective여야
한다. Standalone vLLM 산출물의 mean-token negative-logprobability
`rq_proxy`는 진단 표에는 남기지만 RQ control을 충족한 것으로 취급하지
않으며, adjusted claim gate를 자동으로 실패시킨다.

현재 데이터의 smoke test만 재현하려면 다음 diagnostic 명령을 쓸 수 있다.

```bash
$PY scripts/run_expansion_hypothesis.py analyze-generation \
  --instance-manifest \
    rq_output/reasoning_expansion_pilot/instance_manifest.jsonl \
  --representations \
    rq_output/reasoning_expansion_pilot/representations.npz \
  --mode pilot \
  --validity-policy code_valid \
  --use-parent-surrogate-archive \
  --output-dir \
    rq_output/reasoning_expansion_pilot/generation_diagnostic
```

`--use-parent-surrogate-archive` 결과는
`parent_instances_surrogate_not_archive_snapshot`으로 명시되며,
archive-relative novelty 주장에 사용할 수 없다.

## 5. Confirmatory 데이터 수집 단위

다음 값은 최소 실행 gate이며 최종 parent 수는 pilot effect와
parent-level variance를 이용한 power analysis로 다시 정한다.

- independent evolution run: 최소 3, 가능하면 5 이상
- held-out parent: 최소 20개
- parent당 paired generator draw: 최소 3개
- generator당 shared instance seed: 최소 5개
- plain calibration generator: `max(20, 5 × retained PCA rank)` 이상
- archive: 서로 다른 generator family를 포함해 권장 50개 이상

두 condition은 parent, operator, concept contract, iteration, generator
call count, token/call budget, sampling seed range, instance 수 및 acceptance
budget이 같아야 한다. 생성 성공률 자체는 별도의 end-to-end yield로
보고하고, conditional geometry에서는 양쪽이 모두 유효한 matched
generator만 사용한다.

새 generator draw를 수집할 때
`scripts/compare_mutation_methods_vllm.py --plain-baseline two_stage
--llm-seed <draw-seed>`를 명시한다. 이것이 현재 기본값이며 plain과
reasoning 모두 plan+code 2회를 사용한다. 두 planner의 schema,
temperature/top-p/max-token 및 code prompt는 같고, reasoning 조건만
clean Solver contrast를 추가로 받는다. 이 contrast의 단순 길이 효과를
줄이기 위해 plain plan prompt에는 정보가 없는 padding을 넣어 두 plan
입력 token 수를 맞춘다. 생성된 plan 자체의 길이 차이로 code 입력이
달라질 수 있으므로 그 차이는 저장하며, 64 token을 넘는 pair는
confirmatory compute-parity gate에서 제외한다.

출력 `manifest.json`은 comparison design, evidence gate, model, prompt
hash, stage별 temperature/top-p/token budget, evaluation seed와
`llm_seed`, 그리고 operator/stage별 paired request seed를 저장한다.
각 draw는 별도 output directory에 저장한 뒤
`prepare`/`merge`한다. `one_stage_legacy`는 과거 artifact 재현용이며
confirmatory compute-parity gate를 통과할 수 없다. Adapter는 저장된
stage artifact와 실제/configured call count를 함께 기록하므로 manifest
값만 같게 적어 우회할 수 없다.

`scripts/test copy.ipynb`는 위 CLI의 thin front end이며
`scripts/build_test_notebook.py`로 재생성한다. Notebook 안에 비교
로직을 복제하지 않는다.

## 6. Fixed training data와 same-compute audit

Strict accepted instance만 고정 JSONL로 내보낸다.

```bash
$PY scripts/run_expansion_hypothesis.py prepare-training \
  --instance-manifest \
    rq_output/expansion/confirmatory_merged/instance_manifest.jsonl \
  --tokenizer /data1/yhoon113/qwen3-8b-base \
  --validity-policy strict \
  --output-dir rq_output/expansion/training_data
```

이 명령은 조건별 instance 수와 실제 Solver prompt token 수가 다르면
`training_allowed=false`를 기록한다. 작은 dataset을 batch size까지
몰래 반복해 같아 보이게 만들지 않는다.

Optimizer, learning rate, updates, batch, verifier와 총 compute까지 기록한
manifest는 `configs/expansion_compute_manifest.example.json` 형식을 따른다.
Independent run마다 별도 manifest를 만들고, training JSONL/log의 실제
SHA-256과 평가 checkpoint ID(또는 immutable registry ref)를 연결한다.
각 manifest의 `generation_linkage`에는 같은 run의 generation report,
experiment/generator/instance manifest, 원본 comparison manifest의 실제
SHA-256, frozen Solver checkpoint 경로와 그 immutable SHA-256을 기록한다.
두 condition의 `base_checkpoint_sha256`도 같은 digest여야 한다.
Capability 분석은 이 lineage가 generation report와 정확히 일치하지
않으면 strong claim을 자동으로 차단한다.

```bash
$PY scripts/run_expansion_hypothesis.py audit-training \
  --compute-manifest \
    /path/to/run01_compute_manifest.json \
    /path/to/run02_compute_manifest.json \
    /path/to/run03_compute_manifest.json \
  --plain-jsonl rq_output/expansion/training_data/training_plain.jsonl \
  --reasoning-jsonl rq_output/expansion/training_data/training_reasoning.jsonl \
  --token-count-field prompt_token_count \
  --output rq_output/expansion/training_data/final_compute_audit.json
```

두 학습은 같은 \(\pi_t\)에서 시작하고 `resume_mode=disable`, 서로 다른
output directory를 사용해야 한다. Static JSONL mode를 사용해야 하며
학습 도중 Evolver가 새 문제를 섞으면 조건 비교가 오염된다.

`configs/rq_evolve_base.yaml`을 조건별로 복사하고 다음을 설정한다.

```yaml
training_data:
  static_training_jsonl: /absolute/path/to/training_plain.jsonl
  static_condition: plain
  static_expected_rows: null
  static_expected_tokens: null
  static_epochs: 1

verl_config:
  actor_rollout_ref:
    actor:
      optim:
        total_training_steps: <same exact integer>
  trainer:
    total_epochs: 1
    # rows / generation_batch_size * static_epochs
    total_training_steps: <exact integer>
    resume_mode: disable
    default_local_dir: /unique/path/plain
```

Reasoning config는 JSONL, `static_condition`, experiment/output 이름만
reasoning용으로 바꾸고 나머지 compute 필드는 동일하게 둔다. 먼저
실제 training tokenizer로 audit한다.

```bash
$PY scripts/train_with_verl.py \
  --config /path/to/expansion_plain.yaml \
  --audit-static-data
```

출력된 `source_rows`와 `token_count`를 각각
`static_expected_rows`/`static_expected_tokens`에 고정한 뒤 같은 명령에서
`--audit-static-data`만 제거하면 학습이 시작된다. Fit 경로는 이 검사를
Ray/GPU 시작 전에 다시 수행하며, static mode에서는 RQEvolver,
MAP-Elites archive, bootstrap mutation을 만들지 않는다.

## 7. Held-out capability set

Schema 예시는 `configs/expansion_heldout.example.jsonl`에 있다. 각
`target_reasoning_move`마다 다음을 학습 전에 고정한다.

- `in_family`: 같은 generator family, 다른 instance
- `structural`: 같은 move, 다른 concept type
- `cross_domain`: 같은 move, 다른 concept group
- `archive` / `benchmark`: forgetting control

숫자, 표면 문장, construction seed는 training과 달라야 한다. latent
distance가 가깝다는 이유만으로 held-out을 선정하지 않는다.
Confirmatory schema에서는 `construction_seed`,
`split_frozen_before_training=true`와 함께 `heldout_provenance`의
`provenance_id`, `construction_method`, `frozen_at_utc`,
`freeze_manifest_sha256`를 필수로 둔다. `freeze_manifest_sha256`은 학습
시작 전에 별도로 저장한 held-out freeze manifest의 실제 SHA-256으로
교체한다.

Base/plain/reasoning checkpoint를 같은 decoding 설정으로 각각 평가한다.
한 호출은 한 `independent_run_id`의 checkpoint만 평가하며
`--checkpoint-run-id`로 그 대응을 명시한다.

```bash
for CONDITION in base plain reasoning; do
  $PY scripts/run_expansion_hypothesis.py evaluate-heldout \
    --heldout /path/to/frozen_heldout.jsonl \
    --checkpoint /path/to/${CONDITION}_checkpoint \
    --checkpoint-id ${CONDITION}-run01 \
    --checkpoint-run-id run01 \
    --checkpoint-provenance registry://solver/${CONDITION}/run01@sha256:<digest> \
    --condition ${CONDITION} \
    --vllm-sampler-backend pytorch \
    --temperature 0 \
    --top-p 1 \
    --rollouts 1 \
    --output rq_output/expansion/eval_${CONDITION}.jsonl
done
```

각 JSONL row와 함께 `<output>.manifest.json`이 생성된다. 여기에는 frozen
held-out 파일 hash, 문제/정답 hash, decoding parameter와 contract hash,
checkpoint ID↔run 대응, tokenizer ID 및 결과 파일 hash가 기록된다.
기본 `pytorch` sampler 선택과 실제
`VLLM_USE_FLASHINFER_SAMPLER` 값도 decoding contract에 포함되므로,
CUDA toolkit과 FlashInfer JIT가 맞지 않는 환경에서도 세 checkpoint가
동일한 backend로 평가된다.
분석은 세 조건에서 문제/정답/move/family/transfer/construction seed와
모든 decoding 설정이 정확히 같은지 다시 검사한다. 조건 하나라도 빠진
run/problem unit은 평균에서 조용히 제외하지 않고 claim gate를 실패시킨다.

그 뒤 capability difference-in-differences를 계산한다.

```bash
$PY scripts/run_expansion_hypothesis.py analyze-capability \
  --results \
    rq_output/expansion/eval_base.jsonl \
    rq_output/expansion/eval_plain.jsonl \
    rq_output/expansion/eval_reasoning.jsonl \
  --compute-manifest /path/to/filled_compute_manifest.json \
  --generation-report rq_output/expansion/generation_result/generation_report.json \
  --output-dir rq_output/expansion/capability_result
```

Primary \(\Delta_{\mathrm{cap}}\)은 `in_family`, `structural`,
`cross_domain`에 대해서만 계산한다. `archive`와 `benchmark`는 primary
평균에 섞지 않고 forgetting으로 별도 보고한다.
같은 compute manifest는 maximum rollout length까지 동일해야 하며,
base checkpoint, condition별 training dataset, training log, output
checkpoint 각각에 실제 SHA-256 또는
`{"artifact_id": ..., "immutable_ref": ...}` 형태의 immutable provenance가
있어야 통과한다.
분석기는 각 manifest의 training JSONL과 log hash를 실제 파일에서 다시
계산하고, 평가 checkpoint ID/immutable ref가 해당 run의 학습 산출물과
연결되는지도 검사한다. Training JSONL은 held-out과 stable ID, 정규화된
문제 문장, construction/sampling seed 및 전체 숫자 signature가 겹치면
claim gate를 실패시킨다. Blinding과 독립 generator 사용 여부는 frozen
held-out provenance로 별도 강제한다.

Capability-only evidence gate는 overall, in-family,
structural/cross-domain 및 각 target reasoning move의 bootstrap CI
하한이 모두 0보다 큰지 확인한다. 최종 strong claim은 여기에
`generation_report.json`의 generation-space, 인접 layer/pooling
robustness, independent-run reproduction gate까지 모두 통과해야 한다.
`--generation-report`를 생략해도 capability 분석값은 만들지만 최종
capability-expansion claim은 항상 차단한다.

## 판정 문구

- Generation criteria 1–3만 통과:
  “reasoning-informed Evolver가 기존 scalar-objective expansion과 다른
  model-relative problem region을 탐색했다.”
- Equal-compute training, independent structural/cross-domain transfer,
  run/layer 재현성과 no-serious-forgetting까지 통과:
  “Solver의 reasoning capability space가 새로운 방향으로 확장됐다.”
- 그 외:
  descriptive pilot 또는 insufficient evidence로 보고한다.

StALT는 primary 판정에는 포함하지 않는다. 대신
`rq_evolve.expansion_trajectory`에 response-token hidden state
(`[token, embedding+layers, hidden]`)를 넘기면 temporal/layer transition,
trajectory length, layer concentration, correct–wrong divergence onset과
StALT를 계산할 수 있는 optional API를 제공한다. 현재 비교 산출물에는
token별 layer hidden state가 저장되어 있지 않으므로 이번 pilot에서는
실행하지 않으며, semantic novelty나 capability-expansion 근거로 해석하지
않는다. Correct–wrong onset은 응답 길이를 상대 token 위치로 보간한 뒤
decoder-layer별 hidden vector의 L2 차이를 RMS 집계해 계산하며, StALT
scalar amplitude 차이는 별도 진단으로만 남긴다. 이 보간은 같은 의미의
token을 정렬한다는 뜻이 아니다. SAE는 기존 infrastructure가 있을 때만
같은 분석을 반복하는 optional 단계다.

## 방법론 출처와 구분

- [Manifold Bandits](https://arxiv.org/html/2606.19750v1): frozen policy의
  normalized mean intermediate-layer prompt representation을 사용하는
  선택의 참고다.
- [Deep Nearest Neighbors](https://proceedings.mlr.press/v162/sun22d.html):
  feature-space non-parametric kNN distance 사용의 참고다.
- [Hierarchical bootstrap](https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290/):
  nested observation을 독립 표본처럼 다루지 않는 resampling의 참고다.
- [StALT](https://arxiv.org/html/2605.01853v1): response decoding 중
  adjacent token/layer hidden-state transition을 요약하는 optional
  internal-dynamics 진단의 참고다.
- [CKA](https://proceedings.mlr.press/v97/kornblith19a.html): layer 간
  representation 구조 비교를 추가할 때의 optional 참고다.

\(O\), plain-only expansion subspace, residual novelty, coverage의 구체적
정의는 이 실험의 operationalization이다. 위 논문이 동일한 확장 metric을
제안했다고 기술하지 않는다.
