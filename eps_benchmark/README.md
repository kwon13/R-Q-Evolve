# Group-balanced EPS benchmark

이 폴더는 현재 R-Q-Evolve의 여섯 수학 `GROUP`을 동일 가중치로 평가하기
위한 고정 OOD 벤치마크입니다.

- 벤치마크 이름: `evolved_performance_group_balanced_ood_v2`
- 총 문제 수: **480**
- 구성: **6 GROUP × 2 programs × 40 instances**
- GROUP별 문제 수: **정확히 80**
- 답 형식: 단일 base-10 정수
- 고정 생성 seed 시작값: `5,000,000`
- 모든 480개 정규화 문제문은 서로 다름

현재 EPS 구현은 프로그램별 정확도의 macro 평균을 사용합니다. 이
artifact는 모든 프로그램이 40문제이고 모든 GROUP이 두 프로그램이므로
다음 세 값이 정확히 같습니다.

```text
EPS = program macro accuracy
    = GROUP macro accuracy
    = 480-problem micro accuracy
```

## 먼저 확인해야 할 저장소 불일치

요청에는 현재 seed program이 10개라고 적혀 있지만, 이 artifact를 만든
시점의 실제 `수식 증명/R-Q-Evolve/seed_programs/`에는 **8개**가 있습니다.
모든 실행 config도 그 폴더를 가리키며 여섯 GROUP은 다음처럼 분포합니다.

| GROUP | 현재 seed program의 SKILL | 개수 |
|---|---|---:|
| algebra | invariant | 1 |
| combinatorics | construction, induction | 2 |
| geometry | extremal_principle | 1 |
| inequality | counting | 1 |
| number_theory | casework, transformation | 2 |
| sequence | contradiction | 1 |

정확한 파일명·소스 SHA256·program ID는 `manifest.json`의
`reference_seed_inventory`에 고정했습니다. 의도한 나머지 두 seed가 다른
경로에 있다면 이 benchmark를 사용하기 전에 그 파일을 추가하고 설계를
다시 감사해야 합니다.

## 12개 held-out 생성기

각 생성기는 답을 구하는 주 경로와 별도의 exact cross-check 경로를
`assert answer == check`로 연결합니다. 현재 8개 seed뿐 아니라 구
`challenge_seed_programs/structural_ood_v2`의 6개 구조도 재사용하지
않았습니다.

| GROUP | 프로그램 / SKILL | 답을 결정하는 구조 | 독립 검산 경로 |
|---|---|---|---|
| algebra | `algebra_01_rank_one_determinant` / transformation | 대각행렬의 rank-one update determinant | 전체 Leibniz determinant 전개 |
| algebra | `algebra_quadratic_root_product_resultant` / transformation | cubic을 quadratic으로 나눈 나머지와 Vieta 관계 | 5×5 Sylvester determinant |
| combinatorics | `combinatorics_complete_tripartite_spanning_trees` / counting | complete tripartite spanning-tree 폐쇄식 | Laplacian cofactor의 Bareiss determinant |
| combinatorics | `combinatorics_labeled_trees_exact_leaves` / counting | Prüfer word의 누락 문자와 inclusion–exclusion | distinct-symbol 동적계획법 |
| geometry | `geometry_circle_chord_regions` / counting | chord와 4-endpoint 교점의 조합적 계수 | 점을 하나씩 추가하는 점화식 |
| geometry | `geometry_descartes_circle_curvature` / transformation | 세 곡률에 대한 Descartes relation의 큰 근 | integral quadruple reflection과 relation 대입 |
| inequality | `inequality_boxed_cubic_concentration` / extremal_principle | box와 고정합에서 cubic mass concentration | continuous box-slice의 모든 vertex 완전열거 |
| inequality | `inequality_expanded_even_quartic` / transformation | 전개된 quartic의 recentering과 square 치환 | 모든 derivative critical point 직접 평가 |
| number_theory | `number_theory_gcd_sum` / transformation | divisor–totient 합성곱 | `gcd(k,n)` 직접 합산 |
| number_theory | `number_theory_binomial_prime_valuation` / transformation | factorial valuation의 차 | base-p 덧셈의 carry 수 |
| sequence | `sequence_01_repeated_root_recurrence` / transformation | repeated characteristic root의 `(A+Bn)r^n` 형식 | 명시된 2차 점화식 직접 반복 |
| sequence | `sequence_quadratic_double_angle_modulo` / transformation | `z^(2^n)+z^(-2^n)`로 nonlinear recurrence 변환 | 명시된 합동 점화식 직접 반복 |

구조적으로는 다음을 피했습니다.

- algebra seed의 quadratic-translation discriminant orbit과 구 Newton power-sum 문제
- combinatorics seed의 separating family·end-insertion inversion과 구 banded path 문제
- geometry seed의 minimum empty-triangle extremal argument
- inequality seed의 AM-GM feasible-threshold count와 구 shifted Cauchy minimum
- number-theory seed의 CRT residue casework·primitive-root coordinates와 구 power tower
- sequence seed의 prime arithmetic-progression contradiction과 구 weighted binary recurrence

계열 선택 과정에서는 저장소에 남아 있는 진화 archive도 수동으로 대조해,
accepted program과 명확히 같은 핵심 풀이 구조를 가진 후보는 교체했습니다.
다만 이는 현재 로컬에 보존된 기록에 대한 family-level 감사이며, 보존되지
않은 8B/R-Zero 실행이나 모든 가능한 의미 중복을 증명하는 절차는 아닙니다.

## 폴더 구성

```text
EPS_Benchmark/
  benchmark.jsonl              fixed 480 rows used for inference
  manifest.json                counts, source hashes, two artifact hashes
  validation_report.json       last full validation result
  programs/                    12 reproducible held-out generators
  artifact_integrity.py        immutable v1 digest lock and row hash
  build_benchmark.py           deterministic builder
  validate_benchmark.py        strict source/data/overlap validator
  summarize_group_scores.py    checkpoint summaries -> six GROUP scores
```

각 JSONL row는 기존 evaluator가 요구하는 다음 필드를 그대로 사용합니다.

```text
benchmark, sample_id, program_name, program_id, program_sha256,
group, skill, seed, problem, answer, instance_sha256, index
```

## 재생성과 검증

workspace 최상위에서 실행합니다.

```bash
python3 EPS_Benchmark/build_benchmark.py --force
python3 EPS_Benchmark/validate_benchmark.py
```

고정 출력은 다음 digest를 가져야 합니다.

```text
benchmark_sha256 = 7094ddcde5f2dc08211d65e8b992f928187cae6789df0eb84dce0bcf61375537
artifact_sha256  = 4ae707a98277227607881e2ad6dcbdef3ec97048ff329ce99dfcea793ad8f8d3
```

### v2 변경 사항 (v1 대비)

v1은 `v1_archive/`에 보존되어 있으며 다음 두 결함으로 대체되었습니다.

1. **답 형식 지시 제거.** v1의 480문항 중 400문항이 `State only the integer.`로
   끝났으나 학습용 seed program은 이 문구를 쓰지 않아 학습–평가 간 형식 분포가
   어긋났고, 벤치마크 내부에서도 2개 프로그램만 문구가 없어 불일치했습니다. v2는
   모든 문항에서 이 문구를 제거했습니다. 답 형식 지시는 평가 하네스의 system
   prompt가 담당해야 하며, **학습과 동일한 system prompt를 쓰는지 확인**하십시오.
2. **답 크기 상한.** `combinatorics_labeled_trees_exact_leaves`의 답이 최대 12자리에
   달해 공식 회상이 아니라 대수 계산 정밀도를 재는 문항이 섞였습니다. 해당 생성기의
   허용 답 범위를 `< 1e9`로 낮춰 전체 최대 자릿수가 12 → 9로 줄었습니다. 나머지
   프로그램(최대 8자리)은 그대로 두었습니다.

알려진 한계: 고유 마스킹 템플릿이 480문항에 47개이고 12개 프로그램 중 9개가 템플릿
1개이므로, 정오는 프로그램 단위로 강하게 뭉칩니다. 유효 표본 수는 문항 수가 아니라
**프로그램 수(12)**에 가까우므로, 집계 정확도의 신뢰구간은 문항이 아니라 **프로그램을
재표집하는 부트스트랩**으로 계산하고 프로그램별 정확도를 함께 보고해야 합니다.

`benchmark_sha256`는 기존 R-Q-Evolve loader/evaluator와의 호환 digest입니다.
그 legacy hash는 `group`, `skill`, `index`, `benchmark`를 포함하지 않으므로,
이 폴더의 `artifact_sha256`는 그 필드까지 포함해 별도로 잠급니다.

validator는 다음을 검사합니다.

- 12 programs, 480 rows, 80 rows/GROUP, 40 rows/program
- 480개 단독 문제문 중복 없음, 단일 canonical 정수 답
- 문제문 안에 정답과 같은 standalone integer token이 노출되지 않음
- generator source lint, 허용 라벨, 독립 answer cross-check 계약
- 각 JSONL row를 원 source와 seed로 다시 실행한 결과의 byte-level 일치
- source SHA256/program ID와 manifest 일치
- 문서화한 순차 seed selection을 독립적으로 다시 빌드한 480 row와 byte-level 일치
- 두 digest의 immutable v1 lock 및 builder/validator/R-Q-Evolve module provenance
- 현재 8개 seed 및 구 OOD v2의 낮은 seed와 benchmark 인접 seed 범위에서
  source 재사용 0, exact normalized problem-text overlap 0
- 저장된 5개 기존 benchmark(2,672 rows)와 exact text 및 숫자를 가린 template overlap 0

마지막 항목은 exact overlap 감사이지 의미론적 독립성의 자동 증명은
아닙니다. 의미 수준의 차이는 위 12개 family 설계표로 별도 감사했습니다.

## 기존 evaluator로 실행

`run_evolved_performance.sh`의 prebuilt benchmark 경로를 사용하면 evaluator를
수정하지 않아도 됩니다. 아래 경로와 run/model 설정을 실제 서버에 맞게
바꿉니다.

```bash
cd "/path/to/workspace/수식 증명/R-Q-Evolve"

RUN_DIR=/path/to/rq_output/run_name
EPS_DIR=../../EPS_Benchmark

python3 "$EPS_DIR/validate_benchmark.py"

PY=/path/to/vllm-env/bin/python \
BASE="$RUN_DIR" \
BASE_MODEL=/exact/path/to/Qwen3-8B-Base \
BENCH_DIR="$EPS_DIR" \
BENCHMARK_NAME=evolved_performance_group_balanced_ood_v2 \
PREBUILT_BENCHMARK=1 \
RESULTS_DIR="$RUN_DIR/evolved_performance_group_balanced_ood_v2" \
GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/run_evolved_performance.sh
```

4B와 8B는 각각 정확한 `BASE`, `BASE_MODEL`, `RESULTS_DIR`로 별도 실행해야
합니다. 특히 wrapper의 기본값은 4B config와 특정 서버의 Python 경로이므로
예시처럼 `PY`와 `BASE_MODEL`을 명시하지 않으면 안 됩니다. 이 wrapper는
`mapfile`과 GNU `find -printf`를 사용하므로 Linux의 Bash 4 이상을 전제로
합니다.

그룹별 trajectory 표는 inference 후 다음처럼 만듭니다.

```bash
python3 ../../EPS_Benchmark/summarize_group_scores.py \
  "$RUN_DIR/evolved_performance_group_balanced_ood_v2"
```

`group_scores.json`과 `group_scores.md`가 생기며, 각 checkpoint에서 GROUP
macro와 evaluator EPS가 같은지도 검사합니다. GROUP 결과는 `details.jsonl`
전체를 다시 읽어 label-sensitive `artifact_sha256`까지 검증하며, 출력 JSON에
legacy hash와 이 추가 hash를 모두 기록합니다.

## Figure 3 / Table 3 해석 주의

이 artifact는 기존 논문의 `240 Seed-ID + 240 Structural-OOD` union과 다른
**480문제 전부 held-out, 6-GROUP-balanced** 분포입니다. 따라서 다음 원칙이
필수입니다.

1. Figure 3의 RQ-Evolve와 R-Zero를 포함해 비교하는 모든 checkpoint를 이
   동일 hash로 다시 추론합니다.
2. Figure 4와 Table 3의 full/flat/no-reeval/no-uncertainty/no-variance 및
   R-Zero reference도 모두 새 hash로 다시 측정합니다. 이전 EPS 수치나
   delta를 새 벤치마크와 섞지 않습니다.
3. 기존 Table 2의 Seed-ID 대 Structural-OOD gap은 이 OOD-only artifact로
   계산할 수 없습니다. 그 causal claim을 유지하려면 별도의 새 matched
   Seed-ID half가 필요합니다.
4. 저장소에는 현재 Figure 3의 완성 PDF만 있고, RQ-Evolve/R-Zero를 합치는
   전용 plotting source는 확인되지 않았습니다. 기존 generic script는
   한 run의 EPS plot은 만들지만 두 시스템 합성 Figure 3 자체는 별도
   plotting 코드가 필요합니다.

모든 모델·ablation 결과에서 `benchmark_sha256`가 위 값과 동일한지 확인한
뒤에만 한 그림이나 표로 합쳐야 합니다.
