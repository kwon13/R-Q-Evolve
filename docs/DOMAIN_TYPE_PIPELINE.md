# Domain × Problem Type Pipeline

이 문서는 새 학습 런의 descriptor와 admission contract를 정의합니다. 이전
`GROUP × SKILL` 결과는 분석 자료로만 보존하며, 새 런의 archive나 checkpoint로
resume하지 않습니다.

## 1. MAP descriptor

새 MAP은 다음 두 질문에 답합니다.

- **DOMAIN**: 이 문제를 푸는 데 어떤 수학적 구조가 핵심인가?
- **PROBLEM_TYPE**: 문제는 어떤 형태의 결과를 출력하라고 요구하는가?

두 번째 축은 domain의 하위 분류가 아닙니다. 예를 들어 Number Theory 문제도
Yes/No, 해의 집합, 해의 개수, 최솟값, 일반 값 계산을 각각 요구할 수 있습니다.
따라서 두 축은 설계상 서로 다른 속성을 기술합니다. 다만 이것이 실제 corpus에서
두 라벨이 통계적으로 독립이라는 뜻은 아닙니다. 분포의 편향은 별도로 측정해야
합니다.

### DOMAIN: Omni-MATH top-level 7개

코드 값과 순서는 `src/rq_evolve/concepts.py`의 `DOMAINS`가 유일한 기준입니다.

1. `algebra`
2. `geometry`
3. `number_theory`
4. `discrete_mathematics`
5. `applied_mathematics`
6. `calculus`
7. `precalculus`

`Mathematics → Geometry → Plane Geometry`처럼 Omni-MATH 경로가 주어졌다면 이
MAP은 `Mathematics` 바로 아래의 `Geometry`에 해당하는 `geometry`만 사용합니다.
그 아래의 Plane Geometry는 별도 행이나 계층 cell을 만들지 않습니다.

### PROBLEM_TYPE: 계산적 출력 계약 5개

코드 값과 순서는 같은 파일의 `PROBLEM_TYPES`가 기준입니다.

1. `decision`: 진리값 또는 Yes/No를 출력한다.
2. `search`: 조건을 만족하는 해·구성의 완전한 유한 집합을 출력한다.
3. `counting`: 유효한 객체·방법·해의 개수를 출력한다.
4. `optimization`: 최댓값·최솟값 같은 극값을 출력한다.
5. `function`: 위 네 종류가 아닌, 지정된 값이나 유일하게 결정되는 스칼라를
   계산한다.

`function`은 함수론이라는 수학 domain을 뜻하지 않고, 나머지 네 출력 계약에
속하지 않는 일반적인 value evaluation/recovery를 뜻합니다. 동사 하나로 분류하지
않고 요청의 의미를 우선합니다. 예를 들어 “find the smallest n”은
`optimization`, “find all n”은 `search`, “how many n”은 `counting`입니다.
증명이나 설명만을 요구해 exact answer가 없는 문제는 이 다섯 유형 밖이므로
수용하지 않습니다.

`DOMAINS`와 `PROBLEM_TYPES` tuple은 행/열의 어휘와 표시 순서를 정할 뿐입니다.
코드 순서가 라벨을 판정하지 않습니다. DOMAIN은 fixed child family의 단일 선언을
검증하고, PROBLEM_TYPE은 statement의 출력 요구와 답안 verifier를 함께 보는
결정적 규칙으로 판정합니다.

## 2. Full 7 × 5 archive

runtime archive는 35개 Cartesian-product cell을 항상 모두 만듭니다. benchmark에
관측된 조합만 허용하는 `supported-cell mask`, 최소 빈도, allowlist는 없습니다.
드문 조합도 생성되고 검증되면 정상 cell입니다.

프로그램 하나는 정확히 한 cell에만 들어갑니다. 생성된 child는 다음 로컬 검증을
모두 통과한 경우에만 좌표를 얻습니다.

- stage 2 source가 허용 어휘의 top-level DOMAIN을 정확히 하나 선언한다.
- 각 verification seed의 statement에서 출력 계약이 모호하지 않게 판정된다.
- 그 판정과 normalized verifier/reference answer가 호환된다.
- 모든 verification seed가 동일한 high-confidence PROBLEM_TYPE을 만든다.

DOMAIN 의미를 별도 모델로 재판정하지 않습니다. stage 1에서 family가 이미
확정된 뒤 stage 2가 하나의 허용값을 선언하며, 코드는 선언의 개수·위치·어휘를
fail closed로 검사합니다. 다중 라벨, metadata override, secondary cell 복제,
fallback bin은 사용하지 않습니다. 이 계약은 DOMAIN 의미 판정을 외부 모델의
변동성에서 분리하는 대신, 선언 자체를 auditable source contract로 취급합니다.

초기 seed는 빈 archive를 시작하기 위한 최소 bootstrap이며 mask가 아닙니다.
표의 PROBLEM_TYPE은 source 상수가 아니라 같은 로컬 규칙으로 확인되는 기대
결과입니다. 다섯 type을 한 번씩 쓰는 core diagonal에 남은 두 domain seed를 더해
총 7개 domain을 모두 엽니다.

| Seed file | DOMAIN | PROBLEM_TYPE |
|---|---|---|
| `00_algebra_decision.py` | `algebra` | `decision` |
| `01_geometry_search.py` | `geometry` | `search` |
| `02_number_theory_counting.py` | `number_theory` | `counting` |
| `03_discrete_mathematics_optimization.py` | `discrete_mathematics` | `optimization` |
| `04_applied_mathematics_function.py` | `applied_mathematics` | `function` |
| `05_calculus_function.py` | `calculus` | `function` |
| `06_precalculus_optimization.py` | `precalculus` | `optimization` |

## 3. Untargeted mutation과 local descriptor validation

mutation은 cell coverage 방향을 입력받지 않습니다. `target_cell`, 축 유지/변경,
특정 domain/type 값으로 바꾸라는 지시를 generation prompt에 넣지 않습니다.

1. **Stage 1 — family design**: parent의 parameterized problem family와 한
   instance를 보고, descriptor 어휘 없이 구조적으로 다른 child family를 자연어로
   제안합니다. 선택적 structural inspiration을 사용할 때도 donor의 문제
   skeleton만 보여 줍니다.
2. **Stage 2 — generator compilation**: 확정된 child family를 deterministic
   `generate(seed)` 코드로 옮기고 허용 어휘에서 DOMAIN 하나를 선언합니다. parent
   코드는 코드 형태의 참고 자료일 뿐이며 parent label과 target cell은 전달되지
   않습니다.
3. **Generic verification**: source, 독립 `answer == check`, 여러 seed 실행,
   statement/answer 일관성을 descriptor와 무관하게 검사합니다.
4. **Local descriptor validation**: DOMAIN 선언을 정적으로 검증하고, 모든 rendered
   seed에 대해 statement+verifier 규칙으로 PROBLEM_TYPE을 판정합니다. 한 seed라도
   모호하거나 결과가 다르면 거절합니다.
5. **Scoring and insertion**: fresh instances에서 rollout을 만들고 `R_Q`로 해당
   cell champion과 경쟁합니다. 같은 cell에 머무는 구조적 변이도 허용됩니다.

좌표 검증은 순수 로컬 코드이므로 업데이트되는 solver policy나 별도 모델의 응답에
의존하지 않습니다. `structural_inspiration`은 label-free 선택 사항이고 첫 clean
run에서는 꺼져 있습니다.

## 4. Declarative verifier와 type rules

generator는 `(problem, str(answer), verifier)`를 반환합니다. `verifier`는 실행
코드가 아니라 JSON-safe 선언이며 네 mode만 허용합니다.

| mode | 의미 | 비교 방식 |
|---|---|---|
| `expression` | 하나의 수식/값 | `math_verify` symbolic equivalence |
| `boolean` | Yes/No | 보수적인 canonical Boolean 비교 |
| `one_of` | 여러 완전한 정답 문자열 중 하나 | 각 후보와 symbolic equivalence |
| `set` | 완전한 유한 unordered set | 원소별 symbolic equivalence와 exact set equality |

verifier mode 하나만으로 PROBLEM_TYPE을 정하지 않습니다. 예를 들어
`expression`은 counting, optimization, function을 구별할 수 없습니다. 먼저
statement의 요청 구문을 precedence가 있는 보수적 규칙으로 판정하고, 다음
mode/reference 제약으로 교차검증합니다.

| 판정 type | statement 의미 | 필요한 verifier/reference 계약 |
|---|---|---|
| `decision` | whether, Yes/No 형태의 판정 | `boolean`, gold가 Yes 또는 No |
| `search` | 모든 해/구성의 완전한 유한 집합 | `set` |
| `counting` | 객체·해의 개수 | `expression`, nonnegative integer gold |
| `optimization` | maximum/minimum/optimal 값 | `expression` |
| `function` | 그 밖의 명확한 scalar value | `expression` |

proof/설명 요구, 일반적인 `find`만 있어 출력 계약이 불명확한 문장, statement와
verifier가 충돌하는 경우에는 추측하지 않고 abstain합니다. `one_of`은 일반 verifier
schema에서 수학적으로 동치인 표기 대안을 표현할 수 있지만 production type을
단독으로 인증하지 않습니다. 임의 witness 하나를 허용하는 predicate로 사용하지도
않습니다.

따라서 다섯 problem type 모두 **exact-answer 형태로 작성된 경우** 자동 채점할
수 있습니다. 반대로 임의의 witness predicate, 자연어 증명, 설명의 완전성처럼
네 mode로 표현할 수 없는 정답은 지원하지 않습니다. generated callable이나
임의 Python predicate를 grader에서 실행하는 escape hatch도 두지 않습니다.

## 5. Admission gates

자유 변이는 수용 기준을 없앤다는 뜻이 아닙니다. child는 archive 전에 다음
방어선을 통과합니다.

- static lint와 seeded determinism (`random.Random(seed)`);
- 독립 계산 경로를 잇는 `assert answer == check`의 AST contract;
- hard-killable sandbox에서 여러 seed 실행, visible problem variation, 같은 문장에
  서로 다른 gold가 붙지 않는지 확인;
- 선언형 verifier schema 및 reference answer consistency;
- exact-one/in-vocabulary DOMAIN source contract;
- 전 verification seed의 deterministic PROBLEM_TYPE high-confidence 합의;
- optional donor-copy rejection;
- seed-variation, behavior duplicate, template duplicate, near-template duplicate,
  structural duplicate rejection.

빈 cell에는 `R_Q = 0` 후보도 들어갈 수 있지만 training frontier가 all-correct와
all-wrong instance를 제외합니다. 점수가 있는 후보가 오면 strict cell competition으로
교체됩니다.

## 6. Fresh run contract

새 production 설정과 launcher는 다음입니다. 두 설정은 descriptor/admission
contract가 같고 GPU compute geometry와 독립 output identity만 다릅니다.

```bash
# 8 GPUs
bash scripts/run_train_domain_type_8gpu.sh

# 4 GPUs, detached under nohup with checkpoint auto-merge
bash scripts/run_train_domain_type_4gpu.sh --gpus 0,1,2,3 --detach
```

- configs: `configs/rq_evolve_4b_{4,8}gpu_domain_type.yaml`
- seeds: `seed_programs_domain_type/*.py` 정확히 7개
- outputs: `rq_output/rq_evolve_4b_domain_type_35cell_{4,8}gpu`
- coordinate validation: local code only; 별도 model/API/credential 없음

launcher는 output directory가 이미 비어 있지 않으면 시작하지 않으며, config는
trainer checkpoint resume을 비활성화합니다. archive snapshot도 schema 이름,
version, 두 axis, 전체 vocabulary 순서, binning을 정확히 대조하므로 이전
`GROUP × SKILL` archive를 부분 변환하거나 조용히 재배치하지 않습니다. 새 run을
반복하려면 기존 directory를 재사용하지 말고 새로운 run identity와 빈 output
directory를 지정해야 합니다.
