# Grading Paths

이 프로젝트에는 답을 채점하는 경로가 **3개** 있습니다. 셋 다 `math_verify`
(`parse` + `verify`)와 `\boxed` 추출이라는 같은 토대 위에 있지만, **R-Zero
parity를 위해 의도적으로 다릅니다.** 처음 보는 사람이 "평가 숫자가 왜 다르지?"에
막히지 않도록, 3 경로를 **학습용(reward) / 평가용(measurement)** 2그룹으로 정리합니다.

핵심: **이 셋은 코드로 합치면 안 됩니다.** offline eval은 R-Zero가 발표한 숫자에
byte 단위로 맞추기 위해 존재하고(보존된 버그 포함), trainer val은 학습 reward에서
length guard만 뺀 것입니다. 각자 다른 목적을 가진 세 경로입니다.

## 그룹 1 — 학습 reward (training)

| | |
|---|---|
| 함수 | `answers_match` |
| 위치 | [`src/rq_evolve/reward.py:109`](../src/rq_evolve/reward.py#L109) |
| 소비처 | `compute_score` (verl `reward_function`) → RL/PPO reward. rollout `correct` 판정(`async_rollout.py`, `verl_backend.py`)에도 사용 |
| boxed 추출 | `reward.extract_boxed` — **LAST** `\boxed`, 브레이스 매칭(임의 중첩) |
| match 규칙 | `verify(parse("\boxed{gt}"), parse("\boxed{pred}"))` — 양쪽 `\boxed`-wrap |
| guard/부가 | **>200자 length guard** → `normalize_answer` 문자열 비교로 폴백(worker가 sympy에 물리는 것 방지). worker 스레드용 SIGALRM 재구현(`_ensure_math_verify_thread_safe`) |
| 라이브러리 | `math_verify` + `normalize_answer` |
| quirk | length guard가 길지만 맞는 답을 문자열 불일치로 놓칠 수 있음. `math_verify` hard-import(누락 시 loud fail) |

## 그룹 2 — 평가 (measurement)

두 경로가 있고, 둘 다 length guard가 **없습니다**(측정 정확도 우선).

### 2a. trainer val-core (in-training)

| | |
|---|---|
| 함수 | `grade_eval` |
| 위치 | [`src/rq_evolve/math_eval.py:61`](../src/rq_evolve/math_eval.py#L61) |
| 소비처 | `RQValidatingTrainer._validate` ([`eval_trainer.py:86`](../src/rq_evolve/eval_trainer.py#L86)) → `val-core/<data_source>/acc`. 트레이너 **MAIN 스레드**에서 채점(math_verify SIGALRM이 동작) |
| boxed 추출 | `reward.extract_boxed` — **LAST** `\boxed`, 브레이스 매칭 |
| match 규칙 | `verify(parse("\boxed{gt}"), parse("\boxed{pred}"))` — 양쪽 `\boxed`-wrap |
| guard/부가 | 없음. `answers_match`에서 length guard만 제거한 것 |
| quirk | 아래 caveat 참고 — offline(2b)과 "같은 숫자"를 표방하지만 byte-identical 아님 |

### 2b. offline checkpoint eval (R-Zero parity)

| | |
|---|---|
| 함수 | `rzero_score` / `rzero_compare_answer` |
| 위치 | [`scripts/eval_vllm_math.py:108`](../scripts/eval_vllm_math.py#L108) / [`:92`](../scripts/eval_vllm_math.py#L92) |
| 소비처 | 체크포인트 오프라인 평가(6개 R-Zero math 벤치마크). R-Zero 발표 숫자와의 대조용 |
| boxed 추출 | `rzero_extract_answer` — **FIRST** `\boxed`, 정규식 `(?i)\\boxed\s*{([^\n]+)}`(개행 경계, 단일 레벨) |
| match 규칙 | `verify(parse(gt), parse(pred))` — **raw, wrap 없음** |
| guard/부가 | length guard 없음. **GPT-4o recheck**: score<0.5 행을 gpt-4o "yes"면 1로 승격(R-Zero `results_recheck.py` 포팅) |
| 라이브러리 | `math_verify` + OpenAI gpt-4o |
| quirk | **보존된 버그**: 추출 실패 시 `str(None)` → 문자열 `"None"`이 되어 `is None` 체크가 안 걸리고 `verify(parse(gt), parse("None"))`으로 흘러감. R-Zero와 점수를 맞추려고 일부러 그대로 둠. raw(unwrap) verify라 `\dfrac`↔`\frac`/분수↔소수 동치를 놓침 |

## ⚠️ Caveat — val-core와 offline 숫자는 byte-identical이 아님

`grade_eval`(2a)과 `rzero_score`(2b)는 둘 다 "the eval number"를 표방하지만 실제로는
다릅니다:

- **추출**: LAST `\boxed`(브레이스 매칭) vs FIRST `\boxed`(정규식, 개행 경계)
- **wrap**: `\boxed`-wrap 후 verify vs raw verify

따라서 다중 box / bare-fragment / 분수·소수 동치 같은 케이스에서 **val-core와 offline
체크포인트 숫자가 갈릴 수 있습니다.** 둘은 서로를 추적하지만 동일하지 않습니다. 관련
주석: [`math_eval.py:44`](../src/rq_evolve/math_eval.py#L44).

## 참고 — 중복/보조 채점 함수

- `grade` dispatcher ([`math_eval.py:82`](../src/rq_evolve/math_eval.py#L82)): `"math_verify"` → `grade_eval`, 그 외 → `answers_match`. 오프라인 편의 래퍼.
- `normalize_answer` ([`reward.py:101`](../src/rq_evolve/reward.py#L101)): 그룹 1의 length-guard 폴백에서만 쓰는 문자열 정규화.
- `async_rollout._normalize_response`: **중복 응답 탐지**용이지 답 매칭이 아님(혼동 주의).
- (제거됨) `backends.extract_boxed`: 예전 정규식 중복 구현. 모든 실사용처는
  `reward.extract_boxed`를 쓰므로 삭제함.
