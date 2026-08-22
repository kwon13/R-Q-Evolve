# Pipeline Notes

이 문서는 `evo-sample`의 구현을 직접 다시 만들기 위한 대응표입니다.

## Source-To-Skeleton Mapping

| evo-sample | R-Q-Evolve | 역할 |
|---|---|---|
| `rq_questioner/program.py` | `src/rq_evolve/program.py` | `generate(seed)` 프로그램 실행 단위 |
| `rq_questioner/map_elites.py` | `src/rq_evolve/archive.py` | GROUP×SKILL MAP-Elites archive |
| `rq_questioner/rq_score.py` | `src/rq_evolve/scoring.py` | `R_Q = p(1-p)U` |
| `prompts/*` | `src/rq_evolve/prompts.py`, `prompt_templates/` | mutation / solver prompt 생성 |
| `rq_questioner/code_utils.py` | `src/rq_evolve/code_utils.py` | LLM 코드 추출과 lint |
| `rq_questioner/verl_dataset.py` | `src/rq_evolve/dataset.py` | champion → training examples |
| `rq_questioner/verl_trainer.py` | `src/rq_evolve/evolution.py`, `src/rq_evolve/verl_adapter.py`, `src/rq_evolve/verl_backend.py` | evolution loop와 verl hook 경계 |
| `reward_fn.py` | `src/rq_evolve/reward.py` | boxed-answer reward |
| `run_verl.py` | `scripts/train_with_verl.py` | 학습 entry point |
| `verl/` | 복제하지 않음 | pip 설치된 `verl` 사용 |

## Core Loop

1. Seed programs are loaded as `ProblemProgram`.
2. Each program is verified across multiple seeds.
3. The backend generates G solver rollouts for one representative instance.
4. `p_hat`, uncertainty, and `R_Q` are computed.
5. The program competes for its MAP-Elites cell, which is exactly its declared
   (GROUP, SKILL) pair.
6. Parent programs are sampled from occupied cells.
7. The mutation model is shown the parent source plus both label vocabularies
   and writes the child generator directly. One model call, no planning stage
   and no operator: it chooses what to change and labels the result itself.
8. The child is verified statically and by execution, gated by the taxonomy
   judge, scored, and inserted if elite.
9. Frontier champions render new training examples.
10. The installed `verl` trainer consumes those examples and updates the solver.
11. The next outer iteration re-scores every champion against the new weights
    before any of them is re-binned, replaced, or removed.

## Archive Axes

The grid is GROUP x SKILL: 6 mathematical domains by 8 reasoning skills, 48
cells. Both coordinates are read off the program's own top-level `GROUP` and
`SKILL` constants, so a cell is a (domain, reasoning) pair. There are no bin
counts to configure; the shape is the two vocabularies in `concepts.py`.

Nothing steers a child to a chosen cell. The operator pair that did -- one held
GROUP and moved SKILL, the other the mirror -- was removed because ordering a
label change makes the label a target the child is written to satisfy rather
than a description of what it produced, and a child filed under a label its
statement does not support corrupts the coordinates themselves. Coverage is now
a consequence of mathematical variety plus the judge, not of an instruction.

Uncertainty is not an axis. H remains the uncertainty factor of
`R_Q = p(1-p)H`, which decides who holds a cell and which champion is drawn as
a mutation parent. It was dropped as a coordinate because no operator can aim
at it: a child lands in an H bin only as a side effect of how hard it turned
out to be, so binning on it spent grid capacity on a dimension evolution could
not steer. Champions that differ only in difficulty now compete in one cell,
and the higher R_Q wins.

A program whose GROUP or SKILL falls outside the vocabulary has no cell and is
rejected (`archive_status: unlabelled_rejected`) rather than hashed into a
shared bin, which would make one cell the contest for every mislabelled
generator.

`stats()` reports `group_coverage` and `skill_coverage` separately: a single
coverage number cannot distinguish "we only ever work in two domains" from "we
only ever exercise two reasoning moves", and those are the two failure modes
the operators exist to fix. The 14 seed programs cover 6/6 groups but only 3/8
skills, so operator A carries the early exploration.

Snapshots written before this change store an H coordinate and no SKILL label.
They cannot be placed and are dropped with a message at load; a resume from one
bootstraps from `seed_programs/` instead.

## Mutation Contract

Mutation is one-stage and free-form. `build_mutation_task` renders the parent
source into `in_depth.txt` / `in_breadth.txt` and the model returns the child
generator in one ```python``` block. There is no plan schema, no registered
family compiler, and no quarantine path.

Four gates stand between a generated child and the archive:

## Outer Iteration (Algorithm 1)

```
sync θ_t into the rollout engine
for each elite:                       # Re-scoring == replay 생성
    n개 fresh seed (비재사용 stream) × m rollout
    (x_z, y_zj, r_zj, h_zj) 저장 → replay buffer
    L̂_z = m/(m-1)·ŝ_z(1-ŝ_z),  Û_z,  R̂_Q = mean_z(L̂_z·Û_z),  D̂
for 32 candidates:                    # inner batch, 단일 변이 연산자
    mutate → validity → judge → n×m 채점 → cell 경쟁
T_t ← lagged R̂_Q (t-1 까지의 EWMA) 로 고른 frontier elite
batch ← T_t 의 이번 iteration replay rollout      # 별도 샘플링 패스 없음
θ_{t+1} ← RLOO 1회 업데이트 (인스턴스 LOO baseline)
```

`n=5, m=2` 는 구판 단일 인스턴스 `G=10` 과 같은 rollout 예산이며, 재점수화와
학습 샘플링이 통합되므로 예산이 이중 지출되지 않습니다. 선택을 지난 iteration의
점수로 미루는 이유는 winner's curse입니다 — 같은 rollout으로 점수를 매겨 고르고
그 rollout으로 학습하면 측정 노이즈가 유리하게 실현된 표본이 선택적으로 남습니다.

1. **Static lint** — `lint_generator_source` plus the determinism checks in
   `lint_mutation_generator_source`: a seeded `rng = random.Random(seed)`, no
   direct `random.*` calls, and `sorted()` around dict/set iteration, so a
   program yields the same instance in the verifier and in every rollout
   process. The compiler-notation flags (`MAX_ATTEMPTS = 200`, canonical
   `instance_data`, `str(sympy.Integer(answer))`) stay OFF: free-form
   generation does not satisfy them, and enforcing them rejected sound
   programs on shape alone.
2. **Structural contract** — `ast_contract.py` decides whether the child's own
   `assert answer == check` is a real cross-check. It never proves two routes
   independent (undecidable); it refutes independence three ways — the routes
   are the same program modulo renaming (A3v), the check was read off the
   answer (A4d), or the check side is a constant (A5c) — plus "no assert at
   all" (A1), "no assert straddles the answer" (A2), and a statement parameter
   with no dependency path to the answer (P1). The check is *existential*: one
   certifying assert passes the program, because seeds legitimately carry
   guards and invariants alongside their cross-check. Measured on 456 archived
   champions it flags 45%, and 0 of the 14 clean programs (6 seeds + 8 verified
   fixtures). `evolution.ast_contract` is `off` | `shadow` | `enforce`; shadow
   records the verdict in `program.metadata["ast_contract"]` without rejecting.
   Unlike the notation flags in gate 1, this is a dataflow predicate, not a
   shape predicate — which is why it lives in its own module and is ON.
3. **Execution** — `verify_program` runs seeds 0..N-1, requires a base-10
   integer answer from every generated mutation, and rejects a program whose
   visible problem does not vary across seeds. This carries the guarantee the
   static notation contract used to. Under `enforce`, each rendered statement
   is also checked for handing the solver its own technique (P2).
4. **Taxonomy judge** — the judge is shown the seed-0 problem and the supplied
   answer, and nothing else: not the source, not the declared labels, not the
   parent. It runs its own completeness and answer-consistency gates, derives
   the shortest clean solution, and returns GROUP and SKILL with a mandatory
   concrete witness for each. The child is archived only when both match what
   it declared; `none`, an out-of-vocabulary value and an unreadable reply all
   reject. `use_evaluator` gates it, and because it is the only gate that reads
   the rendered statement as prose, `use_evaluator: false` requires
   `ast_contract` to be on — `EvolutionConfig.__post_init__` refuses the
   combination where neither runs. The verdict is written to
   `metadata["judge"]` whether it passes or fails, so the disagreement rate
   between the Evolver's self-labelling and an independent reading is a
   measurable quantity rather than an assumption.

All repository-owned standalone vLLM evaluation/comparison entry points default
to `--vllm-sampler-backend pytorch`.
FlashInfer 0.6.x attempts to JIT-compile CUDA 12-only sampling extensions, while
some hosts use a CUDA 12 PyTorch wheel with an older CUDA 11.8 system `nvcc`.
The native sampler avoids that engine-startup failure; the selected backend is
recorded in the run manifest.

## Evolution Candidate State

`EvolutionEngine.inner_iteration_batch` ([`src/rq_evolve/evolution.py:193`](../src/rq_evolve/evolution.py#L193))는
한 batch의 candidate를 처리합니다. 여기서 `entries`는 `list[dict]`이고, **각 dict의
"상태"는 어떤 키를 들고 있는지로 표현**됩니다(별도의 status 필드 없음). entry는 세 가지
shape를 거칩니다:

| shape (dict 키) | 의미 |
|---|---|
| `{"task", "child", "inst"}` | verified child — rollout 대상으로 살아있음 |
| `{"_retry": {task, output, reason}}` | parse는 됐으나 verify 실패 — 1회 Reflexion self-fix 대상 |
| `{"report": CandidateReport}` | terminal — 결과를 담음 |

```text
                       backend.mutate(tasks)
                               │
              ┌────────────────┼────────────────────┐
       inst!=None        source!=None            둘 다 None
        (verify OK)     (parse OK/verify 실패)    (추출 실패)
              │                │                     │
        {task,child,inst}   {_retry}         {report: mutation_failed|no_code}
              │                │
              │        _resolve_retries()
              │         fix_retry off ─────────► {report: verify_failed}
              │         fix 성공 ─► {task,child,inst}(fixed_after_retry)
              │         fix 실패 ─► {report: verify_failed "[after fix]"}
              │                │
              └───────┬────────┘
                _apply_judge()   (라벨 불일치 / none 판정 시)
                      │  ├────────────────► {report: judge_rejected}
                      │  └── judge 오류 ───► 즉시 중단 (예외 전파)
                      │ (declared == judged → child 유지)
              generate_rollouts(child)
                      │
        모든 rollout reject ─────────────► {report: rollout_failed}
        archive.try_insert()   (R_Q=0도 셀 경쟁에 참가)
              ├── elite ───────────────────► {report: inserted}
              ├── s_hat <= 0 ──────────────► {report: s_hat_zero}
              ├── R_Q <= 0 (other) ────────► {report: rq_zero}
              └── non-elite ───────────────► {report: rejected_non_elite}
```

parent가 없으면 batch 진입 직후 `{report: no_parent}` 하나로 조기 반환됩니다.

**`CandidateReport.status` 전체 어휘** ([`evolution.py:29`](../src/rq_evolve/evolution.py#L29)):
`no_parent`, `mutation_failed`, `no_code`, `verify_failed`,
`judge_rejected`, `judge_input_too_large`, `rollout_failed`,
`s_hat_zero`, `rq_zero`, `inserted`, `rejected_non_elite`.

`s_hat_zero` / `rq_zero`는 이제 **조기 반환이 아니라 삽입 실패의 사유**입니다.
R_Q = 0인 후보도 `try_insert`에 들어가 빈 셀을 차지할 수 있습니다. 고전 MAP-Elites는
fitness만으로 셀 점유를 막지 않고, "아직 아무것도 없음"은 정책이 못 푸는 프로그램보다
엄격히 나쁘기 때문입니다. 이전 게이트는 p=0/p=1 후보를 전부 거절해 부트스트랩에서
시드 8개 중 3개를 잃었고, 그와 함께 geometry·inequality GROUP과
induction·extremal_principle SKILL의 유일한 대표가 사라졌습니다.
학습 데이터 오염은 별개 장치가 막습니다 — `dataset.py`의 frontier band
(`low < p_hat < high`)가 p=0과 p=1을 이미 제외하고, 셀 경쟁이 엄격한 `>`라서
R_Q=0 champion은 점수 있는 프로그램이 오는 즉시 자리를 내줍니다.

각 report는 `source_code`(`_MAX_LOGGED_SOURCE_CHARS`에서 절단)와 `ast_findings`도
함께 싣습니다. 아카이브 스냅샷에는 champion만 들어가므로, 이 두 필드가 없으면 거절된
자식의 프로그램을 복구할 수 없고 게이트의 오탐률을 그것이 거절한 모집단에 대해 측정할
방법이 없습니다. `{"_retry": {...}}` 페이로드도 `source`/`child_id`/`ast_findings`를
**중첩해서** 나릅니다 — `_resolve_retries`는 `_apply_evaluator`와 달리 `e.clear()`를
호출하지 않으므로, 최상위 `"child"` 키는 `to_eval` 선택자로 새어나가 rollout group을
어긋나게 만듭니다.
Evaluator 인증·호출 오류는 report로 기록하지 않고 실험을 즉시 중단한다.
모든 report는 `append_evolution_log`가 `evolution_log.jsonl`에 append합니다.

## Implementation Milestones

### Milestone 1: Archive Correctness

- Compare `archive.selection_strategy=ucb` with `archive.selection_strategy=random`.
- Add behavior-signature duplicate rejection.
- Add template-signature duplicate rejection.
- Save and load archive snapshots.
- Add deterministic parent-selection tests.

### Milestone 2: Mutation Quality

- Edit `prompt_templates/mutation_system_prompt.txt` and
  `prompt_templates/mutation_user_prompt.txt`.
- Keep `prompt_templates/shots/*.txt` as offline fixtures; the live loop reads
  nothing from that directory.
- Add score-aware feedback from parent `p_hat` and uncertainty.
- Add execution-failure feedback for rejected children.

### Milestone 3: Real Backend

- Implement an OpenAI-compatible mutation backend.
- Implement a vLLM/Ollama rollout backend.
- Decide whether entropy is token entropy, span-max entropy, or semantic entropy.

### Milestone 4: verl Integration

- `VerlTrainerAdapter.fit()` wires the currently installed `.venv` `verl` into the project.
- `VerlDynamicDataset` converts `DynamicProblemDataset.snapshot()` rows to the installed verl dataset shape.
- `EvolvingSampler` runs the outer-iteration hook:
  `reevaluate champions -> inner evolution -> refresh dataset -> solver update`.
- Save archive and used-seed state next to verl checkpoints.

### Milestone 5: Evaluation

- Add math benchmark loaders.
- Add post-training evaluation scripts.
- Save per-rollout logs for failed examples.
