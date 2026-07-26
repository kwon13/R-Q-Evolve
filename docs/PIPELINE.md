# Pipeline Notes

이 문서는 `evo-sample`의 구현을 직접 다시 만들기 위한 대응표입니다.

## Source-To-Skeleton Mapping

| evo-sample | R-Q-Evolve | 역할 |
|---|---|---|
| `rq_questioner/program.py` | `src/rq_evolve/program.py` | `generate(seed)` 프로그램 실행 단위 |
| `rq_questioner/map_elites.py` | `src/rq_evolve/archive.py` | H×D MAP-Elites archive |
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
5. The program competes for a MAP-Elites cell.
6. Parent programs are sampled from occupied cells.
7. Existing R_Q rollouts contribute at most one shortest-correct and one
   lowest-entropy-wrong trace; no additional Solver rollout is requested.
8. The mutation model first produces a validated `MutationPlan` JSON and then
   implements that plan as Python.
9. The child is verified, evaluated against its plan, and inserted if elite.
10. Frontier champions render new training examples.
11. The installed `verl` trainer consumes those examples and updates the solver.
12. The next outer iteration snapshots champion pass rates, re-scores the fixed
    cohort, computes delta-p, and only then moves/replaces/removes champions.

## Metacognitive Control

`metacognition.enabled=true` does not create a second archive. The single live
MAP remains the curriculum and fitness state; the extra data is telemetry
attached to its programs plus a small operator EMA.

```text
existing G rollouts
  -> shortest correct + lowest-entropy wrong (each <= 4096 tokens)
  -> Monitoring + MutationPlan JSON
  -> planned Python generator
  -> static/runtime/LLM evaluation
  -> existing R_Q insertion rule

next Solver update
  -> snapshot p_hat for the current champion cohort
  -> re-evaluate that same cohort
  -> compute delta-p globally/by group/by type/by creating operator
  -> update depth/breadth EMA
  -> only now re-bin, replace, or remove R_Q=0 champions
```

The plan schema requires independent insight/brute routes, a live
confident-wrong decoy, `MAX_ATTEMPTS=200`, seed-local deterministic randomness,
and one base-10 integer answer. Accidental `decoy == answer` collisions are
resampled before the accepted instance asserts inequality.

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
                _apply_evaluator()   (INVALID 판정 시)
                      │  ├────────────────► {report: evaluator_rejected}
                      │  └── 예외 ─────────► {report: evaluator_error}
                      │ (valid → child 유지)
              generate_rollouts(child)
                      │
        모든 rollout reject ─────────────► {report: rollout_failed}
        p_hat <= 0 ──────────────────────► {report: p_hat_zero}
        R_Q <= 0 (other boundary) ───────► {report: rq_zero}
        archive.try_insert()
              ├── elite ───────────────────► {report: inserted}
              └── non-elite ───────────────► {report: rejected_non_elite}
```

parent가 없으면 batch 진입 직후 `{report: no_parent}` 하나로 조기 반환됩니다.

**`CandidateReport.status` 전체 어휘** ([`evolution.py:29`](../src/rq_evolve/evolution.py#L29)):
`no_parent`, `mutation_failed`, `no_code`, `verify_failed`, `evaluator_error`,
`evaluator_rejected`, `rollout_failed`, `p_hat_zero`, `rq_zero`, `inserted`,
`rejected_non_elite`.
모든 report는 `append_evolution_log`가 `evolution_log.jsonl`에 append합니다.

## Implementation Milestones

### Milestone 1: Archive Correctness

- Compare `archive.selection_strategy=ucb` with `archive.selection_strategy=random`.
- Add behavior-signature duplicate rejection.
- Add template-signature duplicate rejection.
- Save and load archive snapshots.
- Add deterministic parent-selection tests.

### Milestone 2: Mutation Quality

- Edit `prompt_templates/in_depth.txt` and `prompt_templates/in_breadth.txt`.
- Edit `prompt_templates/shots/in_depth.txt` and `prompt_templates/shots/in_breadth.txt` for mutation-specific few-shot examples.
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
