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
7. The mutation model is shown the parent source under the `in_depth` or
   `in_breadth` template and writes the child generator directly. One model
   call, no planning stage.
8. The child is verified statically and by execution, gated by the LLM
   coherence evaluator, scored, and inserted if elite.
9. Frontier champions render new training examples.
10. The installed `verl` trainer consumes those examples and updates the solver.
11. The next outer iteration re-scores every champion against the new weights
    before any of them is re-binned, replaced, or removed.

## Mutation Contract

Mutation is one-stage and free-form. `build_mutation_task` renders the parent
source into `in_depth.txt` / `in_breadth.txt` and the model returns the child
generator in one ```python``` block. There is no plan schema, no registered
family compiler, and no quarantine path.

Three gates stand between a generated child and the archive:

1. **Static lint** — `lint_generator_source` plus the determinism checks in
   `lint_mutation_generator_source`: a seeded `rng = random.Random(seed)`, no
   direct `random.*` calls, and `sorted()` around dict/set iteration, so a
   program yields the same instance in the verifier and in every rollout
   process. The compiler-notation flags (`MAX_ATTEMPTS = 200`, canonical
   `instance_data`, `str(sympy.Integer(answer))`) stay OFF: free-form
   generation does not satisfy them, and enforcing them rejected sound
   programs on shape alone.
2. **Execution** — `verify_program` runs seeds 0..N-1, requires a base-10
   integer answer from every generated mutation, and rejects a program whose
   visible problem does not vary across seeds. This carries the guarantee the
   static notation contract used to.
3. **Operator contract + evaluator** — `in_depth` must preserve the parent's
   CONCEPT_GROUP and CONCEPT_TYPE, `in_breadth` must change CONCEPT_GROUP, and
   `use_evaluator` gates seed-0 coherence before any solver rollout is spent.

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
                _apply_evaluator()   (INVALID 판정 시)
                      │  ├────────────────► {report: evaluator_rejected}
                      │  └── evaluator 오류 ─► 즉시 중단 (예외 전파)
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
`no_parent`, `mutation_failed`, `no_code`, `verify_failed`,
`evaluator_rejected`, `rollout_failed`, `p_hat_zero`, `rq_zero`, `inserted`,
`rejected_non_elite`.
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

- Edit `prompt_templates/in_depth.txt` and `prompt_templates/in_breadth.txt`.
- Keep `prompt_templates/shots/*.txt` as offline fixtures; live mutation calls
  omit content-rich shots to avoid example copying.
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
