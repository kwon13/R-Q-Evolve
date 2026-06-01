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
7. A mutation prompt is built.
8. The backend returns generated Python source.
9. The child is verified, evaluated, and inserted if elite.
10. Frontier champions render new training examples.
11. The installed `verl` trainer consumes those examples and updates the solver.
12. The next outer iteration re-scores champions under the updated solver.

## Implementation Milestones

### Milestone 1: Archive Correctness

- Compare `archive.selection_strategy=ucb` with `archive.selection_strategy=random`.
- Add behavior-signature duplicate rejection.
- Add template-signature duplicate rejection.
- Save and load archive snapshots.
- Add deterministic parent-selection tests.

### Milestone 2: Mutation Quality

- Edit `prompt_templates/in_depth.txt`, `prompt_templates/in_breadth.txt`, and `prompt_templates/crossover.txt`.
- Edit `prompt_templates/shots/in_depth.txt`, `prompt_templates/shots/in_breadth.txt`, and `prompt_templates/shots/crossover.txt` for mutation-specific few-shot examples.
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
