# Evolved Performance Score

This benchmark measures the Solver at every saved training checkpoint on one
fixed in-distribution anchor set. It is deliberately different from the public
MATH benchmarks: the generator families are the six files in `seed_programs/`,
while the rendered instances are fixed once in a hashed JSONL.

## Score

The default benchmark contains 40 unique instances from each of six seed
programs (240 total). Each response is greedy-decoded and graded by the same
last-`\\boxed{}` + `answers_match` path used by R-Q-Evolve.

```text
EPS(step) = 100 * mean(per-seed-program accuracy at step)
```

Because every program contributes 40 rows, this macro average is also the raw
number correct divided by 240.

## One-command run

The default points at the completed 4B base run and infers the base-model path
from `configs/rq_evolve_4b_base.yaml`:

```bash
cd /data1/yhoon113/R-Q-Evolve
bash scripts/run_evolved_performance.sh
```

For another run:

```bash
BASE=/path/to/run \
BASE_MODEL=/path/to/starting-model \
CONFIG=/path/to/run-config.yaml \
GPU_LIST=0,1,2,3 \
bash scripts/run_evolved_performance.sh
```

Useful controls:

```bash
# Only selected saved models
STEPS_LIST=32,64,128 GPU_LIST=0,1 bash scripts/run_evolved_performance.sh

# Re-evaluate results that already have summary.json
FORCE=1 bash scripts/run_evolved_performance.sh

# Exclude the step-0 base model
INCLUDE_BASE=0 bash scripts/run_evolved_performance.sh
```

The script reuses existing `hf_merged/` models, merges actor shards only when
needed, and fans one vLLM process per selected GPU.

## Structural-OOD companion benchmark

The evaluation-only programs in
`challenge_seed_programs/structural_ood_v2/` preserve the original six
`GROUP/SKILL` pairs but require different mathematical solution structures.
They were not part of the completed run's training generator directory. The
fixed companion benchmark contains 40 examples per program (240 total) and is
stored separately at `benchmarks/evolved_performance_structural_ood_v2/`.

Evaluate the base model and every saved checkpoint with:

```bash
cd /data1/yhoon113/R-Q-Evolve
GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_evolved_performance_challenge.sh
```

This performs new checkpoint inference because the 240 challenge prompts are
new, but it does not retrain any model. Results are written to
`<RUN>/evolved_performance_structural_ood_v2/` and never overwrite the Seed-ID
results. Report this trajectory as structural transfer/OOD rather than ID.

Once an OOD trajectory exists, combine it with the Seed-ID result without any
additional inference:

```bash
python scripts/plot_combined_evolved_performance.py \
  --id-results-dir rq_output/rq_evolve_base_4b/evolved_performance \
  --ood-results-dir rq_output/rq_evolve_base_4b/evolved_performance_structural_ood_v2 \
  --output-dir rq_output/rq_evolve_base_4b/evolved_performance_combined_id_ood_v2
```

The balanced score gives the two 240-item distributions equal weight:
`0.5 × Seed-ID EPS + 0.5 × Structural-OOD EPS`.

## Canonical 480-problem benchmark

To evaluate Seed-ID and Structural-OOD v2 as one fixed benchmark and produce
the original single-curve performance figure, run:

```bash
cd /data1/yhoon113/R-Q-Evolve
GPU_LIST=0,1,2,3,4,5,6,7 \
  bash scripts/run_evolved_performance_480.sh
```

This creates and validates the immutable union at
`benchmarks/evolved_performance_480_v1/` and evaluates all 480 prompts in one
vLLM job per saved model. The single EPS is both the macro average over the 12
equally sized generators and `100 × correct / 480`. Checkpoint annotations
aggregate the two generators sharing each GROUP/SKILL label, so every concept
score is measured over 80 fixed problems. Results and the original-style plot
are written to `<RUN>/evolved_performance_480_v1/`.

## Outputs

```text
benchmarks/evolved_performance_seed_id_v1/
  benchmark.jsonl       immutable 240 examples
  manifest.json         source hashes + benchmark SHA256

<RUN>/evolved_performance/
  global_step_0/        base-model details + summary
  global_step_*/        checkpoint details + summary
  overlap_audit.json    retrospective known-overlap audit
  trajectory.json       plot-ready checkpoint/evolution join
  scores.md             EPS trajectory + absolute concept scores per checkpoint
  high_rq_problems.md/json  prominent high-R_Q GROUP/SKILL events
  concept_score_changes.md/json  per-concept accuracy deltas between checkpoints
  evolved_performance.png
  evolved_performance.svg
```

The plot uses:

- bottom X: saved-model `global_step`;
- left Y: checkpoint EPS and best-so-far EPS;
- right Y: cumulative evaluated inner proposals.

Up to eight amber callouts identify prominent high-R_Q events on the evolution
timeline. They are selected from the top 10% of per-outer-iteration R_Q maxima,
independently of saved-model checkpoints. Each callout contains only R_Q and
the generated program's GROUP/SKILL; no problem statement is rendered.

Separate blue callouts at saved-model checkpoints show the two largest absolute
Seed-ID label-score changes from the preceding checkpoint, for example
`NT/Trans: 67.5→87.5 (+20.0%p)`. The complete concept report records every
improved, unchanged, and declining label.

The problem report can also be generated before checkpoint EPS evaluation:

```bash
python scripts/summarize_high_rq_problems.py \
  --run-dir rq_output/rq_evolve_base_4b
```

The evolution overlay is not inferred from checkpoint spacing.
`rollout_metrics.jsonl` supplies each outer iteration's actual trainer step,
and `evolution_log.jsonl` supplies its attempted/inserted counts.

## Leakage interpretation

High seed numbers guarantee numerical seed separation from the normal early
training sequence, but not instance separation: finite generators can map two
different seeds to the same problem. For completed runs the command writes an
exact retrospective audit by reconstructing all consumed seeds of the original
seed programs from `rq_used_seeds.json`.

That audit is conservative but incomplete. It catches exact collisions from
the original seed generators; it cannot reconstruct a semantically equivalent
problem or an identical statement emitted by a mutated generator. Therefore a
retrospective score should be described as a fixed **Seed-ID performance
anchor**, not as a contamination-free held-out test score.
