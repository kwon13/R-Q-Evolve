# Async rollout pipeline

How R-Q-Evolve's evolve-phase rollouts work after the streaming refactor, what
is (and deliberately is not) overlapped, and how to read the new metrics.

## Was the old pipeline async?

No. verl 0.7.x's rollout *engine* is asynchronous internally (agent-loop
asyncio tasks per request against the vLLM server), but the R-Q-Evolve driver
consumed it through ONE blocking `AgentLoopManager.generate_sequences` call per
inner batch: mutation, self-fix, evaluator gate, solver rollout, entropy, and
dataset refresh were six sequential whole-batch barriers, and verification of
the first completed sample waited for the LAST sample of the whole batch.

## What changed

With `async_rollout.streaming_enabled: true` (default), solver rollouts run
through `src/rq_evolve/async_rollout.py`:

- The batch is split into chunks of `chunk_size` problems (default 1 problem =
  its `num_rollouts` generations). Chunks are submitted **round-robin directly
  to the AgentLoopWorker Ray actors** (asyncio actors: any batch size, no
  divisor padding; per-request load balancing is unchanged because it happens
  in the shared request load balancer, one level below).
- At most `max_in_flight_chunks` chunks are outstanding; submission also pauses
  while the verify queue is full (`queue_maxsize`) — backpressure in both
  directions.
- The scheduler consumes chunks with `ray.wait(num_returns=1)`: **the moment a
  problem's own rollouts finish, its decoding + `\boxed{}` extraction +
  math_verify + filtering + JSONL logging run on the verify thread pool** while
  other chunks are still generating. A short problem is never blocked behind a
  long one (granularity: one chunk).
- Each chunk has a `request_timeout_s` budget, `max_retries` resubmissions, and
  on final failure produces explicit rejected records (reason `timeout` /
  `worker_error:*`). Nothing is silently dropped.
- `streaming_enabled: false` restores the legacy whole-batch path bit-for-bit.

### What stays sequential, and why

- **Entropy (uncertainty)** is one batched FSDP `compute_log_prob` forward
  AFTER the last chunk drains (`entropy_mode: deferred`). It runs on the same
  GPUs vLLM occupies (sleep mode is off on this machine), so overlapping it
  with generation would contend for compute/memory. Same memory profile as
  before.
- **Training vs generation**: verl HYBRID mode time-shares all 8 GPUs
  (generate → sync → train each step; evolve phase at each epoch boundary).
  The evolve phase *is* the trainer idle time (logged as
  `evolve/trainer_idle_s`). True rollout/training overlap is intentionally NOT
  enabled — correctness first. If it ever is, the pieces are ready: per-sample
  policy versions + the staleness gate below, and verl's own
  `algorithm.rollout_correction.*` (truncated importance sampling) +
  `rollout.calculate_log_probs` are the off-policy correction knobs.
- **This sequential fallback is also the "model needs all GPUs" answer**: a
  model that fills every GPU during rollout still trains, because rollout and
  training never run concurrently; memory pressure is tuned via
  `rollout.gpu_memory_utilization` (vLLM share) vs FSDP offload flags.

## Async-RL correctness (staleness)

Every trainer→vLLM weight sync bumps a monotonic `policy_version` (and
`adapter_version` when LoRA is on) via `PolicyVersionTracker`, which wraps
`trainer.checkpoint_manager.update_weights` — covering verl's per-step sync,
the initial sync, and the evolve-phase push. Every rollout sample is stamped
with the versions + global step + source checkpoint + start/finish times.

The consumer applies the staleness gate per sample:

- `staleness_mode: strict` — lag must be 0, else rejected (`stale_policy`);
- `staleness_mode: bounded` — lag ≤ `max_policy_lag` (default 1) allowed.

The active mode is printed at startup ("rollout mode: streaming
producer-consumer (staleness=bounded, max_policy_lag=1, ...)"). In today's
sequential flow the lag is 0 by construction (weights are static across an
evolve phase; a `sync_weights` during in-flight chunks is an assertion error),
so the gate is an invariant check that makes future overlap modes safe — stale
samples can never be scored silently.

## Filtering semantics (who leaves the p̂ denominator)

Rejected samples are excluded from p̂/R_Q; every rejection carries a reason and
a JSONL line. Because p̂ drives the curriculum, each filter that could bias it
is explicit config:

| filter | default | rationale |
|---|---|---|
| `timeout` / `worker_error` | always reject | no response exists |
| `stale_policy` | always reject (per mode/lag) | wrong policy's sample |
| `overlong` (`reject_overlong`) | reject | truncated → ambiguous outcome |
| `duplicate` (`reject_duplicates`) | detect only | identical rollouts are legitimate policy samples; dropping them biases p̂ on low-entropy problems |
| `invalid_answer` (`reject_invalid_answer`) | detect only | a boxless response DID fail; removing it would overestimate p̂ |

"Detect only" filters still increment `flag_*` metrics and are visible per
sample in the JSONL. A child whose rollouts are ALL rejected becomes a
`rollout_failed` candidate report (with the dominant reason) instead of a
silent `p_hat_zero`.

## Artifacts and metrics

Under `<default_local_dir>/rq_archive/`:

- `rollout_samples.jsonl` — one line per rollout sample, accepted AND
  rejected: status, reject_reason, correct, predicted answer, response tokens,
  latency, ts_start/ts_end, policy/adapter version, global step, source
  checkpoint. (`entropy` is null here: the deferred pass runs after logging;
  in-memory records used for scoring DO get entropy backfilled.)
- `rollout_metrics.jsonl` — one line per outer iteration: submitted/completed/
  failed/timeout/retried chunks, accepted/rejected (+ per-reason), tokens/s,
  chunk & per-sample latency avg/p50/p95, max pending, max queue depth,
  rollout idle time, verify/entropy durations, trainer idle, policy version.
- `evolution_log.jsonl` — existing problem-level log, now also with per-champion
  `frontier` decisions (`in_frontier` / `p_hat_out_of_range`).

The same metrics go to W&B per outer iteration under `rollout/*` and
`evolve/trainer_idle_s` (merged into verl's next step commit, `commit=False`).
Reward mean / KL / entropy / grad-norm are verl's own `actor/*` keys (mirrored
under `solver/*`).

## vLLM sleep/weight-sync on this machine

`free_cache_engine: false` + `enable_sleep_mode: false` stay REQUIRED: the
cumem sleep/wake allocator crashes after ~3 cycles on this box. Weight sync
does not need sleep — verl pushes weights over IPC (`update_weights`) into the
resident engine every step; **vLLM is never restarted**. With LoRA, the first
push carries the full base weights (engine boots with `load_format=dummy`) and
subsequent pushes carry only the adapter (vLLM `add_lora`).

## Known limitations

- **Timeout cancellation is best-effort.** `ray.cancel` reaches the agent-loop
  asyncio task, but the underlying vLLM request may run to completion (its
  result is discarded by chunk id). Additionally, if a per-sample task is
  cancelled exactly inside the load-balancer acquire window, verl's
  `GlobalRequestLoadBalancer` in-flight counter for one server can stay
  over-counted, slightly biasing request routing for the rest of the run.
  Watch `rollout/chunks_timeout`; if timeouts are frequent, raise
  `request_timeout_s` (the budget covers engine queueing, not just decoding)
  rather than relying on cancellation.
- **Sample JSONL `entropy` is always null** (the deferred entropy pass runs
  after logging); per-sample entropy lives only in the in-memory records used
  for scoring. Iteration-level entropy timing is in `rollout_metrics.jsonl`.
- **A verify-side crash degrades, not aborts**: a chunk whose verification
  raises (disk full, etc.) becomes all-rejected records with reason
  `worker_error:verify_error:*` and a `flag_verify_error` metric — check those
  before trusting an iteration with unusually high rejection counts.

## Commands

```bash
conda activate azr-bw-blackwell
cd /data1/yhoon113/R-Q-Evolve

# CPU, anytime (GPUs may be busy):
python -m pytest tests/                                   # unit tests
python scripts/dry_run_async.py --mode mock               # streaming proof
python scripts/preflight_check.py \
    --model-path /data1/yhoon113/qwen3-8b-base \
    --config configs/rq_evolve_base.yaml                  # model/config gate

# GPU, when free:
python scripts/dry_run_async.py --mode live --config configs/rq_evolve_smoke_lora.yaml
python scripts/train_with_verl.py --config configs/rq_evolve_smoke_lora.yaml --smoke
bash scripts/run_train_rq_evolve_base.sh                  # first real run
```
