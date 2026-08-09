# Results

Everything the runs produced except model weights: curricula, per-candidate
telemetry, benchmark generations and their grades. Mirrors the layout of
`rq_output/` (which is gitignored) with `hf_merged/`, `actor/`, `ray_tmp/` and
every `.safetensors` / `.pt` removed — 182 GB down to 1.1 GB.

## Layout

    <run>/rq_archive/
        archive_iter<N>.json      champions after outer iteration N: source,
                                  lineage, p_hat, h_score, rq_score, niche
        archive.json              resume state (latest)
        evolution_log.jsonl       one line per outer iteration: metrics plus a
                                  CandidateReport per attempt, including the
                                  rejected child's source_code and ast_findings
        rollout_samples.jsonl     per-rollout grade, reject reason, entropy
    <run>/global_step_<N>/
        eval/<bench>/summary.json     pass@1, pass_at_1_pre_gpt, gpt_flips
        eval/<bench>/details.jsonl    every prompt, response and grade
        eval_general/<bench>/...      same shape, plus unparsed/truncated rates
    model_bench/<model>/            standalone HF checkpoints, same shape
    visualize/, evolved_performance*/  figures and the fixed Seed-ID benchmark

## Runs

| Directory | What it is |
|---|---|
| `rq_evolve_base_4b` | Qwen3-4B-Base, 256 steps. The control every ablation is measured against. |
| `rq_evolve_4b_ablate_novar` | `select_ignores_variance` — R_Q ranked by H alone |
| `rq_evolve_4b_ablate_nounc` | `select_ignores_uncertainty` — R_Q ranked by s(1-s) alone |
| `rq_evolve_4b_ablate_noreeval` | champions never rescored or evicted |
| `rq_evolve_4b_ablate_flat` | no GROUP x SKILL binning; 48 slots as a top-K pool |
| `model_bench` | downloaded checkpoints (INFUSER, Miner, LTE, Qwen3 bases) |

## Reading the numbers

**`pass_at_1` is post-GPT.** The math suite grades with math_verify and then
asks gpt-4o about every example it scored 0, so `pass_at_1_pre_gpt` and
`pass_at_1` can differ by 30 points on minerva_math. Compare like with like.

**Check `gpt_flips` before trusting a post-GPT number.** A judge that could not
be reached scores everything "No", which is indistinguishable from agreement —
`gpt_recheck_degraded: true` marks the cases that were caught. Five benchmarks
under `model_bench/LTE-Qwen3-8B-Base/eval/` are in that state and are
effectively pre-GPT.

**`gpu_memory_utilization` belongs to the comparison.** Re-evaluating one
checkpoint at 0.85 twice moved amc23 by 0.16 points; at 0.45 it moved 2.19.
KV-cache size changes batching, batching changes floating-point reduction
order. Runs here used 0.85 except `model_bench`, which used 0.45.

**On general-domain accuracy, read `unparsed_rate` alongside it.** An answer
the grader cannot parse still scores 1/10 of the time through the random-letter
fallback, so accuracy alone cannot be told apart from guessing.
