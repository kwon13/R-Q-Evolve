# Evolved Performance Ablation (480 problems)

| global step | R-Q-Evolve (full) | Flat archive (no MAP bins) | Without reevaluation | Without uncertainty | Without variance |
|---:|---:|---:|---:|---:|---:|
| 0 | 39.79% | 39.79% | 39.79% | 39.79% | 39.79% |
| 32 | 49.38% | 52.92% | 54.38% | 53.13% | 52.71% |
| 64 | 49.79% | 55.21% | 54.38% | 53.12% | 54.58% |
| 96 | 51.46% | 56.04% | 55.42% | 53.13% | 55.42% |
| 128 | 56.46% | 54.17% | 55.21% | 54.38% | 55.62% |

## Difference from R-Q-Evolve (full) at step 128

| ablation | EPS | delta vs. Full |
|---|---:|---:|
| Flat archive (no MAP bins) | 54.17% | -2.29%p |
| Without reevaluation | 55.21% | -1.25%p |
| Without uncertainty | 54.38% | -2.08%p |
| Without variance | 55.62% | -0.83%p |

## Standard Math Benchmarks at step 128

Final pass@1 after the stored R-Zero-aligned GPT-4o re-check. AVG is the macro average over the seven benchmark columns.

| method | math500 | gsm8k | amc23 | aime24 | aime25 | minerva_math | olympiadbench | AVG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R-Q-Evolve (full) | 81.80% | 92.12% | 53.59% | 20.42% | 14.06% | 61.40% | 48.15% | 53.08% |
| Flat archive (no MAP bins) | 66.80% | 92.04% | 57.19% | 15.94% | 6.98% | 41.54% | 46.37% | 46.69% |
| Without reevaluation | 68.20% | 92.19% | 53.83% | 14.58% | 13.02% | 40.81% | 44.00% | 46.66% |
| Without uncertainty | 78.40% | 91.81% | 55.16% | 16.67% | 13.23% | 54.78% | 45.63% | 50.81% |
| Without variance | 79.40% | 92.04% | 53.98% | 12.29% | 4.90% | 56.99% | 45.93% | 49.36% |
