# R-Q-Evolve eval — pass@1 (%) , R-Zero-aligned grading

Base: `/data1/yhoon113/R-Q-Evolve/rq_output/rq_evolve_4b_ablate_noreeval`  |  steps: 32, 64, 96, 128

## pass@1 (final — includes GPT-4o re-check)

| step | math500 | gsm8k | amc23 | aime24 | aime25 | minerva_math | olympiadbench | AVG |
|---|---|---|---|---|---|---|---|---|
| 32 | 79.20 | 91.96 | 56.64 | 17.60 | 8.75 | 55.51 | 45.48 | 50.74 |
| 64 | 77.60 | 92.42 | 55.08 | 14.17 | 11.67 | 56.25 | 32.59 | 48.54 |
| 96 | 65.80 | 92.65 | 54.53 | 12.60 | 11.15 | 23.90 | 32.74 | 41.91 |
| 128 | 68.20 | 92.19 | 53.83 | 14.58 | 13.02 | 40.81 | 44.00 | 46.66 |

## pass@1 (pre-GPT — math_verify only; AVG cell shows total GPT flips)

| step | math500 | gsm8k | amc23 | aime24 | aime25 | minerva_math | olympiadbench | AVG |
|---|---|---|---|---|---|---|---|---|
| 32 | 67.40 | 91.66 | 55.78 | 13.65 | 8.23 | 25.00 | 32.74 | 42.07 (+286) |
| 64 | 66.40 | 92.19 | 53.91 | 14.17 | 11.46 | 23.90 | 32.59 | 42.09 (+164) |
| 96 | 65.80 | 92.65 | 54.53 | 12.60 | 11.15 | 23.90 | 32.74 | 41.91 |
| 128 | 68.20 | 92.19 | 52.89 | 11.46 | 10.52 | 24.26 | 31.41 | 41.56 (+196) |
