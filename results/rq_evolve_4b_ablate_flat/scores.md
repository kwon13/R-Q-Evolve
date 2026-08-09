# R-Q-Evolve eval — pass@1 (%) , R-Zero-aligned grading

Base: `/data1/yhoon113/R-Q-Evolve/rq_output/rq_evolve_4b_ablate_flat`  |  steps: 32, 64, 96, 128

## pass@1 (final — includes GPT-4o re-check)

| step | math500 | gsm8k | amc23 | aime24 | aime25 | minerva_math | olympiadbench | AVG |
|---|---|---|---|---|---|---|---|---|
| 32 | 77.40 | 91.96 | 51.72 | 16.56 | 8.75 | 56.25 | 43.85 | 49.50 |
| 64 | 78.20 | 92.12 | 61.41 | 8.75 | 10.10 | 55.15 | 32.59 | 48.33 |
| 96 | 67.20 | 92.04 | 58.13 | 14.17 | 9.48 | 25.74 | 32.00 | 42.68 |
| 128 | 66.80 | 92.04 | 57.19 | 15.94 | 6.98 | 41.54 | 46.37 | 46.69 |

## pass@1 (pre-GPT — math_verify only; AVG cell shows total GPT flips)

| step | math500 | gsm8k | amc23 | aime24 | aime25 | minerva_math | olympiadbench | AVG |
|---|---|---|---|---|---|---|---|---|
| 32 | 67.20 | 91.66 | 51.41 | 11.98 | 8.54 | 21.69 | 31.11 | 40.51 (+285) |
| 64 | 67.20 | 92.04 | 60.78 | 8.75 | 9.17 | 24.63 | 32.59 | 42.17 (+156) |
| 96 | 67.20 | 92.04 | 58.13 | 14.17 | 9.48 | 25.74 | 32.00 | 42.68 |
| 128 | 66.80 | 92.04 | 57.11 | 13.96 | 3.75 | 25.00 | 32.44 | 41.59 (+190) |
