# Overall benchmark scores

AVG is the equal-weight mean over all ten benchmarks; M-AVG is the seven-math-benchmark mean; G-AVG is the three-general-benchmark mean.

| MODEL | AVG | M-AVG | G-AVG | MATH | GSM8K | AMC | AIME24 | AIME25 | MINERVA | OLYMPIAD | SUPER-GPQA | MMLU-PRO | BBEH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RQ Evolve (step 160) | 47.41 | 53.66 | 32.83 | 82.20 | 92.26 | 54.76 | 20.42 | 14.27 | 61.03 | 50.66 | 29.23 | 57.20 | 12.06 |
| INFUSER-Qwen3-4B-base | 44.53 | 48.96 | 34.20 | 77.20 | 92.04 | 51.95 | 17.50 | 6.88 | 52.57 | 44.59 | 29.83 | 59.30 | 13.47 |
| INFUSER-Qwen3-8B-base | 52.97 | 59.04 | 38.80 | 85.00 | 94.16 | 66.48 | 22.60 | 24.79 | 61.40 | 58.81 | 36.94 | 66.80 | 12.66 |
| LTE-Qwen3-4B-Base | 52.50 | 59.86 | 35.33 | 86.60 | 93.25 | 61.17 | 26.98 | 25.00 | 66.91 | 59.11 | 32.63 | 60.60 | 12.76 |
| LTE-Qwen3-8B-Base | 45.52 | 48.75 | 38.00 | 89.80 | 95.15 | 59.61* | 21.88* | 18.33* | 24.63* | 31.85* | 34.73 | 66.70 | 12.56 |
| Miner-4B | 48.35 | 55.04 | 32.73 | 83.40 | 93.18 | 60.78 | 18.65 | 16.25 | 59.56 | 53.48 | 28.63 | 56.10 | 13.47 |
| Miner-8B | 51.40 | 57.52 | 37.13 | 85.20 | 94.16 | 67.42 | 20.42 | 21.35 | 60.29 | 53.78 | 33.13 | 64.30 | 13.97 |

RQ Evolve step 160 uses the user-confirmed curated row: User-confirmed selected scores for the step-160 model.

`*` indicates a stored standalone result whose GPT re-check was degraded; that cell is effectively the pre-GPT score.
