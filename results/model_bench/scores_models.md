# Standalone model benchmarks

Root: `/data1/yhoon113/R-Q-Evolve/rq_output/model_bench`

## math — pass@1 (%), R-Zero grading with GPT-4o re-check

| model | math500 | gsm8k | amc23 | aime24 | aime25 | minerva_math | olympiadbench | AVG |
|---|---|---|---|---|---|---|---|---|
| `INFUSER-Qwen3-4B-base` | 77.20 | 92.04 | 51.95 | 17.50 | 6.88 | 52.57 | 44.59 | 48.96 |
| `INFUSER-Qwen3-8B-base` | 85.00 | 94.16 | 66.48 | 22.60 | 24.79 | 61.40 | 58.81 | 59.04 |
| `LTE-Qwen3-4B-Base` | 86.60 | 93.25 | 61.17 | 26.98 | 25.00 | 66.91 | 59.11 | 59.86 |
| `LTE-Qwen3-8B-Base` | 89.80 | 95.15 | 59.61* | 21.88* | 18.33* | 24.63* | 31.85* | 48.75 |
| `Miner-4B` | 83.40 | 93.18 | 60.78 | 18.65 | 16.25 | 59.56 | 53.48 | 55.04 |
| `Miner-8B` | 85.20 | 94.16 | 67.42 | 20.42 | 21.35 | 60.29 | 53.78 | 57.52 |

`*` = the GPT judge failed on more than 10% of its calls for that benchmark, so the value is effectively pre-GPT. Repair with scripts/rerun_gpt_recheck.py before comparing.

## math — pass@1 (%), math_verify only

| model | math500 | gsm8k | amc23 | aime24 | aime25 | minerva_math | olympiadbench | AVG |
|---|---|---|---|---|---|---|---|---|
| `INFUSER-Qwen3-4B-base` | 65.80 | 92.04 | 51.95 | 14.37 | 5.94 | 22.06 | 30.52 | 40.38 |
| `INFUSER-Qwen3-8B-base` | 65.20 | 92.80 | 62.58 | 17.60 | 14.90 | 26.10 | 31.26 | 44.35 |
| `LTE-Qwen3-4B-Base` | 68.40 | 92.12 | 54.77 | 23.33 | 19.27 | 20.59 | 31.41 | 44.27 |
| `LTE-Qwen3-8B-Base` | 68.40 | 94.16 | 59.61 | 21.88 | 18.33 | 24.63 | 31.85 | 45.55 |
| `Miner-4B` | 67.60 | 87.95 | 54.84 | 15.73 | 14.90 | 22.79 | 35.85 | 42.81 |
| `Miner-8B` | 71.40 | 93.71 | 64.22 | 17.71 | 17.08 | 26.10 | 36.74 | 46.71 |

## general-domain — accuracy (%)

| model | MMLU-Pro | SuperGPQA | BBEH | AVG |
|---|---|---|---|---|
| `INFUSER-Qwen3-4B-base` | 59.30 | 29.83 | 13.47 | 34.20 |
| `INFUSER-Qwen3-8B-base` | 66.80 | 36.94 | 12.66 | 38.80 |
| `LTE-Qwen3-4B-Base` | 60.60 | 32.63 | 12.76 | 35.33 |
| `LTE-Qwen3-8B-Base` | 66.70 | 34.73 | 12.56 | 38.00 |
| `Miner-4B` | 56.10 | 28.63 | 13.47 | 32.73 |
| `Miner-8B` | 64.30 | 33.13 | 13.97 | 37.13 |

## general-domain — unparsed answer rate (%)

| model | MMLU-Pro | SuperGPQA | BBEH | AVG |
|---|---|---|---|---|
| `INFUSER-Qwen3-4B-base` | 0.20 | 0.70 | 10.05 | 3.65 |
| `INFUSER-Qwen3-8B-base` | 0.40 | 0.30 | 9.55 | 3.42 |
| `LTE-Qwen3-4B-Base` | 3.20 | 5.91 | 27.24 | 12.11 |
| `LTE-Qwen3-8B-Base` | 2.20 | 2.60 | 27.84 | 10.88 |
| `Miner-4B` | 8.40 | 9.11 | 17.69 | 11.73 |
| `Miner-8B` | 1.90 | 3.30 | 17.49 | 7.56 |

## general-domain — truncated rate (%)

| model | MMLU-Pro | SuperGPQA | BBEH | AVG |
|---|---|---|---|---|
| `INFUSER-Qwen3-4B-base` | 0.20 | 0.40 | 9.25 | 3.28 |
| `INFUSER-Qwen3-8B-base` | 0.10 | 0.20 | 8.84 | 3.05 |
| `LTE-Qwen3-4B-Base` | 3.50 | 6.61 | 28.84 | 12.98 |
| `LTE-Qwen3-8B-Base` | 2.50 | 2.70 | 29.55 | 11.58 |
| `Miner-4B` | 1.50 | 1.90 | 15.98 | 6.46 |
| `Miner-8B` | 1.20 | 2.30 | 17.49 | 7.00 |

