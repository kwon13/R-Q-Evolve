# Balanced Evolve Performance

`Balanced = 0.5 × Seed-ID EPS + 0.5 × Structural-OOD-v2 EPS`. Both sets contain 240 balanced examples, so this is also pooled accuracy over 480 problems.

| global step | Seed-ID | OOD-v2 | Balanced | best Balanced | correct |
|---:|---:|---:|---:|---:|---:|
| 0 | 62.50% | 15.83% | 39.17% | 39.17% | 188/480 |
| 32 | 79.58% | 15.83% | 47.71% | 47.71% | 229/480 |
| 64 | 81.25% | 15.42% | 48.33% | 48.33% | 232/480 |
| 96 | 77.92% | 22.08% | 50.00% | 50.00% | 240/480 |
| 128 | 80.00% | 35.42% | 57.71% | 57.71% | 277/480 |
| 160 | 84.58% | 32.92% | 58.75% | 58.75% | 282/480 |
| 192 | 79.17% | 30.42% | 54.79% | 58.75% | 263/480 |
| 224 | 79.58% | 28.33% | 53.96% | 58.75% | 259/480 |
| 256 | 79.17% | 23.33% | 51.25% | 58.75% | 246/480 |

## Balanced Concept Scores

Each cell is the 50:50 average of the corresponding Seed-ID and OOD-v2 concept accuracies.

| global step | algebra/casework | algebra/transformation | combinatorics/counting | inequality/transformation | number_theory/transformation | sequence/induction |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 75.0% | 41.2% | 17.5% | 55.0% | 25.0% | 21.2% |
| 32 | 71.2% | 53.8% | 18.8% | 56.2% | 36.2% | 50.0% |
| 64 | 76.2% | 52.5% | 20.0% | 56.2% | 33.8% | 51.2% |
| 96 | 77.5% | 58.8% | 22.5% | 60.0% | 31.2% | 50.0% |
| 128 | 72.5% | 61.2% | 18.8% | 88.8% | 41.2% | 63.8% |
| 160 | 71.2% | 57.5% | 25.0% | 87.5% | 48.8% | 62.5% |
| 192 | 78.8% | 53.8% | 12.5% | 93.8% | 38.8% | 51.2% |
| 224 | 78.8% | 53.8% | 15.0% | 85.0% | 40.0% | 51.2% |
| 256 | 76.2% | 52.5% | 11.2% | 65.0% | 46.2% | 56.2% |
