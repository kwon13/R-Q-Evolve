# Evolved Performance Score

Benchmark SHA256: `b3c7c3f93a3d8d777ac7a70563b851defbe1ad4c5508fb58d3e4f9ea99a24ba7`

| global step | EPS (%) | best (%) | correct | active outer | cumulative inner | inserted |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 39.79 | 39.79 | 191/480 | bootstrap | 0 | 0 |
| 32 | 53.13 | 53.13 | 255/480 | 11 | 352 | 35 |
| 64 | 53.12 | 53.13 | 255/480 | 21 | 672 | 64 |
| 96 | 53.13 | 53.13 | 255/480 | 31 | 992 | 98 |
| 128 | 54.38 | 54.38 | 261/480 | 38 | 1216 | 122 |

## Concept Scores by Checkpoint

Each cell is accuracy on the fixed benchmark examples for that seed-program concept.

| global step | algebra/casework | algebra/transformation | combinatorics/counting | inequality/transformation | number_theory/transformation | sequence/induction |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 77.5% | 38.8% | 17.5% | 57.5% | 25.0% | 22.5% |
| 32 | 77.5% | 48.8% | 26.2% | 72.5% | 43.8% | 50.0% |
| 64 | 78.8% | 50.0% | 22.5% | 77.5% | 40.0% | 50.0% |
| 96 | 75.0% | 52.5% | 22.5% | 87.5% | 30.0% | 51.2% |
| 128 | 81.2% | 53.8% | 25.0% | 87.5% | 27.5% | 51.2% |
