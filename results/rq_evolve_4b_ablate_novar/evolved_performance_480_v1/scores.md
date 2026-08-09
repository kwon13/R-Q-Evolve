# Evolved Performance Score

Benchmark SHA256: `b3c7c3f93a3d8d777ac7a70563b851defbe1ad4c5508fb58d3e4f9ea99a24ba7`

| global step | EPS (%) | best (%) | correct | active outer | cumulative inner | inserted |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 39.79 | 39.79 | 191/480 | bootstrap | 0 | 0 |
| 32 | 52.71 | 52.71 | 253/480 | 10 | 320 | 31 |
| 64 | 54.58 | 54.58 | 262/480 | 20 | 640 | 70 |
| 96 | 55.42 | 55.42 | 266/480 | 27 | 864 | 107 |
| 128 | 55.62 | 55.62 | 267/480 | 36 | 1152 | 152 |

## Concept Scores by Checkpoint

Each cell is accuracy on the fixed benchmark examples for that seed-program concept.

| global step | algebra/casework | algebra/transformation | combinatorics/counting | inequality/transformation | number_theory/transformation | sequence/induction |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 77.5% | 38.8% | 17.5% | 57.5% | 25.0% | 22.5% |
| 32 | 81.2% | 52.5% | 21.2% | 66.2% | 45.0% | 50.0% |
| 64 | 75.0% | 51.2% | 25.0% | 73.8% | 52.5% | 50.0% |
| 96 | 82.5% | 53.8% | 21.2% | 66.2% | 55.0% | 53.8% |
| 128 | 87.5% | 52.5% | 16.2% | 72.5% | 53.8% | 51.2% |
