# Evolved Performance Score

Benchmark SHA256: `b3c7c3f93a3d8d777ac7a70563b851defbe1ad4c5508fb58d3e4f9ea99a24ba7`

| global step | EPS (%) | best (%) | correct | active outer | cumulative inner | inserted |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 39.79 | 39.79 | 191/480 | bootstrap | 0 | 0 |
| 32 | 54.38 | 54.38 | 261/480 | 9 | 288 | 27 |
| 64 | 54.38 | 54.38 | 261/480 | 15 | 480 | 36 |
| 96 | 55.42 | 55.42 | 266/480 | 19 | 608 | 42 |
| 128 | 55.21 | 55.42 | 265/480 | 23 | 736 | 46 |

## Concept Scores by Checkpoint

Each cell is accuracy on the fixed benchmark examples for that seed-program concept.

| global step | algebra/casework | algebra/transformation | combinatorics/counting | inequality/transformation | number_theory/transformation | sequence/induction |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 77.5% | 38.8% | 17.5% | 57.5% | 25.0% | 22.5% |
| 32 | 80.0% | 51.2% | 27.5% | 75.0% | 42.5% | 50.0% |
| 64 | 76.2% | 51.2% | 25.0% | 85.0% | 40.0% | 48.8% |
| 96 | 76.2% | 51.2% | 23.8% | 92.5% | 37.5% | 51.2% |
| 128 | 78.8% | 52.5% | 25.0% | 93.8% | 27.5% | 53.8% |
