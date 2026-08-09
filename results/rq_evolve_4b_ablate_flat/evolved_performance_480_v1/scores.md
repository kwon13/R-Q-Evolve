# Evolved Performance Score

Benchmark SHA256: `b3c7c3f93a3d8d777ac7a70563b851defbe1ad4c5508fb58d3e4f9ea99a24ba7`

| global step | EPS (%) | best (%) | correct | active outer | cumulative inner | inserted |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 39.79 | 39.79 | 191/480 | bootstrap | 0 | 0 |
| 32 | 52.92 | 52.92 | 254/480 | 9 | 288 | 34 |
| 64 | 55.21 | 55.21 | 265/480 | 14 | 448 | 65 |
| 96 | 56.04 | 56.04 | 269/480 | 18 | 576 | 87 |
| 128 | 54.17 | 56.04 | 260/480 | 23 | 736 | 118 |

## Concept Scores by Checkpoint

Each cell is accuracy on the fixed benchmark examples for that seed-program concept.

| global step | algebra/casework | algebra/transformation | combinatorics/counting | inequality/transformation | number_theory/transformation | sequence/induction |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 77.5% | 38.8% | 17.5% | 57.5% | 25.0% | 22.5% |
| 32 | 70.0% | 52.5% | 27.5% | 62.5% | 55.0% | 50.0% |
| 64 | 70.0% | 51.2% | 25.0% | 76.2% | 57.5% | 51.2% |
| 96 | 77.5% | 48.8% | 21.2% | 77.5% | 61.3% | 50.0% |
| 128 | 77.5% | 52.5% | 17.5% | 66.2% | 60.0% | 51.2% |
