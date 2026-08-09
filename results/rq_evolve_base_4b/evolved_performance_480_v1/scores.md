# Evolved Performance Score

Benchmark SHA256: `b3c7c3f93a3d8d777ac7a70563b851defbe1ad4c5508fb58d3e4f9ea99a24ba7`

| global step | EPS (%) | best (%) | correct | active outer | cumulative inner | inserted |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 39.79 | 39.79 | 191/480 | bootstrap | 0 | 0 |
| 32 | 49.38 | 49.38 | 237/480 | 12 | 384 | 28 |
| 64 | 49.79 | 49.79 | 239/480 | 22 | 704 | 53 |
| 96 | 51.46 | 51.46 | 247/480 | 30 | 960 | 91 |
| 128 | 56.46 | 56.46 | 271/480 | 40 | 1280 | 118 |
| 160 | 57.92 | 57.92 | 278/480 | 47 | 1504 | 145 |
| 192 | 55.21 | 57.92 | 265/480 | 57 | 1824 | 182 |
| 224 | 52.71 | 57.92 | 253/480 | 69 | 2208 | 229 |
| 256 | 52.08 | 57.92 | 250/480 | 77 | 2464 | 271 |

## Concept Scores by Checkpoint

Each cell is accuracy on the fixed benchmark examples for that seed-program concept.

| global step | algebra/casework | algebra/transformation | combinatorics/counting | inequality/transformation | number_theory/transformation | sequence/induction |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 77.5% | 38.8% | 17.5% | 57.5% | 25.0% | 22.5% |
| 32 | 73.8% | 52.5% | 20.0% | 60.0% | 40.0% | 50.0% |
| 64 | 76.2% | 53.8% | 20.0% | 58.8% | 38.8% | 51.2% |
| 96 | 75.0% | 57.5% | 21.2% | 57.5% | 46.2% | 51.2% |
| 128 | 78.8% | 60.0% | 13.8% | 86.2% | 37.5% | 62.5% |
| 160 | 77.5% | 58.8% | 21.2% | 83.8% | 45.0% | 61.3% |
| 192 | 78.8% | 58.8% | 13.8% | 95.0% | 33.8% | 51.2% |
| 224 | 71.2% | 55.0% | 16.2% | 87.5% | 35.0% | 51.2% |
| 256 | 80.0% | 53.8% | 13.8% | 63.7% | 43.8% | 57.5% |
