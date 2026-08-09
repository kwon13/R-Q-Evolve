# Evolved Performance Score

Benchmark SHA256: `e2076da43023e4a4b139ffa1a7374e4248e2e558f270c44d03212e8bb36767c4`

| global step | EPS (%) | best (%) | correct | active outer | cumulative inner | inserted |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 15.83 | 15.83 | 38/240 | bootstrap | 0 | 0 |
| 32 | 15.83 | 15.83 | 38/240 | 12 | 384 | 28 |
| 64 | 15.42 | 15.83 | 37/240 | 22 | 704 | 53 |
| 96 | 22.08 | 22.08 | 53/240 | 30 | 960 | 91 |
| 128 | 35.42 | 35.42 | 85/240 | 40 | 1280 | 118 |
| 160 | 32.92 | 35.42 | 79/240 | 47 | 1504 | 145 |
| 192 | 30.42 | 35.42 | 73/240 | 57 | 1824 | 182 |
| 224 | 28.33 | 35.42 | 68/240 | 69 | 2208 | 229 |
| 256 | 23.33 | 35.42 | 56/240 | 77 | 2464 | 271 |

## Concept Scores by Checkpoint

Each cell is accuracy on the fixed benchmark examples for that seed-program concept.

| global step | sequence/induction | algebra/transformation | inequality/transformation | number_theory/transformation | combinatorics/counting | algebra/casework |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.5% | 7.5% | 10.0% | 20.0% | 0.0% | 50.0% |
| 32 | 0.0% | 12.5% | 12.5% | 27.5% | 0.0% | 42.5% |
| 64 | 2.5% | 10.0% | 12.5% | 15.0% | 0.0% | 52.5% |
| 96 | 0.0% | 20.0% | 20.0% | 37.5% | 0.0% | 55.0% |
| 128 | 27.5% | 25.0% | 77.5% | 37.5% | 0.0% | 45.0% |
| 160 | 25.0% | 17.5% | 75.0% | 37.5% | 0.0% | 42.5% |
| 192 | 2.5% | 10.0% | 87.5% | 25.0% | 0.0% | 57.5% |
| 224 | 2.5% | 10.0% | 70.0% | 27.5% | 0.0% | 60.0% |
| 256 | 12.5% | 12.5% | 30.0% | 30.0% | 0.0% | 55.0% |
