# Trusted Stage-2 Luna audit

This audit checks whether the two-stage mutation interface is understandable,
whether the trusted assembler accepts sound cores and rejects malformed ones,
and whether untargeted children differ from their parents. It does not run a
local model or occupy a GPU.

The samples were written by independent Luna audit turns against the exact
production prompt builders, then passed through `parse_family_plan`,
`compile_stage2_reply`, and `RQEvolver.verify_program` with five verification
seeds. This is a prompt-compliance audit, not an estimate of Qwen training-time
yield.

## Latest fresh replicate

Replicate B generated one fresh untargeted child for each of the seven diagonal
seed programs. No target domain, problem type, or cell was supplied.

| Metric | Result |
| --- | ---: |
| Stage-1 parse | 7/7 |
| Stage-2 compile after the control-dependency fix | 7/7 |
| Five-seed production verification | 7/7 |
| Exact family uniqueness | 100% |
| Structural family uniqueness | 100% |
| Pairwise family similarity, mean / median / max | 0.401 / 0.374 / 0.732 |
| Family-parent similarity, mean / median | 0.269 / 0.251 |
| Family-parent token Jaccard, mean / median | 0.103 / 0.100 |
| Source-parent similarity, mean | 0.079 |
| Model-owned skeleton-parent similarity, mean | 0.473 |
| Child stayed in its exact parent cell | 2/7 |
| Child cell absent from the seven seed cells | 5/7 |

| Parent cell | Child cell | Family-parent similarity | Final result |
| --- | --- | ---: | --- |
| algebra x decision | algebra x function | 0.241 | pass |
| geometry x search | geometry x decision | 0.127 | pass |
| number_theory x counting | number_theory x search | 0.311 | pass |
| discrete_mathematics x optimization | discrete_mathematics x counting | 0.488 | pass |
| applied_mathematics x function | applied_mathematics x function | 0.130 | pass |
| calculus x function | calculus x function | 0.336 | pass |
| precalculus x optimization | precalculus x function | 0.251 | pass |

The discrete-mathematics and precalculus cores initially triggered false `P1`
findings because the dependency graph did not connect a loop bound to a
loop-carried accumulator or recurrence. Both cores already used legitimate
independent routes. After adding narrowly scoped control-dependency edges for
accumulators in bounded `for` and `while` loops, the unchanged replies compiled
and passed all five verification seeds. Constant loop bodies, identical
constant branches, and unrelated stated parameters remain rejectable in the
regression suite.

## Earlier replicate

The first seven Luna candidates also parsed, compiled, and verified 7/7 with
100% exact and structural family uniqueness. Their mean family-parent
similarity was 0.366, but all seven stayed in their parent's cell. This is why
protocol compliance and MAP expansion are reported separately.

## Interpretation limits

The trusted assembler verifies output shape, bounded execution, deterministic
generation, independent-check structure, verifier compatibility, and exact
domain vocabulary. Per the no-classifier design, it does not independently
prove that a self-declared domain is mathematically correct. Domain accuracy
therefore remains an audit statistic rather than a second model gate.
