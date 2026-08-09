# Evolved Performance Concept Score Changes

Accuracy-point changes are measured against the immediately previous saved model on the same fixed benchmark rows.

| checkpoint | EPS delta | improved concepts | declined concepts |
|---:|---:|---|---|
| 0 → 32 | +14.58%p | sequence/induction (+27.5%p), inequality/transformation (+17.5%p), number_theory/transformation (+17.5%p), algebra/transformation (+12.5%p), combinatorics/counting (+10.0%p), algebra/casework (+2.5%p) | - |
| 32 → 64 | +0.00%p | inequality/transformation (+10.0%p) | sequence/induction (-1.3%p), number_theory/transformation (-2.5%p), combinatorics/counting (-2.5%p), algebra/casework (-3.8%p) |
| 64 → 96 | +1.04%p | inequality/transformation (+7.5%p), sequence/induction (+2.5%p) | combinatorics/counting (-1.3%p), number_theory/transformation (-2.5%p) |
| 96 → 128 | -0.21%p | algebra/casework (+2.5%p), sequence/induction (+2.5%p), algebra/transformation (+1.3%p), combinatorics/counting (+1.3%p), inequality/transformation (+1.2%p) | number_theory/transformation (-10.0%p) |
