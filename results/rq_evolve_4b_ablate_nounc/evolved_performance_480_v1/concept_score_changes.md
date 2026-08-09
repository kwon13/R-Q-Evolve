# Evolved Performance Concept Score Changes

Accuracy-point changes are measured against the immediately previous saved model on the same fixed benchmark rows.

| checkpoint | EPS delta | improved concepts | declined concepts |
|---:|---:|---|---|
| 0 → 32 | +13.33%p | sequence/induction (+27.5%p), number_theory/transformation (+18.8%p), inequality/transformation (+15.0%p), algebra/transformation (+10.0%p), combinatorics/counting (+8.8%p) | - |
| 32 → 64 | -0.00%p | inequality/transformation (+5.0%p), algebra/transformation (+1.3%p), algebra/casework (+1.2%p) | number_theory/transformation (-3.7%p), combinatorics/counting (-3.8%p) |
| 64 → 96 | +0.00%p | inequality/transformation (+10.0%p), algebra/transformation (+2.5%p), sequence/induction (+1.2%p) | algebra/casework (-3.7%p), number_theory/transformation (-10.0%p) |
| 96 → 128 | +1.25%p | algebra/casework (+6.2%p), combinatorics/counting (+2.5%p), algebra/transformation (+1.2%p) | number_theory/transformation (-2.5%p) |
