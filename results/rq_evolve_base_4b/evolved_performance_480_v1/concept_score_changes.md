# Evolved Performance Concept Score Changes

Accuracy-point changes are measured against the immediately previous saved model on the same fixed benchmark rows.

| checkpoint | EPS delta | improved concepts | declined concepts |
|---:|---:|---|---|
| 0 → 32 | +9.58%p | sequence/induction (+27.5%p), number_theory/transformation (+15.0%p), algebra/transformation (+13.8%p), combinatorics/counting (+2.5%p), inequality/transformation (+2.5%p) | algebra/casework (-3.7%p) |
| 32 → 64 | +0.42%p | algebra/casework (+2.5%p), algebra/transformation (+1.2%p), sequence/induction (+1.2%p) | inequality/transformation (-1.2%p), number_theory/transformation (-1.3%p) |
| 64 → 96 | +1.67%p | number_theory/transformation (+7.5%p), algebra/transformation (+3.7%p), combinatorics/counting (+1.2%p) | algebra/casework (-1.2%p), inequality/transformation (-1.3%p) |
| 96 → 128 | +5.00%p | inequality/transformation (+28.8%p), sequence/induction (+11.3%p), algebra/casework (+3.7%p), algebra/transformation (+2.5%p) | combinatorics/counting (-7.5%p), number_theory/transformation (-8.8%p) |
| 128 → 160 | +1.46%p | number_theory/transformation (+7.5%p), combinatorics/counting (+7.5%p) | algebra/casework (-1.2%p), algebra/transformation (-1.2%p), sequence/induction (-1.2%p), inequality/transformation (-2.5%p) |
| 160 → 192 | -2.71%p | inequality/transformation (+11.2%p), algebra/casework (+1.2%p) | combinatorics/counting (-7.5%p), sequence/induction (-10.0%p), number_theory/transformation (-11.2%p) |
| 192 → 224 | -2.50%p | combinatorics/counting (+2.5%p), number_theory/transformation (+1.2%p) | algebra/transformation (-3.7%p), algebra/casework (-7.5%p), inequality/transformation (-7.5%p) |
| 224 → 256 | -0.63%p | algebra/casework (+8.8%p), number_theory/transformation (+8.8%p), sequence/induction (+6.2%p) | algebra/transformation (-1.3%p), combinatorics/counting (-2.5%p), inequality/transformation (-23.8%p) |
