# Evolved Performance Concept Score Changes

Accuracy-point changes are measured against the immediately previous saved model on the same fixed benchmark rows.

| checkpoint | EPS delta | improved concepts | declined concepts |
|---:|---:|---|---|
| 0 → 32 | +13.12%p | number_theory/transformation (+30.0%p), sequence/induction (+27.5%p), algebra/transformation (+13.8%p), combinatorics/counting (+10.0%p), inequality/transformation (+5.0%p) | algebra/casework (-7.5%p) |
| 32 → 64 | +2.29%p | inequality/transformation (+13.7%p), number_theory/transformation (+2.5%p), sequence/induction (+1.2%p) | algebra/transformation (-1.3%p), combinatorics/counting (-2.5%p) |
| 64 → 96 | +0.83%p | algebra/casework (+7.5%p), number_theory/transformation (+3.8%p), inequality/transformation (+1.3%p) | sequence/induction (-1.2%p), algebra/transformation (-2.5%p), combinatorics/counting (-3.8%p) |
| 96 → 128 | -1.88%p | algebra/transformation (+3.8%p), sequence/induction (+1.2%p) | number_theory/transformation (-1.3%p), combinatorics/counting (-3.8%p), inequality/transformation (-11.3%p) |
