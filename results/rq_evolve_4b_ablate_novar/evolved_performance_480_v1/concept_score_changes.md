# Evolved Performance Concept Score Changes

Accuracy-point changes are measured against the immediately previous saved model on the same fixed benchmark rows.

| checkpoint | EPS delta | improved concepts | declined concepts |
|---:|---:|---|---|
| 0 → 32 | +12.92%p | sequence/induction (+27.5%p), number_theory/transformation (+20.0%p), algebra/transformation (+13.8%p), inequality/transformation (+8.8%p), combinatorics/counting (+3.8%p), algebra/casework (+3.7%p) | - |
| 32 → 64 | +1.87%p | inequality/transformation (+7.5%p), number_theory/transformation (+7.5%p), combinatorics/counting (+3.8%p) | algebra/transformation (-1.3%p), algebra/casework (-6.2%p) |
| 64 → 96 | +0.83%p | algebra/casework (+7.5%p), sequence/induction (+3.7%p), algebra/transformation (+2.5%p), number_theory/transformation (+2.5%p) | combinatorics/counting (-3.8%p), inequality/transformation (-7.5%p) |
| 96 → 128 | +0.21%p | inequality/transformation (+6.2%p), algebra/casework (+5.0%p) | algebra/transformation (-1.2%p), number_theory/transformation (-1.3%p), sequence/induction (-2.5%p), combinatorics/counting (-5.0%p) |
