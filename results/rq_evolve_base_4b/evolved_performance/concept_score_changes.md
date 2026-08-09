# Seed-ID Concept Score Changes

Accuracy-point changes are measured against the immediately previous saved model on the same fixed benchmark rows.

| checkpoint | EPS delta | improved concepts | declined concepts |
|---:|---:|---|---|
| 0 → 32 | +17.08%p | sequence/induction (+65.0%p), algebra/transformation (+20.0%p), number_theory/transformation (+15.0%p), combinatorics/counting (+2.5%p) | - |
| 32 → 64 | +1.67%p | number_theory/transformation (+7.5%p), combinatorics/counting (+2.5%p) | - |
| 64 → 96 | -3.33%p | combinatorics/counting (+5.0%p), algebra/transformation (+2.5%p) | number_theory/transformation (-27.5%p) |
| 96 → 128 | +2.08%p | number_theory/transformation (+20.0%p) | combinatorics/counting (-7.5%p) |
| 128 → 160 | +4.58%p | number_theory/transformation (+15.0%p), combinatorics/counting (+12.5%p) | - |
| 160 → 192 | -5.42%p | - | number_theory/transformation (-7.5%p), combinatorics/counting (-25.0%p) |
| 192 → 224 | +0.42%p | combinatorics/counting (+5.0%p) | algebra/casework (-2.5%p) |
| 224 → 256 | -0.42%p | number_theory/transformation (+10.0%p) | algebra/transformation (-5.0%p), combinatorics/counting (-7.5%p) |
