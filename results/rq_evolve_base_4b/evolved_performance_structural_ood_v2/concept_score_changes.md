# Seed-ID Concept Score Changes

Accuracy-point changes are measured against the immediately previous saved model on the same fixed benchmark rows.

| checkpoint | EPS delta | improved concepts | declined concepts |
|---:|---:|---|---|
| 0 → 32 | +0.00%p | number_theory/transformation (+7.5%p), algebra/transformation (+5.0%p), inequality/transformation (+2.5%p) | sequence/induction (-7.5%p), algebra/casework (-7.5%p) |
| 32 → 64 | -0.42%p | algebra/casework (+10.0%p), sequence/induction (+2.5%p) | algebra/transformation (-2.5%p), number_theory/transformation (-12.5%p) |
| 64 → 96 | +6.67%p | number_theory/transformation (+22.5%p), algebra/transformation (+10.0%p), inequality/transformation (+7.5%p), algebra/casework (+2.5%p) | sequence/induction (-2.5%p) |
| 96 → 128 | +13.33%p | inequality/transformation (+57.5%p), sequence/induction (+27.5%p), algebra/transformation (+5.0%p) | algebra/casework (-10.0%p) |
| 128 → 160 | -2.50%p | - | sequence/induction (-2.5%p), inequality/transformation (-2.5%p), algebra/casework (-2.5%p), algebra/transformation (-7.5%p) |
| 160 → 192 | -2.50%p | algebra/casework (+15.0%p), inequality/transformation (+12.5%p) | algebra/transformation (-7.5%p), number_theory/transformation (-12.5%p), sequence/induction (-22.5%p) |
| 192 → 224 | -2.08%p | number_theory/transformation (+2.5%p), algebra/casework (+2.5%p) | inequality/transformation (-17.5%p) |
| 224 → 256 | -5.00%p | sequence/induction (+10.0%p), algebra/transformation (+2.5%p), number_theory/transformation (+2.5%p) | algebra/casework (-5.0%p), inequality/transformation (-40.0%p) |
