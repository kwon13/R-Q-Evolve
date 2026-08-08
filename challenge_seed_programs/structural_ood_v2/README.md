# Structural-OOD v2 seed programs

Evaluation-only generators, one per `GROUP` / `SKILL` pair in
`structural_ood_v1`, holding both labels fixed and raising what the problem
demands. They must not be added to `configs/*seed_programs_dir`.

Difficulty here means one more step of reasoning, not larger numbers. Each v1
problem has a route a strong solver can name on sight -- Stern's diatomic
sequence, Newton's two-variable identity, plain Cauchy-Schwarz, Euler plus CRT,
Bertrand's ballot formula. Every v2 problem breaks that recall route while
staying inside the same skill.

| Program | GROUP / SKILL | v1 structure | v2 structure | What the extra step is |
|---|---|---|---|---|
| `01_weighted_binary_recurrence.py` | sequence / induction | Stern's diatomic recurrence | the same halving recurrence with weighted branches | at `alpha = beta = 1` the sequence can be recalled; weighting the branches forces the induction to be carried out |
| `02_symmetric_power_sum_three.py` | algebra / transformation | `p_n` from `e1, e2` | `p_n` from `e1, e2, e3` | the three-variable Newton identity needs two previous terms and an alternating sign, so the two-variable recurrence does not extend by analogy |
| `03_shifted_quadratic_minimum.py` | inequality / transformation | weighted Cauchy minimum | the same minimum after a translation | the shifts have to be removed before Cauchy applies; applying it to the stated variables gives the wrong bound |
| `04_three_level_power_tower.py` | number_theory / transformation | `b^(c^d)` mod 1000 | `b^(c^(d^e))` mod 1000 | the middle exponent needs its own reduction modulo `lambda(100)`, which the two-level tower never required |
| `05_banded_lattice_paths.py` | combinatorics / counting | one-sided ballot paths | walks confined to a two-sided band | one wall closes with a single reflection; two walls need an alternating sum over the reflection group |
| `06_weighted_absolute_centers.py` | algebra / casework | 3 equal-weight centers | 5 centers with distinct, odd-total weights | each of the six regions has its own nonzero slope, so every region must be checked separately |

## Contract

Every generator satisfies the same contract as the training seeds, verified by
`lint_generator_source`, `ast_contract.check_generator_contract`,
`lint_problem_instance` and `validate_label_decl`:

* one integer answer, and five seeds give five distinct problems and answers;
* a cross-check computed by a route genuinely independent of the answer route
  -- typically closed form against enumeration, or the reverse -- so a wrong
  derivation fails the `assert` rather than shipping a wrong answer;
* nothing at module level except imports, helpers and the two labels.

## Reporting

Same caveat as v1: none of these structures occur in the six training seeds, so
results are structural transfer, not a Seed-ID score. v1 and v2 differ only in
difficulty within a fixed (GROUP, SKILL), so the pair also measures how quickly
accuracy falls off as a problem gets harder inside a skill the model has seen.
