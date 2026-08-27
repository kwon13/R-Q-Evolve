# Omni-MATH computational problem-type pilot

This pilot asks whether the existing Omni-MATH top-level domain labels can be
crossed with the following five computational problem types:

1. `decision`
2. `search`
3. `counting`
4. `optimization`
5. `function`

The vocabulary is the five-way computational-problem distinction stated in
Pramod Ganapathi's *Algorithmically Hard Problems*, p. 12:
<https://www3.cs.stonybrook.edu/~pramod.ganapathi/doc/theory-of-computation/AlgorithmicallyHardProblems.pdf>.
The application of that vocabulary to competition-math statements is our
annotation layer; it is not metadata shipped by Omni-MATH.

## Method

`src/rq_evolve/problem_type.py` implements a conservative statement-only pilot
annotator. It uses only the requested output in `problem`, never `solution`,
`answer`, difficulty, or a model's performance. It abstains on proof prompts,
damaged statements, and generic requests that cannot be assigned reliably.
Audit output records the local ruleset name and source hash so later rule edits
cannot be confused with these counts.

Run the audit with:

```bash
python scripts/audit_omni_problem_types.py \
  --input /path/to/Omni-Math.jsonl \
  --output-dir /tmp/omni-type-audit/full
```

The audit was run on official Omni-MATH commit
`23be225c8e268df51990f6c5c1448f34d3b56911`.

## Pilot results

### Full Omni-MATH (4,428 rows)

| Type | Classified rows | Base-10 integer answers |
|---|---:|---:|
| decision | 170 | 3 |
| search | 744 | 71 |
| counting | 707 | 608 |
| optimization | 679 | 400 |
| function | 1,347 | 632 |

- Classified by high-precision surface rules: 3,647 / 4,428 (82.4%).
- Sent to review instead of forcing a label: 781 / 4,428 (17.6%).
- Expanded multi-domain memberships occur in all 35 domain/type combinations.
- Exactly-one-domain rows occur in 34 / 35 combinations.
- These are descriptive frequencies only; no count threshold masks a MAP cell.
- Expanded descriptive association: NMI 0.075, bias-corrected Cramer's V 0.233.
- Exactly-one-domain association: NMI 0.098, bias-corrected Cramer's V 0.260.

The small NMI is encouraging but is not proof of independence. Domain imbalance
and surface-rule label error remain. Expanded-domain association is descriptive
only because duplicated memberships are not independent observations.

### Omni-MATH-Rule (2,821 rows)

| Type | Classified rows | Base-10 integer answers |
|---|---:|---:|
| decision | 4 | 3 |
| search | 120 | 70 |
| counting | 656 | 603 |
| optimization | 540 | 396 |
| function | 1,099 | 585 |

The rule-verifiable subset nearly eliminates natural decision problems. This is
the key selection effect: judging the axis only on Omni-MATH-Rule would make a
real problem type look absent. Increasing type coverage therefore requires a
Boolean verifier and broader symbolic/witness verification, not only new labels.

## Domain caveat

Omni-MATH's `domain` field is multi-label. In the full dataset, 3,359 rows have
one top-level domain, 1,027 have two, 40 have three, and 2 have none. Treating
the first path as a primary domain would add a convention not supplied by the
benchmark. The audit therefore reports both expanded memberships and the clean
exactly-one-domain subset.

## Required validation before changing the MAP

1. Build a stratified human-reviewed gold sample across all five types and
   seven domains; report per-type precision and confusion, not only coverage.
2. Adjudicate the 781 abstentions with the same statement-only rubric.
3. Keep all 35 Cartesian-product cells; corpus frequency is analysis metadata,
   not a runtime allowlist or supported-cell mask.
4. Use verifier dispatch rather than an integer-only answer gate:
   expression equivalence for function/counting/optimal value, canonical
   Boolean comparison for decision, and complete exact-set verification for
   search. Arbitrary executable predicates remain forbidden.
