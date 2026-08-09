# Reasoning-informed expansion: generation-space report

- Status: `ok_descriptive`
- Mode: `pilot`
- Validity policy: `code_valid`
- Primary representation: `layer_023__masked_mean`
- Inferentially valid: `False`
- Mean paired O difference (reasoning - plain): `-0.0875851217465117`
- CI: `[None, None]`
- Limited generation-space claim allowed: `False`

## Claim gates

- `preregistered_protocol_and_pair_design_passed`: `False`
- `orthogonal_effect_positive_with_ci_above_zero`: `False`
- `controls_adjusted_effect_positive_with_ci_above_zero`: `False`
- `strict_valid_matched_generators_present`: `False`
- `independent_runs_and_parents_gate`: `False`
- `layer_pooling_direction_consistent`: `False`
- `limited_generation_space_claim_allowed`: `False`

## Warnings

- pilot_fit_evaluation_reuse: plain PCA is fitted and described on the same parent set; no inferential claim is permitted.
- diagnostic_validity_policy: code-valid rows include rejected or operator-contract-invalid generators; selection/confounding precludes a causal method comparison.
- pilot_auto_k: requested k=5, using k=4 for archive_size=5
- genuine_scalar_rq_missing: rq is absent or is a mean-NLL proxy; the R_Q-controls-adjusted confirmatory claim is disabled.
- missing_prespecified_controls: rq; adjusted claims are disabled.
- controls_adjusted_effect_unavailable: too few independent paired generators (or inferential gate failed) for the prespecified controls.

> Orthogonal projection is this experiment's operational definition. It is not a metric proposed by Manifold Bandits.
