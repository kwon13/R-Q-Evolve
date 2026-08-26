# Prompt Templates

The files in this directory drive the evolution loop. The code reads them
verbatim and substitutes only the documented placeholders.

## Mutation (the Evolver)

- `diff_problem_system_prompt.txt` / `diff_problem_user_prompt.txt` — stage 1,
  which designs a child problem family
- `gen_program_system_prompt.txt` / `gen_program_user_prompt.txt` — stage 2,
  which implements that already-fixed family as a generator
- `structural_inspiration_system_note.txt` /
  `structural_inspiration_user_block.txt` — optional stage-1-only donor rules
  and its single `$inspiration_template` placeholder

When `evolution.structural_inspiration` is enabled, a donor is sampled from a
different primary-parent lineage, preferring candidates whose GROUP and SKILL
descriptors both differ. Only its statement-only parameterized problem skeleton
is rendered as untrusted quoted data. Its source, answer/check, concrete
instances, GROUP/SKILL, score, metadata and identifiers never enter the prompt,
and stage 2 never sees it. Chat-control/role/label markers make a donor
ineligible, and an assigned-donor copy is rejected before solver rollouts. The
feature-off prompt contains neither the donor block nor its system rules.

There is **one** mutation operator. The retired pair (`in_depth` held GROUP and
moved SKILL, `in_breadth` the mirror) forced a label change on one axis, which
made the label a target the child was written to satisfy rather than a
description of what it produced. The archived GROUP/SKILL then disagreed with
what the visible problem actually demanded — the one error a MAP cannot absorb,
because the coordinates stop meaning anything and both the parent sampler and
the coverage metric read a fiction.

The child now decides for itself what to change, and labels the result from its
own shortest solution.

Stage-1 placeholders:

- `$parent_template`
- `$parent_problem`
- `$allowed_groups`   (from `group_definitions.txt`)
- `$allowed_skills`   (from `skill_definitions.txt`)
- `$inspiration_template` (optional donor block only)

Stage-2 placeholders:

- `$parent_template`
- `$parent_source` (primary parent only, with labels removed)
- `$new_problem`
- `$target_skill`
- `$target_skill_definition`

## Judge (the label gate)

- `mutation_judge_system_prompt.txt` — validity gates, then the taxonomy rubric
- `mutation_judge_user_prompt.txt` — the one (problem, answer) pair

The judge runs `generate(seed=0)` output only. It never sees the generator
source, the declared labels, or the parent — that independence is the whole
point, because a judge shown the declared label cannot disagree with it. It
returns seven fields:

```text
GROUP: ...
GROUP_EVIDENCE: ...
SKILL: ...
SKILL_WITNESS: ...
CLOSEST_ALTERNATIVE: ...
WHY_NOT_ALTERNATIVE: ...
FAILURE_REASON: ...
```

The rubric keeps two validity gates and then judges both axes against the same
one-line definitions above. `evolution.judge_rubric` selects the file, so an
alternative rubric can be measured against this one without a code change.

An earlier draft additionally required a SKILL to survive a routineness test, a
mandatory named witness and a closest-alternative challenge. Over 41 items that
took both-axis agreement from 41% to 2% and returned `SKILL: none` for 6 of 8
hand-labelled seeds, so a "declared == judged" gate built on it rejected ground
truth. See `analysis/judge_pipeline_v2/`.

A child is archived only when the judge's GROUP **and** SKILL both equal what it
declared. `none` on either axis, an out-of-vocabulary value, and an unreadable
reply are all the same answer: reject. The verdict is recorded on the child
either way, under `metadata["judge"]`, so the disagreement rate is measurable.

Placeholders:

- `$problem_text`
- `$answer`

## Axis definitions

- `group_definitions.txt`
- `skill_definitions.txt`

One file per axis, shared by the mutation prompt and the judge rubric, so a
label cannot mean one thing when it is chosen and another when it is checked.

## Directory overrides

```bash
export RQ_EVOLVE_PROMPT_DIR=/path/to/your/templates
```

`shots/` holds offline format fixtures only. Nothing in the live loop reads it.

Templates use Python `string.Template` placeholders, not `{format}`, so a Python
example containing `{...}` needs no brace escaping. An unresolved placeholder is
an error, never a silent passthrough.

## Sampling

Mutation sampling is separate from solver rollout sampling. Configure
`evolution.code_temperature` / `evolution.code_top_p` and
`evolution.judge_temperature` / `evolution.judge_top_p` in the R_Q config. The
judge is read greedily (temperature 0, top_p 1): it is a measurement, and
sampling noise there shows up as label disagreement the generator did not cause.
Solver diversity continues to use
`verl_config.actor_rollout_ref.rollout.temperature`.
