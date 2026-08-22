# Prompt Templates

Four files drive the evolution loop. Edit them to change its behaviour; the code
reads them verbatim and substitutes only the placeholders listed below.

## Mutation (the Evolver)

- `mutation_system_prompt.txt` — role, output discipline, the answer/check rule
- `mutation_user_prompt.txt` — the live parent and both label vocabularies

There is **one** mutation operator. The retired pair (`in_depth` held GROUP and
moved SKILL, `in_breadth` the mirror) forced a label change on one axis, which
made the label a target the child was written to satisfy rather than a
description of what it produced. The archived GROUP/SKILL then disagreed with
what the visible problem actually demanded — the one error a MAP cannot absorb,
because the coordinates stop meaning anything and both the parent sampler and
the coverage metric read a fiction.

The child now decides for itself what to change, and labels the result from its
own shortest solution.

Placeholders:

- `$parent_source`
- `$parent_group`
- `$parent_skill`
- `$allowed_groups`   (from `group_definitions.txt`)
- `$allowed_skills`   (from `skill_definitions.txt`)

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
