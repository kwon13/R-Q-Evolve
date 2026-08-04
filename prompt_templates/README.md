# Prompt Templates

Edit these files to control mutation prompts:

- `in_depth.txt`: same-domain deeper mutation
- `in_breadth.txt`: different-domain breadth mutation

Put few-shot examples here:

- `shots/in_depth.txt`
- `shots/in_breadth.txt`
- `shots/evaluator.txt`

Mutation is one-stage: the parent source goes into the code-writing prompt and
the model returns the child generator directly. There is no planning call and
no plan schema.

Few-shots are offline format fixtures, not reusable mutation content. Live
mutation generation substitutes `$few_shot_examples` with the empty string,
because greedy decoding was copying an example instead of mutating the live
parent. The evaluator shots are the exception: `shots/evaluator.txt` is injected
into the coherence-gate conversation.

The code reads this directory by default. To use another directory, set:

```bash
export RQ_EVOLVE_PROMPT_DIR=/path/to/your/templates
```

To use another shot directory, set:

```bash
export RQ_EVOLVE_SHOT_DIR=/path/to/your/shots
```

Templates use Python `string.Template` placeholders, not `{format}`.
That means you can paste Python examples containing `{...}` without escaping
their braces.

Available placeholders:

- `$parent_id`
- `$few_shot_examples`
- `$parent_generation`
- `$parent_source`
- `$parent_concept_group`
- `$parent_concept_type`
- `$allowed_breadth_groups`
- `$parent_p_hat`
- `$parent_h_score`
- `$parent_rq_score`

Mutation sampling is separate from solver rollout sampling. Configure
`evolution.code_temperature` and `evolution.evaluator_temperature` (plus their
`*_top_p` values) in the R_Q config. Solver diversity continues to use
`verl_config.actor_rollout_ref.rollout.temperature`.
