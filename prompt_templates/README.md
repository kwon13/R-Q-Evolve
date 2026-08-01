# Prompt Templates

Edit these files to control mutation prompts:

- `in_depth.txt`: same-domain deeper mutation
- `in_breadth.txt`: different-domain breadth mutation
- `metacognitive_plan.txt`: shared plain/reasoning planning schema; reasoning
  condition additionally receives monitoring evidence
- `planned_in_depth.txt`: validated in-depth plan to Python
- `planned_in_breadth.txt`: validated in-breadth plan to Python

Put few-shot examples here:

- `shots/in_depth.txt`
- `shots/in_breadth.txt`
- `shots/metacognitive_in_depth.txt`
- `shots/metacognitive_in_breadth.txt`
- `shots/planned_in_depth.txt`
- `shots/planned_in_breadth.txt`

Few-shots are offline format fixtures, not reusable mutation content. Live
generation omits content-rich shots from the chat conversation to prevent
greedy decoding from copying an example. The paired comparison uses the same
schema-v5 family catalog and resolver for both conditions: plain planning sees
only the parent, while reasoning-informed planning additionally receives one
clean same-problem correct/wrong pair. Schema version 5 adds
`generator_family` and typed `family_config` to the single `answer_route`
contract. Registered families are compiled into the mechanical Python
generator skeleton and checked by family-specific mathematical validators, so
no code-generation model call is needed. Unknown `free_form.*` ideas may be
retained for diagnostics in hybrid mode, but are quarantined from the archive
and training data. Legacy schema-v4 plans remain parseable only for this
quarantined compatibility path.

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
- `$parent_p_hat`
- `$parent_h_score`
- `$parent_rq_score`

Metacognitive templates additionally receive:

- `$operator`
- `$operator_contract`
- `$target_label_contract`
- `$necessity_contract`
- `$planning_condition`
- `$evidence_contract`
- `$inherited_reasoning_move`
- `$behavioral_evidence`
- `$meta_progress`
- `$registered_generator_families`
- `$mutation_plan`
- `$plan_id`

Mutation sampling is separate from solver rollout sampling. Configure
`evolution.plan_temperature`, `evolution.code_temperature`, and
`evolution.evaluator_temperature` (plus their `*_top_p` values) in the R_Q
config. Solver diversity continues to use
`verl_config.actor_rollout_ref.rollout.temperature`.
