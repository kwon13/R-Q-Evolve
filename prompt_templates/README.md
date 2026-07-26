# Prompt Templates

Edit these files to control mutation prompts:

- `in_depth.txt`: same-domain deeper mutation
- `in_breadth.txt`: different-domain breadth mutation
- `metacognitive_plan.txt`: monitoring evidence + pre-update delta-p to JSON plan
- `planned_in_depth.txt`: validated in-depth plan to Python
- `planned_in_breadth.txt`: validated in-breadth plan to Python

Put few-shot examples here:

- `shots/in_depth.txt`
- `shots/in_breadth.txt`
- `shots/metacognitive_in_depth.txt`
- `shots/metacognitive_in_breadth.txt`
- `shots/planned_in_depth.txt`
- `shots/planned_in_breadth.txt`

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
- `$inherited_reasoning_move`
- `$behavioral_evidence`
- `$meta_progress`
- `$mutation_plan`
- `$plan_id`
