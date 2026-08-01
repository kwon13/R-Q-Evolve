from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from rq_evolve.metacognition import EVIDENCE_QUALITY_VERSION
from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import parse_mutation_plan


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_mutation_methods_vllm.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "rq_compare_mutation_methods_vllm",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
comparison = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = comparison
_SPEC.loader.exec_module(comparison)


def _add_evidence_provenance(rows):
    for row in rows:
        row.update(
            {
                "source": "observed",
                "origin_program_id": "parent",
                "origin_concept_group": "algebra",
                "origin_concept_type": "algebra.sum",
                "entropy": 0.1,
                "policy_version": 0,
                "iteration": 0,
                "response_tokens": 8,
                "evidence_quality_version": EVIDENCE_QUALITY_VERSION,
            }
        )
    return rows


def test_resolve_evaluation_seeds_defaults_to_every_verify_seed():
    args = Namespace(verify_seeds=5, evaluation_seeds=None)
    assert comparison._resolve_evaluation_seeds(args) == [0, 1, 2, 3, 4]

    args.evaluation_seeds = [0, 2, 4]
    assert comparison._resolve_evaluation_seeds(args) == [0, 2, 4]

    args.evaluation_seeds = [0, 0]
    with pytest.raises(ValueError, match="duplicates"):
        comparison._resolve_evaluation_seeds(args)


def test_vllm_sampler_backend_defaults_to_pytorch_and_is_explicit(
    monkeypatch,
):
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    assert comparison._configure_vllm_sampler_backend("pytorch") == "0"

    assert comparison._configure_vllm_sampler_backend("flashinfer") == "1"
    assert comparison._configure_vllm_sampler_backend("auto") == "1"

    with pytest.raises(ValueError, match="sampler backend"):
        comparison._configure_vllm_sampler_backend("unknown")


def test_vllm_wrapper_accepts_one_sampling_params_object_per_request(
    monkeypatch,
):
    class FakeStructuredOutputsParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLLM:
        def __init__(self):
            self.params = None

        def chat(self, conversations, *, sampling_params, **kwargs):
            self.params = sampling_params
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text=f"response-{index}",
                            token_ids=[index + 1],
                            cumulative_logprob=-1.0,
                        )
                    ]
                )
                for index, _ in enumerate(conversations)
            ]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=FakeSamplingParams),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(
            StructuredOutputsParams=FakeStructuredOutputsParams
        ),
    )
    inference = comparison.VLLMChatInference.__new__(
        comparison.VLLMChatInference
    )
    inference.llm = FakeLLM()
    inference.chat_template = None
    inference.default_sampling_params = {"temperature": 1.0, "max_tokens": 8}
    conversations = [
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
    ]

    outputs = inference.generate_detailed(
        conversations,
        sampling_params=[
            {
                "seed": 11,
                "structured_outputs": {
                    "json": {"type": "object"}
                },
            },
            {"seed": 22},
        ],
    )

    assert isinstance(outputs, list)
    assert [params.kwargs["seed"] for params in inference.llm.params] == [11, 22]
    assert isinstance(
        inference.llm.params[0].kwargs["structured_outputs"],
        FakeStructuredOutputsParams,
    )
    with pytest.raises(ValueError, match="length must match"):
        inference.generate_detailed(
            conversations,
            sampling_params=[{"seed": 11}],
        )


def test_evaluator_and_child_rollouts_use_condition_shared_request_seeds():
    class FakeInference:
        def __init__(self):
            self.evaluator_sampling = None
            self.solver_sampling = None
            self.tokenizer = SimpleNamespace(
                encode=lambda text, **kwargs: text.split()
            )

        def generate(self, messages, *, sampling_params, **kwargs):
            self.evaluator_sampling = sampling_params
            return ["reason: sound\nverdict: VALID" for _ in messages]

        def generate_detailed(self, messages, *, sampling_params, **kwargs):
            self.solver_sampling = sampling_params
            answers = ("1", "1", "2", "2")
            return [
                comparison.GeneratedText(
                    text=rf"Calculation gives \boxed{{{answer}}}.",
                    token_ids=[1, 2, 3],
                    cumulative_logprob=-0.3,
                    mean_negative_logprob=0.1,
                )
                for answer in answers
            ]

    program = ProblemProgram(
        source_code='''
def generate(seed):
    return f"Compute {seed} + 1.", str(seed + 1)

CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.integer_addition"
'''
    )
    inference = FakeInference()
    evaluation = comparison._evaluate_program_seeds(
        inference,
        program,
        seeds=[0, 1],
        mutation_plan=None,
        max_tokens=32,
        temperature=0.0,
        top_p=1.0,
        chat_template_kwargs={},
        operator="in_depth",
        llm_seed=17,
    )
    expected_evaluator = [
        comparison._evaluator_request_seed(17, "in_depth", seed)
        for seed in (0, 1)
    ]
    assert [
        params["seed"] for params in inference.evaluator_sampling
    ] == expected_evaluator
    assert [
        row["sampling_seed"] for row in evaluation["per_seed"]
    ] == expected_evaluator

    scored = comparison._score_program_seeds(
        inference,
        program,
        seeds=[0, 1],
        count=2,
        max_tokens=32,
        temperature=1.0,
        top_p=0.95,
        chat_template_kwargs={},
        operator="in_depth",
        llm_seed=17,
    )
    expected_solver = [
        comparison._child_solver_request_seed(
            17,
            "in_depth",
            seed,
            rollout_idx,
        )
        for seed in (0, 1)
        for rollout_idx in (0, 1)
    ]
    assert [
        params["seed"] for params in inference.solver_sampling
    ] == expected_solver
    assert [
        rollout["sampling_seed"]
        for seed_row in scored["per_seed"]
        for rollout in seed_row["rollouts"]
    ] == expected_solver


def test_fixed_evidence_requires_one_same_seed_same_problem_contrast():
    evidence = _add_evidence_provenance([
        {
            "seed": 0,
            "problem": "Problem zero",
            "role": "success",
            "correct": True,
            "response": r"Correct route gives \boxed{1}.",
            "predicted_answer": "1",
        },
        {
            "seed": 1,
            "problem": "Problem one",
            "role": "failure",
            "correct": False,
            "response": r"Wrong route gives \boxed{2}.",
            "predicted_answer": "2",
        },
    ])
    assert comparison._select_same_instance_contrast(
        evidence,
        preferred_seed=0,
        expected_problems={0: "Problem zero", 1: "Problem one"},
    ) == []

    evidence.append(_add_evidence_provenance([
        {
            "seed": 0,
            "problem": "Problem zero",
            "role": "failure",
            "correct": False,
            "response": r"Wrong route gives \boxed{2}.",
            "predicted_answer": "2",
        }
    ])[0])
    selected = comparison._select_same_instance_contrast(
        evidence,
        preferred_seed=0,
        expected_problems={0: "Problem zero", 1: "Problem one"},
    )
    assert [item["role"] for item in selected] == ["success", "failure"]
    assert {item["seed"] for item in selected} == {0}


def test_fixed_evidence_rejects_stale_problem_text():
    evidence = _add_evidence_provenance([
        {
            "seed": 0,
            "problem": "Old problem",
            "role": "success",
            "correct": True,
            "response": r"Correct route gives \boxed{1}.",
            "predicted_answer": "1",
        },
        {
            "seed": 0,
            "problem": "Old problem",
            "role": "failure",
            "correct": False,
            "response": r"Wrong route gives \boxed{2}.",
            "predicted_answer": "2",
        },
    ])
    assert comparison._select_same_instance_contrast(
        evidence,
        preferred_seed=0,
        expected_problems={0: "Current problem"},
    ) == []


def test_rollout_summary_keeps_per_seed_and_recomputes_one_pooled_score():
    grouped = [
        {
            "instance": SimpleNamespace(seed=0, problem="p0", answer="0"),
            "records": [
                SimpleNamespace(correct=True, entropy=0.2),
                SimpleNamespace(correct=False, entropy=0.4),
            ],
            "rollouts": [{}, {}],
        },
        {
            "instance": SimpleNamespace(seed=1, problem="p1", answer="1"),
            "records": [
                SimpleNamespace(correct=True, entropy=0.6),
                SimpleNamespace(correct=True, entropy=0.8),
            ],
            "rollouts": [{}, {}],
        },
    ]

    result = comparison._summarize_rollout_groups(grouped, seeds=[0, 1])

    assert result["num_scored_seeds"] == 2
    assert result["num_correct"] == 3
    assert result["num_rollouts"] == 4
    assert result["p_hat"] == pytest.approx(0.75)
    assert result["uncertainty_proxy"] == pytest.approx(0.5)
    assert [item["p_hat"] for item in result["per_seed"]] == [0.5, 1.0]
    assert [
        rollout["seed"]
        for item in result["per_seed"]
        for rollout in item["rollouts"]
    ] == [0, 0, 1, 1]


def test_cached_parent_rollouts_can_be_subset_and_truncated(tmp_path):
    instances = [
        SimpleNamespace(seed=0, problem="p0", answer="0"),
        SimpleNamespace(seed=1, problem="p1", answer="1"),
    ]
    rows = []
    for instance in instances:
        rows.append(
            {
                "seed": instance.seed,
                "problem": instance.problem,
                "answer": instance.answer,
                "rollouts": [
                    {
                        "correct": index % 2 == 0,
                        "confidence_proxy": 0.1 * (index + 1),
                    }
                    for index in range(4)
                ],
            }
        )
    # An extra cached seed is allowed when a development run requests a subset.
    rows.append(
        {
            "seed": 2,
            "problem": "p2",
            "answer": "2",
            "rollouts": [],
        }
    )
    cache = tmp_path / "parent_rollouts.json"
    cache.write_text(json.dumps({"per_seed": rows}), encoding="utf-8")

    scores = comparison._load_cached_parent_scores(
        cache,
        instances=instances,
        seeds=[0, 1],
        rollouts_per_seed=3,
    )

    assert scores["status"] == "cached"
    assert scores["num_rollouts"] == 6
    assert scores["num_correct"] == 4
    assert all(len(row["rollouts"]) == 3 for row in scores["per_seed"])


def test_belief_json_extraction_and_rq_selection_are_strictly_separated():
    raw = (
        'echoed example: {"schema_version":6,"operator":"in_depth",'
        '"attributed_hypothesis":"<one hypothesis_id from the list above>",'
        '"evidence_quote":null}\n'
        "<think>choose a grounded hypothesis</think>\n```json\n"
        '{"schema_version":6,"operator":"in_depth",'
        '"attributed_hypothesis":"unweighted_sum_then_divide",'
        '"evidence_quote":null}\n```'
    )
    plan = comparison._extract_json_mapping(raw)
    assert plan is not None
    assert plan["schema_version"] == 6

    candidates = [
        {
            "candidate_index": 0,
            "status": "eligible_scored",
            "attribution_niche": "a::belief",
            "attributed_hypothesis": "belief",
            "rq_proxy": 0.1,
            "p_hat": 0.2,
            "eligibility": {"eligible": True},
            "method_dir": "/tmp/0",
        },
        {
            "candidate_index": 1,
            "status": "eligible_scored",
            "attribution_niche": "a::belief",
            "attributed_hypothesis": "belief",
            "rq_proxy": 0.4,
            "p_hat": 0.5,
            "eligibility": {"eligible": True},
            "method_dir": "/tmp/1",
        },
        {
            "candidate_index": 2,
            "status": "probe_ineligible",
            "attribution_niche": "a::other",
            "attributed_hypothesis": "other",
            "rq_proxy": 99.0,
            "p_hat": 0.5,
            "eligibility": {"eligible": False},
            "method_dir": "/tmp/2",
        },
    ]
    selection = comparison._select_belief_candidates(candidates)

    assert selection["eligible_candidate_count"] == 2
    assert selection["selected_candidate_index"] == 1
    assert selection["niche_champion_count"] == 1


def test_belief_structured_output_allows_only_catalog_ids_and_trace_lines():
    schema = comparison._belief_structured_output_schema(
        generator_family="linear_system_aggregate",
        operator="in_depth",
        wrong_trace="First exact line.\n\nSecond exact line.",
    )

    properties = schema["properties"]
    assert properties["schema_version"]["const"] == 6
    assert properties["operator"]["const"] == "in_depth"
    assert "unweighted_sum_then_divide" in properties[
        "attributed_hypothesis"
    ]["enum"]
    assert properties["evidence_quote"]["enum"] == [
        "First exact line.",
        "Second exact line.",
    ]
    assert schema["additionalProperties"] is False

    plain = comparison._belief_structured_output_schema(
        generator_family="linear_system_aggregate",
        operator="in_depth",
        wrong_trace=None,
    )
    assert plain["properties"]["evidence_quote"] == {"type": "null"}


def test_belief_v6_plan_only_runs_python_gates_without_evaluator(
    tmp_path,
):
    class FakeInference:
        tokenizer = None

        def generate(self, messages, *, sampling_params, **kwargs):
            assert sampling_params["seed"] == comparison._paired_request_seed(
                13,
                "in_depth",
                "belief_plan:0",
            )
            return json.dumps(
                {
                    "schema_version": 6,
                    "operator": "in_depth",
                    "attributed_hypothesis": "unweighted_sum_then_divide",
                    "evidence_quote": (
                        "Adding the equations directly gives the aggregate"
                    ),
                }
            )

    parent = ProblemProgram(
        source_code=(
            "def generate(seed):\n"
            "    return 'Find x + y + z from the system.', '3'\n"
            'CONCEPT_GROUP = "algebra"\n'
            'CONCEPT_TYPE = "algebra.linear_system_sum"\n'
        )
    )
    args = Namespace(
        # Five seeds: this route is silent when the division is not whole, so a
        # three-seed set leaves only one discriminating instance.
        evaluation_seeds=[0, 1, 2, 3, 4],
        verify_seeds=5,
        instance_seed=0,
        prompt_dir=_SCRIPT_PATH.parents[1] / "prompt_templates",
        plan_temperature=0.3,
        plan_top_p=0.95,
        plan_max_tokens=256,
        llm_seed=13,
        plan_only=True,
        child_rollouts=0,
    )
    result = comparison._run_belief_candidate(
        FakeInference(),
        method="metacognitive",
        op="in_depth",
        parent=parent,
        output_dir=tmp_path,
        args=args,
        chat_template_kwargs={},
        planning_evidence=[
            {
                "role": "failure",
                "correct": False,
                "response": (
                    "Adding the equations directly gives the aggregate, "
                    "so the right-hand sides can be summed."
                ),
            }
        ],
        candidate_index=0,
    )

    assert result["status"] == "plan_only_eligible"
    assert result["eligibility"]["eligible"] is True
    assert result["family_semantics_valid"] is True
    assert result["probe_diagnosticity_valid"] is True
    assert not (Path(result["method_dir"]) / "evaluator.json").exists()


def test_standalone_validator_enforces_operator_contract_for_both_methods():
    parent = ProblemProgram(
        source_code="\n".join(
            [
                "def generate(seed):",
                "    return 'Find 1 + 1.', '2'",
                'CONCEPT_GROUP = "algebra"',
                'CONCEPT_TYPE = "algebra.sum"',
            ]
        )
    )
    same_domain_child = """```python
import random
import sympy

MAX_ATTEMPTS = 200

def generate(seed):
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        value = rng.randint(1, 5)
        instance_data = {"value": value}
        answer = instance_data["value"] + 1
        assert answer - instance_data["value"] == 1
        problem = f"Find {instance_data['value']} + 1."
        return problem, str(sympy.Integer(answer))
    else:
        raise RuntimeError("failed")

CONCEPT_REASON = "Add one to an integer."
CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.sum"
```"""
    child, result = comparison._validate_generated_program(
        same_domain_child,
        parent=parent,
        op="in_breadth",
        mutation_plan=None,
        plan_id=None,
        verify_seeds=2,
    )
    assert child is not None
    assert result["valid"] is False
    assert result["operator_contract_valid"] is False
    assert "must change CONCEPT_GROUP" in result["reason"]


def test_standalone_validator_rejects_historical_stale_coefficient_child():
    parent = ProblemProgram(
        source_code="""
def generate(seed):
    return "Find x + y from x=1 and y=2.", "3"

CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.linear_system_sum"
"""
    )
    stale_child = """```python
import random
import sympy

MAX_ATTEMPTS = 200

def generate(seed):
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        x, y, z = 1, 2, 3
        g, h, i = 1, 1, 1
        rows = [[1, 0, 0], [0, 1, 0], [g, h, i]]
        rows[2] = [2, 2, 2]
        rhs = [sum(row[j] * [x, y, z][j] for j in range(3)) for row in rows]
        answer = sum(rhs) // 3
        problem = f"x={x}, y={y}, z={z}; final row {g},{h},{i}. Find x+y+z."
        return problem, str(sympy.Integer(answer))
    else:
        raise RuntimeError("failed")

CONCEPT_REASON = "Combine a linear system."
CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.linear_system_sum"
```"""
    child, result = comparison._validate_generated_program(
        stale_child,
        parent=parent,
        op="in_depth",
        mutation_plan=None,
        plan_id=None,
        verify_seeds=2,
    )
    assert child is not None
    assert result["valid"] is False
    assert "canonical `instance_data`" in result["reason"]


def test_default_comparison_uses_equal_two_stage_plain_and_reasoning_calls(
    tmp_path,
):
    class FakeInference:
        def __init__(self, outputs):
            self.outputs = list(outputs)
            self.calls = []

        def generate(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return self.outputs.pop(0)

    parent = ProblemProgram(
        source_code='''
def generate(seed):
    return f"Compute {seed} + 1.", str(seed + 1)

CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.integer_addition"
'''
    )
    shot = Path(
        "prompt_templates/shots/metacognitive_in_depth.txt"
    ).read_text(encoding="utf-8")
    reasoning_plan, reason = parse_mutation_plan(shot, "in_depth")
    assert reasoning_plan is not None, reason
    reasoning_plan["target_concept_group"] = "algebra"
    reasoning_plan["target_concept_type"] = "algebra.integer_addition"
    plain_plan = dict(reasoning_plan)
    for field in (
        "failure_summary",
        "correct_wrong_contrast",
        "predicted_pre_behavior",
        "predicted_post_behavior",
    ):
        plain_plan[field] = None
    evidence = _add_evidence_provenance([
        {
            "seed": 0,
            "problem": "Compute 0 + 1.",
            "role": "success",
            "correct": True,
            "response": r"0+1=1, so \boxed{1}.",
            "predicted_answer": "1",
        },
        {
            "seed": 0,
            "problem": "Compute 0 + 1.",
            "role": "failure",
            "correct": False,
            "response": r"I add incorrectly and get \boxed{2}.",
            "predicted_answer": "2",
        },
    ])
    args = Namespace(
        plain_baseline="two_stage",
        plan_max_tokens=1024,
        plan_temperature=0.3,
        plan_top_p=0.95,
        code_temperature=0.2,
        code_top_p=0.95,
        mutation_max_tokens=5000,
        verify_seeds=2,
        evaluation_seeds=[0, 1],
        llm_seed=17,
    )

    seeds_by_method = {}
    for method, plan in (
        ("legacy", plain_plan),
        ("metacognitive", reasoning_plan),
    ):
        inference = FakeInference([json.dumps(plan), "not Python"])
        result = comparison._run_method(
            inference,
            method=method,
            op="in_depth",
            parent=parent,
            output_dir=tmp_path,
            args=args,
            chat_template_kwargs={},
            planning_evidence=evidence,
            meta_progress=comparison._empty_meta_progress(),
        )
        assert result["status"] == "invalid_code"
        assert result["configured_llm_generation_call_count"] == 2
        assert result["llm_generation_call_count"] == 2
        assert len(inference.calls) == 2
        seeds_by_method[method] = [
            call[1]["sampling_params"]["seed"] for call in inference.calls
        ]
        assert (tmp_path / "in_depth" / method / "01_plan_prompt.json").exists()
    assert seeds_by_method["legacy"] == seeds_by_method["metacognitive"]
    assert seeds_by_method["legacy"][0] != seeds_by_method["legacy"][1]
    assert seeds_by_method["legacy"] == [
        comparison._paired_request_seed(17, "in_depth", "plan"),
        comparison._paired_request_seed(17, "in_depth", "code"),
    ]


def test_hybrid_registered_family_compiles_without_code_model_call(
    tmp_path,
    monkeypatch,
):
    class FakeInference:
        tokenizer = None

        def __init__(self, plan):
            self.plan = plan
            self.calls = []

        def generate(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return json.dumps(self.plan)

    parent = ProblemProgram.from_file(
        Path("seed_programs/09_linear_algebra.py")
    )
    shot = Path(
        "prompt_templates/shots/metacognitive_in_depth.txt"
    ).read_text(encoding="utf-8")
    reasoning_plan, reason = parse_mutation_plan(shot, "in_depth")
    assert reasoning_plan is not None, reason
    reasoning_plan.update(
        {
            "target_concept_group": "algebra",
            "target_concept_type": "algebra.linear_system_sum",
            "generator_family": "linear_system_aggregate",
            "family_config": {
                "coefficient_min": -3,
                "coefficient_max": 3,
                "solution_min": -3,
                "solution_max": 3,
            },
        }
    )
    plain_plan = dict(reasoning_plan)
    for field in (
        "failure_summary",
        "correct_wrong_contrast",
        "predicted_pre_behavior",
        "predicted_post_behavior",
    ):
        plain_plan[field] = None
    evidence = _add_evidence_provenance(
        [
            {
                "seed": 0,
                "problem": parent.execute(0).problem,
                "role": "success",
                "correct": True,
                "response": r"Correct row-space reasoning gives \boxed{2}.",
                "predicted_answer": "2",
            },
            {
                "seed": 0,
                "problem": parent.execute(0).problem,
                "role": "failure",
                "correct": False,
                "response": r"A shortcut gives \boxed{3}.",
                "predicted_answer": "3",
            },
        ]
    )
    args = Namespace(
        plain_baseline="two_stage",
        mutation_code_backend="hybrid",
        plan_max_tokens=1024,
        plan_temperature=0.3,
        plan_top_p=0.95,
        code_temperature=0.2,
        code_top_p=0.95,
        mutation_max_tokens=5000,
        verify_seeds=2,
        evaluation_seeds=[0, 1],
        llm_seed=17,
        evaluator_max_tokens=128,
        evaluator_temperature=0.0,
        evaluator_top_p=1.0,
        child_rollouts=0,
        solver_max_tokens=128,
        child_temperature=1.0,
        child_top_p=0.95,
    )
    monkeypatch.setattr(
        comparison,
        "_evaluate_program_seeds",
        lambda *args, **kwargs: {
            "valid": True,
            "num_valid": 2,
            "num_seeds": 2,
            "per_seed": [
                {"seed": 0, "valid": True},
                {"seed": 1, "valid": True},
            ],
        },
    )

    source_hashes = set()
    for method, plan in (
        ("legacy", plain_plan),
        ("metacognitive", reasoning_plan),
    ):
        inference = FakeInference(plan)
        result = comparison._run_method(
            inference,
            method=method,
            op="in_depth",
            parent=parent,
            output_dir=tmp_path,
            args=args,
            chat_template_kwargs={},
            planning_evidence=evidence,
            meta_progress=comparison._empty_meta_progress(),
        )
        assert result["status"] == "ok"
        assert result["generation_path"] == "registered_compiled"
        assert result["generator_family"] == "linear_system_aggregate"
        assert result["configured_llm_generation_call_count"] == 1
        assert result["llm_generation_call_count"] == 1
        assert len(inference.calls) == 1
        source_hashes.add(result["compiler_source_hash"])
        method_dir = tmp_path / "in_depth" / method
        assert (method_dir / "04_compiler_validation.json").exists()
        assert not (method_dir / "04_code_prompt.json").exists()
        assert not (method_dir / "05_code_raw.txt").exists()
        assert (method_dir / "child.py").exists()
    assert len(source_hashes) == 1


def test_mock_evaluator_receives_the_verified_family_contract_per_seed():
    """End-to-end evaluator wiring with a mock, no GPU and no vLLM session.

    Regression for the ``evaluator_rejected`` run: every seed's evaluator prompt
    must carry the compiler-verified family contract as the authoritative claim,
    with the planner's prose demoted, and a recorded ``target_move_required: NO``
    must still reject the candidate.
    """
    from rq_evolve.mutation_compiler import (
        CompilationStatus,
        MutationSpec,
        compile_mutation_spec,
        family_contract_payload,
        validate_compiled_family_semantics,
    )

    compilation = compile_mutation_spec(
        MutationSpec(
            generator_family="linear_system_aggregate",
            operator="in_depth",
        )
    )
    assert compilation.status is CompilationStatus.COMPILED
    seeds = [0, 1, 2]
    semantics = validate_compiled_family_semantics(compilation, seeds)
    assert semantics.valid, semantics.reasons
    contract = family_contract_payload(compilation, semantics)

    plan = {
        "schema_version": 5,
        "operator": "in_depth",
        "generator_family": "linear_system_aggregate",
        # The parent-shaped prose that produced the false NO verdicts.
        "target_reasoning_move": (
            "adding all three equations gives a multiple of x + y + z"
        ),
    }

    class ContractRecordingInference:
        def __init__(self, raw_output):
            self.prompts = []
            self._raw_output = raw_output
            self.tokenizer = SimpleNamespace(
                encode=lambda text, **kwargs: text.split()
            )

        def generate(self, messages, *, sampling_params, **kwargs):
            self.prompts.extend(messages)
            return [self._raw_output for _ in messages]

    program = compilation.to_problem_program()

    approving = ContractRecordingInference(
        "reason: the hidden row-space combination is required\n\n"
        "target_move_required: YES\n\nverdict: VALID"
    )
    evaluation = comparison._evaluate_program_seeds(
        approving,
        program,
        seeds=seeds,
        mutation_plan=plan,
        family_contract=contract,
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        chat_template_kwargs={},
        operator="in_depth",
        llm_seed=11,
    )
    assert evaluation["valid"] is True
    assert evaluation["num_valid"] == len(seeds)
    assert len(approving.prompts) == len(seeds)
    for messages in approving.prompts:
        user = messages[1]["content"]
        assert "Verified family contract" in user
        assert contract["target_reasoning_move"] in user
        # The plan never reaches the evaluator on the registered path, so the
        # evaluator input cannot differ between plain and reasoning conditions.
        assert "Mutation plan" not in user
        assert plan["target_reasoning_move"] not in user

    # The gate is untouched: the run's recorded NO still rejects every seed.
    refusing = ContractRecordingInference(
        "reason: the answer is correct\n\n"
        "target_move_required: NO\n\nverdict: VALID"
    )
    rejected = comparison._evaluate_program_seeds(
        refusing,
        program,
        seeds=seeds,
        mutation_plan=plan,
        family_contract=contract,
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        chat_template_kwargs={},
        operator="in_depth",
        llm_seed=11,
    )
    assert rejected["valid"] is False
    assert rejected["num_valid"] == 0
    assert all(
        "target reasoning move is not necessary" in row["reason"]
        for row in rejected["per_seed"]
    )
