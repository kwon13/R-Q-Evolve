from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.backends import PendingRollouts, RolloutRecord
from rq_evolve.code_utils import lint_metacognitive_generator_source
from rq_evolve.config import MetacognitionConfig
from rq_evolve.evolution import RQEvolver
from rq_evolve.metacognition import (
    clean_and_grade_solver_rollout,
    collect_planning_evidence,
    compute_meta_progress,
    reasoning_trace_quality_issues,
    sanitize_solver_trace,
    select_reasoning_evidence,
    validate_reasoning_contrast,
)
from rq_evolve.mutation_compiler import (
    CompilationStatus,
    compile_mutation_plan,
)
from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import parse_mutation_plan
from rq_evolve.scoring import RQResult


class ScriptedBackend:
    def __init__(self, grouped):
        self.grouped = grouped
        self.generate_calls = 0

    def begin_session(self):
        pass

    def end_session(self):
        pass

    def sync_weights(self):
        pass

    def mutate(self, tasks):
        return [None] * len(tasks)

    def generate_rollouts(self, instances, n_rollouts):
        self.generate_calls += 1
        return PendingRollouts(
            instances=list(instances),
            n_rollouts=n_rollouts,
            grouped=self.grouped,
        )

    def finalize_rollouts(self, pending):
        return pending.grouped


class PlanningBackend(ScriptedBackend):
    def __init__(self, plan_output):
        super().__init__([])
        self.plan_output = plan_output
        self.mutation_batches = []

    def mutate(self, tasks):
        self.mutation_batches.append(list(tasks))
        return [self.plan_output] * len(tasks)


def _record(response, correct, entropy):
    return RolloutRecord(
        response=response,
        predicted_answer="11" if correct else "12",
        correct=correct,
        entropy=entropy,
        policy_version=3,
    )


def _program(program_id="p"):
    return ProblemProgram(
        source_code='''
import sympy


def generate(seed):
    left = seed + 10
    right = seed + 1
    answer = left + right
    problem = f"Compute {left} plus {right}. State only the integer."
    return problem, str(sympy.Integer(answer))


CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.integer_addition"
''',
        program_id=program_id,
    )


def test_evidence_reuses_one_correct_and_one_confident_wrong_trace():
    program = _program()
    instance = program.execute(0)
    assert instance is not None
    rollouts = [
        _record(r"long correct trace with many words \boxed{11}", True, 0.7),
        _record(r"short correct \boxed{11}", True, 0.8),
        _record(r"uncertain wrong \boxed{12}", False, 0.9),
        _record(r"confident wrong route gives \boxed{12}", False, 0.1),
    ]

    evidence = select_reasoning_evidence(
        rollouts,
        program=program,
        instance=instance,
        iteration=4,
        max_tokens=10,
    )

    assert [item.role for item in evidence] == ["success", "failure"]
    assert evidence[0].response == r"short correct \boxed{11}"
    assert evidence[1].entropy == 0.1
    assert evidence[1].policy_version == 3
    assert "...[trace truncated]..." not in evidence[1].response


def test_evidence_trims_second_chat_and_regrades_first_answer():
    program = _program()
    instance = program.execute(0)
    assert instance is not None
    contaminated = _record(
        (
            r"Correct route gives \boxed{11}."
            "\nPlease reason step by step, and put your final answer within boxed."
            "\nAssistant: user\nUnrelated problem. "
            r"\boxed{12}"
        ),
        False,
        0.05,
    )
    contaminated.predicted_answer = "12"
    actual_wrong = _record(r"Wrong arithmetic gives \boxed{12}.", False, 0.2)

    evidence = select_reasoning_evidence(
        [contaminated, actual_wrong],
        program=program,
        instance=instance,
        iteration=0,
        max_tokens=100,
    )

    assert [item.role for item in evidence] == ["success", "failure"]
    assert evidence[0].predicted_answer == "11"
    assert "Unrelated problem" not in evidence[0].response
    assert sanitize_solver_trace(contaminated.response).endswith(r"\boxed{11}.")


def test_repetitive_low_entropy_loop_is_not_selected_as_wrong_evidence():
    program = _program()
    instance = program.execute(0)
    assert instance is not None
    loop = _record("\n".join(["3x + y = 6"] * 12), False, 0.001)
    clean_wrong = _record(
        r"I add 10 and 1 incorrectly and obtain \boxed{12}.",
        False,
        0.2,
    )

    evidence = select_reasoning_evidence(
        [
            _record(r"10+1=11, so \boxed{11}.", True, 0.5),
            loop,
            clean_wrong,
        ],
        program=program,
        instance=instance,
        iteration=0,
        max_tokens=100,
    )

    assert [item.role for item in evidence] == ["success", "failure"]
    assert "incorrectly" in evidence[1].response
    assert "repetitive_no_progress_loop" in reasoning_trace_quality_issues(
        loop.response,
        loop.predicted_answer,
    )


def test_incomplete_wrong_trace_does_not_form_reasoning_contrast():
    program = _program()
    instance = program.execute(0)
    assert instance is not None
    incomplete = _record("I will start solving but do not finish.", False, 0.01)
    incomplete.predicted_answer = None

    evidence = select_reasoning_evidence(
        [
            _record(r"10+1=11, so \boxed{11}.", True, 0.5),
            incomplete,
        ],
        program=program,
        instance=instance,
        iteration=0,
        max_tokens=100,
    )

    assert evidence == []


def test_reasoning_contrast_rejects_cross_policy_pair():
    common = {
        "seed": 0,
        "problem": "Compute 10+1.",
        "origin_program_id": "p",
    }
    issues = validate_reasoning_contrast(
        [
            {
                **common,
                "role": "success",
                "response": r"Correct \boxed{11}.",
                "predicted_answer": "11",
                "policy_version": 1,
            },
            {
                **common,
                "role": "failure",
                "response": r"Wrong \boxed{12}.",
                "predicted_answer": "12",
                "policy_version": 2,
            },
        ]
    )
    assert "contrast_policy_version_mismatch" in issues


def test_stored_evidence_is_regraded_against_live_parent_and_roles_overwritten():
    program = _program()
    instance = program.execute(0)
    assert instance is not None
    stored = [
        asdict(item)
        for item in select_reasoning_evidence(
            [
                _record(r"Correct arithmetic gives \boxed{11}.", True, 0.5),
                _record(r"Wrong arithmetic gives \boxed{12}.", False, 0.1),
            ],
            program=program,
            instance=instance,
            iteration=2,
            max_tokens=100,
        )
    ]
    for item in stored:
        item["correct"] = not item["correct"]
        item["role"] = "success" if item["correct"] else "failure"
        item["predicted_answer"] = "999"
    program.metadata["reasoning_evidence"] = stored

    evidence = collect_planning_evidence(
        program,
        "in_depth",
        [program],
        total_tokens=100,
    )

    assert [item["role"] for item in evidence] == ["success", "failure"]
    assert [item["correct"] for item in evidence] == [True, False]
    assert [item["predicted_answer"] for item in evidence] == ["11", "12"]


def test_stored_evidence_requires_exact_parent_provenance_and_is_not_truncated():
    program = _program()
    instance = program.execute(0)
    assert instance is not None
    stored = [
        asdict(item)
        for item in select_reasoning_evidence(
            [
                _record(r"Correct route gives \boxed{11}.", True, 0.5),
                _record(r"Wrong route gives \boxed{12}.", False, 0.1),
            ],
            program=program,
            instance=instance,
            iteration=2,
            max_tokens=100,
        )
    ]
    program.metadata["reasoning_evidence"] = stored
    assert collect_planning_evidence(
        program,
        "in_depth",
        [program],
        total_tokens=1,
    ) == []
    assert all("...[trace truncated]..." not in item["response"] for item in stored)

    stored[0]["origin_program_id"] = "not-the-live-parent"
    assert collect_planning_evidence(
        program,
        "in_depth",
        [program],
        total_tokens=100,
    ) == []


def test_cleaned_rollout_grade_uses_first_conversation_answer():
    program = _program()
    instance = program.execute(0)
    assert instance is not None
    contaminated = _record(
        (
            r"Correct route gives \boxed{11}."
            "\nAssistant: user\nUnrelated problem. "
            r"\boxed{12}"
        ),
        False,
        0.05,
    )
    contaminated.predicted_answer = "12"

    cleaned, predicted, correct = clean_and_grade_solver_rollout(
        contaminated,
        instance,
    )

    assert cleaned == r"Correct route gives \boxed{11}."
    assert predicted == "11"
    assert correct is True


def test_meta_progress_uses_pre_update_fixed_cohort():
    first = _program("first")
    second = _program("second")
    first.metadata["op"] = "in_depth"
    second.metadata["op"] = "in_breadth"
    before = {
        "first": {
            "p_hat": 0.25,
            "concept_group": "algebra",
            "concept_type": "algebra.integer_addition",
            "operator": "in_depth",
        },
        "second": {
            "p_hat": 0.75,
            "concept_group": "algebra",
            "concept_type": "algebra.integer_addition",
            "operator": "in_breadth",
        },
    }
    results = [
        RQResult(0.1, 0.50, 0.25, 0.4, 4, 2),
        RQResult(0.1, 1.00, 0.00, 0.4, 4, 4),
    ]

    progress = compute_meta_progress(before, [first, second], results, iteration=7)

    assert progress.global_progress.pre_mean_p == 0.5
    assert progress.global_progress.post_mean_p == 0.75
    assert progress.global_progress.delta_p == 0.25
    assert progress.by_operator["in_depth"].delta_p == 0.25
    assert progress.by_operator["in_breadth"].delta_p == 0.25


def test_reevaluation_measures_delta_before_zero_rq_removes_champion():
    champion = _program("champion")
    champion.p_hat = 0.5
    champion.h_score = 1.0
    champion.rq_score = 0.25
    champion.fitness = 0.25
    archive = MAPElitesArchive()
    assert archive.try_insert(
        champion,
        h_value=1.0,
        problem_text="bootstrap",
        rq_score=0.25,
    )
    backend = ScriptedBackend(
        [[_record("answer 11", True, 0.5), _record("also 11", True, 0.5)]]
    )
    evolver = RQEvolver(
        archive=archive,
        backend=backend,
        metacognition_config=MetacognitionConfig(enabled=True),
    )
    evolver.current_iteration = 2

    evolver.reevaluate_champions()

    assert backend.generate_calls == 1
    assert evolver.last_meta_progress["global_progress"]["pre_mean_p"] == 0.5
    assert evolver.last_meta_progress["global_progress"]["post_mean_p"] == 1.0
    assert evolver.last_meta_progress["global_progress"]["delta_p"] == 0.5
    assert archive.champions() == []
    assert any(
        event["event"] == "champion_removed_after_reevaluation"
        and event["reason"] == "rq_zero"
        for event in evolver.events
    )


def test_metacognitive_controller_state_round_trips_with_archive(tmp_path):
    champion = _program("champion")
    champion.p_hat = 0.5
    champion.h_score = 1.0
    champion.rq_score = 0.25
    archive = MAPElitesArchive()
    assert archive.try_insert(champion, 1.0, "bootstrap", 0.25)
    original = RQEvolver(
        archive=archive,
        backend=ScriptedBackend([]),
        metacognition_config=MetacognitionConfig(enabled=True),
    )
    original.last_meta_progress = {
        "global_progress": {"count": 1, "delta_p": 0.25}
    }
    original.operator_ema = {"in_depth": 0.2, "in_breadth": -0.1}
    original.current_iteration = 9
    original.save_state(tmp_path)

    restored = RQEvolver(
        archive=MAPElitesArchive(),
        backend=ScriptedBackend([]),
        metacognition_config=MetacognitionConfig(enabled=True),
    )
    assert restored.load_state(tmp_path)
    assert restored.last_meta_progress == original.last_meta_progress
    assert restored.operator_ema == original.operator_ema
    assert restored.current_iteration == 9


def test_synthetic_plan_shots_follow_schema_v5_without_comparison_routes():
    shot_dir = Path("prompt_templates/shots")
    for filename, op in (
        ("metacognitive_in_depth.txt", "in_depth"),
        ("metacognitive_in_breadth.txt", "in_breadth"),
    ):
        plan, reason = parse_mutation_plan(
            (shot_dir / filename).read_text(encoding="utf-8"),
            op,
        )
        assert plan is not None, reason
        assert plan["schema_version"] == 5
        assert isinstance(plan["generator_family"], str)
        assert isinstance(plan["family_config"], dict)
        assert plan["answer_route"]
        assert plan["why_target_reasoning_move_is_necessary"]
        assert not {
            "insight_route",
            "brute_route",
            "equivalence_assertion",
            "route_independence_reason",
            "decoy_route",
            "decoy_assertion",
        } & plan.keys()


def test_breadth_plan_preserves_inherited_reasoning_move_as_exact_join_key():
    output = Path(
        "prompt_templates/shots/metacognitive_in_breadth.txt"
    ).read_text(encoding="utf-8")
    plan, reason = parse_mutation_plan(
        output,
        "in_breadth",
        required_target_reasoning_move="a different move",
    )
    assert plan is None
    assert "exactly match" in reason


def test_plan_parser_skips_reasoning_braces_before_valid_json():
    shot = Path(
        "prompt_templates/shots/metacognitive_in_depth.txt"
    ).read_text(encoding="utf-8")
    output = '<think>{"draft": true}</think>\n' + shot
    plan, reason = parse_mutation_plan(output, "in_depth")
    assert plan is not None, reason
    assert plan["schema_version"] == 5


def test_plan_parser_rejects_removed_or_unexpected_route_fields():
    shot = Path(
        "prompt_templates/shots/metacognitive_in_depth.txt"
    ).read_text(encoding="utf-8")
    plan, reason = parse_mutation_plan(shot, "in_depth")
    assert plan is not None, reason

    import json

    for field_name, expected_reason in (
        ("brute_route", "removed schema_version 5 plan field"),
        ("decoy", "unexpected schema_version 5 plan field"),
        ("answer_brute", "unexpected schema_version 5 plan field"),
        ("wrong_answer_route", "unexpected schema_version 5 plan field"),
    ):
        contaminated = {**plan, field_name: "obsolete extra computation"}
        rejected, reason = parse_mutation_plan(
            json.dumps(contaminated),
            "in_depth",
        )
        assert rejected is None
        assert f"{expected_reason}: {field_name}" in reason


def test_plan_required_text_fields_reject_null_and_non_strings():
    import json

    shot = Path(
        "prompt_templates/shots/metacognitive_in_depth.txt"
    ).read_text(encoding="utf-8")
    plan, reason = parse_mutation_plan(shot, "in_depth")
    assert plan is not None, reason

    for field_name, invalid_value in (
        ("target_reasoning_move", None),
        ("answer_route", ["not", "a", "string"]),
        ("why_target_reasoning_move_is_necessary", {"text": "not a string"}),
        ("failure_summary", None),
    ):
        rejected, reason = parse_mutation_plan(
            json.dumps({**plan, field_name: invalid_value}),
            "in_depth",
        )
        assert rejected is None
        assert "non-empty string" in reason


def test_plan_requires_executable_necessity_and_contract_shapes():
    import json

    shot = Path(
        "prompt_templates/shots/metacognitive_in_depth.txt"
    ).read_text(encoding="utf-8")
    plan, reason = parse_mutation_plan(shot, "in_depth")
    assert plan is not None, reason

    malformed_plans = (
        (
            {**plan, "guards": ["gcd(a,m)=1"]},
            "guards[0] must start with `necessity:`",
        ),
        (
            {**plan, "parameters": [{"name": "a"}]},
            "parameters[0].domain",
        ),
        (
            {
                **plan,
                "problem_output_contract": {
                    **plan["problem_output_contract"],
                    "states": [],
                },
            },
            "problem_output_contract.states",
        ),
    )
    for malformed, expected in malformed_plans:
        rejected, reason = parse_mutation_plan(
            json.dumps(malformed),
            "in_depth",
        )
        assert rejected is None
        assert expected in reason


def test_legacy_schema_v4_parses_but_compiler_quarantines_it():
    import json

    shot = Path(
        "prompt_templates/shots/metacognitive_in_depth.txt"
    ).read_text(encoding="utf-8")
    plan, reason = parse_mutation_plan(shot, "in_depth")
    assert plan is not None, reason
    legacy_plan = dict(plan)
    legacy_plan["schema_version"] = 4
    legacy_plan.pop("generator_family")
    legacy_plan.pop("family_config")

    parsed, reason = parse_mutation_plan(
        json.dumps(legacy_plan),
        "in_depth",
    )
    assert parsed == legacy_plan, reason
    compiled = compile_mutation_plan(parsed, _program(), "in_depth")
    assert compiled.status is CompilationStatus.UNSUPPORTED
    assert "quarantine" in "; ".join(compiled.reasons)


def test_plain_plan_requires_null_observation_fields_and_live_targets():
    import json

    parent = _program()
    shot = Path(
        "prompt_templates/shots/metacognitive_in_depth.txt"
    ).read_text(encoding="utf-8")
    plan, reason = parse_mutation_plan(shot, "in_depth")
    assert plan is not None, reason
    plan["target_concept_group"] = parent.get_concept_group()
    plan["target_concept_type"] = parent.get_concept_type()
    for field in (
        "failure_summary",
        "correct_wrong_contrast",
        "predicted_pre_behavior",
        "predicted_post_behavior",
    ):
        plan[field] = None

    parsed, reason = parse_mutation_plan(
        json.dumps(plan),
        "in_depth",
        reasoning_informed=False,
        parent=parent,
    )
    assert parsed is not None, reason
    contaminated = {**plan, "failure_summary": "invented failure"}
    rejected, reason = parse_mutation_plan(
        json.dumps(contaminated),
        "in_depth",
        reasoning_informed=False,
        parent=parent,
    )
    assert rejected is None
    assert "plain plan field must be null" in reason


def test_prepare_mutation_runs_planning_before_code_task_without_solver_rollout():
    parent = _program("parent")
    parent.p_hat = 0.5
    parent.h_score = 0.5
    parent.rq_score = 0.125
    parent.metadata["reasoning_evidence"] = [
        asdict(item)
        for item in select_reasoning_evidence(
                [
                    _record(r"short correct \boxed{11}", True, 0.5),
                    _record(r"confident wrong gives \boxed{12}", False, 0.1),
            ],
            program=parent,
            instance=parent.execute(0),
            iteration=0,
            max_tokens=4096,
        )
    ]
    plan_output = Path(
        "prompt_templates/shots/metacognitive_in_depth.txt"
    ).read_text(encoding="utf-8")
    parsed_plan, reason = parse_mutation_plan(plan_output, "in_depth")
    assert parsed_plan is not None, reason
    parsed_plan["target_concept_group"] = parent.get_concept_group()
    parsed_plan["target_concept_type"] = parent.get_concept_type()
    import json

    plan_output = json.dumps(parsed_plan)
    backend = PlanningBackend(plan_output)
    evolver = RQEvolver(
        archive=MAPElitesArchive(),
        backend=backend,
        metacognition_config=MetacognitionConfig(enabled=True),
    )

    tasks = evolver._prepare_mutation_tasks([("in_depth", parent)])

    assert backend.generate_calls == 0
    assert len(backend.mutation_batches) == 1
    assert backend.mutation_batches[0][0].stage == "plan"
    assert backend.mutation_batches[0][0].max_output_tokens == 1024
    assert backend.mutation_batches[0][0].temperature == 0.7
    assert backend.mutation_batches[0][0].top_p == 0.95
    assert backend.mutation_batches[0][0].messages[-1]["role"] == "user"
    assert "PARENT_PROGRAM:" in backend.mutation_batches[0][0].messages[-1]["content"]
    assert "Few-shot examples:" not in backend.mutation_batches[0][0].messages[-1]["content"]
    assert tasks[0].stage == "code"
    assert tasks[0].temperature == 0.2
    assert tasks[0].top_p == 0.95
    assert tasks[0].mutation_plan["schema_version"] == 5
    assert tasks[0].plan_status == "planned"
    assert tasks[0].messages[-1]["content"].rstrip().endswith("CHILD_PROGRAM:")
    planned_prompt = "\n".join(message["content"] for message in tasks[0].messages)
    assert "answer_route" in planned_prompt
    assert "answer_brute" not in planned_prompt
    assert "decoy_route" not in planned_prompt


def test_default_mutation_lint_accepts_one_answer_route_without_assert():
    source = '''
import random
import sympy
MAX_ATTEMPTS = 200
def generate(seed):
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        n = rng.randint(5, 20)
        answer = n * (n + 1) // 2
        problem = f"Find the sum of the integers from 1 through {n}."
        return problem, str(sympy.Integer(answer))
    else:
        raise RuntimeError("exhausted")
CONCEPT_REASON = "count"
CONCEPT_GROUP = "combinatorics"
CONCEPT_TYPE = "combinatorics.count"
'''
    assert lint_metacognitive_generator_source(source) == []
    assert any(
        "answer_insight" in reason
        for reason in lint_metacognitive_generator_source(
            source,
            require_assert=True,
            require_answer_routes=True,
        )
    )


def test_canonical_instance_contract_rejects_stale_rendering_and_mutation():
    stale_rendering = '''
import random
import sympy
MAX_ATTEMPTS = 200
def generate(seed):
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        g = rng.randint(1, 4)
        rows = [[1, 0], [0, 1]]
        rows[1] = [2, 3]
        rhs = [4, 5]
        instance_data = {"rows": rows, "rhs": rhs}
        answer = sum(instance_data["rhs"])
        assert answer == sum(instance_data["rhs"])
        problem = f"Use row {g} with right side {rhs[1]}."
        return problem, str(sympy.Integer(answer))
    else:
        raise RuntimeError("exhausted")
CONCEPT_REASON = "combine rows"
CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.rows"
'''
    reasons = lint_metacognitive_generator_source(
        stale_rendering,
        require_canonical_instance_data=True,
    )
    assert any("names outside" in reason for reason in reasons)
    assert any("g" in reason and "rhs" in reason for reason in reasons)

    post_canonical_mutation = stale_rendering.replace(
        'answer = sum(instance_data["rhs"])',
        'instance_data["rhs"].append(6)\n        answer = sum(instance_data["rhs"])',
    ).replace(
        'problem = f"Use row {g} with right side {rhs[1]}."',
        "problem = f\"Use rows {instance_data['rows']}.\"",
    )
    reasons = lint_metacognitive_generator_source(
        post_canonical_mutation,
        require_canonical_instance_data=True,
    )
    assert any("may not mutate `instance_data`" in reason for reason in reasons)


def test_canonical_instance_contract_accepts_one_shared_visible_object():
    source = '''
import random
import sympy
MAX_ATTEMPTS = 200
def generate(seed):
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        n = rng.randint(5, 20)
        instance_data = {"n": n}
        answer = instance_data["n"] * (instance_data["n"] + 1) // 2
        assert answer == sum(range(1, instance_data["n"] + 1))
        problem = (
            f"Find the sum of the integers from 1 through "
            f"{instance_data['n']}."
        )
        return problem, str(sympy.Integer(answer))
    else:
        raise RuntimeError("exhausted")
CONCEPT_REASON = "count"
CONCEPT_GROUP = "combinatorics"
CONCEPT_TYPE = "combinatorics.count"
'''
    assert (
        lint_metacognitive_generator_source(
            source,
            require_canonical_instance_data=True,
        )
        == []
    )
    repeated_answer_expression = source.replace(
        'sum(range(1, instance_data["n"] + 1))',
        'instance_data["n"] * (instance_data["n"] + 1) // 2',
    )
    assert any(
        "without repeating the answer assignment" in reason
        for reason in lint_metacognitive_generator_source(
            repeated_answer_expression,
            require_canonical_instance_data=True,
        )
    )


def test_compatibility_dual_route_lint_rejects_identical_assignments():
    source = '''
import random
import sympy
MAX_ATTEMPTS = 200
def generate(seed):
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        value = rng.randint(2, 20)
        answer_insight = value
        answer_brute = value
        assert answer_insight == answer_brute
        answer = answer_insight
        break
    else:
        raise RuntimeError("exhausted")
    return "Compute the hidden integer.", str(sympy.Integer(answer))
CONCEPT_REASON = "count"
CONCEPT_GROUP = "combinatorics"
CONCEPT_TYPE = "combinatorics.count"
'''
    assert any(
        "identical assignments" in reason
        for reason in lint_metacognitive_generator_source(
            source,
            require_assert=True,
            require_answer_routes=True,
        )
    )


def test_verify_program_uses_same_single_answer_contract_with_or_without_plan():
    source = '''
import random
import sympy

MAX_ATTEMPTS = 200

def generate(seed):
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        n = rng.randint(5, 20)
        instance_data = {"n": n}
        answer = (
            instance_data["n"] * (instance_data["n"] + 1) // 2
        )
        assert answer == sum(range(1, instance_data["n"] + 1))
        problem = (
            "Find the sum of all positive integers from 1 through "
            f"{instance_data['n']}. "
            "State only the integer."
        )
        return problem, str(sympy.Integer(answer))
    else:
        raise RuntimeError("exhausted")

CONCEPT_REASON = "Sum a finite arithmetic sequence."
CONCEPT_GROUP = "sequence"
CONCEPT_TYPE = "sequence.arithmetic_sum"
'''
    evolver = RQEvolver(
        archive=MAPElitesArchive(),
        backend=ScriptedBackend([]),
        metacognition_config=MetacognitionConfig(enabled=False),
    )

    for metadata in (
        {"op": "in_depth"},
        {
            "op": "in_depth",
            "mutation_plan": {"schema_version": 4, "answer_route": "sum 1..n"},
        },
    ):
        child = ProblemProgram(source_code=source, metadata=metadata)
        instance, reason = evolver.verify_program(child, n_seeds=5)
        assert instance is not None, reason


def test_operator_contract_applies_to_plain_and_planned_tasks_equally():
    parent = _program()
    same_group_child = _program("child")
    for mutation_plan in (None, {"target_concept_group": "sequence"}):
        task = SimpleNamespace(
            op="in_breadth",
            parent=parent,
            mutation_plan=mutation_plan,
        )
        reason = RQEvolver._validate_mutation_contract(
            task,
            same_group_child,
        )
        assert reason is not None
        assert "must change CONCEPT_GROUP" in reason
