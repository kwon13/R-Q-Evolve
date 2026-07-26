from dataclasses import asdict
from pathlib import Path

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.backends import PendingRollouts, RolloutRecord
from rq_evolve.code_utils import lint_metacognitive_generator_source
from rq_evolve.config import MetacognitionConfig
from rq_evolve.evolution import RQEvolver
from rq_evolve.metacognition import (
    compute_meta_progress,
    select_reasoning_evidence,
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
        _record("long correct trace with many unnecessary words answer 11", True, 0.7),
        _record("short correct 11", True, 0.8),
        _record("uncertain wrong 12", False, 0.9),
        _record("confident wrong route gives 12", False, 0.1),
    ]

    evidence = select_reasoning_evidence(
        rollouts,
        program=program,
        instance=instance,
        iteration=4,
        max_tokens=4,
    )

    assert [item.role for item in evidence] == ["success", "failure"]
    assert evidence[0].response == "short correct 11"
    assert evidence[1].entropy == 0.1
    assert evidence[1].policy_version == 3
    assert len(evidence[1].response.split()) <= 4  # marker is inside the budget


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


def test_synthetic_plan_shots_follow_schema_v2():
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
        assert plan["schema_version"] == 2
        assert "continue" in plan["decoy_assertion"]


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
    assert plan["schema_version"] == 2


def test_prepare_mutation_runs_planning_before_code_task_without_solver_rollout():
    parent = _program("parent")
    parent.p_hat = 0.5
    parent.h_score = 0.5
    parent.rq_score = 0.125
    parent.metadata["reasoning_evidence"] = [
        asdict(item)
        for item in select_reasoning_evidence(
            [
                _record("short correct 11", True, 0.5),
                _record("confident wrong gives 12", False, 0.1),
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
    assert backend.mutation_batches[0][0].messages[-1]["role"] == "user"
    assert "PARENT_PROGRAM:" in backend.mutation_batches[0][0].messages[-1]["content"]
    assert "Few-shot examples:" not in backend.mutation_batches[0][0].messages[-1]["content"]
    assert tasks[0].stage == "code"
    assert tasks[0].mutation_plan["schema_version"] == 2
    assert tasks[0].plan_status == "planned"
    assert tasks[0].messages[-1]["content"].rstrip().endswith("CHILD_PROGRAM:")


def test_planned_lint_requires_decoy_collision_resampling():
    valid = '''
import random
import sympy
MAX_ATTEMPTS = 200
def generate(seed):
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        answer_insight = rng.randint(2, 20)
        answer_brute = sum(1 for _ in range(answer_insight))
        assert answer_insight == answer_brute
        decoy = answer_insight - 1
        if decoy == answer_insight:
            continue
        assert decoy != answer_insight
        answer = answer_insight
        break
    else:
        raise RuntimeError("exhausted")
    return "Compute the hidden integer.", str(sympy.Integer(answer))
CONCEPT_REASON = "count"
CONCEPT_GROUP = "combinatorics"
CONCEPT_TYPE = "combinatorics.count"
'''
    assert lint_metacognitive_generator_source(valid) == []
    invalid = valid.replace(
        "        if decoy == answer_insight:\n            continue\n",
        "",
    )
    assert any(
        "resample accidental decoy collisions" in reason
        for reason in lint_metacognitive_generator_source(invalid)
    )
