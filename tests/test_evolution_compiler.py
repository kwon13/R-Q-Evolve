import json

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.backends import PendingRollouts
from rq_evolve.config import EvolutionConfig, MetacognitionConfig
from rq_evolve.evolution import CandidateReport, RQEvolver
from rq_evolve.mutation_compiler import (
    MUTATION_FAMILY_REGISTRY_VERSION,
    compile_mutation_plan,
)
from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import MutationTask, build_planned_mutation_task


class RecordingBackend:
    def __init__(self, plan_output: str | None = None, code_output: str | None = None):
        self.plan_output = plan_output
        self.code_output = code_output
        self.mutation_batches: list[list[MutationTask]] = []
        self.rollout_batches: list[list] = []

    def begin_session(self):
        pass

    def end_session(self):
        pass

    def sync_weights(self):
        pass

    def mutate(self, tasks):
        self.mutation_batches.append(list(tasks))
        return [
            self.plan_output if task.stage == "plan" else self.code_output
            for task in tasks
        ]

    def generate_rollouts(self, instances, n_rollouts):
        instances = list(instances)
        self.rollout_batches.append(instances)
        return PendingRollouts(
            instances=instances,
            n_rollouts=n_rollouts,
            grouped=[[] for _ in instances],
        )

    def finalize_rollouts(self, pending):
        return pending.grouped or []


def _parent() -> ProblemProgram:
    return ProblemProgram(
        source_code='''
import sympy

def generate(seed):
    value = seed + 3
    return f"State the integer {value}.", str(sympy.Integer(value))

CONCEPT_REASON = "test fixture"
CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.linear_system_sum"
''',
        program_id="parent",
    )


def _plan(
    *,
    family: str = "linear_system_aggregate",
    family_config: dict | None = None,
) -> dict:
    return {
        "schema_version": 5,
        "operator": "in_depth",
        "failure_summary": "The wrong trace assumes the hidden state is unique.",
        "target_reasoning_move": "identify the invariant fixed by all solutions",
        "correct_wrong_contrast": (
            "The correct trace checks identifiability; the wrong trace does not."
        ),
        "target_concept_group": "algebra",
        "target_concept_type": "algebra.linear_system_sum",
        "generator_family": family,
        "family_config": family_config or {},
        "why_target_reasoning_move_is_necessary": (
            "The full state is underdetermined while one requested sum is fixed."
        ),
        "preserve": ["CONCEPT_GROUP algebra", "CONCEPT_TYPE linear system sum"],
        "change": ["make only the requested aggregate identifiable"],
        "forbidden_changes": ["leak the invariant in the problem statement"],
        "parameters": [{"name": "coefficients", "domain": "-3..3"}],
        "guards": [
            "necessity: rank is deficient and the target lies in the row space"
        ],
        "max_sampling_attempts": 200,
        "answer_route": "express the requested aggregate as a row combination",
        "problem_output_contract": {
            "states": ["two equations in x, y, z"],
            "withholds": ["the row combination"],
            "closing_instruction": "State only the integer value.",
        },
        "answer_output_contract": {
            "type": "integer",
            "range": "[-30,30]",
            "serialization": "str(sympy.Integer(answer))",
        },
        "predicted_pre_behavior": "The Solver reports infinitely many answers.",
        "predicted_post_behavior": "The Solver extracts the fixed aggregate.",
        "heldout_evaluation": "Compare held-out pass rate before and after training.",
    }


def _evolver(backend, *, code_backend: str = "hybrid", fix_retry: bool = False):
    return RQEvolver(
        archive=MAPElitesArchive(),
        backend=backend,
        evolution_config=EvolutionConfig(
            mutation_code_backend=code_backend,
            fix_retry=fix_retry,
            use_evaluator=False,
        ),
        metacognition_config=MetacognitionConfig(enabled=True),
    )


def test_registered_plan_is_precompiled_after_the_single_plan_call(monkeypatch):
    parent = _parent()
    backend = RecordingBackend(plan_output=json.dumps(_plan()))
    evolver = _evolver(backend)
    monkeypatch.setattr(
        "rq_evolve.evolution.collect_planning_evidence",
        lambda *args, **kwargs: [{"role": "correct"}, {"role": "wrong"}],
    )

    prepared = evolver._prepare_mutation_tasks([("in_depth", parent)])

    assert len(backend.mutation_batches) == 1
    assert backend.mutation_batches[0][0].stage == "plan"
    assert len(prepared) == 1
    task = prepared[0]
    assert isinstance(task, MutationTask)
    assert task.precompiled_source
    assert task.generation_path == "registered_compiled"
    assert task.generator_family == "linear_system_aggregate"
    assert task.plan_status == "compiled_family"
    assert task.compiler_version == str(MUTATION_FAMILY_REGISTRY_VERSION)
    # The deterministic family-semantic gate ran and certified both axes, and
    # its contract is attached for the evaluator's necessity judgement.
    semantics = task.compiler_diagnostics["family_semantics"]
    assert semantics["valid"] is True
    assert semantics["reasons"] == []
    assert task.family_contract is not None
    assert task.family_contract["target_reasoning_move"]
    verification = task.family_contract["deterministic_verification"]
    assert verification["answer_oracle_agrees"] is True
    assert verification["necessity_holds"] is True


def test_family_semantics_failure_is_terminal_before_the_evaluator(monkeypatch):
    """A degenerate compiled family must never reach an LLM evaluator call."""
    from rq_evolve.mutation_compiler import FamilySemanticValidation

    parent = _parent()
    backend = RecordingBackend(plan_output=json.dumps(_plan()))
    evolver = _evolver(backend)
    monkeypatch.setattr(
        "rq_evolve.evolution.collect_planning_evidence",
        lambda *args, **kwargs: [{"role": "correct"}, {"role": "wrong"}],
    )
    monkeypatch.setattr(
        "rq_evolve.evolution.validate_compiled_family_semantics",
        lambda result, seeds: FamilySemanticValidation(
            valid=False,
            generator_family=result.generator_family,
            seeds=tuple(seeds),
            reasons=("seed=0: necessity: plain-sum bypass",),
        ),
    )

    prepared = evolver._prepare_mutation_tasks([("in_depth", parent)])

    assert len(prepared) == 1
    report = prepared[0]
    assert not isinstance(report, MutationTask)
    assert report.status == "family_semantics_rejected"
    assert "plain-sum bypass" in report.reason
    # Only the single plan call happened; no code or evaluator call followed.
    assert len(backend.mutation_batches) == 1
    assert any(
        event["event"] == "family_semantics_rejected"
        for event in evolver.events
    )


def test_invalid_registered_spec_is_terminal_without_code_call(monkeypatch):
    parent = _parent()
    bad_plan = _plan(
        family_config={
            "coefficient_min": 4,
            "coefficient_max": -4,
            "solution_min": -2,
            "solution_max": 2,
        }
    )
    backend = RecordingBackend(plan_output=json.dumps(bad_plan))
    evolver = _evolver(backend)
    monkeypatch.setattr(
        "rq_evolve.evolution.collect_planning_evidence",
        lambda *args, **kwargs: [{"role": "correct"}, {"role": "wrong"}],
    )

    prepared = evolver._prepare_mutation_tasks([("in_depth", parent)])

    assert len(backend.mutation_batches) == 1
    assert len(prepared) == 1
    report = prepared[0]
    assert isinstance(report, CandidateReport)
    assert report.status == "invalid_spec"
    assert report.generation_path == "registered_compiled"


def test_precompiled_task_skips_code_model_and_compiler_failure_never_retries(
    monkeypatch,
):
    parent = _parent()
    plan = _plan()
    compiled = compile_mutation_plan(plan, parent, "in_depth")
    assert compiled.compiled
    task = build_planned_mutation_task("in_depth", parent, plan)
    task.precompiled_source = compiled.source_code
    task.generation_path = "registered_compiled"
    task.generator_family = compiled.generator_family
    task.compiler_version = str(MUTATION_FAMILY_REGISTRY_VERSION)
    task.compiler_diagnostics = {
        "status": "compiled",
        "source_hash": compiled.source_hash,
        "family_config": dict(compiled.family_config),
    }
    task.plan_status = "compiled_family"

    backend = RecordingBackend()
    evolver = _evolver(backend, fix_retry=True)
    monkeypatch.setattr(evolver.archive, "sample_parent", lambda: parent)
    monkeypatch.setattr(evolver, "_sample_operator", lambda _parent: "in_depth")
    monkeypatch.setattr(evolver, "_prepare_mutation_tasks", lambda _requests: [task])

    reports = evolver.inner_iteration_batch(1)

    assert backend.mutation_batches == []
    assert len(backend.rollout_batches) == 1
    assert len(backend.rollout_batches[0]) == 1
    assert reports[0].generation_path == "registered_compiled"

    # Exercise the verifier-failure branch with retry enabled: it must remain
    # terminal and make no model call.
    task.precompiled_source = "def generate(seed):\n    return '', ''\n"
    backend.mutation_batches.clear()
    backend.rollout_batches.clear()
    reports = evolver.inner_iteration_batch(1)
    assert backend.mutation_batches == []
    assert reports[0].status == "compiler_error"
    assert "failed verification" in (reports[0].reason or "")


def test_hybrid_unknown_family_is_diagnostic_only_and_never_archived(monkeypatch):
    parent = _parent()
    free_form_plan = _plan(family="free_form.partial_identifiability")
    registered = compile_mutation_plan(_plan(), parent, "in_depth")
    assert registered.compiled
    backend = RecordingBackend(
        plan_output=json.dumps(free_form_plan),
        code_output=f"```python\n{registered.source_code}\n```",
    )
    evolver = _evolver(backend)
    monkeypatch.setattr(
        "rq_evolve.evolution.collect_planning_evidence",
        lambda *args, **kwargs: [{"role": "correct"}, {"role": "wrong"}],
    )
    monkeypatch.setattr(evolver.archive, "sample_parent", lambda: parent)
    monkeypatch.setattr(evolver, "_sample_operator", lambda _parent: "in_depth")

    reports = evolver.inner_iteration_batch(1)

    assert [batch[0].stage for batch in backend.mutation_batches] == [
        "plan",
        "code",
    ]
    assert reports[0].status == "quarantined_freeform"
    assert reports[0].quarantined is True
    assert backend.rollout_batches == [[]]
    assert list(evolver.archive.champions()) == []
