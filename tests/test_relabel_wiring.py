"""The retired relabel path cannot influence local descriptor authority."""

import inspect

import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.config import EvolutionConfig
from rq_evolve.evolution import RQEvolver
from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import MUTATION_OP


class _NoDescriptorBackend:
    """Any backend call during descriptor verification is a test failure."""

    def __init__(self) -> None:
        self.calls = 0

    def mutate(self, _tasks):
        self.calls += 1
        raise AssertionError("descriptor verification must not call mutate")

    def rollout(self, _instances, _n_rollouts):
        self.calls += 1
        raise AssertionError("descriptor verification must not call rollout")


def _program(
    *,
    domain: str | None = "geometry",
    declare_problem_type: bool = False,
    stale_metadata: bool = False,
) -> ProblemProgram:
    declarations = []
    if domain is not None:
        declarations.append(f'DOMAIN = "{domain}"')
    if declare_problem_type:
        declarations.append('PROBLEM_TYPE = "decision"')
    header = "\n".join(declarations)
    if header:
        header += "\n"
    metadata = {"op": MUTATION_OP}
    if stale_metadata:
        metadata.update(
            {
                "domain": "algebra",
                "problem_type": "decision",
                "descriptor_contract": {"authority": "stale"},
            }
        )
    return ProblemProgram(
        source_code=(
            "import random\n"
            + header
            + "\n"
            + "def generate(seed):\n"
            + "    rng = random.Random(seed)\n"
            + "    n = rng.randint(2, 20)\n"
            + "    answer = n + 1\n"
            + "    check = sum((n, 1))\n"
            + '    assert answer == check, "independent addition check"\n'
            + '    problem = f"What is the value of {n} + 1?"\n'
            + '    return problem, str(answer), {"mode": "expression"}\n'
        ),
        metadata=metadata,
    )


def _evolver(backend: _NoDescriptorBackend) -> RQEvolver:
    return RQEvolver(
        archive=MAPElitesArchive(),
        backend=backend,
        evolution_config=EvolutionConfig(verify_seeds=3, ast_contract="off"),
    )


def test_relabel_compatibility_knob_fails_fast():
    with pytest.raises(ValueError, match="relabel_skill is retired"):
        EvolutionConfig(relabel_skill=True)


def test_live_candidate_pipeline_has_no_relabel_or_remote_verdict_stage():
    source = inspect.getsource(RQEvolver.inner_iteration_batch)
    retired = (
        "_apply_relabel",
        "_apply_judge",
        "_run_openai_judge",
        "_drop_oversized_judge_inputs",
    )
    assert [name for name in retired if name in source] == []
    assert [name for name in retired if hasattr(RQEvolver, name)] == []


def test_evolution_config_has_no_remote_descriptor_settings():
    cfg = EvolutionConfig(two_stage_mutation=True)
    retired = (
        "use_evaluator",
        "evaluator_provider",
        "evaluator_model",
        "evaluator_reasoning_effort",
        "evaluator_timeout_s",
        "evaluator_max_output_tokens",
        "evaluator_concurrency",
        "judge_rubric",
        "judge_temperature",
        "judge_top_p",
    )
    assert [name for name in retired if hasattr(cfg, name)] == []


def test_source_domain_and_local_type_rules_override_stale_metadata():
    backend = _NoDescriptorBackend()
    program = _program(stale_metadata=True)
    instance, reason = _evolver(backend).verify_program(program)

    assert reason is None and instance is not None
    assert backend.calls == 0
    assert program.declared_domain() == "geometry"
    assert program.declared_problem_type() is None
    assert program.get_domain() == "geometry"
    assert program.get_problem_type() == "function"
    assert instance.domain == "geometry"
    assert instance.problem_type == "function"

    contract = program.metadata["descriptor_contract"]
    assert contract["domain_authority"] == "source_exact_one_literal"
    assert (
        contract["problem_type_authority"]
        == "deterministic_statement_and_verifier"
    )
    assert contract["domain"] == "geometry"
    assert contract["problem_type"] == "function"
    assert contract["verified_seeds"] == 3
    assert len(contract["problem_type_ruleset_sha256"]) == 64


def test_missing_domain_is_rejected_without_a_backend_call():
    backend = _NoDescriptorBackend()
    program = _program(domain=None)
    instance, reason = _evolver(backend).verify_program(program)
    assert instance is None
    assert "exactly one top-level literal DOMAIN" in reason
    assert backend.calls == 0


def test_model_declared_problem_type_is_rejected_not_trusted():
    backend = _NoDescriptorBackend()
    program = _program(declare_problem_type=True)
    instance, reason = _evolver(backend).verify_program(program)
    assert instance is None
    assert "PROBLEM_TYPE" in reason
    assert backend.calls == 0
