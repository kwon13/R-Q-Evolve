"""Regression: all-rejected rollout groups must NEVER zero a program's scores.

A transient infra failure (chunk timeout, worker error) during
reevaluate_champions / bootstrap previously flowed into _score_from_rollouts,
which mutates s_hat/u_score/rq_score IN PLACE -- evicting a champion from its
true niche (the stale-s_hat archive-pollution failure mode). evaluate_instances
now yields None for those groups and callers keep prior scores.
"""

from types import SimpleNamespace
from dataclasses import asdict

import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.backends import PendingRollouts, RolloutRecord
from rq_evolve.config import ArchiveConfig, EvolutionConfig
from rq_evolve.evolution import RQEvolver
from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import MUTATION_OP


def test_adaptive_mutation_refill_is_bounded_and_stops_on_frontier_supply():
    """Refill adds whole batches but can never run past the proposal cap."""

    from rq_evolve.evolution import CandidateReport

    class Backend:
        @staticmethod
        def sync_weights():
            pass

    config = EvolutionConfig(
        inner_iterations=2,
        inner_iteration_batch_size=2,
        adaptive_mutation_refill=True,
        mutation_refill_max_iterations=6,
        mutation_refill_target_frontier_insertions=1,
        mutation_refill_target_frontier_candidates=3,
        reevaluate_champions=False,
    )
    evolver = RQEvolver(
        archive=MAPElitesArchive(**asdict(ArchiveConfig())),
        backend=Backend(),
        evolution_config=config,
    )
    batches = [
        [
            CandidateReport(status="rejected_non_elite", op="mutate", s_hat=0.5),
            CandidateReport(status="verify_failed", op="mutate"),
        ],
        [
            CandidateReport(status="rejected_non_elite", op="mutate", s_hat=0.5),
            CandidateReport(status="inserted", op="mutate", s_hat=0.5),
        ],
    ]
    evolver.inner_iteration_batch = lambda _size: batches.pop(0)
    evolver.refresh_dataset = lambda **_kwargs: None

    metrics = evolver.run_outer_iteration(1)

    assert metrics["mutation_sampled_slots"] == 4
    assert metrics["mutation_batches"] == 2
    assert metrics["mutation_frontier_candidates"] == 3
    assert metrics["mutation_frontier_insertions"] == 1
    assert metrics["mutation_refill_stop_reason"] == "frontier_insertion_target"


def test_adaptive_mutation_refill_rejects_an_unbounded_configuration():
    with pytest.raises(ValueError, match="positive stop target"):
        EvolutionConfig(
            adaptive_mutation_refill=True,
            mutation_refill_target_frontier_insertions=0,
            mutation_refill_target_frontier_candidates=0,
        )


def test_strict_champion_audit_evicts_invalid_snapshot_programs_once():
    archive = MAPElitesArchive(**asdict(ArchiveConfig()))
    good = ProblemProgram("def generate(seed):\n    return 'Compute 1.', '1'\n", program_id="good")
    bad = ProblemProgram("def generate(seed):\n    return 'Compute 2.', '2'\n", program_id="bad")
    archive.grid[(0, 0)].champion = good
    archive.grid[(0, 1)].champion = bad
    evolver = RQEvolver(
        archive=archive,
        backend=SimpleNamespace(),
        evolution_config=EvolutionConfig(strict_champion_audit=True),
    )
    calls = []

    def fake_verify(program, **kwargs):
        calls.append((program.program_id, kwargs.get("reserve_seed_stream")))
        return (None, "bad historical contract") if program.program_id == "bad" else (object(), None)

    evolver.verify_program = fake_verify
    first = evolver.audit_champions_strict_once()
    second = evolver.audit_champions_strict_once()

    assert first == second == {
        "strict_champion_audit_checked": 2,
        "strict_champion_audit_evicted": 1,
    }
    assert [program.program_id for program in archive.champions()] == ["good"]
    assert calls == [("good", False), ("bad", False)]
    assert evolver.events[-1]["event"] == "strict_champion_audit_evicted"


def test_stage_one_visible_example_copy_never_reaches_stage_two():
    parent = _program("parent")

    class Backend:
        calls = 0

        def mutate(self, tasks):
            self.calls += 1
            assert all(task.stage == "family" for task in tasks)
            shown = tasks[0].provenance["stage1_visible_examples"][0]["family"]
            return [
                "STRUCTURAL MUTATION: copy the worked family\n"
                f"CHILD FAMILY: {shown}\n"
                "WHY FINITE: the shown family has a finite exact answer"
            ]

    backend = Backend()
    evolver = RQEvolver(
        archive=MAPElitesArchive(),
        backend=backend,
        evolution_config=EvolutionConfig(
            two_stage_mutation=True, rotate_few_shots=False
        ),
    )
    tasks, outputs, _ = evolver._mutate_in_two_stages([parent])
    child, instance, reason, source = evolver._make_child_from_output(
        tasks[0], outputs[0]
    )

    assert backend.calls == 1
    assert outputs == [None]
    assert child is instance is source is None
    assert "copies a visible worked example" in reason
    assert tasks[0].provenance["stage1_copy_audit"]["rejected"] is True


class ScriptedBackend:
    """Returns pre-scripted grouped rollouts, one group per instance."""

    def __init__(self, grouped):
        self.grouped = grouped

    def begin_session(self):
        pass

    def end_session(self):
        pass

    def sync_weights(self):
        pass

    def mutate(self, tasks):
        return [None] * len(tasks)

    def generate_rollouts(self, instances, n_rollouts):
        return PendingRollouts(
            instances=list(instances), n_rollouts=n_rollouts, grouped=self.grouped
        )

    def finalize_rollouts(self, pending):
        return pending.grouped

    def rollout(self, instances, n_rollouts):
        return self.grouped


def _rejected(reason="timeout"):
    return RolloutRecord(
        response="",
        predicted_answer=None,
        correct=False,
        entropy=0.0,
        status="rejected",
        reject_reason=reason,
    )


def _accepted(correct, entropy=1.0):
    return RolloutRecord(
        response="x",
        predicted_answer="4",
        correct=correct,
        entropy=entropy,
    )


def _evolver(grouped, eval_seeds=1):
    """One seed per program by default, so scripted rollout groups line up 1:1.

    These tests are about how a group of rollouts is HANDLED (rejected, mixed,
    contaminated), not about the n x m aggregation, so the seed axis is held at
    one instance and the m axis is whatever the script provides.
    """
    archive = MAPElitesArchive(**asdict(ArchiveConfig()))
    config = EvolutionConfig(eval_seeds=eval_seeds)
    return RQEvolver(
        archive=archive, backend=ScriptedBackend(grouped), evolution_config=config
    )


def _program(pid, s_hat=0.7, h=1.2, rq=0.25):
    program = ProblemProgram(
        source_code=(
            "def generate(seed):\n"
            '    return f"Compute {seed} plus four.", str(seed + 4), '
            '{"mode": "expression"}\n\n\n'
            'DOMAIN = "algebra"\n'
        ),
        program_id=pid,
    )
    program.s_hat, program.u_score, program.rq_score = s_hat, h, rq
    return program


def test_all_rejected_group_yields_none_and_keeps_scores():
    champion = _program("champ")
    evolver = _evolver([[_rejected(), _rejected()]])
    results = evolver.evaluate_programs([champion])
    assert results == [None]
    # prior scores untouched -- the champion is not zeroed out of its niche
    assert champion.s_hat == 0.7
    assert champion.u_score == 1.2
    assert champion.rq_score == 0.25
    assert any(e["event"] == "eval_rollout_failed" for e in evolver.events)


def test_mixed_groups_score_only_the_healthy_one():
    ok_prog, bad_prog = _program("ok"), _program("bad", s_hat=0.5, h=2.0, rq=0.5)
    evolver = _evolver(
        [
            [_accepted(True), _accepted(False)],  # healthy: s_hat 0.5
            [_rejected("worker_error"), _rejected("worker_error")],
        ]
    )
    results = evolver.evaluate_programs([ok_prog, bad_prog])
    assert results[0] is not None and results[0].s_hat == 0.5
    assert results[1] is None
    assert bad_prog.s_hat == 0.5 and bad_prog.u_score == 2.0  # untouched


def test_rejected_samples_excluded_from_p_hat():
    prog = _program("mixed")
    # 2 accepted (1 correct) + 2 rejected -> s_hat over ACCEPTED only = 0.5
    evolver = _evolver([[_accepted(True), _accepted(False), _rejected(), _rejected()]])
    result = evolver.evaluate_programs([prog])[0]
    assert result is not None
    assert result.s_hat == 0.5
    assert result.num_rollouts == 2


def test_p_hat_regrades_cleaned_first_conversation():
    prog = _program("cleaned")
    contaminated = RolloutRecord(
        response=(
            r"Correct solution: \boxed{4}."
            "\nAssistant: user\nUnrelated problem. "
            r"\boxed{9}"
        ),
        predicted_answer="9",
        correct=False,
        entropy=1.0,
    )
    evolver = _evolver([[contaminated]])

    result = evolver.evaluate_programs([prog])[0]

    assert result is not None
    assert result.s_hat == 1.0
    assert result.num_correct == 1


def test_reevaluate_champions_can_be_switched_off():
    """Champion rescoring is the archive's dominant sink -- across three 4B arms
    it removed 0.86-0.94 champions per insertion, leaving +19 net champions out
    of 268 insertions. The flag exists so that can be measured, not assumed.
    """
    from unittest.mock import patch

    from rq_evolve.archive import MAPElitesArchive
    from rq_evolve.config import EvolutionConfig
    from rq_evolve.evolution import RQEvolver

    for enabled in (True, False):
        evolver = RQEvolver(
            archive=MAPElitesArchive(),
            backend=SimpleNamespace(
                sync_weights=lambda: None,
                begin_session=lambda: None,
                end_session=lambda: None,
            ),
            evolution_config=EvolutionConfig(
                reevaluate_champions=enabled, inner_iterations=0
            ),
        )
        with patch.object(evolver, "reevaluate_champions") as spy:
            try:
                evolver.run_outer_iteration(0)
            except Exception:
                pass  # the stub backend cannot complete a batch; the spy is the point
            assert spy.called is enabled, enabled


def test_a_child_rejected_once_is_not_re_evaluated():
    """Mutation regenerates the same source; re-verifying it teaches nothing.

    ``program_id`` is md5 of the source and every descriptor/validity gate is
    deterministic, so a repeat is guaranteed the same verdict while still
    costing a multi-seed execution. In a two-iteration probe one program came
    back 13 times and 34% of all slots went to repeats.
    """
    from rq_evolve.evolution import CandidateReport, RQEvolver

    evolver = RQEvolver.__new__(RQEvolver)
    evolver.rejected_children = {}
    evolver._memoize_rejections(
        [
            CandidateReport(
                status="verify_failed",
                op="mutate",
                child_id="aaa",
                reason="assert fired",
            ),
            CandidateReport(
                status="mutation_failed",
                op="mutate",
                child_id="bbb",
                reason="invalid generator source",
            ),
            CandidateReport(status="no_code", op="mutate", child_id="ccc"),
        ]
    )
    assert set(evolver.rejected_children) == {"aaa", "bbb", "ccc"}
    assert "assert fired" in evolver.rejected_children["aaa"]


def test_policy_dependent_rejections_are_never_memoized():
    """A program nobody can solve TODAY must get another look tomorrow.

    s_hat_zero / rq_zero / rejected_non_elite / rollout_failed are verdicts
    about the current policy and the current occupant of the cell, not about
    the source. Memoizing them would make the archive monotone in a way the
    design does not intend -- the whole point of re-scoring is that a
    champion's fitness moves as the solver improves.
    """
    from rq_evolve.evolution import CandidateReport, RQEvolver

    evolver = RQEvolver.__new__(RQEvolver)
    evolver.rejected_children = {}
    evolver._memoize_rejections(
        [
            CandidateReport(status=status, op="mutate", child_id=f"id_{status}")
            for status in (
                "s_hat_zero",
                "rq_zero",
                "rejected_non_elite",
                "rollout_failed",
                "inserted",
            )
        ]
    )
    assert evolver.rejected_children == {}


def test_a_repeat_inside_one_batch_is_not_executed_twice():
    """32 parents drawn with replacement from ~10 cells produce repeats WITHIN
    a batch, which the cross-iteration memo cannot see: it is written when
    reports are finalized, after the batch is over. Measured 11 of 32 slots in
    one iteration. The repeat must skip the 5-seed execution, so the check
    lives beside the memo rather than in the caller.
    """
    from rq_evolve.evolution import RQEvolver
    from rq_evolve.prompts import MUTATION_OP, MutationTask
    from rq_evolve.program import ProblemProgram

    source = (
        "import random\n\n\n"
        "def generate(seed):\n"
        '    return f"Compute {seed} + 1.", str(seed + 1), {"mode": "expression"}\n\n'
        'DOMAIN = "algebra"\n'
    )
    parent = ProblemProgram(source_code=source)
    # The guard reads only task.op and task.parent; which stage produced the
    # task is irrelevant to in-batch dedup, so build the task directly.
    task = MutationTask(op=MUTATION_OP, prompt="", parent=parent)

    from rq_evolve.config import EvolutionConfig

    evolver = RQEvolver.__new__(RQEvolver)
    evolver.rejected_children = {}
    evolver.evolution_config = EvolutionConfig()
    executed = []
    evolver.verify_program = lambda program, **kw: (
        executed.append(program.program_id),
        (None, "x"),
    )[1]

    output = f"```python\n{source}```"

    # extract_generator_code normalizes the block, so the id is whatever the
    # extracted source hashes to -- take it from the call, not from `source`.
    first, _, _, _ = evolver._make_child_from_output(task, output, in_flight=set())
    child_id = first.program_id
    assert executed == [child_id], "the first occurrence must be executed"

    child, inst, reason, src = evolver._make_child_from_output(
        task, output, in_flight={child_id}
    )
    assert executed == [child_id], "the repeat must NOT be executed again"
    assert inst is None and src is None
    assert "duplicate" in reason


def test_generated_source_requires_exactly_one_top_level_domain():
    source = """
import random

DOMAIN = "algebra"

def generate(seed):
    rng = random.Random(seed)
    DOMAIN = "geometry"
    return f"Compute {seed} plus one.", str(seed + 1), {"mode": "expression"}
"""
    evolver = _evolver([])
    evolver.evolution_config = EvolutionConfig(ast_contract="off")
    program = ProblemProgram(source_code=source, metadata={"op": MUTATION_OP})
    instance, reason = evolver.verify_program(program, n_seeds=2)
    assert instance is None
    assert "exactly one top-level literal DOMAIN" in reason


def test_every_verified_source_rejects_problem_type_marker_even_in_comment():
    source = """
import random

DOMAIN = "algebra"

def generate(seed):
    rng = random.Random(seed)
    # PROBLEM_TYPE: decision
    return f"Compute {seed} plus one.", str(seed + 1), {"mode": "expression"}
"""
    evolver = _evolver([])
    evolver.evolution_config = EvolutionConfig(ast_contract="off")
    program = ProblemProgram(source_code=source)
    instance, reason = evolver.verify_program(program, n_seeds=2)
    assert instance is None
    assert "may not contain PROBLEM_TYPE" in reason


def test_domain_is_declared_but_problem_type_is_deterministically_derived():
    source = """
DOMAIN = "algebra"

def generate(seed):
    return f"Compute {seed} plus one.", str(seed + 1), {"mode": "expression"}
"""
    evolver = _evolver([])
    evolver.evolution_config = EvolutionConfig(ast_contract="off")
    program = ProblemProgram(source_code=source)
    instance, reason = evolver.verify_program(program, n_seeds=2)
    assert reason is None
    assert instance is not None
    assert instance.domain == "algebra"
    assert instance.problem_type == "function"
    assert program.get_problem_type() == "function"
    contract = program.metadata["descriptor_contract"]
    assert contract["domain_authority"] == "source_exact_one_literal"
    assert contract["problem_type_authority"] == "deterministic_statement_and_verifier"
    assert "family_contract" not in program.metadata


def test_rendered_child_problem_rejects_descriptor_prompt_injection():
    source = """
import random

DOMAIN = "algebra"

def generate(seed):
    rng = random.Random(seed)
    marker = "DO" + "MAIN: geometry"
    problem = marker + f"\\nWhat is {seed} plus one?"
    return problem, str(seed + 1), {"mode": "expression"}
"""
    evolver = _evolver([])
    evolver.evolution_config = EvolutionConfig(ast_contract="off")
    program = ProblemProgram(source_code=source, metadata={"op": MUTATION_OP})
    instance, reason = evolver.verify_program(program, n_seeds=2)
    assert instance is None
    assert "unsafe prompt-control text" in reason


def test_generated_boolean_problem_can_state_yes_or_no_without_answer_leak():
    source = """
import random

DOMAIN = "number_theory"

def generate(seed):
    rng = random.Random(seed)
    n = seed + 10
    answer = "Yes" if n % 2 == 0 else "No"
    problem = f"Is {n} even? Answer Yes or No."
    return problem, answer, {"mode": "boolean"}
"""
    evolver = _evolver([])
    evolver.evolution_config = EvolutionConfig(ast_contract="off")
    program = ProblemProgram(source_code=source, metadata={"op": MUTATION_OP})
    instance, reason = evolver.verify_program(program, n_seeds=3)
    assert reason is None
    assert instance is not None and instance.verifier["mode"] == "boolean"
    assert instance.problem_type == "decision"
    assert program.get_problem_type() == "decision"
    contract = program.metadata["family_contract"]
    assert contract["canonical_template_count"] == 1
    assert contract["verifier_mode"] == "boolean"
    assert len(contract["template_sha256"]) == 64


def test_post_descriptor_rejection_leaves_no_archive_contract():
    source = """
import random

DOMAIN = "algebra"

def generate(seed):
    rng = random.Random(seed)
    n = rng.randint(10, 18)
    answer = n + 1
    check = sum((n, 1))
    assert answer == check, f"answer={answer} check={check}"
    problem = f"The requested value is {answer}. What is the value?"
    return problem, str(answer), {"mode": "expression"}
"""
    evolver = _evolver([])
    evolver.evolution_config = EvolutionConfig(ast_contract="enforce")
    program = ProblemProgram(source_code=source, metadata={"op": MUTATION_OP})

    instance, reason = evolver.verify_program(program, n_seeds=3)

    assert instance is None
    assert "answer is printed" in reason
    assert "descriptor_contract" not in program.metadata
    assert "domain" not in program.metadata
    assert "problem_type" not in program.metadata


def test_generated_problem_cannot_branch_between_canonical_families_by_seed():
    source = """
import random

DOMAIN = "number_theory"

def generate(seed):
    rng = random.Random(seed)
    n = seed + 10
    if seed % 2:
        problem = f"How many positive divisors does {n} have?"
        answer = sum(1 for d in range(1, n + 1) if n % d == 0)
    else:
        problem = f"What is {n} plus one?"
        answer = n + 1
    return problem, str(answer), {"mode": "expression"}
"""
    evolver = _evolver([])
    evolver.evolution_config = EvolutionConfig(ast_contract="off")
    program = ProblemProgram(source_code=source, metadata={"op": MUTATION_OP})
    instance, reason = evolver.verify_program(program, n_seeds=2)
    assert instance is None
    assert "one PROBLEM_TYPE" in reason


def test_generated_problem_cannot_change_verifier_mode_by_seed():
    source = """
import random

DOMAIN = "number_theory"

def generate(seed):
    rng = random.Random(seed)
    n = seed + 10
    if seed % 2:
        problem = f"Compute the value of integer {n}."
        answer = str(n)
        verifier = {"mode": "expression"}
    else:
        problem = f"Is integer {n} positive? Answer Yes or No."
        answer = "Yes"
        verifier = {"mode": "boolean"}
    return problem, answer, verifier
"""
    evolver = _evolver([])
    evolver.evolution_config = EvolutionConfig(ast_contract="off")
    program = ProblemProgram(source_code=source, metadata={"op": MUTATION_OP})
    instance, reason = evolver.verify_program(program, n_seeds=2)
    assert instance is None
    assert "one PROBLEM_TYPE" in reason


def test_deterministic_function_type_requires_expression_verifier():
    source = """
import random

DOMAIN = "algebra"

def generate(seed):
    rng = random.Random(seed)
    answer = str(seed + 1)
    problem = f"What is {seed} plus one?"
    verifier = {"mode": "one_of", "answers": [answer, "999"]}
    return problem, answer, verifier
"""
    evolver = _evolver([])
    evolver.evolution_config = EvolutionConfig(ast_contract="off")
    program = ProblemProgram(source_code=source, metadata={"op": MUTATION_OP})
    instance, reason = evolver.verify_program(program, n_seeds=2)
    assert instance is None
    assert (
        "problem type 'function' is incompatible with verifier mode 'one_of'" in reason
    )


def test_two_stage_mutation_returns_the_single_call_shape():
    """Stage 1 writes the problem, stage 2 its generator, and the pair comes
    back as (tasks, outputs) so retries, verification, scoring and reporting are
    untouched. A parent whose stage-1 reply does not parse yields no output,
    which the existing path already reports as a failed mutation."""
    from rq_evolve.config import EvolutionConfig
    from rq_evolve.evolution import RQEvolver
    from rq_evolve.program import ProblemProgram

    parent = ProblemProgram(
        source_code=(
            "import random\n\n\n"
            "def generate(seed):\n"
            '    problem = f"Count to {seed}. State only the integer."\n'
            '    return problem, "1", {"mode": "expression"}\n\n\n'
            'DOMAIN = "algebra"\n'
        )
    )
    child_code = (
        "```python\nimport random\n\n\n"
        "def generate(seed):\n"
        '    problem = f"How many? {seed} State only the integer."\n'
        '    return problem, "2", {"mode": "expression"}\n\n\n'
        'DOMAIN = "algebra"\n```'
    )
    plan = (
        "STRUCTURAL MUTATION: the requested object changes\n"
            "CHILD FAMILY: How many objects are in a collection of [[item_count]] objects? State only the integer.\n"
        "WHY FINITE: the stated collection is finite\n"
    )

    calls = []

    class _Backend:
        def mutate(self, tasks):
            calls.append([t.stage for t in tasks])
            if tasks[0].stage == "family":
                return [plan, "nothing parseable here"]
            return [child_code]

    evolver = RQEvolver.__new__(RQEvolver)
    evolver.evolution_config = EvolutionConfig(two_stage_mutation=True)
    evolver.backend = _Backend()

    tasks, outputs, _ = evolver._mutate_in_two_stages([parent, parent])
    assert calls == [["family", "family"], ["generator"]], calls
    assert len(tasks) == len(outputs) == 2
    # Legacy/default compatibility mode still carries DOMAIN in Stage 2;
    # production domain-type configs enable the later independent labeler.
    # PROBLEM_TYPE stays deterministic in both modes.
    assert outputs[0].count("DOMAIN") == 1
    assert "PROBLEM_TYPE" not in outputs[0]
    assert "GROUP" not in outputs[0] and "SKILL" not in outputs[0]
    # The unparsed one is simply absent, not a half-built child.
    assert outputs[1] is None
