import pytest

from rq_evolve.dataset import DynamicProblemDataset, VerlDynamicDataset
from rq_evolve.program import ProblemProgram
from rq_evolve.reward import answers_match, compute_score
from rq_evolve.verifier import (
    MAX_VERIFIER_ITEMS,
    canonical_boolean,
    normalize_verifier,
)


def test_legacy_generator_defaults_to_expression_verifier():
    program = ProblemProgram(
        'def generate(seed):\n    return "Compute one plus one.", "2"\n'
    )
    instance = program.execute(0)
    assert instance is not None
    assert instance.verifier == {"mode": "expression"}


def test_generator_propagates_declarative_verifier_and_axes():
    program = ProblemProgram(
        source_code=r'''
DOMAIN = "algebra"

def generate(seed):
    return "Is zero even?", "Yes", {"mode": "boolean"}
''',
        metadata={"domain": "number_theory", "problem_type": "decision"},
    )
    instance = program.execute(0)
    assert instance is not None
    assert instance.verifier == {"mode": "boolean"}
    # DOMAIN is source-authoritative; the deterministic type cache is metadata.
    assert program.declared_domain() == "algebra"
    assert program.get_domain() == "algebra"
    assert instance.domain == "algebra"
    assert instance.problem_type == "decision"


def test_generator_rejects_executable_or_malformed_verifier():
    program = ProblemProgram(
        source_code='''
def generate(seed):
    return "Choose anything.", "1", {"mode": "predicate", "fn": lambda x: True}
'''
    )
    assert program.execute(0) is None
    assert "ValueError" in str(program.last_execution_error)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Yes", True),
        (r"\text{NO}", False),
        (1, True),
        (0, False),
        ("tr ue", None),
        ("y.e.s", None),
    ],
)
def test_boolean_normalization_is_conservative(value, expected):
    assert canonical_boolean(value) is expected


def test_boolean_verifier_accepts_canonical_aliases_only():
    spec = {"mode": "boolean"}
    assert answers_match(r"\text{Yes}", "1", spec)
    assert answers_match("false", "No", spec)
    assert not answers_match("probably", "Yes", spec)


def test_one_of_uses_expression_equivalence_within_one_worker_request():
    spec = {"mode": "one_of", "answers": ["1/2", "2"]}
    assert answers_match("0.5", "1/2", spec)
    assert answers_match("2", "1/2", spec)
    assert not answers_match("3", "1/2", spec)


def test_set_is_order_independent_complete_and_symbolic():
    spec = {"mode": "set", "elements": ["1/2", "2"]}
    gold = r"\{1/2,2\}"
    assert answers_match(r"\{2,0.5\}", gold, spec)
    assert not answers_match(r"\{2\}", gold, spec)
    assert not answers_match(r"\{2,0.5,3\}", gold, spec)


def test_set_rejects_semantic_duplicates_in_contract_or_prediction():
    duplicate_contract = {"mode": "set", "elements": ["1", "2-1"]}
    assert not answers_match(r"\{1,2-1\}", r"\{1,2-1\}", duplicate_contract)
    ordinary = {"mode": "set", "elements": ["1", "2"]}
    assert not answers_match(r"\{1,2-1\}", r"\{1,2\}", ordinary)


def test_empty_set_contract():
    spec = {"mode": "set", "elements": []}
    assert answers_match(r"\emptyset", r"\emptyset", spec)
    assert not answers_match(r"\{0\}", r"\emptyset", spec)


def test_compute_score_dispatches_from_extra_info_and_batch():
    one = compute_score(
        data_source="rq_evolved",
        solution_str=r"answer: \boxed{No}",
        ground_truth="0",
        extra_info={"verifier": {"mode": "boolean"}},
    )
    assert one["accuracy"] == 1.0

    batch = compute_score(
        response_str_list=[r"\boxed{2}", r"\boxed{No}"],
        ground_truth_list=["1", "Yes"],
        extra_infos=[
            {"verifier": {"mode": "one_of", "answers": ["1", "2"]}},
            {"verifier": {"mode": "boolean"}},
        ],
    )
    assert [row["accuracy"] for row in batch] == [1.0, 0.0]


def test_dynamic_dataset_carries_verifier_to_both_verl_reward_channels():
    contract = {"mode": "one_of", "answers": ["1", "2"]}
    dynamic = DynamicProblemDataset(
        [{"problem": "Choose a valid value.", "answer": "1", "verifier": contract}]
    )
    row = VerlDynamicDataset(dynamic, tokenizer=object())[0]
    assert row["reward_model"]["ground_truth"] == "1"
    assert row["reward_model"]["verifier"] == contract
    assert row["extra_info"]["verifier"] == contract


def test_verifier_schema_fails_closed_and_is_bounded():
    with pytest.raises(ValueError, match="unknown verifier mode"):
        normalize_verifier({"mode": "predicate", "code": "return True"})
    with pytest.raises(ValueError, match="unknown verifier field"):
        normalize_verifier({"mode": "expression", "code": "return True"})
    with pytest.raises(ValueError, match="include the reference"):
        normalize_verifier(
            {"mode": "one_of", "answers": ["2"]}, answer="1"
        )
    with pytest.raises(ValueError, match="must render exactly"):
        normalize_verifier(
            {"mode": "set", "elements": ["1", "2"]}, answer=r"\{1,3\}"
        )
    with pytest.raises(ValueError, match="must contain <="):
        normalize_verifier(
            {
                "mode": "one_of",
                "answers": [str(i) for i in range(MAX_VERIFIER_ITEMS + 1)],
            }
        )


def test_problem_program_round_trip_preserves_canonical_niches():
    program = ProblemProgram(
        source_code='DOMAIN="geometry"\n'
        'def generate(seed):\n    return "Find x.", "2"\n',
        niche_domain=1,
        niche_problem_type=1,
        metadata={"domain": "geometry", "problem_type": "search"},
    )
    restored = ProblemProgram.from_dict(program.to_dict())
    assert restored.niche_domain == 1
    assert restored.niche_problem_type == 1
    assert restored.get_domain() == "geometry"
    assert restored.get_problem_type() == "search"
