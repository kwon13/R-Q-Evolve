from rq_evolve.concepts import validate_concept_decl


def test_concept_type_prefix_must_match_declared_group():
    assert validate_concept_decl(
        "algebra.linear_system",
        "algebra",
    ) == []
    errors = validate_concept_decl(
        "algebra.linear_system",
        "sequence",
    )
    assert any("prefix must match" in reason for reason in errors)
