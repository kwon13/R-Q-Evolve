import pytest

from rq_evolve.concepts import (
    DOMAINS,
    GROUPS,
    PROBLEM_TYPES,
    SKILLS,
    axis_index,
    axis_values,
    validate_label_decl,
)


def test_valid_declaration_has_no_errors():
    assert validate_label_decl("number_theory", "counting") == []


def test_both_labels_are_required():
    assert validate_label_decl(None, "counting") == ["missing DOMAIN"]
    assert validate_label_decl("number_theory", None) == ["missing PROBLEM_TYPE"]
    assert validate_label_decl(None, None) == [
        "missing DOMAIN",
        "missing PROBLEM_TYPE",
    ]


def test_each_label_is_checked_against_its_own_vocabulary():
    errors = validate_label_decl("counting", "number_theory")
    assert len(errors) == 2
    assert "unknown DOMAIN: 'counting'" in errors[0]
    assert "unknown PROBLEM_TYPE: 'number_theory'" in errors[1]


def test_axes_are_the_complete_cartesian_product():
    for domain in DOMAINS:
        for problem_type in PROBLEM_TYPES:
            assert validate_label_decl(domain, problem_type) == []


def test_no_hierarchical_or_retired_label_is_accepted():
    errors = validate_label_decl("geometry", "geometry.plane_geometry")
    assert len(errors) == 1
    assert "unknown PROBLEM_TYPE" in errors[0]


def test_axis_index_locates_a_label_and_reports_unknown_as_none():
    assert axis_index("domain", "algebra") == 0
    assert axis_index("domain", "number_theory") == DOMAINS.index("number_theory")
    assert axis_index("problem_type", "decision") == 0
    assert axis_index("problem_type", "counting") == PROBLEM_TYPES.index("counting")
    assert axis_index("domain", "topology") is None
    assert axis_index("problem_type", None) is None


def test_retired_axis_names_are_not_valid_axes():
    with pytest.raises(ValueError, match="unknown axis"):
        axis_values("group")
    with pytest.raises(ValueError, match="unknown axis"):
        axis_values("skill")


def test_grid_shape_and_vocabularies():
    assert len(DOMAINS) == 7
    assert len(PROBLEM_TYPES) == 5
    assert set(DOMAINS).isdisjoint(PROBLEM_TYPES)


def test_temporary_import_aliases_do_not_create_a_second_vocabulary():
    assert GROUPS is DOMAINS
    assert SKILLS is PROBLEM_TYPES
