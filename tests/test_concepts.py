import pytest

from rq_evolve.concepts import (
    GROUPS,
    SKILLS,
    axis_index,
    axis_values,
    validate_label_decl,
)


def test_valid_declaration_has_no_errors():
    assert validate_label_decl("number_theory", "counting") == []


def test_both_labels_are_required():
    assert validate_label_decl(None, "counting") == ["missing GROUP"]
    assert validate_label_decl("number_theory", None) == ["missing SKILL"]
    assert validate_label_decl(None, None) == ["missing GROUP", "missing SKILL"]


def test_each_label_is_checked_against_its_own_vocabulary():
    errors = validate_label_decl("counting", "number_theory")
    assert len(errors) == 2
    assert "unknown GROUP: 'counting'" in errors[0]
    assert "unknown SKILL: 'number_theory'" in errors[1]


def test_axes_are_independent_so_any_pair_is_accepted():
    """No cross-label rule: the grid is a product, not a partition of GROUP."""
    for group in GROUPS:
        for skill in SKILLS:
            assert validate_label_decl(group, skill) == [], (group, skill)


def test_the_retired_prefix_rule_is_gone():
    """``geometry.trig_area`` was a valid CONCEPT_TYPE; it is not a SKILL."""
    errors = validate_label_decl("geometry", "geometry.trig_area")
    assert len(errors) == 1
    assert "unknown SKILL" in errors[0]


def test_axis_index_locates_a_label_and_reports_unknown_as_none():
    assert axis_index("group", "number_theory") == 0
    assert axis_index("skill", "casework") == 0
    assert axis_index("skill", "counting") == SKILLS.index("counting")
    # Unknown must not silently collapse into bin 0.
    assert axis_index("group", "topology") is None
    assert axis_index("skill", None) is None


def test_unknown_axis_is_an_error_not_an_empty_vocabulary():
    with pytest.raises(ValueError, match="unknown axis"):
        axis_values("concept_type")


def test_grid_shape():
    assert len(GROUPS) == 6
    assert len(SKILLS) == 8
    assert set(GROUPS).isdisjoint(SKILLS)
