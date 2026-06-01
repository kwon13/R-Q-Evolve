"""Controlled diversity labels used by the MAP-Elites D-axis."""

from __future__ import annotations

CONCEPT_GROUPS: tuple[str, ...] = (
    "number_theory",
    "combinatorics",
    "sequence",
    "algebra",
    "geometry",
    "inequality",
)

DEFAULT_CONCEPT_TYPES: tuple[str, ...] = (
    "number_theory.gcd_lcm",
    "number_theory.modular_arithmetic",
    "combinatorics.counting",
    "sequence.recurrence",
    "algebra.quadratic",
    "geometry.euclidean",
    "inequality.am_gm",
)


def concept_group_for_type(concept_type: str | None) -> str | None:
    """Infer the coarse group from a ``group.name`` concept type."""
    if not concept_type or "." not in concept_type:
        return None
    group = concept_type.split(".", 1)[0]
    return group if group in CONCEPT_GROUPS else None


def validate_concept_decl(
    concept_type: str | None,
    concept_group: str | None,
) -> list[str]:
    """Return validation errors for a generated program's concept labels."""
    reasons: list[str] = []
    if not concept_type:
        reasons.append("missing CONCEPT_TYPE")
    if not concept_group:
        reasons.append("missing CONCEPT_GROUP")
    elif concept_group not in CONCEPT_GROUPS:
        reasons.append(f"unknown CONCEPT_GROUP: {concept_group}")
    return reasons


def axis_labels(diversity_axis: str) -> list[str]:
    if diversity_axis == "concept_group":
        return list(CONCEPT_GROUPS)
    if diversity_axis == "concept_type":
        return list(DEFAULT_CONCEPT_TYPES)
    return []

