"""Closed vocabularies for the two MAP-Elites descriptors.

Every accepted generator receives exactly one value on each independent axis::

    DOMAIN = "number_theory"
    # PROBLEM_TYPE is deterministically inferred as "counting"

Generated ``DOMAIN`` is assigned from the fixed family through seven
label-blind binary checks; Stage 2 never emits it. Hand-authored bootstrap
seeds retain a file-pinned source declaration. ``PROBLEM_TYPE`` is inferred from
the output contract requested by the visible problem and checked against its
verifier on every verification seed. Neither axis refines the other, and the
runtime archive is the complete Cartesian product: all 35 domain/type pairs
exist, with no mask or frequency threshold.
"""

# Axis 1 -- top-level Omni-MATH mathematical domain.
DOMAINS: tuple[str, ...] = (
    "algebra",
    "geometry",
    "number_theory",
    "discrete_mathematics",
    "applied_mathematics",
    "calculus",
    "precalculus",
)

# Axis 2 -- computational problem type requested by the statement.
PROBLEM_TYPES: tuple[str, ...] = (
    "decision",
    "search",
    "counting",
    "optimization",
    "function",
)

AXES: tuple[str, ...] = ("domain", "problem_type")

# Temporary import aliases for callers migrating in the same release. They do
# not revive the retired axis names: ``axis_values('group')`` and old snapshot
# metadata remain invalid, and no serialized schema writes GROUP/SKILL fields.
GROUPS = DOMAINS
SKILLS = PROBLEM_TYPES


def axis_values(axis: str) -> tuple[str, ...]:
    """Return the complete ordered vocabulary for one descriptor axis."""
    if axis == "domain":
        return DOMAINS
    if axis == "problem_type":
        return PROBLEM_TYPES
    raise ValueError(f"unknown axis: {axis!r} (expected one of {AXES})")


def axis_index(axis: str, value: str | None) -> int | None:
    """Return ``value``'s coordinate, or ``None`` for an unknown label.

    Unknown labels are never folded into a fallback bin. A generator without
    one unambiguous in-vocabulary label on both axes has no archive cell.
    """
    values = axis_values(axis)
    if value in values:
        return values.index(value)
    return None


def axis_labels(axis: str) -> list[str]:
    """Return display labels for a known axis, otherwise an empty list."""
    if axis in AXES:
        return list(axis_values(axis))
    return []


def validate_label_decl(
    domain: str | None,
    problem_type: str | None,
) -> list[str]:
    """Return validation errors for one DOMAIN / PROBLEM_TYPE assignment.

    Both labels are required and each must come from its own closed vocabulary.
    There is deliberately no cross-label rule: every one of the 7 x 5 pairs is
    a real runtime cell, including pairs rare or absent in Omni-MATH.
    """
    reasons: list[str] = []

    if not domain:
        reasons.append("missing DOMAIN")
    elif domain not in DOMAINS:
        reasons.append(
            f"unknown DOMAIN: {domain!r} (expected one of {', '.join(DOMAINS)})"
        )

    if not problem_type:
        reasons.append("missing PROBLEM_TYPE")
    elif problem_type not in PROBLEM_TYPES:
        reasons.append(
            "unknown PROBLEM_TYPE: "
            f"{problem_type!r} (expected one of {', '.join(PROBLEM_TYPES)})"
        )

    return reasons
