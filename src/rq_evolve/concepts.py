"""Label vocabulary for the two MAP-Elites axes.

A program declares exactly two labels at module top level::

    GROUP = "number_theory"
    SKILL = "counting"

The two axes are deliberately independent. One GROUP holds many SKILLs (the
same domain reached by different reasoning), and one SKILL spans many GROUPs
(the same reasoning move transplanted across domains). That independence is
what makes a GROUP x SKILL grid fillable: a mutation can hold the domain and
move the reasoning, or hold the reasoning and move the domain.

This replaces the old ``CONCEPT_GROUP`` / ``CONCEPT_TYPE`` pair, where the type
was required to carry the group as a literal prefix (``number_theory.crt_count``).
That made the second label a refinement of the first rather than an independent
axis, so the grid could only ever be a partition of the group axis.
"""

# Axis 1 -- mathematical domain.
GROUPS: tuple[str, ...] = (
    "number_theory",
    "combinatorics",
    "sequence",
    "algebra",
    "geometry",
    "inequality",
)

# Axis 2 -- reasoning skill the visible problem demands.
SKILLS: tuple[str, ...] = (
    "casework",
    "induction",
    "contradiction",
    "invariant",
    "extremal_principle",
    "counting",
    "transformation",
    "construction",
)

AXES: tuple[str, ...] = ("group", "skill")

# Pre-migration names, still imported by archive.py and prompts.py. Both are
# aliases, not copies: there is exactly one vocabulary per axis.
CONCEPT_GROUPS = GROUPS
SKILL_GROUPS = SKILLS

def axis_values(axis: str) -> tuple[str, ...]:
    """Return the full ordered vocabulary of one axis."""
    if axis == "group":
        return GROUPS
    if axis == "skill":
        return SKILLS
    raise ValueError(f"unknown axis: {axis!r} (expected one of {AXES})")


def axis_index(axis: str, value: str | None) -> int | None:
    """Return the grid coordinate of ``value`` on ``axis``, or None if unknown.

    None is a real answer, not an error: it means the label is outside the
    vocabulary and the caller must decide whether to reject the program or bin
    it elsewhere. Silently folding an unknown label into bin 0 would let every
    mislabelled program compete in one cell.
    """
    values = axis_values(axis)
    if value in values:
        return values.index(value)
    return None


def axis_labels(axis: str) -> list[str]:
    """Display labels for one axis; empty for an axis with no fixed vocabulary."""
    if axis in AXES:
        return list(axis_values(axis))
    return []


def validate_label_decl(
    group: str | None,
    skill: str | None,
) -> list[str]:
    """Return validation errors for a program's GROUP / SKILL declaration.

    Both labels are required and each must come from its own closed vocabulary.
    There is no cross-label constraint: any GROUP may pair with any SKILL, which
    is precisely what lets the two axes be independent.
    """
    reasons: list[str] = []

    if not group:
        reasons.append("missing GROUP")
    elif group not in GROUPS:
        reasons.append(
            f"unknown GROUP: {group!r} (expected one of {', '.join(GROUPS)})"
        )

    if not skill:
        reasons.append("missing SKILL")
    elif skill not in SKILLS:
        reasons.append(
            f"unknown SKILL: {skill!r} (expected one of {', '.join(SKILLS)})"
        )

    return reasons
