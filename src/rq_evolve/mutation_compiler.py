"""Compile structured mutation plans into verified problem programs.

The language model chooses a small, typed mathematical specification.  A
registered family owns the repetitive Python program shape, concept labels,
renderer, answer oracle, and executable mathematical guards.  This keeps
mechanical code-format requirements out of the model's creative mutation step.

Only registered families are compiled.  An unknown or ``free_form`` family
returns an explicit :class:`CompilationStatus.UNSUPPORTED` result so callers can
route it to a quarantined exploratory path instead of silently treating it as a
verified program.
"""

from __future__ import annotations

import hashlib
import json
import math
import textwrap
from dataclasses import asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol, TypeAlias

import sympy

from .code_utils import (
    lint_generator_source,
    lint_metacognitive_generator_source,
)
from .program import ProblemProgram

MutationOperator: TypeAlias = Literal["in_depth", "in_breadth"]
MUTATION_FAMILY_REGISTRY_VERSION = 1


class CompilationStatus(str, Enum):
    COMPILED = "compiled"
    UNSUPPORTED = "unsupported"
    INVALID_SPEC = "invalid_spec"
    COMPILER_ERROR = "compiler_error"


@dataclass(frozen=True, slots=True)
class MutationSpec:
    """Family-neutral, JSON-compatible input to the compiler."""

    generator_family: str
    operator: MutationOperator
    family_config: Mapping[str, Any] = field(default_factory=dict)
    target_reasoning_move: str = ""
    target_concept_group: str | None = None
    target_concept_type: str | None = None
    family_variant: str | None = None


@dataclass(frozen=True, slots=True)
class LinearSystemAggregateConfig:
    """Sampling bounds for a rank-deficient integer linear system.

    ``aggregate_multiplier_max`` and ``combination_weight_max`` bound the
    row-space combination the solver has to discover.  The compiler pins the
    multiplier at two or more so the aggregate is never recoverable by simply
    adding the printed equations; see :func:`_linear_necessity`.
    """

    coefficient_min: int = -3
    coefficient_max: int = 3
    solution_min: int = -3
    solution_max: int = 3
    aggregate_multiplier_min: int = 2
    aggregate_multiplier_max: int = 4
    combination_weight_min: int = 1
    combination_weight_max: int = 3


@dataclass(frozen=True, slots=True)
class ModularLinearSystemAggregateConfig:
    """Configuration for a small, independently brute-forceable modulus."""

    modulus: int = 7
    multiplier_min: int = 2


FamilyConfig: TypeAlias = (
    LinearSystemAggregateConfig | ModularLinearSystemAggregateConfig
)


@dataclass(frozen=True, slots=True)
class InstanceValidation:
    """Result of a family-specific mathematical validation."""

    valid: bool
    reasons: tuple[str, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FamilyVariant:
    """One registry-owned specialization of a family's construction.

    Variants exist so a plan can make a *consequential* choice. Without them both
    experimental conditions copied the published defaults, compiled to the same
    source hash, and produced identical problems -- which makes "reasoning-informed
    planning beats plain planning" untestable no matter how good the evaluator is.

    Each variant names the solver failure mode it stresses, so a planner holding a
    real failure trace has a documented basis for choosing one, while a planner
    with no trace has none. The config override is registry-owned and still runs
    through the family's parser, correctness oracle, and necessity check.
    """

    variant_id: str
    targets_failure_mode: str
    config_override: Mapping[str, Any]

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "variant_id": self.variant_id,
            "targets_failure_mode": self.targets_failure_mode,
            "family_config": dict(self.config_override),
        }


@dataclass(frozen=True, slots=True)
class FamilyReasoningContract:
    """The reasoning move a registered family provably forces.

    The planner writes free-form narration, but a fixed compiler cannot honour
    an arbitrary claim.  The registry therefore owns the move, the witness that
    makes it necessary, and the bypasses the construction structurally excludes.
    Downstream necessity judgements are made against *this* contract, never
    against the plan's prose.
    """

    target_reasoning_move: str
    necessity_witness: str
    excluded_bypasses: tuple[str, ...]
    variants: tuple[FamilyVariant, ...] = ()
    default_variant_id: str = "balanced"

    def variant(self, variant_id: str | None) -> FamilyVariant | None:
        wanted = (variant_id or self.default_variant_id).strip()
        for candidate in self.variants:
            if candidate.variant_id == wanted:
                return candidate
        return None

    @property
    def variant_ids(self) -> tuple[str, ...]:
        return tuple(variant.variant_id for variant in self.variants)

    def to_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "target_reasoning_move": self.target_reasoning_move,
                "necessity_witness": self.necessity_witness,
                "excluded_bypasses": list(self.excluded_bypasses),
            }
        )


@dataclass(frozen=True, slots=True)
class FamilySemanticValidation:
    """Deterministic per-seed family check that runs before any LLM evaluator."""

    valid: bool
    generator_family: str
    seeds: tuple[int, ...] = ()
    per_seed: tuple[Mapping[str, Any], ...] = ()
    reasons: tuple[str, ...] = ()

    def to_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "valid": self.valid,
                "generator_family": self.generator_family,
                "seeds": list(self.seeds),
                "per_seed": [dict(entry) for entry in self.per_seed],
                "reasons": list(self.reasons),
            }
        )


@dataclass(frozen=True, slots=True)
class FamilyCatalogEntry:
    generator_family: str
    supported_operator: MutationOperator
    concept_group: str
    concept_type: str
    config_fields: tuple[str, ...]
    validator: str
    target_reasoning_move: str


@dataclass(frozen=True, slots=True)
class FamilyDescriptor:
    """Prompt-safe descriptor for the one registered route allowed by an op."""

    generator_family: str
    operator: MutationOperator
    concept_group: str
    concept_type: str
    config_schema: Mapping[str, Any]
    default_config: Mapping[str, Any]
    validator: str
    reasoning_contract: FamilyReasoningContract


@dataclass(frozen=True, slots=True)
class CompilationResult:
    status: CompilationStatus
    generator_family: str
    operator: str
    family_variant: str | None = None
    variant_selected_by_plan: bool = False
    variant_targets_failure_mode: str | None = None
    # Keys the plan tried to set that the registry-owned variant overrode.
    variant_overridden_plan_keys: tuple[str, ...] = ()
    source_code: str | None = None
    concept_group: str | None = None
    concept_type: str | None = None
    family_config: Mapping[str, Any] = field(default_factory=dict)
    source_hash: str | None = None
    reasons: tuple[str, ...] = ()
    reasoning_contract: FamilyReasoningContract | None = None

    @property
    def compiled(self) -> bool:
        return self.status is CompilationStatus.COMPILED

    def to_problem_program(
        self,
        *,
        parent_id: str = "",
        generation: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProblemProgram:
        """Materialize a compiled source as the existing runtime abstraction."""
        if not self.compiled or self.source_code is None:
            detail = "; ".join(self.reasons) or self.status.value
            raise ValueError(f"mutation was not compiled: {detail}")
        program_metadata = dict(metadata or {})
        program_metadata.update(
            {
                "op": self.operator,
                "generator_family": self.generator_family,
                "family_variant": self.family_variant,
                "variant_selected_by_plan": self.variant_selected_by_plan,
                "family_config": dict(self.family_config),
                "compiler_registry_version": MUTATION_FAMILY_REGISTRY_VERSION,
                "compiler_source_hash": self.source_hash,
                "concept_group": self.concept_group,
                "concept_type": self.concept_type,
                "family_reasoning_contract": (
                    dict(self.reasoning_contract.to_payload())
                    if self.reasoning_contract is not None
                    else None
                ),
            }
        )
        return ProblemProgram(
            source_code=self.source_code,
            parent_id=parent_id,
            generation=generation,
            metadata=program_metadata,
        )


class _ConfigParser(Protocol):
    def __call__(self, raw: Mapping[str, Any]) -> FamilyConfig: ...


@dataclass(frozen=True, slots=True)
class _FamilyDefinition:
    generator_family: str
    supported_operator: MutationOperator
    concept_group: str
    concept_type: str
    config_fields: tuple[str, ...]
    config_schema: Mapping[str, Any]
    default_config: Mapping[str, Any]
    validator_name: str
    parse_config: _ConfigParser
    render_source: Callable[[FamilyConfig], str]
    reasoning_contract: FamilyReasoningContract
    validate_instance: Callable[[Mapping[str, Any]], InstanceValidation]
    check_necessity: Callable[[Mapping[str, Any]], InstanceValidation]


class MutationFamilyRegistry:
    """Immutable-at-use registry of trusted mutation family compilers."""

    def __init__(self) -> None:
        self._families: dict[str, _FamilyDefinition] = {}

    def register(self, family: _FamilyDefinition) -> None:
        if family.generator_family in self._families:
            raise ValueError(
                f"duplicate mutation family: {family.generator_family}"
            )
        self._families[family.generator_family] = family

    def get(self, generator_family: str) -> _FamilyDefinition | None:
        return self._families.get(generator_family)

    def catalog(self) -> Mapping[str, Any]:
        """Return a stable, JSON-compatible catalog for prompts/manifests."""
        entries = tuple(
            asdict(
                FamilyCatalogEntry(
                    generator_family=family.generator_family,
                    supported_operator=family.supported_operator,
                    concept_group=family.concept_group,
                    concept_type=family.concept_type,
                    config_fields=family.config_fields,
                    validator=family.validator_name,
                    target_reasoning_move=(
                        family.reasoning_contract.target_reasoning_move
                    ),
                )
            )
            for family in sorted(
                self._families.values(),
                key=lambda item: item.generator_family,
            )
        )
        return MappingProxyType(
            {
                "version": MUTATION_FAMILY_REGISTRY_VERSION,
                "families": entries,
            }
        )

    def compile(
        self,
        spec: MutationSpec,
        *,
        parent: ProblemProgram | None = None,
    ) -> CompilationResult:
        family = self.get(spec.generator_family)
        if family is None:
            return CompilationResult(
                status=CompilationStatus.UNSUPPORTED,
                generator_family=spec.generator_family,
                operator=spec.operator,
                reasons=(
                    f"unsupported generator_family {spec.generator_family!r}; "
                    "route this plan through the quarantined free-form path",
                ),
            )

        errors: list[str] = []
        if spec.operator != family.supported_operator:
            errors.append(
                f"{family.generator_family} supports "
                f"{family.supported_operator}, not {spec.operator}"
            )
        if (
            spec.target_concept_group is not None
            and spec.target_concept_group != family.concept_group
        ):
            errors.append(
                "target_concept_group is registry-derived and must be "
                f"{family.concept_group!r}"
            )
        if (
            spec.target_concept_type is not None
            and spec.target_concept_type != family.concept_type
        ):
            errors.append(
                "target_concept_type is registry-derived and must be "
                f"{family.concept_type!r}"
            )

        if parent is not None:
            parent_group = parent.get_concept_group()
            parent_type = parent.get_concept_type()
            if spec.operator == "in_depth" and (
                parent_group != family.concept_group
                or parent_type != family.concept_type
            ):
                errors.append(
                    "in_depth compiled family must preserve the parent's exact "
                    f"concept labels; parent={parent_group!r}/{parent_type!r}, "
                    f"family={family.concept_group!r}/{family.concept_type!r}"
                )
            if (
                spec.operator == "in_breadth"
                and parent_group == family.concept_group
            ):
                errors.append(
                    "in_breadth compiled family must change the parent's "
                    f"concept group from {parent_group!r}"
                )

        # Resolve the registry-owned variant first; its override is the baseline
        # the plan's own family_config is layered onto, so a variant choice
        # actually changes the compiled construction.
        contract = family.reasoning_contract
        variant = contract.variant(spec.family_variant)
        variant_requested = (spec.family_variant or "").strip()
        variant_selected = bool(variant_requested)
        overridden_plan_keys: tuple[str, ...] = ()
        if variant is None:
            errors.append(
                f"unknown family_variant {variant_requested!r} for "
                f"{family.generator_family}; valid ids are "
                + ", ".join(contract.variant_ids)
            )
            merged_config: Mapping[str, Any] = dict(spec.family_config)
        else:
            # The variant wins on the keys it declares. Planners observably echo
            # the published defaults verbatim, and letting that echo override a
            # variant would silently neutralize the plan's only consequential
            # choice. The ignored keys are recorded rather than dropped quietly.
            plan_config = dict(spec.family_config)
            overridden_plan_keys = tuple(
                sorted(
                    key
                    for key, value in plan_config.items()
                    if key in variant.config_override
                    and value != variant.config_override[key]
                )
            )
            merged_config = {
                **plan_config,
                **dict(variant.config_override),
            }

        try:
            config = family.parse_config(merged_config)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
            config = None

        variant_fields: dict[str, Any] = {
            "family_variant": variant.variant_id if variant else variant_requested,
            "variant_selected_by_plan": variant_selected,
            "variant_targets_failure_mode": (
                variant.targets_failure_mode if variant else None
            ),
            "variant_overridden_plan_keys": overridden_plan_keys,
        }

        if errors or config is None:
            return CompilationResult(
                status=CompilationStatus.INVALID_SPEC,
                generator_family=spec.generator_family,
                operator=spec.operator,
                concept_group=family.concept_group,
                concept_type=family.concept_type,
                reasons=tuple(errors),
                **variant_fields,
            )

        source = family.render_source(config)
        source_errors = lint_generator_source(source)
        source_errors.extend(
            lint_metacognitive_generator_source(
                source,
                require_assert=False,
                reject_trivial_assert=True,
                reject_unbounded_sampling=True,
                require_answer_routes=False,
                require_canonical_instance_data=True,
            )
        )
        normalized_config = asdict(config)
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        if source_errors:
            return CompilationResult(
                status=CompilationStatus.COMPILER_ERROR,
                generator_family=spec.generator_family,
                operator=spec.operator,
                concept_group=family.concept_group,
                concept_type=family.concept_type,
                family_config=normalized_config,
                source_hash=source_hash,
                reasons=tuple(source_errors),
                reasoning_contract=family.reasoning_contract,
                **variant_fields,
            )
        return CompilationResult(
            status=CompilationStatus.COMPILED,
            generator_family=spec.generator_family,
            operator=spec.operator,
            source_code=source,
            concept_group=family.concept_group,
            concept_type=family.concept_type,
            family_config=normalized_config,
            source_hash=source_hash,
            reasoning_contract=family.reasoning_contract,
            **variant_fields,
        )


def _json_ready(value: Any) -> Any:
    """Deep-convert mappings/tuples so ``json.dumps`` emits real JSON.

    ``MappingProxyType`` is not JSON-serializable. Passing one to ``json.dumps``
    with a ``default=`` fallback silently produced a Python ``repr`` wrapped in a
    JSON *string*, which is unreadable as structure by a downstream evaluator, so
    every payload crossing a prompt or artifact boundary is normalized here.
    """
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"family_config.{name} must be an integer")
    return value


def _reject_unknown_config(
    raw: Mapping[str, Any],
    allowed: set[str],
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            "unknown family_config field(s): " + ", ".join(unknown)
        )


def _parse_linear_config(
    raw: Mapping[str, Any],
) -> LinearSystemAggregateConfig:
    allowed = {
        "coefficient_min",
        "coefficient_max",
        "solution_min",
        "solution_max",
        "aggregate_multiplier_min",
        "aggregate_multiplier_max",
        "combination_weight_min",
        "combination_weight_max",
    }
    _reject_unknown_config(raw, allowed)
    defaults = LinearSystemAggregateConfig()
    values = {
        name: _integer(raw.get(name, getattr(defaults, name)), name=name)
        for name in allowed
    }
    config = LinearSystemAggregateConfig(**values)
    if config.aggregate_multiplier_min < 2:
        raise ValueError(
            "family_config.aggregate_multiplier_min must be at least 2 so the "
            "aggregate is never the plain sum of the printed equations"
        )
    if config.aggregate_multiplier_max > 12:
        raise ValueError(
            "family_config.aggregate_multiplier_max must stay at most 12"
        )
    if config.aggregate_multiplier_min > config.aggregate_multiplier_max:
        raise ValueError(
            "family_config.aggregate_multiplier_min must not exceed "
            "aggregate_multiplier_max"
        )
    if config.combination_weight_min < 1:
        raise ValueError(
            "family_config.combination_weight_min must be at least 1 so the "
            "hidden row weight is never zero"
        )
    if config.combination_weight_max > 6:
        raise ValueError(
            "family_config.combination_weight_max must stay at most 6"
        )
    if config.combination_weight_min > config.combination_weight_max:
        raise ValueError(
            "family_config.combination_weight_min must not exceed "
            "combination_weight_max"
        )
    if config.coefficient_min >= config.coefficient_max:
        raise ValueError(
            "family_config coefficient_min must be less than coefficient_max"
        )
    if config.solution_min >= config.solution_max:
        raise ValueError(
            "family_config solution_min must be less than solution_max"
        )
    if max(
        abs(config.coefficient_min),
        abs(config.coefficient_max),
        abs(config.solution_min),
        abs(config.solution_max),
    ) > 20:
        raise ValueError("linear family_config bounds must stay within [-20, 20]")
    return config


def _parse_modular_config(
    raw: Mapping[str, Any],
) -> ModularLinearSystemAggregateConfig:
    _reject_unknown_config(raw, {"modulus", "multiplier_min"})
    modulus = _integer(raw.get("modulus", 7), name="modulus")
    if modulus > 11 or not sympy.isprime(modulus):
        raise ValueError(
            "family_config.modulus must be a prime at most 11 so the "
            "independent brute-force oracle stays bounded"
        )
    if modulus < 5:
        raise ValueError("family_config.modulus must be at least 5")
    multiplier_min = _integer(raw.get("multiplier_min", 2), name="multiplier_min")
    if multiplier_min < 2:
        raise ValueError(
            "family_config.multiplier_min must be at least 2 so a real modular "
            "inversion is always required"
        )
    if multiplier_min > modulus - 1:
        raise ValueError(
            "family_config.multiplier_min must leave at least one invertible "
            "multiplier below the modulus"
        )
    return ModularLinearSystemAggregateConfig(
        modulus=modulus,
        multiplier_min=multiplier_min,
    )


def _matrix_payload(
    payload: Mapping[str, Any],
) -> tuple[sympy.Matrix, sympy.Matrix, sympy.Matrix]:
    rows = sympy.Matrix(payload["rows"])
    rhs = sympy.Matrix(payload["rhs"])
    target = sympy.Matrix([payload.get("target", [1, 1, 1])])
    return rows, rhs, target


def validate_linear_system_aggregate_instance(
    payload: Mapping[str, Any],
) -> InstanceValidation:
    """Check rank deficiency, consistency, rowspace, and nullspace witnesses."""
    reasons: list[str] = []
    facts: dict[str, Any] = {}
    try:
        rows, rhs, target = _matrix_payload(payload)
    except (KeyError, TypeError, ValueError) as exc:
        return InstanceValidation(False, (f"malformed linear instance: {exc}",))

    if rows.shape != (2, 3):
        reasons.append("rows must be a 2x3 matrix")
    if rhs.shape != (2, 1):
        reasons.append("rhs must contain two values")
    if target.shape != (1, 3):
        reasons.append("target must contain three coefficients")
    if reasons:
        return InstanceValidation(False, tuple(reasons))

    rank = int(rows.rank())
    facts["rank"] = rank
    facts["nullity"] = rows.cols - rank
    if not 0 < rank < rows.cols:
        reasons.append("system must be nonzero and rank-deficient")
    if rows.row_join(rhs).rank() != rank:
        reasons.append("system must be consistent")
    if rows.col_join(target).rank() != rank:
        reasons.append("target functional must lie in the rowspace")

    nullspace = rows.nullspace()
    if not nullspace:
        reasons.append("system must have a nontrivial nullspace")
    elif any(sympy.simplify((target * vector)[0]) != 0 for vector in nullspace):
        reasons.append("target functional must annihilate the nullspace")

    witnesses = payload.get("witnesses")
    target_values: list[sympy.Expr] = []
    if not isinstance(witnesses, (list, tuple)) or len(witnesses) < 2:
        reasons.append("two distinct full-solution witnesses are required")
    else:
        parsed_witnesses: list[sympy.Matrix] = []
        for witness in witnesses[:2]:
            vector = sympy.Matrix(witness)
            parsed_witnesses.append(vector)
            if vector.shape != (3, 1) or rows * vector != rhs:
                reasons.append("each witness must be a full solution")
            else:
                target_values.append(sympy.simplify((target * vector)[0]))
        if (
            len(parsed_witnesses) == 2
            and parsed_witnesses[0] == parsed_witnesses[1]
        ):
            reasons.append("solution witnesses must be distinct")
        if len(target_values) == 2 and target_values[0] != target_values[1]:
            reasons.append("target must agree across distinct full solutions")

    try:
        weights = rows.T.gauss_jordan_solve(target.T)[0]
        answer_expr = sympy.simplify((weights.T * rhs)[0])
        if answer_expr.free_symbols or answer_expr.is_integer is not True:
            reasons.append("rowspace oracle must produce one integer")
        else:
            answer = int(answer_expr)
            facts["answer"] = answer
            if (
                "answer" in payload
                and int(payload["answer"]) != answer
            ):
                reasons.append("provided answer disagrees with rowspace oracle")
            if target_values and any(value != answer_expr for value in target_values):
                reasons.append("witness target disagrees with rowspace oracle")
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        reasons.append(f"rowspace answer oracle failed: {exc}")

    return InstanceValidation(
        valid=not reasons,
        reasons=tuple(reasons),
        facts=MappingProxyType(facts),
    )


def check_linear_system_aggregate_necessity(
    payload: Mapping[str, Any],
) -> InstanceValidation:
    """Prove the row-space combination is not answerable by inspection.

    Correctness alone is not the family's contract: an instance whose two rows
    already sum to the target functional lets the solver read the aggregate off
    ``rhs[0] + rhs[1]``.  This check rejects that degeneracy, the single-equation
    leak, and any unit-weight combination, so the declared move stays necessary.
    """
    reasons: list[str] = []
    facts: dict[str, Any] = {}
    try:
        rows, rhs, target = _matrix_payload(payload)
    except (KeyError, TypeError, ValueError) as exc:
        return InstanceValidation(False, (f"malformed linear instance: {exc}",))
    if rows.shape != (2, 3) or target.shape != (1, 3):
        return InstanceValidation(False, ("rows/target shape is not 2x3/1x3",))

    plain_row_sum = [
        sympy.simplify(rows[0, column] + rows[1, column])
        for column in range(3)
    ]
    facts["plain_row_sum"] = [int(value) for value in plain_row_sum]
    if all(
        sympy.simplify(plain_row_sum[column] - target[0, column]) == 0
        for column in range(3)
    ):
        reasons.append(
            "adding the two printed equations already yields the target "
            "functional, so the aggregate is answerable by inspection"
        )

    for index in range(2):
        single = sympy.Matrix([[rows[index, column] for column in range(3)]])
        if single.col_join(target).rank() < 2:
            reasons.append(
                f"row {index} is proportional to the target functional, so one "
                "equation leaks the aggregate"
            )

    try:
        weights = rows.T.gauss_jordan_solve(target.T)[0]
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        return InstanceValidation(False, (f"rowspace weights failed: {exc}",))
    if weights.free_symbols:
        reasons.append("row-space combination is not uniquely determined")
    else:
        facts["rowspace_weights"] = [str(value) for value in weights]
        if all(sympy.simplify(value - 1) == 0 for value in weights):
            reasons.append(
                "row-space weights are all one, which is the plain-sum bypass"
            )
        if any(sympy.simplify(value) == 0 for value in weights):
            reasons.append(
                "a zero row-space weight leaves an unused printed equation"
            )

    declared = payload.get("aggregate_multiplier")
    if declared is not None:
        facts["aggregate_multiplier"] = int(declared)
        if int(declared) < 2:
            reasons.append(
                "aggregate_multiplier must be at least 2 so the combination "
                "requires a division"
            )
    return InstanceValidation(
        valid=not reasons,
        reasons=tuple(reasons),
        facts=MappingProxyType(facts),
    )


def check_modular_linear_system_aggregate_necessity(
    payload: Mapping[str, Any],
) -> InstanceValidation:
    """Require a genuine modular inversion rather than a bare congruence sum."""
    reasons: list[str] = []
    facts: dict[str, Any] = {}
    try:
        modulus = int(payload["modulus"])
        multiplier = int(payload["multiplier"]) % modulus
        rows = tuple(
            tuple(int(value) % modulus for value in row)
            for row in payload["rows"]
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return InstanceValidation(False, (f"malformed modular instance: {exc}",))

    facts["multiplier"] = multiplier
    if multiplier == 1:
        reasons.append(
            "multiplier 1 makes the aggregate the plain sum of the printed "
            "congruences, so no modular inverse is required"
        )
    if multiplier == 0:
        reasons.append("multiplier 0 is not invertible modulo the modulus")
    for index, row in enumerate(rows):
        if len(set(row)) == 1 and row[0] % modulus == 1:
            reasons.append(
                f"row {index} is the all-ones functional, so one congruence "
                "leaks the aggregate residue"
            )
    return InstanceValidation(
        valid=not reasons,
        reasons=tuple(reasons),
        facts=MappingProxyType(facts),
    )


def validate_modular_linear_system_aggregate_instance(
    payload: Mapping[str, Any],
    *,
    max_brute_modulus: int = 11,
) -> InstanceValidation:
    """Cross-check modular inversion with exhaustive small-domain solutions."""
    reasons: list[str] = []
    facts: dict[str, Any] = {}
    try:
        modulus = int(payload["modulus"])
        multiplier = int(payload["multiplier"])
        rows = tuple(
            tuple(int(value) % modulus for value in row)
            for row in payload["rows"]
        )
        rhs = tuple(int(value) % modulus for value in payload["rhs"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return InstanceValidation(False, (f"malformed modular instance: {exc}",))

    if modulus < 2 or modulus > max_brute_modulus:
        reasons.append(
            f"modulus must be between 2 and {max_brute_modulus} for brute force"
        )
        return InstanceValidation(False, tuple(reasons))
    if len(rows) != 2 or any(len(row) != 3 for row in rows):
        reasons.append("rows must be a 2x3 matrix")
    if len(rhs) != 2:
        reasons.append("rhs must contain two residues")
    if math.gcd(multiplier, modulus) != 1:
        reasons.append("multiplier must be invertible modulo the modulus")
    if reasons:
        return InstanceValidation(False, tuple(reasons))

    expected_column_sum = multiplier % modulus
    if any(
        (rows[0][column] + rows[1][column]) % modulus
        != expected_column_sum
        for column in range(3)
    ):
        reasons.append(
            "each column sum must equal the common invertible multiplier"
        )

    try:
        inverse = pow(multiplier, -1, modulus)
    except ValueError as exc:
        reasons.append(f"modular inverse failed: {exc}")
        inverse = 0
    expected = (sum(rhs) * inverse) % modulus
    facts["modular_inverse"] = inverse
    facts["answer"] = expected

    solution_count = 0
    brute_answers: set[int] = set()
    for x in range(modulus):
        for y in range(modulus):
            for z in range(modulus):
                values = (x, y, z)
                if all(
                    sum(
                        rows[row_index][column] * values[column]
                        for column in range(3)
                    )
                    % modulus
                    == rhs[row_index]
                    for row_index in range(2)
                ):
                    solution_count += 1
                    brute_answers.add(sum(values) % modulus)
    facts["solution_count"] = solution_count
    facts["brute_answers"] = tuple(sorted(brute_answers))
    if solution_count < 2:
        reasons.append("system must have at least two full modular solutions")
    if brute_answers != {expected}:
        reasons.append(
            "brute-force solutions do not have one residue matching the "
            "modular-inverse oracle"
        )
    if "answer" in payload and int(payload["answer"]) % modulus != expected:
        reasons.append("provided answer disagrees with modular-inverse oracle")

    return InstanceValidation(
        valid=not reasons,
        reasons=tuple(reasons),
        facts=MappingProxyType(facts),
    )


_LINEAR_SOURCE = r'''
import math
import random
import sympy

MAX_ATTEMPTS = 200

def _rowspace_weights(data):
    rows = sympy.Matrix(data["rows"])
    target = sympy.Matrix([data["target"]])
    return rows.T.gauss_jordan_solve(target.T)[0]

def _rowspace_answer(data):
    rhs = sympy.Matrix(data["rhs"])
    weights = _rowspace_weights(data)
    return int(sympy.Integer(sympy.simplify((weights.T * rhs)[0])))

def _symbolic_answer(data):
    rows = sympy.Matrix(data["rows"])
    rhs = sympy.Matrix(data["rhs"])
    target = data["target"]
    solution = next(iter(sympy.linsolve((rows, rhs))))
    value = sympy.simplify(sum(target[index] * solution[index] for index in range(3)))
    if value.free_symbols:
        raise ValueError("target is not uniquely identified")
    return int(sympy.Integer(value))

def _necessity_holds(data):
    rows = sympy.Matrix(data["rows"])
    target_row = sympy.Matrix([data["target"]])
    if data["aggregate_multiplier"] < 2:
        return False
    for index in range(2):
        single = sympy.Matrix([[rows[index, column] for column in range(3)]])
        if single.col_join(target_row).rank() < 2:
            return False
    if all(
        sympy.simplify(rows[0, column] + rows[1, column] - target_row[0, column]) == 0
        for column in range(3)
    ):
        return False
    weights = _rowspace_weights(data)
    if weights.free_symbols:
        return False
    if all(sympy.simplify(value - 1) == 0 for value in weights):
        return False
    return not any(sympy.simplify(value) == 0 for value in weights)

def _valid_linear_instance(data):
    rows = sympy.Matrix(data["rows"])
    rhs = sympy.Matrix(data["rhs"])
    target_row = sympy.Matrix([data["target"]])
    if rows.shape != (2, 3) or rhs.shape != (2, 1):
        return False
    rank = rows.rank()
    if not 0 < rank < rows.cols:
        return False
    if rows.row_join(rhs).rank() != rank:
        return False
    if rows.col_join(target_row).rank() != rank:
        return False
    nullspace = rows.nullspace()
    if not nullspace:
        return False
    if any(sympy.simplify((target_row * vector)[0]) != 0 for vector in nullspace):
        return False
    first = sympy.Matrix(data["witnesses"][0])
    second = sympy.Matrix(data["witnesses"][1])
    if first == second or rows * first != rhs or rows * second != rhs:
        return False
    if (target_row * first)[0] != (target_row * second)[0]:
        return False
    if not _necessity_holds(data):
        return False
    return _rowspace_answer(data) == _symbolic_answer(data)

def _render_linear_problem(data):
    names = ("x", "y", "z")
    equations = []
    for row, rhs in zip(data["rows"], data["rhs"]):
        expression = " + ".join(
            f"({row[index]}){names[index]}" for index in range(3)
        )
        equations.append(f"{expression} = {rhs}")
    return (
        "The following consistent integer linear system does not determine "
        "x, y, and z individually:\n"
        + "\n".join(equations)
        + "\nDetermine the uniquely fixed value of x + y + z. "
        "State only the integer."
    )

def _sample_linear_candidate(rng):
    target = [1, 1, 1]
    multiplier = rng.randint(
        __AGGREGATE_MULTIPLIER_MIN__, __AGGREGATE_MULTIPLIER_MAX__
    )
    weight = rng.choice(
        [
            sign * magnitude
            for magnitude in range(
                __COMBINATION_WEIGHT_MIN__, __COMBINATION_WEIGHT_MAX__ + 1
            )
            for sign in (-1, 1)
        ]
    )
    first_row = [
        rng.randint(__COEFFICIENT_MIN__, __COEFFICIENT_MAX__)
        for _ in range(3)
    ]
    second_row = [
        multiplier * target[index] - weight * first_row[index]
        for index in range(3)
    ]
    rows = [first_row, second_row]
    null_vector = [
        first_row[1] * second_row[2] - first_row[2] * second_row[1],
        first_row[2] * second_row[0] - first_row[0] * second_row[2],
        first_row[0] * second_row[1] - first_row[1] * second_row[0],
    ]
    divisor = math.gcd(
        math.gcd(abs(null_vector[0]), abs(null_vector[1])),
        abs(null_vector[2]),
    )
    first_solution = [
        rng.randint(__SOLUTION_MIN__, __SOLUTION_MAX__) for _ in range(3)
    ]
    if divisor == 0:
        return None
    null_vector = [value // divisor for value in null_vector]
    second_solution = [
        first_solution[index] + null_vector[index] for index in range(3)
    ]
    rhs = [
        sum(rows[row][column] * first_solution[column] for column in range(3))
        for row in range(2)
    ]
    candidate = {
        "rows": rows,
        "rhs": rhs,
        "target": target,
        "aggregate_multiplier": multiplier,
        "combination_weight": weight,
        "witnesses": [first_solution, second_solution],
    }
    if not _valid_linear_instance(candidate):
        return None
    return candidate

def build_instance_data(seed):
    """Canonical structured instance; shares one RNG stream with generate."""
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        candidate_data = _sample_linear_candidate(rng)
        if candidate_data is None:
            continue
        return candidate_data
    raise RuntimeError("failed to sample a valid linear-system aggregate")

def generate(seed):
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        candidate_data = _sample_linear_candidate(rng)
        if candidate_data is None:
            continue
        instance_data = candidate_data
        answer = _rowspace_answer(instance_data)
        assert answer == _symbolic_answer(instance_data)
        problem = _render_linear_problem(instance_data)
        return problem, str(sympy.Integer(answer))
    else:
        raise RuntimeError("failed to sample a valid linear-system aggregate")

CONCEPT_REASON = "Identify a row-space invariant that fixes an aggregate even though the latent variables are not individually identifiable."
CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.linear_system_sum"
'''


_MODULAR_SOURCE = r'''
import math
import random
import sympy

MAX_ATTEMPTS = 200

def _modular_answer(data):
    modulus = data["modulus"]
    inverse = pow(data["multiplier"], -1, modulus)
    return (sum(data["rhs"]) * inverse) % modulus

def _brute_force_answer(data):
    modulus = data["modulus"]
    answers = set()
    solution_count = 0
    for x in range(modulus):
        for y in range(modulus):
            for z in range(modulus):
                values = (x, y, z)
                if all(
                    sum(
                        data["rows"][row][column] * values[column]
                        for column in range(3)
                    )
                    % modulus
                    == data["rhs"][row]
                    for row in range(2)
                ):
                    solution_count += 1
                    answers.add(sum(values) % modulus)
    if solution_count < 2 or len(answers) != 1:
        raise ValueError("aggregate residue is not uniquely fixed")
    return next(iter(answers))

def _modular_necessity_holds(data):
    modulus = data["modulus"]
    multiplier = data["multiplier"] % modulus
    if multiplier < __MULTIPLIER_MIN__ or multiplier <= 1:
        return False
    for row in data["rows"]:
        if len(set(value % modulus for value in row)) == 1 and row[0] % modulus == 1:
            return False
    return True

def _valid_modular_instance(data):
    modulus = data["modulus"]
    multiplier = data["multiplier"]
    if math.gcd(multiplier, modulus) != 1:
        return False
    if any(
        (data["rows"][0][column] + data["rows"][1][column]) % modulus
        != multiplier % modulus
        for column in range(3)
    ):
        return False
    if not _modular_necessity_holds(data):
        return False
    return _modular_answer(data) == _brute_force_answer(data)

def _render_modular_problem(data):
    names = ("x", "y", "z")
    equations = []
    for row, rhs in zip(data["rows"], data["rhs"]):
        expression = " + ".join(
            f"{row[index]}{names[index]}" for index in range(3)
        )
        equations.append(
            f"{expression} is congruent to {rhs} modulo {data['modulus']}"
        )
    return (
        "Let x, y, and z be residue classes satisfying both congruences:\n"
        + "\n".join(equations)
        + "\nDetermine the unique residue of x + y + z modulo "
        + f"{data['modulus']}. State only its least nonnegative integer."
    )

def _sample_modular_candidate(rng):
    modulus = __MODULUS__
    multiplier = rng.randint(__MULTIPLIER_MIN__, modulus - 1)
    first_row = [rng.randrange(modulus) for _ in range(3)]
    second_row = [
        (multiplier - first_row[index]) % modulus for index in range(3)
    ]
    rows = [first_row, second_row]
    witness = [rng.randrange(modulus) for _ in range(3)]
    if math.gcd(multiplier, modulus) != 1:
        return None
    rhs = [
        sum(rows[row][column] * witness[column] for column in range(3))
        % modulus
        for row in range(2)
    ]
    candidate = {
        "modulus": modulus,
        "multiplier": multiplier,
        "rows": rows,
        "rhs": rhs,
    }
    if not _valid_modular_instance(candidate):
        return None
    return candidate

def build_instance_data(seed):
    """Canonical structured instance; shares one RNG stream with generate."""
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        candidate_data = _sample_modular_candidate(rng)
        if candidate_data is None:
            continue
        return candidate_data
    raise RuntimeError("failed to sample a valid modular-system aggregate")

def generate(seed):
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        candidate_data = _sample_modular_candidate(rng)
        if candidate_data is None:
            continue
        instance_data = candidate_data
        answer = _modular_answer(instance_data)
        assert answer == _brute_force_answer(instance_data)
        problem = _render_modular_problem(instance_data)
        return problem, str(sympy.Integer(answer))
    else:
        raise RuntimeError("failed to sample a valid modular-system aggregate")

CONCEPT_REASON = "Combine congruences and invert a common coefficient to identify a unique aggregate residue."
CONCEPT_GROUP = "number_theory"
CONCEPT_TYPE = "number_theory.modular_linear_system_sum"
'''


def _render_linear_source(config: FamilyConfig) -> str:
    if not isinstance(config, LinearSystemAggregateConfig):
        raise TypeError("linear compiler received the wrong config type")
    return (
        textwrap.dedent(_LINEAR_SOURCE)
        .replace("__COEFFICIENT_MIN__", str(config.coefficient_min))
        .replace("__COEFFICIENT_MAX__", str(config.coefficient_max))
        .replace("__SOLUTION_MIN__", str(config.solution_min))
        .replace("__SOLUTION_MAX__", str(config.solution_max))
        .replace(
            "__AGGREGATE_MULTIPLIER_MIN__",
            str(config.aggregate_multiplier_min),
        )
        .replace(
            "__AGGREGATE_MULTIPLIER_MAX__",
            str(config.aggregate_multiplier_max),
        )
        .replace(
            "__COMBINATION_WEIGHT_MIN__",
            str(config.combination_weight_min),
        )
        .replace(
            "__COMBINATION_WEIGHT_MAX__",
            str(config.combination_weight_max),
        )
        .strip()
        + "\n"
    )


def _render_modular_source(config: FamilyConfig) -> str:
    if not isinstance(config, ModularLinearSystemAggregateConfig):
        raise TypeError("modular compiler received the wrong config type")
    return (
        textwrap.dedent(_MODULAR_SOURCE)
        .replace("__MODULUS__", str(config.modulus))
        .replace("__MULTIPLIER_MIN__", str(config.multiplier_min))
        .strip()
        + "\n"
    )


LINEAR_SYSTEM_AGGREGATE_CONTRACT = FamilyReasoningContract(
    target_reasoning_move=(
        "Find the scaled linear combination of the two printed equations whose "
        "coefficient vector is (1, 1, 1), then divide that combination by its "
        "aggregate multiplier to read off x + y + z."
    ),
    necessity_witness=(
        "The two equations leave a one-dimensional solution set, so x, y and z "
        "are individually unidentifiable, while the target functional (1, 1, 1) "
        "lies in the row space and annihilates the null space. Recovering the "
        "aggregate therefore requires the specific weight pair the construction "
        "hides, and those weights are never both 1."
    ),
    excluded_bypasses=(
        "adding the two printed equations (their column sums are a multiple of "
        "the target, never the target itself)",
        "reading the aggregate off a single equation (no printed row is "
        "proportional to (1, 1, 1))",
        "solving for x, y and z individually (the system is rank deficient)",
    ),
    variants=(
        FamilyVariant(
            variant_id="balanced",
            targets_failure_mode=(
                "no specific observed failure; a general mix of combination "
                "weights and aggregate multipliers"
            ),
            config_override={},
        ),
        FamilyVariant(
            variant_id="heavy_division",
            targets_failure_mode=(
                "the solver finds a correct row combination but drops or "
                "mishandles the final division by the aggregate multiplier"
            ),
            config_override={
                "aggregate_multiplier_min": 4,
                "aggregate_multiplier_max": 7,
            },
        ),
        FamilyVariant(
            variant_id="asymmetric_combination",
            targets_failure_mode=(
                "the solver only ever tries adding the two equations with equal "
                "weight and never searches for an unequal row combination"
            ),
            config_override={
                "combination_weight_min": 2,
                "combination_weight_max": 5,
            },
        ),
        FamilyVariant(
            variant_id="wide_coefficients",
            targets_failure_mode=(
                "the solver's method is sound but it makes arithmetic slips on "
                "larger coefficients while eliminating variables"
            ),
            config_override={
                "coefficient_min": -7,
                "coefficient_max": 7,
                "solution_min": -6,
                "solution_max": 6,
            },
        ),
    ),
    default_variant_id="balanced",
)

MODULAR_LINEAR_SYSTEM_AGGREGATE_CONTRACT = FamilyReasoningContract(
    target_reasoning_move=(
        "Add the two congruences to obtain a common multiplier times "
        "x + y + z, then multiply by that multiplier's inverse modulo the "
        "modulus to obtain the unique residue."
    ),
    necessity_witness=(
        "Every column sum equals one common multiplier that is at least 2 and "
        "invertible modulo the prime modulus, so the summed congruence fixes "
        "the aggregate only after an explicit modular inversion. Bounded "
        "exhaustive search confirms at least two full solutions share exactly "
        "one aggregate residue."
    ),
    excluded_bypasses=(
        "adding the congruences alone (the multiplier is never 1, so the sum "
        "is not yet the aggregate)",
        "reading the residue off a single congruence (no printed row is the "
        "all-ones functional)",
        "solving for x, y and z individually (the congruence system is "
        "underdetermined)",
    ),
    variants=(
        FamilyVariant(
            variant_id="balanced",
            targets_failure_mode=(
                "no specific observed failure; the smallest bounded prime with "
                "any invertible multiplier"
            ),
            config_override={},
        ),
        FamilyVariant(
            variant_id="hard_inverse",
            targets_failure_mode=(
                "the solver sums the congruences correctly but skips or "
                "mis-computes the modular inverse of the common multiplier"
            ),
            config_override={"modulus": 11, "multiplier_min": 4},
        ),
        FamilyVariant(
            variant_id="small_modulus",
            targets_failure_mode=(
                "the solver loses track of residue reduction; a tiny modulus "
                "keeps arithmetic short so only the inversion step can fail"
            ),
            config_override={"modulus": 5, "multiplier_min": 2},
        ),
    ),
    default_variant_id="balanced",
)

DEFAULT_MUTATION_FAMILY_REGISTRY = MutationFamilyRegistry()
DEFAULT_MUTATION_FAMILY_REGISTRY.register(
    _FamilyDefinition(
        generator_family="linear_system_aggregate",
        supported_operator="in_depth",
        concept_group="algebra",
        concept_type="algebra.linear_system_sum",
        config_fields=(
            "coefficient_min",
            "coefficient_max",
            "solution_min",
            "solution_max",
            "aggregate_multiplier_max",
            "combination_weight_max",
        ),
        config_schema={
            "coefficient_min": {"type": "integer", "minimum": -20},
            "coefficient_max": {"type": "integer", "maximum": 20},
            "solution_min": {"type": "integer", "minimum": -20},
            "solution_max": {"type": "integer", "maximum": 20},
            "aggregate_multiplier_max": {
                "type": "integer",
                "minimum": 2,
                "maximum": 12,
                "description": (
                    "upper bound on the aggregate multiplier; the compiler pins "
                    "the lower bound at 2 so the plain row sum never answers"
                ),
            },
            "combination_weight_max": {
                "type": "integer",
                "minimum": 1,
                "maximum": 6,
                "description": (
                    "bound on the hidden nonzero weight applied to the first row"
                ),
            },
        },
        default_config=asdict(LinearSystemAggregateConfig()),
        validator_name="rank+consistency+rowspace+nullspace+necessity",
        parse_config=_parse_linear_config,
        render_source=_render_linear_source,
        reasoning_contract=LINEAR_SYSTEM_AGGREGATE_CONTRACT,
        validate_instance=validate_linear_system_aggregate_instance,
        check_necessity=check_linear_system_aggregate_necessity,
    )
)
DEFAULT_MUTATION_FAMILY_REGISTRY.register(
    _FamilyDefinition(
        generator_family="modular_linear_system_aggregate",
        supported_operator="in_breadth",
        concept_group="number_theory",
        concept_type="number_theory.modular_linear_system_sum",
        config_fields=("modulus",),
        config_schema={
            "modulus": {
                "type": "integer",
                "enum": [5, 7, 11],
                "description": "prime modulus bounded for exhaustive checking",
            }
        },
        default_config=asdict(ModularLinearSystemAggregateConfig()),
        validator_name="gcd+modular_inverse+bounded_brute_force+necessity",
        parse_config=_parse_modular_config,
        render_source=_render_modular_source,
        reasoning_contract=MODULAR_LINEAR_SYSTEM_AGGREGATE_CONTRACT,
        validate_instance=validate_modular_linear_system_aggregate_instance,
        check_necessity=check_modular_linear_system_aggregate_necessity,
    )
)


def registered_family_descriptor(
    parent: ProblemProgram,
    op: str,
    *,
    registry: MutationFamilyRegistry = DEFAULT_MUTATION_FAMILY_REGISTRY,
) -> FamilyDescriptor | None:
    """Return the exact registered route compatible with ``parent`` and ``op``.

    Selection depends only on the paired parent/operator, never on whether the
    plan is plain or reasoning-informed.  This lets both experimental
    conditions receive an identical compiler action/configuration space.
    """
    if op not in ("in_depth", "in_breadth"):
        return None
    compatible: list[_FamilyDefinition] = []
    for family in registry._families.values():
        if family.supported_operator != op:
            continue
        if op == "in_depth" and (
            parent.get_concept_group() != family.concept_group
            or parent.get_concept_type() != family.concept_type
        ):
            continue
        if (
            op == "in_breadth"
            and parent.get_concept_group() == family.concept_group
        ):
            continue
        compatible.append(family)
    if len(compatible) != 1:
        return None
    family = compatible[0]
    return FamilyDescriptor(
        generator_family=family.generator_family,
        operator=family.supported_operator,
        concept_group=family.concept_group,
        concept_type=family.concept_type,
        config_schema=MappingProxyType(dict(family.config_schema)),
        default_config=MappingProxyType(dict(family.default_config)),
        validator=family.validator_name,
        reasoning_contract=family.reasoning_contract,
    )


def registered_family_catalog(
    parent: ProblemProgram,
    op: str,
    *,
    registry: MutationFamilyRegistry = DEFAULT_MUTATION_FAMILY_REGISTRY,
) -> str:
    """Serialize the forced family descriptor for direct prompt insertion."""
    descriptor = registered_family_descriptor(parent, op, registry=registry)
    payload: dict[str, Any] = {
        "registry_version": MUTATION_FAMILY_REGISTRY_VERSION,
        "operator": op,
        "supported": descriptor is not None,
    }
    if descriptor is None:
        payload["reason"] = (
            "no unique registered family is compatible with this parent/operator"
        )
    else:
        payload["family"] = {
            "generator_family": descriptor.generator_family,
            "concept_group": descriptor.concept_group,
            "concept_type": descriptor.concept_type,
            "family_config_schema": dict(descriptor.config_schema),
            "family_config_defaults": dict(descriptor.default_config),
            "validator": descriptor.validator,
            "reasoning_contract": dict(
                descriptor.reasoning_contract.to_payload()
            ),
            # The planner's one consequential choice: pick the variant whose
            # targets_failure_mode matches the observed failure.
            "family_variants": [
                _json_ready(variant.to_payload())
                for variant in descriptor.reasoning_contract.variants
            ],
            "default_family_variant": (
                descriptor.reasoning_contract.default_variant_id
            ),
        }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def validate_compiled_family_semantics(
    result: CompilationResult,
    seeds: Any,
    *,
    registry: MutationFamilyRegistry = DEFAULT_MUTATION_FAMILY_REGISTRY,
) -> FamilySemanticValidation:
    """Check every seed's structured instance before any LLM evaluator runs.

    The compiled source is rendered by this module, so it is executed here in an
    isolated namespace to recover the canonical ``instance_data`` the sandbox's
    ``(problem, answer)`` return value hides.  Each instance is then put through
    the family's correctness oracle *and* its necessity check, keeping "the
    answer is right" and "the declared reasoning move is required" separate
    judgements.
    """
    seed_list = tuple(int(seed) for seed in seeds)
    if not result.compiled or result.source_code is None:
        return FamilySemanticValidation(
            valid=False,
            generator_family=result.generator_family,
            seeds=seed_list,
            reasons=("family semantics need a compiled source",),
        )
    family = registry.get(result.generator_family)
    if family is None:
        return FamilySemanticValidation(
            valid=False,
            generator_family=result.generator_family,
            seeds=seed_list,
            reasons=(
                f"unregistered family {result.generator_family!r} has no "
                "semantic contract",
            ),
        )

    namespace: dict[str, Any] = {"__name__": "_compiled_family"}
    try:
        exec(compile(result.source_code, "<compiled-family>", "exec"), namespace)
    except Exception as exc:  # pragma: no cover - compiler renders its own source
        return FamilySemanticValidation(
            valid=False,
            generator_family=result.generator_family,
            seeds=seed_list,
            reasons=(f"compiled source failed to import: {exc}",),
        )
    builder = namespace.get("build_instance_data")
    generator = namespace.get("generate")
    if not callable(builder) or not callable(generator):
        return FamilySemanticValidation(
            valid=False,
            generator_family=result.generator_family,
            seeds=seed_list,
            reasons=(
                "compiled family must expose build_instance_data and generate",
            ),
        )

    per_seed: list[Mapping[str, Any]] = []
    reasons: list[str] = []
    for seed in seed_list:
        entry: dict[str, Any] = {"seed": seed}
        try:
            payload = builder(seed)
            problem, answer = generator(seed)
        except Exception as exc:
            entry.update({"valid": False, "reasons": [f"generation failed: {exc}"]})
            per_seed.append(MappingProxyType(entry))
            reasons.append(f"seed={seed}: generation failed: {exc}")
            continue
        checked = dict(payload)
        checked["answer"] = int(answer)
        correctness = family.validate_instance(checked)
        necessity = family.check_necessity(checked)
        seed_reasons = [
            *(f"correctness: {reason}" for reason in correctness.reasons),
            *(f"necessity: {reason}" for reason in necessity.reasons),
        ]
        entry.update(
            {
                "problem": problem,
                "answer": answer,
                "answer_correct": correctness.valid,
                "necessity_holds": necessity.valid,
                "valid": correctness.valid and necessity.valid,
                "correctness_facts": dict(correctness.facts),
                "necessity_facts": dict(necessity.facts),
                "reasons": seed_reasons,
            }
        )
        per_seed.append(MappingProxyType(entry))
        for reason in seed_reasons:
            reasons.append(f"seed={seed}: {reason}")

    return FamilySemanticValidation(
        valid=bool(per_seed) and not reasons,
        generator_family=result.generator_family,
        seeds=seed_list,
        per_seed=tuple(per_seed),
        reasons=tuple(reasons),
    )


def compile_belief_probe(
    plan: Mapping[str, Any],
    parent: ProblemProgram,
    op: MutationOperator,
    *,
    registry: MutationFamilyRegistry = DEFAULT_MUTATION_FAMILY_REGISTRY,
) -> CompilationResult:
    """Compile the probe a belief attribution implies.

    The planner never chooses a family, a variant, or a config: it names one
    hypothesis, and the registry owns the mapping from hypothesis to the probe
    that falsifies it. This is what makes the analysis *drive* the mutation
    rather than merely accompany it -- under the previous schema both conditions
    picked the published defaults and compiled to the same source hash, so the
    attribution could not affect a single generated problem.
    """
    from .belief_probe import get_hypothesis

    descriptor = registered_family_descriptor(parent, op, registry=registry)
    if descriptor is None:
        return CompilationResult(
            status=CompilationStatus.UNSUPPORTED,
            generator_family="",
            operator=op,
            reasons=(
                "no unique registered family is compatible with this "
                "parent/operator",
            ),
        )
    hypothesis_id = str(plan.get("attributed_hypothesis") or "").strip()
    hypothesis = get_hypothesis(descriptor.generator_family, hypothesis_id)
    if hypothesis is None:
        return CompilationResult(
            status=CompilationStatus.INVALID_SPEC,
            generator_family=descriptor.generator_family,
            operator=op,
            reasons=(
                f"unknown attributed_hypothesis {hypothesis_id!r} for "
                f"{descriptor.generator_family}",
            ),
        )
    return registry.compile(
        MutationSpec(
            generator_family=descriptor.generator_family,
            operator=op,
            family_config={},
            family_variant=hypothesis.probe_variant,
        ),
        parent=parent,
    )


def compiled_family_instances(
    result: CompilationResult,
) -> Callable[[Any], dict[int, tuple[Mapping[str, Any], str]]]:
    """Return a loader for structured instances of a compiled family."""

    def load(seeds: Any) -> dict[int, tuple[Mapping[str, Any], str]]:
        if not result.compiled or result.source_code is None:
            return {}
        namespace: dict[str, Any] = {"__name__": "_compiled_family"}
        exec(compile(result.source_code, "<compiled-family>", "exec"), namespace)
        builder = namespace["build_instance_data"]
        generator = namespace["generate"]
        instances: dict[int, tuple[Mapping[str, Any], str]] = {}
        for seed in seeds:
            seed = int(seed)
            instances[seed] = (builder(seed), generator(seed)[1])
        return instances

    return load


def family_contract_payload(
    result: CompilationResult,
    semantics: FamilySemanticValidation | None = None,
) -> dict[str, Any] | None:
    """Bundle the registry contract with the deterministic verification facts.

    This is what the evaluator is shown for its necessity judgement: a claim the
    compiler owns and has already checked, instead of the planner's free-form
    prose about a construction it does not control.
    """
    if result.reasoning_contract is None:
        return None
    payload: dict[str, Any] = {
        "generator_family": result.generator_family,
        "operator": result.operator,
        "concept_group": result.concept_group,
        "concept_type": result.concept_type,
        "family_config": dict(result.family_config),
        "family_variant": result.family_variant,
        "compiler_registry_version": MUTATION_FAMILY_REGISTRY_VERSION,
        "compiler_source_hash": result.source_hash,
        **dict(result.reasoning_contract.to_payload()),
    }
    if semantics is not None:
        payload["deterministic_verification"] = {
            "valid": semantics.valid,
            "seeds": list(semantics.seeds),
            "answer_oracle_agrees": all(
                bool(entry.get("answer_correct")) for entry in semantics.per_seed
            )
            and bool(semantics.per_seed),
            "necessity_holds": all(
                bool(entry.get("necessity_holds")) for entry in semantics.per_seed
            )
            and bool(semantics.per_seed),
            "reasons": list(semantics.reasons),
        }
    # Plain, JSON-ready dict: this payload is embedded in the evaluator prompt.
    return _json_ready(payload)


def _spec_from_mapping(payload: Mapping[str, Any]) -> MutationSpec:
    schema_version = payload.get("schema_version")
    if schema_version == 5:
        if "generator_family" not in payload:
            raise ValueError("schema v5 requires generator_family")
        if "family_config" not in payload:
            raise ValueError("schema v5 requires family_config")

    family_value = (
        payload.get("generator_family")
        or payload.get("family")
        or payload.get("mutation_family")
    )
    if not isinstance(family_value, str) or not family_value.strip():
        if schema_version == 4:
            raise LookupError(
                "schema v4 has no registered generator_family/family_config; "
                "quarantine it on the legacy free-form path"
            )
        raise ValueError("generator_family must be a non-empty string")

    operator_value = payload.get("operator")
    if operator_value not in ("in_depth", "in_breadth"):
        raise ValueError("operator must be in_depth or in_breadth")

    if "family_config" in payload:
        raw_config = payload["family_config"]
    else:
        raw_config = payload.get("parameters", {})
    if not isinstance(raw_config, Mapping):
        raise ValueError("family_config must be an object")

    target_move = payload.get("target_reasoning_move", "")
    if not isinstance(target_move, str):
        raise ValueError("target_reasoning_move must be a string")
    target_group = payload.get("target_concept_group")
    target_type = payload.get("target_concept_type")
    if target_group is not None and not isinstance(target_group, str):
        raise ValueError("target_concept_group must be a string or null")
    if target_type is not None and not isinstance(target_type, str):
        raise ValueError("target_concept_type must be a string or null")
    variant_value = payload.get("family_variant")
    if variant_value is not None and not isinstance(variant_value, str):
        raise ValueError("family_variant must be a string or null")
    return MutationSpec(
        generator_family=family_value.strip(),
        operator=operator_value,
        family_config=dict(raw_config),
        target_reasoning_move=target_move,
        target_concept_group=target_group,
        target_concept_type=target_type,
        family_variant=(
            variant_value.strip()
            if isinstance(variant_value, str) and variant_value.strip()
            else None
        ),
    )


def compile_mutation_spec(
    spec: MutationSpec | Mapping[str, Any],
    *,
    parent: ProblemProgram | None = None,
    registry: MutationFamilyRegistry = DEFAULT_MUTATION_FAMILY_REGISTRY,
) -> CompilationResult:
    """Compile a typed spec or its JSON mapping representation."""
    if isinstance(spec, Mapping):
        try:
            normalized = _spec_from_mapping(spec)
        except LookupError as exc:
            return CompilationResult(
                status=CompilationStatus.UNSUPPORTED,
                generator_family="legacy_free_form",
                operator=str(spec.get("operator", "")),
                reasons=(str(exc),),
            )
        except (TypeError, ValueError) as exc:
            return CompilationResult(
                status=CompilationStatus.INVALID_SPEC,
                generator_family=str(spec.get("generator_family", "")),
                operator=str(spec.get("operator", "")),
                reasons=(str(exc),),
            )
    elif isinstance(spec, MutationSpec):
        normalized = spec
    else:
        return CompilationResult(
            status=CompilationStatus.INVALID_SPEC,
            generator_family="",
            operator="",
            reasons=("spec must be MutationSpec or a mapping",),
        )
    return registry.compile(normalized, parent=parent)


def compile_mutation_plan(
    plan: Mapping[str, Any],
    parent: ProblemProgram | None = None,
    op: MutationOperator | None = None,
    *,
    operator: MutationOperator | None = None,
    registry: MutationFamilyRegistry = DEFAULT_MUTATION_FAMILY_REGISTRY,
) -> CompilationResult:
    """Plan-oriented adapter for the live schema-v5 routing path."""
    payload = dict(plan)
    if op is not None and operator is not None and op != operator:
        return CompilationResult(
            status=CompilationStatus.INVALID_SPEC,
            generator_family=str(payload.get("generator_family", "")),
            operator=str(payload.get("operator", "")),
            reasons=(
                f"op {op!r} disagrees with operator {operator!r}",
            ),
        )
    requested_operator = operator if operator is not None else op
    if requested_operator is not None:
        existing = payload.get("operator")
        if existing is not None and existing != requested_operator:
            return CompilationResult(
                status=CompilationStatus.INVALID_SPEC,
                generator_family=str(payload.get("generator_family", "")),
                operator=str(existing),
                reasons=(
                    f"plan operator {existing!r} disagrees with requested "
                    f"operator {requested_operator!r}",
                ),
            )
        payload["operator"] = requested_operator
    return compile_mutation_spec(payload, parent=parent, registry=registry)


__all__ = [
    "CompilationResult",
    "CompilationStatus",
    "DEFAULT_MUTATION_FAMILY_REGISTRY",
    "FamilyCatalogEntry",
    "FamilyDescriptor",
    "FamilyReasoningContract",
    "FamilySemanticValidation",
    "FamilyVariant",
    "InstanceValidation",
    "LINEAR_SYSTEM_AGGREGATE_CONTRACT",
    "LinearSystemAggregateConfig",
    "MODULAR_LINEAR_SYSTEM_AGGREGATE_CONTRACT",
    "MUTATION_FAMILY_REGISTRY_VERSION",
    "ModularLinearSystemAggregateConfig",
    "MutationFamilyRegistry",
    "MutationSpec",
    "check_linear_system_aggregate_necessity",
    "check_modular_linear_system_aggregate_necessity",
    "compile_belief_probe",
    "compiled_family_instances",
    "compile_mutation_plan",
    "compile_mutation_spec",
    "family_contract_payload",
    "registered_family_catalog",
    "registered_family_descriptor",
    "validate_compiled_family_semantics",
    "validate_linear_system_aggregate_instance",
    "validate_modular_linear_system_aggregate_instance",
]
