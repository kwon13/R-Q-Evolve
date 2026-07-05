import ast
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any
from omegaconf import OmegaConf


@dataclass(slots=True)
class ArchiveConfig:
    n_h_bins: int = 6
    n_div_bins: int = 6
    h_range: tuple[float, float] = (0.0, 6.0)
    diversity_axis: str = "concept_group"
    epsilon: float = 0.3
    ucb_c: float = 1.0
    selection_strategy: str = "ucb"


@dataclass(slots=True)
class EvolutionConfig:
    seed_programs_dir: str = "seed_programs"
    inner_iterations: int = 8
    inner_iteration_batch_size: int = 4
    num_rollouts: int = 4
    in_depth_ratio: float = 0.5
    verify_seeds: int = 5
    frontier_p_hat_range: tuple[float, float] = (0.1, 0.9)
    # When True, a child that parses but fails verification gets ONE multi-turn
    # self-fix attempt: the model is shown its own program + the rejection reason
    # and asked to fix only that issue.
    fix_retry: bool = True
    # When True, every lint-verified child (including fix-retry survivors) passes
    # an LLM coherence gate on its seed-0 problem before solver rollout / archive
    # insertion. A problem the evaluator marks INVALID is discarded -- a final
    # noise filter against incoherent statements that pass the cheap lint checks.
    use_evaluator: bool = True
    # Ablation: drop the H/uncertainty term ONLY from the priority that drives
    # evolution -- which champions are picked as mutation parents and which are
    # drained into the training batch -- so those decisions rank by s(1-s)
    # (pass-rate variance) instead of s(1-s)*H. The MAP still bins on real H and
    # stores/logs each champion's real R_Q, so the archive snapshots show the
    # true scores; only the selection ranking ignores H. This isolates whether
    # H is actually needed to drive the curriculum. Production keeps this False.
    select_ignores_uncertainty: bool = False

    # Ablation (the mirror of select_ignores_uncertainty): drop the s(1-s)
    # pass-rate-variance term ONLY from the selection/mutation priority, so those
    # decisions rank by H (uncertainty) alone instead of s(1-s)*H. The MAP still
    # bins on real H and stores/logs each champion's real R_Q. Isolates whether
    # the pass-rate-variance term is needed to drive the curriculum. Do NOT set
    # this together with select_ignores_uncertainty (that leaves no signal).
    select_ignores_variance: bool = False


@dataclass(slots=True)
class TrainingDataConfig:
    instances_per_program: int = 8
    training_budget: int | None = None
    strict_anti_reuse: bool = True
    # Order in which frontier champions are drained into the training batch.
    #   False (default) -> highest R_Q first (production behavior).
    #   True            -> lowest R_Q first (ablation: invert the priority so the
    #                      budget binds on the LEAST uncertain/valuable champions).
    select_lowest_rq_first: bool = False
    # ABLATION: ignore R_Q ordering entirely and drain frontier champions in a
    # RANDOM order (no sort). Takes precedence over select_lowest_rq_first when
    # both are set. The shuffle is seeded (select_random_seed + refresh count) so
    # runs are reproducible while still varying across outer iterations.
    select_random_order: bool = False
    select_random_seed: int = 0


@dataclass(slots=True)
class VerlConfig:
    enabled: bool = False
    config_path: str | None = None
    reward_function: str = "./src/rq_evolve/reward.py:compute_score"
    evolve_on_first_epoch: bool = True


@dataclass(slots=True)
class MathEvalConfig:
    """Benchmark validation, ported from evo-sample's math_eval section.

    When enabled, the listed benchmarks are tokenized into a verl validation
    dataset (one ``data_source`` per benchmark). ``RQValidatingTrainer._validate``
    (eval_trainer.py) reports per-benchmark accuracy; grading reuses the training
    ``reward_function`` (sympy ``answers_match``) but runs on the trainer's MAIN
    thread (the agent-loop reward worker skips eval rows) so math_verify's SIGALRM
    timeout works and a pathological boxed answer can't stall the GPU mid-eval.
    GPT-judge is intentionally dropped. Evaluation cadence (before-train / every N
    steps) is controlled by ``trainer.val_before_train`` and ``trainer.test_freq``.
    """

    enabled: bool = False
    benchmarks: tuple[str, ...] = (
        "math500",
        "amc23",
        "aime24",
        "aime25",
        "minerva_math",
        "olympiadbench",
    )
    # Sub-sample per benchmark for quick debugging; -1 = full set (R-Zero parity).
    max_samples_per_benchmark: int = -1
    sample_seed: int = 42
    inflate_x32: bool = False
    grader: str = "sympy"


@dataclass(slots=True)
class RQEvolveConfig:
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    training_data: TrainingDataConfig = field(default_factory=TrainingDataConfig)
    verl: VerlConfig = field(default_factory=VerlConfig)
    math_eval: MathEvalConfig = field(default_factory=MathEvalConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RQEvolveConfig":
        return _dataclass_from_dict(cls, payload)


def load_config(path: str | Path) -> RQEvolveConfig:
    """Load YAML via OmegaConf, with a tiny fallback for this simple config."""
    path = Path(path)
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return RQEvolveConfig.from_dict(raw)


def _load_minimal_yaml(path: Path) -> dict[str, Any]:
    """Parse the small YAML subset used by ``configs/rq_evolve.yaml``.

    This is not a general YAML parser. It supports nested mappings through
    indentation plus inline scalars/lists, which keeps the starter project
    runnable before optional dependencies are installed.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"unsupported config line: {raw_line!r}")
        while stack and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if not value.strip():
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent + 2, child))
            continue
        current[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") or value.startswith(("'", '"')):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _dataclass_from_dict(cls, payload: dict[str, Any]):
    kwargs = {}
    for item in fields(cls):
        if item.name in payload:
            value = payload[item.name]
        elif item.default is not MISSING:
            value = item.default
        else:
            value = item.default_factory()

        if isinstance(value, dict) and item.default_factory is not MISSING:
            default_obj = item.default_factory()
            if is_dataclass(default_obj):
                kwargs[item.name] = _dataclass_from_dict(type(default_obj), value)
                continue
            kwargs[item.name] = value
        elif item.name == "h_range" and isinstance(value, list):
            kwargs[item.name] = tuple(float(x) for x in value)
        elif item.name == "frontier_p_hat_range" and isinstance(value, list):
            kwargs[item.name] = tuple(float(x) for x in value)
        else:
            kwargs[item.name] = value
    return cls(**kwargs)
