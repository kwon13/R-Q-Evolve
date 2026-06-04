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
    crossover_ratio: float = 0.2
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


@dataclass(slots=True)
class TrainingDataConfig:
    instances_per_program: int = 8
    training_budget: int | None = None
    strict_anti_reuse: bool = True


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
    dataset (one ``data_source`` per benchmark). verl's _validate() then reports
    per-benchmark accuracy automatically; grading reuses the training
    ``reward_function`` (sympy ``answers_match``). GPT-judge is intentionally
    dropped. Evaluation cadence (before-train / every N steps) is controlled by
    the verl override's ``trainer.val_before_train`` and ``trainer.test_freq``.
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
    # R-Zero x32-inflates AMC/AIME for greedy pass@1 stability. The in-trainer
    # periodic eval defaults this OFF: the inflated set (~4.6k prompts at -1)
    # grades serially through math_verify and stalls the run with GPU at 0%
    # mid-eval. Set true only if you want full R-Zero parity in periodic eval.
    inflate_x32: bool = False
    # "sympy" reuses reward.answers_match (no extra deps). "math_verify" matches
    # R-Zero numerically but needs math-verify + latex2sympy2_extended installed.
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
