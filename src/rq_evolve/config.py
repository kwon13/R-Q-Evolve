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
    # --- Evolver (Questioner) REINFORCE update -------------------------------
    # When True, after the Solver GRPO update of an outer iteration the shared
    # actor is ALSO updated as the Evolver: a REINFORCE + baseline step over the
    # mutation generations that successfully entered the MAP archive this
    # iteration, rewarded by their normalized R_Q score. This keeps the
    # problem-generation policy improving (otherwise the Evolver is frozen and
    # good problems stop appearing). Default OFF -- existing runs are unaffected.
    evolver_update: bool = False
    # Cap on the number of inserted-this-iteration generations fed to one
    # REINFORCE step. None = use all of them. Keeping it <= the actor
    # ppo_mini_batch_size makes the update a single on-policy mini-batch (verl
    # then sets old_log_prob = log_prob.detach() -> an exact REINFORCE gradient).
    # When more were inserted than the cap, the highest-R_Q ones are kept (logged).
    evolver_max_samples: int | None = None
    # Which generations train the Evolver, and how the [0,1] min-max reward pool
    # is formed:
    #   "inserted_only" -- only champions that entered the MAP (sparse: most outer
    #       iterations insert 0-2, so many updates are no-ops / single-sample).
    #   "valid_all" -- every candidate that produced a valid, solver-scored problem
    #       (inserted + rejected_non_elite + p_hat_zero); reward = each one's R_Q,
    #       normalized across this pool. Far denser signal; does NOT add a -1 for
    #       invalid generations -- it only rewards valid ones by R_Q.
    evolver_reward_mode: str = "inserted_only"

    # How the R_Q uncertainty/capacity factor U is measured per token:
    #   "entropy" -- Shannon entropy -sum p log p (verl's default; an
    #                approximation of the exact capacity factor).
    #   "gini"    -- exact Gini impurity 1 - sum p^2 = (1 - ||pi_t||^2).
    # In "gini" mode verl's per-token entropy is swapped at worker startup
    # (see verl_patches.py), so the change is transparent to the scoring path.
    uncertainty_measure: str = "entropy"

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
    # R-Zero x32-inflates AMC/AIME for greedy pass@1 stability. The in-trainer
    # periodic eval defaults this OFF to keep the set ~1.5k prompts (the inflated
    # set is ~4.6k). Grading no longer stalls the GPU (main-thread grade, see
    # eval_trainer.py); this is now purely an eval wall-time knob. Set true for
    # full R-Zero parity in periodic eval.
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
