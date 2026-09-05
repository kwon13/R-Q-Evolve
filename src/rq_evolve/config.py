import ast
import math
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any
from omegaconf import OmegaConf


@dataclass(slots=True)
class ArchiveConfig:
    # The grid is DOMAIN x PROBLEM_TYPE and its shape is the two vocabularies, so
    # there are no bin counts to configure. Uncertainty is not an axis: H stays
    # in R_Q = p(1-p)H, which decides who holds a cell, not which cell.
    epsilon: float = 0.3
    ucb_c: float = 1.0
    selection_strategy: str = "ucb"
    # Champion competition inside an occupied DOMAIN x PROBLEM_TYPE cell.
    # ``fitness`` is production MAP-Elites (higher selection priority wins).
    # ``random`` is the score-free control: every valid challenger replaces the
    # incumbent with a fixed Bernoulli probability. Empty cells are always
    # filled. The draw counter is checkpointed by MAPElitesArchive, so resume is
    # bit-for-bit reproducible.
    admission_strategy: str = "fitness"
    random_replace_probability: float = 0.5
    random_seed: int = 0
    # Sliding-Window Monte Carlo Elites (SW-MCE) keeps the original MCE
    # binary offspring-survival reward, but estimates each cell's success rate
    # from only the most recent outer iterations.  The window is iteration-
    # based (not offspring-based), so a value of 50 with 32 mutations per
    # iteration retains roughly 1,600 selection outcomes.
    mce_window_iterations: int = 50
    # When enabled, a generated child cannot enter or resume in the archive
    # until the local restricted-token labeler assigns DOMAIN from its fixed
    # family. Hand-authored bootstrap seeds carry an explicit file contract.
    require_domain_labeling: bool = False
    # Duplicated in the archive contract on purpose: snapshot resume must keep
    # the same score thresholds without trusting mutable evolution metadata.
    domain_labeling_min_probability: float = 0.55
    domain_labeling_min_logit_margin: float = 0.50
    # ``epsilon_greedy`` keeps strict fitness wins and admits a non-winning
    # challenger with this probability. Independent of parent-selection epsilon.
    admission_epsilon: float = 0.25

    def __post_init__(self) -> None:
        if self.admission_strategy not in {"fitness", "random", "epsilon_greedy"}:
            raise ValueError(f"unknown admission_strategy: {self.admission_strategy}")
        if isinstance(self.admission_epsilon, bool):
            raise ValueError("admission_epsilon must be a finite number in [0, 1]")
        try:
            value = float(self.admission_epsilon)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "admission_epsilon must be a finite number in [0, 1]"
            ) from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("admission_epsilon must be a finite number in [0, 1]")
        self.admission_epsilon = value


@dataclass(slots=True)
class EvolutionConfig:
    seed_programs_dir: str = "seed_programs"
    inner_iterations: int = 8
    inner_iteration_batch_size: int = 4
    # Optional bounded refill of the mutation pool. ``inner_iterations`` stays
    # the minimum number of proposals, preserving every existing experiment.
    # When enabled, further batches are sampled until either enough genuinely
    # frontier children have entered MAP or enough frontier candidates have
    # at least been measured to show that cell competition/novelty -- rather
    # than generation volume -- is the bottleneck. The hard proposal cap is the
    # only exhaustion condition: wall-clock limits are deliberately absent
    # because throughput varies substantially across GPU environments.
    adaptive_mutation_refill: bool = False
    mutation_refill_max_iterations: int = 96
    mutation_refill_target_frontier_insertions: int = 1
    mutation_refill_target_frontier_candidates: int = 6
    # Run score-independent archive novelty/variation gates after labeling but
    # before solver rollout. Disabled by default because enabling it mid-run is
    # archive-equivalent but not trajectory-equivalent: rejected duplicates no
    # longer consume evaluation seeds or rollout compute.
    archive_preflight_before_rollout: bool = False
    # G: rollouts per instance, and the group the advantage baseline averages
    # over. R_Q is estimated from ONE fresh instance x G rollouts and OVERWRITTEN
    # every re-evaluation -- Dynamic MAP-Elites semantics, and the same choice
    # LILO/SFL makes (Foster & Foerster 2025, Alg. 1: L_SFL rollouts per
    # question, buffer refreshed every step, estimation trajectories REUSED as
    # the training batch; they found refreshing less often overfits the buffer).
    #
    # The seed axis is gone: n=5 x m=2 spread the same 10 rollouts over five
    # instances, which made every training group size 2, and a size-2 binary
    # group is degenerate with probability >= 50% at ANY difficulty. Measured on
    # the 8B run: replay_degenerate_frac 75-83%, i.e. four fifths of the update
    # was exactly zero. At G=10 that falls to 0.2% at s=0.5 and 35% at s=0.9.
    #
    # G >= 2 is not a preference: at G=1 the plug-in learnability is identically
    # 0 for every program and R_Q is unidentifiable.
    group_size: int | None = None
    # Prompts per training step. The frontier is smaller than this in practice
    # (median 18, 10th pct 13 over 67 iterations of the 8B run), so when it
    # falls short the highest-R_Q champions each contribute another FRESH
    # instance until the batch is full. Those extra instances are training-only:
    # R_Q stays a single-instance estimate so it means the same thing for every
    # champion regardless of how many slots it happened to fill.
    train_batch_target: int = 16
    # The trainer repeats each prompt actor_rollout_ref.rollout.n times before
    # generating, so a replayed group only lines up row-for-row when that n
    # equals m. Checked against the verl config at startup rather than left to
    # drift, because a mismatch silently disables replay.
    replay_requires_matching_rollout_n: bool = True
    # Deprecated: the old single-instance group size. Read as the rollout budget.
    num_rollouts: int | None = None
    # Deprecated n x m spelling. Every config and snapshot on disk still carries
    # it, and _dataclass_from_dict silently DROPS unknown keys, so without these
    # the old runs would quietly load at the new defaults instead of their own
    # budget. Resolved in __post_init__ into group_size = n * m, which keeps the
    # rollout spend identical to what the config asked for.
    eval_seeds: int | None = None
    rollouts_per_seed: int | None = None
    verify_seeds: int = 5
    # Candidate-level local verification workers. Each candidate preserves its
    # ordered multi-seed and repeat-execution checks; only independent
    # candidates run concurrently, each through a persistent sandbox process.
    # This is separate from async_rollout.verify_workers, which grades solver
    # responses after GPU rollout and does not execute generated programs.
    program_verify_workers: int = 1
    # Re-run the current deterministic source/statement contracts once for
    # every champion already present when a process starts.  This is primarily
    # a resume/migration guard: newly generated children already pass the same
    # verifier before admission, while an old snapshot may contain champions
    # accepted by weaker historical rules.
    strict_champion_audit: bool = False
    frontier_s_hat_range: tuple[float, float] = (0.1, 0.9)
    # Deprecated spelling, from before the notation matched the paper. Every
    # config on disk still uses it, and _dataclass_from_dict silently DROPS an
    # unknown key -- so without the alias those runs would quietly fall back to
    # the default band instead of the one they asked for. Resolved in
    # __post_init__; None means "not set", which is why it is not a plain tuple.
    frontier_p_hat_range: tuple[float, float] | None = None
    # When True, a child that parses but fails verification gets ONE multi-turn
    # self-fix attempt: the model is shown its own program + the rejection reason
    # and asked to fix only that issue.
    fix_retry: bool = False
    # Structural contract on the child's own answer cross-check (see
    # ``ast_contract.py``). "off" skips it, "shadow" records the verdict in
    # ``program.metadata["ast_contract"]`` without rejecting anything, and
    # "enforce" turns findings into a verify failure. Runs before execution, so
    # a rejection here also saves five sandbox runs and ``num_rollouts`` solver
    # generations.
    ast_contract: str = "enforce"
    # Solver rollouts keep using actor_rollout_ref.rollout.{temperature,top_p}.
    # Mutation code is contract-heavy and deliberately uses its own lower-noise
    # sampling settings.
    code_temperature: float = 0.4
    code_top_p: float = 0.95
    # Mutation in two calls: stage 1 writes the child PROBLEM in prose with no
    # program in front of it, stage 2 writes that problem's generator with the
    # parent shown only as a worked example of the statement-to-program
    # mapping. Measured against the single call that asks for a mutated
    # program: child/parent source similarity 0.99 -> 0.14, and children
    # declaring their parent's own cell 96% -> 0%. Both come from the same
    # thing -- a base policy shown a program to mutate reproduces it.
    two_stage_mutation: bool = False
    # Give stage 1 one label-free problem skeleton sampled from a DIFFERENT
    # primary-parent lineage, preferring a cross-GROUP/cross-SKILL donor and
    # sampling uniformly within that best available tier. This is inspiration,
    # not crossover:
    # the donor's source, answer/check routes, labels, scores and concrete
    # instances never enter the prompt, and stage 2 never sees the donor at
    # all.  Disabled by default so old configs and ablation baselines remain
    # byte-for-byte prompt compatible.
    structural_inspiration: bool = False
    # Independent from Python's process-global RNG. RQEvolver derives one local
    # RNG from this seed and a persisted draw counter for every proposed child.
    structural_inspiration_seed: int = 0
    # Oversized donor templates are omitted rather than handed to the backend's
    # left-truncation policy, which could silently remove the primary parent or
    # the output contract.  Live archive templates are far below this cap.
    structural_inspiration_max_chars: int = 1600
    # Named explicitly in configs/logs so a future semantic-cluster selector
    # cannot silently change the meaning of an existing experiment.
    structural_inspiration_selection: str = "cross_lineage_random"
    # Donor-v2 safety contract. A donor must come from the manually reviewed
    # seed allowlist and must currently have strictly positive R_Q. These affect
    # donor context only; generated children and R_Q=0 champions may still
    # occupy and reproduce from MAP cells.
    structural_inspiration_require_certified_donor: bool = False
    structural_inspiration_require_positive_rq: bool = False
    # Reject a child whose normalized statement-token set overlaps too much
    # with its assigned donor. None preserves the v1 copy gates.
    structural_inspiration_max_token_jaccard: float | None = None
    # Authored seeds eligible to act as structural donors. Generated children
    # never inherit this certification.
    manual_certified_seed_files: tuple[str, ...] = ()
    # Per-proposal few-shot rotation has its own deterministic stream.  Stage-2
    # tasks are created in asynchronous completion order, so using the global
    # RNG there would assign different examples to the same parent solely due
    # to worker timing and would also perturb later parent/target draws.
    mutation_prompt_seed: int = 0
    # Reproductive parent, target-axis and mutation-strategy draws use a second
    # counter-derived stream. Persisting its counter makes a checkpoint resume
    # independent of process-global Python RNG state.
    search_seed: int = 0
    # Stage 2 is transcription of a fixed specification onto a fixed shape, so
    # it wants no exploration; stage 1 keeps code_temperature because it is
    # invention and collapses to one child per parent at 0.
    generator_temperature: float = 0.0
    generator_top_p: float = 1.0
    # Per-stage output caps for the mutation calls. Left unset these fall back
    # to data.max_response_length (5,000 on the live configs), which is what the
    # 4B 8-GPU run did.
    #
    # Measured with the run's own tokenizer over 400 stage-1/stage-2 pairs per
    # policy (rq_output/gate_experiment/raw_baseline*.jsonl):
    #
    #   stage 1 (plan, prose):  p50 171 tok, p90 246,  4.0-7.5% run away
    #   stage 2 (generator):    p50 301 tok, p90 755,  4.0-4.5% exceed 1024
    #
    # So a runaway is ~20x the median reply, and over 32 parents there is a 92%
    # (stage 1) / 77% (stage 2) chance of at least one per iteration.
    #
    # 2048 rather than something tighter, for two reasons. It is the cap the
    # gate experiment itself ran stage 2 under, so every number the prompt work
    # rests on -- empty-cell survival, target compliance, few-shot copying --
    # was measured with exactly this ceiling; tightening it would put the
    # science on an untested setting to buy time. And the barrier this was
    # originally sized against is gone: `mutate_pipelined` submits per parent,
    # so a runaway now occupies one slot out of ~32 instead of holding up the
    # whole stage. What is left to win is that slot's own time, 5,000 -> 2,048,
    # and 0% of the measured replies reach even 2,048 on their own.
    #
    # A truncated stage 2 fails `extract_generator_code` and is reported as a
    # mutation failure -- already the outcome for ~50% of candidates -- so the
    # downside is noise in an existing channel, not a new failure mode.
    family_max_output_tokens: int | None = 2048
    generator_max_output_tokens: int | None = 2048
    # How often a champion whose LAST measurement was degenerate (s_hat exactly
    # 0 or exactly 1) is re-measured. 1 = every outer iteration, which is what
    # the 4B run did; 0 = never, so it keeps its score until something displaces
    # it; k = every k-th iteration.
    #
    # Why 0 is defensible: R_Q = 0 means the frontier band drains the champion
    # from the training set, and with replay_training_batch its re-scoring
    # rollouts are the training batch -- so a degenerate champion's rollouts are
    # generated and then thrown away. Measured on the 4B run that was 35-48% of
    # the whole re-evaluation budget (480 rollouts/iteration at 48 champions,
    # of which 270 went to champions that could not train).
    #
    # What 0 costs, stated plainly because it is large: at n=1 instance the
    # "degenerate" reading is mostly measurement noise, not a property of the
    # program. Over the run, 20.3% of degenerate readings were followed by a
    # live one at the very next re-evaluation, 43% of death runs lasted exactly
    # one iteration, and 111 of 145 programs (77%) were alive again at some
    # point after their first degenerate reading. Freezing on the first such
    # reading therefore mislabels most of the archive, and the observations it
    # would have skipped carry mean learnability s(1-s) = 0.152 against a
    # maximum of 0.25. A k of 3-5 buys most of the saving back with a bounded
    # lag; 1 is the safe setting and the one the measurements were taken under.
    #
    # 1, NOT 0, and this is not a preference. 0 was tried and it killed a run at
    # iteration 65 with "VerlDynamicDataset is empty": a champion that reads
    # degenerate once keeps that score until something displaces it, so
    # degenerate champions accumulate MONOTONICALLY. The archive went from 8
    # champions with 5 on the frontier to 30 with 1, mean R_Q fell 0.059 ->
    # 0.0021, and the frontier finally emptied. The numbers above say why: at
    # n=1 a degenerate reading is mostly noise, and freezing on it turns a
    # transient miss into a permanent one. Re-measurement is what makes the
    # archive a measurement rather than a ratchet.
    reevaluate_degenerate_every: int = 1
    # Retired compatibility knobs. The live pipeline never calls the old
    # policy-relative relabeller and never injects a desired cell. Keeping the
    # fields lets old YAML fail with an explicit message instead of being
    # silently ignored by dataclass loading.
    relabel_skill: bool = False
    target_cell_injection: bool = False
    # Label-blind DOMAIN assignment from the already fixed family. Seven
    # one-token YES/NO calls run as one local-policy batch before solver
    # rollouts; Stage 2 emits no DOMAIN and no external evaluator/API is used.
    independent_domain_labeling: bool = False
    domain_labeling_min_probability: float = 0.55
    domain_labeling_min_logit_margin: float = 0.50
    # Deterministically reshuffle/sample the available few-shot examples per
    # proposal. Rotation never consults DOMAIN, PROBLEM_TYPE, archive occupancy,
    # or any desired destination.
    rotate_few_shots: bool = True
    # Program fitness uncertainty factor. ``standard`` keeps R_Q=L*U;
    # ``reverse_u`` uses L*(C-U); and ``no_u`` fixes U=1, giving R_Q=L.
    # This changes the score stored in MAP and the lagged training priority,
    # unlike ``select_ignores_uncertainty`` below, which changes priority only.
    rq_fitness_mode: str = "standard"
    # Fixed across every Reverse-U round. It is deliberately not estimated
    # from the current population, which would make fitness drift over time.
    rq_reverse_u_constant: float = 2.0
    # Ablation: drop the H/u_score term ONLY from the priority that drives
    # evolution -- which champions are picked as mutation parents and which are
    # drained into the training batch -- so those decisions rank by s(1-s)
    # (pass-rate variance) instead of s(1-s)*H. The MAP still bins on real H and
    # stores/logs each champion's real R_Q, so the archive snapshots show the
    # true scores; only the selection ranking ignores H. This isolates whether
    # H is actually needed to drive the curriculum. Production keeps this False.
    select_ignores_uncertainty: bool = False

    # Ablation (the mirror of select_ignores_uncertainty): drop the s(1-s)
    # pass-rate-variance term ONLY from the selection/mutation priority, so those
    # decisions rank by H (u_score) alone instead of s(1-s)*H. The MAP still
    # bins on real H and stores/logs each champion's real R_Q. Isolates whether
    # the pass-rate-variance term is needed to drive the curriculum. Do NOT set
    # this together with select_ignores_uncertainty (that leaves no signal).
    select_ignores_variance: bool = False

    # Ablation: keep the archive's 35 slots and every validity gate, but stop
    # rescoring champions against the current policy. They keep the R_Q they
    # were admitted with and are never evicted. Measured on three 4B arms, that
    # rescoring removes 0.86-0.94 champions per insertion, so this is the single
    # largest change one flag can make to archive dynamics.
    reevaluate_champions: bool = True

    # Ablation: "flat" removes DOMAIN x PROBLEM_TYPE capacity reservation. Same
    # 35 slots, same gates and parent sampling -- a candidate takes any free slot
    # and then competes against the weakest occupant instead of the champion
    # sharing its post-hoc descriptors.
    archive_binning: str = "grid"

    def __post_init__(self) -> None:
        if self.inner_iterations < 0:
            raise ValueError("evolution.inner_iterations must be >= 0")
        if self.inner_iteration_batch_size < 1:
            raise ValueError(
                "evolution.inner_iteration_batch_size must be >= 1"
            )
        if (
            self.adaptive_mutation_refill
            and self.mutation_refill_max_iterations < self.inner_iterations
        ):
            raise ValueError(
                "evolution.mutation_refill_max_iterations must be >= "
                "evolution.inner_iterations"
            )
        if (
            self.adaptive_mutation_refill
            and self.mutation_refill_max_iterations < 1
        ):
            raise ValueError(
                "evolution.mutation_refill_max_iterations must be >= 1 when "
                "adaptive mutation refill is enabled"
            )
        for name in (
            "mutation_refill_target_frontier_insertions",
            "mutation_refill_target_frontier_candidates",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"evolution.{name} must be >= 0")
        if self.adaptive_mutation_refill and not (
            self.mutation_refill_target_frontier_insertions > 0
            or self.mutation_refill_target_frontier_candidates > 0
        ):
            raise ValueError(
                "adaptive mutation refill needs at least one positive stop target"
            )
        if self.group_size is None:
            # Legacy spellings, in the order they were introduced. n x m spent
            # n*m rollouts per program; num_rollouts spent that many directly.
            if self.rollouts_per_seed is not None:
                self.group_size = int(self.rollouts_per_seed) * int(
                    self.eval_seeds or 1
                )
            elif self.num_rollouts is not None:
                self.group_size = int(self.num_rollouts)
            else:
                self.group_size = 10
        if self.group_size < 2:
            raise ValueError(
                "evolution.group_size (G) must be >= 2: at G=1 the learnability "
                "estimate is identically 0 for every program, so R_Q cannot be "
                "identified from the rollouts at all"
            )
        if self.train_batch_target < 1:
            raise ValueError("evolution.train_batch_target must be >= 1")
        if int(self.verify_seeds) < 1:
            raise ValueError("evolution.verify_seeds must be >= 1")
        if int(self.program_verify_workers) < 1:
            raise ValueError("evolution.program_verify_workers must be >= 1")
        for name in ("family_max_output_tokens", "generator_max_output_tokens"):
            value = getattr(self, name)
            if value is not None and int(value) < 1:
                raise ValueError(f"evolution.{name} must be >= 1 or null")
        if self.structural_inspiration and not self.two_stage_mutation:
            raise ValueError(
                "evolution.structural_inspiration requires "
                "evolution.two_stage_mutation=true: inspiration belongs only "
                "to the stage-1 family-design prompt"
            )
        if self.structural_inspiration_seed < 0:
            raise ValueError("evolution.structural_inspiration_seed must be >= 0")
        if self.mutation_prompt_seed < 0:
            raise ValueError("evolution.mutation_prompt_seed must be >= 0")
        if self.search_seed < 0:
            raise ValueError("evolution.search_seed must be >= 0")
        if self.structural_inspiration_max_chars < 1:
            raise ValueError("evolution.structural_inspiration_max_chars must be >= 1")
        if self.structural_inspiration_selection != "cross_lineage_random":
            raise ValueError(
                "evolution.structural_inspiration_selection must be "
                "'cross_lineage_random', got "
                f"{self.structural_inspiration_selection!r}"
            )
        if self.structural_inspiration_max_token_jaccard is not None:
            threshold = float(self.structural_inspiration_max_token_jaccard)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    "evolution.structural_inspiration_max_token_jaccard must "
                    "be in [0, 1] or null"
                )
        if len(set(self.manual_certified_seed_files)) != len(
            self.manual_certified_seed_files
        ):
            raise ValueError(
                "evolution.manual_certified_seed_files contains duplicates"
            )
        if self.frontier_p_hat_range is not None:
            self.frontier_s_hat_range = tuple(
                float(x) for x in self.frontier_p_hat_range
            )
        if self.target_cell_injection:
            raise ValueError(
                "evolution.target_cell_injection is retired: mutation must be "
                "untargeted"
            )
        if self.relabel_skill:
            raise ValueError(
                "evolution.relabel_skill is retired: DOMAIN is declared once "
                "by the fixed child family and PROBLEM_TYPE is derived by "
                "deterministic statement/verifier rules"
            )
        if not 0.0 <= float(self.domain_labeling_min_probability) <= 1.0:
            raise ValueError(
                "evolution.domain_labeling_min_probability must be in [0, 1]"
            )
        if float(self.domain_labeling_min_logit_margin) < 0.0:
            raise ValueError(
                "evolution.domain_labeling_min_logit_margin must be >= 0"
            )
        if float(self.code_temperature) < 0.0:
            raise ValueError("evolution.code_temperature must be >= 0")
        if not 0.0 < float(self.code_top_p) <= 1.0:
            raise ValueError("evolution.code_top_p must be in (0, 1]")
        if self.archive_binning not in ("grid", "flat"):
            raise ValueError(
                "evolution.archive_binning must be 'grid' or 'flat', got "
                f"{self.archive_binning!r}"
            )
        if self.rq_fitness_mode not in ("standard", "reverse_u", "no_u"):
            raise ValueError(
                "evolution.rq_fitness_mode must be 'standard', 'reverse_u' "
                f"or 'no_u', got {self.rq_fitness_mode!r}"
            )
        if not math.isfinite(float(self.rq_reverse_u_constant)) or float(
            self.rq_reverse_u_constant
        ) <= 0.0:
            raise ValueError(
                "evolution.rq_reverse_u_constant must be a positive finite value"
            )
        if self.ast_contract not in ("off", "shadow", "enforce"):
            raise ValueError(
                "evolution.ast_contract must be 'off', 'shadow' or 'enforce', "
                f"got {self.ast_contract!r}"
            )


@dataclass(slots=True)
class TrainingDataConfig:

    # Build the training batch from the re-scoring rollouts instead of running a
    # second sampling pass over the same programs. The rollouts come from the
    # theta_t the following update starts from, so they are on-policy for
    # exactly that update -- and the pass they replace cost the same again.
    replay_training_batch: bool = True
    # Selection lag: an elite's place in the batch is decided by its exact raw
    # R_Q from the previous iteration, never by the rollouts it is about to
    # train on. Scores from different policies are deliberately not averaged.
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
    # Expansion-study mode: train on one frozen JSONL instead of constructing a
    # MAP archive or invoking the Evolver.  The file is condition-specific and
    # must contain stable run/parent/generator/sample identifiers.
    static_training_jsonl: str | None = None
    static_condition: str | None = None
    # Pin the audited source file shape so a stale/regenerated JSONL cannot
    # silently change a run.  These may be null only for --audit-static-data;
    # fit() requires both values.
    static_expected_rows: int | None = None
    static_expected_tokens: int | None = None
    # Exact number of complete passes over the fixed file.  VERL's configured
    # total steps must equal rows / generation_batch_size * static_epochs.
    static_epochs: int = 1

    def __post_init__(self) -> None:
        if self.static_epochs < 1:
            raise ValueError("training_data.static_epochs must be >= 1")
        for name in ("static_expected_rows", "static_expected_tokens"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"training_data.{name} must be >= 1 or null")
        if self.static_training_jsonl:
            if self.static_condition not in ("plain", "reasoning"):
                raise ValueError(
                    "training_data.static_condition must be 'plain' or "
                    "'reasoning' when static_training_jsonl is set"
                )
        elif self.static_epochs != 1 or any(
            value is not None
            for value in (
                self.static_condition,
                self.static_expected_rows,
                self.static_expected_tokens,
            )
        ):
            raise ValueError(
                "training_data.static_training_jsonl is required when any "
                "other static training field is set"
            )


@dataclass(slots=True)
class VerlConfig:
    enabled: bool = False
    config_path: str | None = None
    reward_function: str = "./src/rq_evolve/reward.py:compute_score"
    evolve_on_first_epoch: bool = True


@dataclass(slots=True)
class AsyncRolloutConfig:
    """Streaming (producer-consumer) evolve-phase rollout settings.

    ``streaming_enabled: false`` restores the legacy whole-batch path exactly
    (one blocking ``generate_sequences`` for the full inner batch). When true,
    solver rollouts are split into chunks of ``chunk_size`` instances (each
    chunk carries ``chunk_size * num_rollouts`` requests), submitted round-robin
    to verl's agent-loop workers with at most ``max_in_flight_chunks`` chunks
    outstanding, and verification/filtering/logging consume each chunk as soon
    as IT finishes -- a short problem's chunk is never blocked behind another
    chunk's long generations.
    """

    streaming_enabled: bool = True
    # Instances per chunk. 1 = finest granularity: one problem's G rollouts per
    # chunk, so its verification starts the moment its own rollouts finish.
    chunk_size: int = 1
    # Backpressure: max chunks outstanding in the rollout engine at once.
    #
    # 32, not 8. At 8 this was the binding constraint on the whole run, not a
    # safety margin: `max_pending_chunks` read exactly 8 in all 243 iterations
    # of the 4B 8-GPU run (rq_output/rq_evolve_4b_8gpu), i.e. the scheduler sat
    # at the cap every moment it had work. 8 chunks x G=10 = 80 sequences over
    # 8 vLLM engines is 10 per engine, and the phase delivered 1,890 tok/s
    # aggregate (236 tok/s per A100). Measured on one resident qwen3-4b-base
    # server (2 GPUs, DP=2), throughput against concurrency is 807 tok/s at 8,
    # 6,073 at 80, and 12,642 at 320 -- the 8-GPU evolve phase was running
    # slower than two GPUs at concurrency ~30.
    #
    # There is KV room by a wide margin: gpu_memory_utilization 0.38 leaves
    # ~20 GB of cache per engine and Qwen3-4B costs 144 KB/token, so ~138k
    # cached tokens -- >100 concurrent sequences per engine at the ~1.1k tokens
    # a solver rollout actually uses. An outer iteration only submits ~37
    # chunks in total, so 32 puts essentially the whole iteration in flight and
    # hands admission control to vLLM's own scheduler, which is the component
    # that is supposed to do it (it admits by free blocks and queues the rest).
    #
    # Staleness is unaffected: weights are pushed once per evolve phase
    # (`sync_weights`) and `_wake` already asserts no chunk is in flight during
    # a sync, so every chunk in a phase sees the same policy at any cap.
    max_in_flight_chunks: int = 32
    # Per-chunk wall-clock budget (covers queueing + generation of the chunk's
    # slowest sample). On expiry the chunk is retried up to max_retries times,
    # then recorded as a failure with reason "timeout" -- never silently dropped.
    request_timeout_s: float = 900.0
    max_retries: int = 1
    # CPU threads consuming completed chunks (math_verify + filtering + JSONL).
    verify_workers: int = 4
    # Bound on the completed-chunk queue between scheduler and verifiers; when
    # full, submission pauses (backpressure) instead of buffering unboundedly.
    queue_maxsize: int = 32
    # Async-RL correctness: "strict" rejects any sample whose policy_version
    # differs from the current one; "bounded" allows lag <= max_policy_lag.
    # During today's evolve phase weights are static (lag 0 by construction);
    # the gate is the invariant check that makes overlap modes safe later.
    staleness_mode: str = "bounded"
    max_policy_lag: int = 1
    # Per-sample JSONL logging (accepted AND rejected) to rq_archive/.
    log_samples: bool = True
    # --- filtering semantics -------------------------------------------------
    # Rejected samples leave the s_hat denominator (s_hat = correct/accepted),
    # so each filter that can bias the curriculum estimate is a knob:
    #   reject_overlong: a truncated response is ambiguous (can't tell if the
    #     model would have solved it) -> rejected by default.
    #   reject_duplicates: identical rollouts are legitimate samples of the
    #     policy; dropping them biases s_hat on low-entropy problems -> default
    #     False (duplicates are DETECTED and counted in metrics/JSONL either way).
    #   reject_invalid_answer: a response with no \boxed{} DID fail the problem;
    #     removing it from the denominator would overestimate s_hat -> default
    #     False (counted as correct=false, flagged in metrics/JSONL either way).
    reject_overlong: bool = True
    reject_duplicates: bool = False
    reject_invalid_answer: bool = False
    # Where the u_score (actor-forward entropy) pass runs. Only "deferred"
    # is implemented: one batched FSDP forward after the last chunk drains,
    # identical memory profile to the legacy path (no contention with vLLM
    # generation on the colocated GPUs).
    entropy_mode: str = "deferred"

    def __post_init__(self) -> None:
        if self.staleness_mode not in ("strict", "bounded"):
            raise ValueError(
                f"async_rollout.staleness_mode must be 'strict' or 'bounded', "
                f"got {self.staleness_mode!r}"
            )
        if self.entropy_mode != "deferred":
            raise ValueError(
                f"async_rollout.entropy_mode: only 'deferred' is implemented, "
                f"got {self.entropy_mode!r}"
            )
        if self.chunk_size < 1:
            raise ValueError("async_rollout.chunk_size must be >= 1")
        if self.max_in_flight_chunks < 1:
            raise ValueError("async_rollout.max_in_flight_chunks must be >= 1")
        if self.max_policy_lag < 0:
            raise ValueError("async_rollout.max_policy_lag must be >= 0")
        # NOTE: unlike queue.Queue, 0 does NOT mean unbounded here -- it would
        # gate every submission off and hang the scheduler, so reject it.
        if self.queue_maxsize < 1:
            raise ValueError(
                "async_rollout.queue_maxsize must be >= 1 (it bounds the "
                "completed-chunk backlog; 0 is not 'unbounded')"
            )
        if self.request_timeout_s <= 0:
            raise ValueError("async_rollout.request_timeout_s must be > 0")
        if self.max_retries < 0:
            raise ValueError("async_rollout.max_retries must be >= 0")
        if self.verify_workers < 1:
            raise ValueError("async_rollout.verify_workers must be >= 1")


@dataclass(slots=True)
class LoraConfig:
    """LoRA settings plumbed into verl (actor_rollout_ref.model.lora_*).

    Disabled in the base configs (existing full-finetune runs unchanged);
    enabled in the smoke/DeepSeek configs. NOTE: the installed verl 0.7.1
    legacy FSDP worker does not plumb lora_dropout into peft -- effective
    dropout is 0.0 and a warning is printed when dropout > 0 is configured.
    """

    enabled: bool = False
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    # Standard attention/MLP projections (Qwen/Llama naming). Models with other
    # module names (e.g. DeepSeek MLA: q_a_proj/q_b_proj/kv_a_proj_with_mqa/
    # kv_b_proj) must override this; scripts/preflight_check.py prints matched
    # modules per pattern and fails early if any pattern matches nothing.
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    # How trainer LoRA weights reach the vLLM rollout engine:
    #   auto           -> native_adapter if vLLM supports LoRA for the model
    #                     class, else fail with guidance (merge_push is a
    #                     documented future path, see docs/deepseek_support_plan.md)
    #   native_adapter -> verl's TensorLoRARequest/add_lora push (Qwen3 etc.)
    #   merge_push     -> NOT IMPLEMENTED (fails early); merged full-weight push
    #                     for models without vLLM LoRA support (DeepSeek MoE)
    sync_mode: str = "auto"

    def __post_init__(self) -> None:
        if self.sync_mode not in ("auto", "native_adapter", "merge_push"):
            raise ValueError(
                f"lora.sync_mode must be auto|native_adapter|merge_push, "
                f"got {self.sync_mode!r}"
            )
        if self.enabled and self.rank <= 0:
            raise ValueError("lora.enabled requires lora.rank > 0")


@dataclass(slots=True)
class MathEvalConfig:
    """Benchmark validation, ported from evo-sample's math_eval section.

    When enabled, the listed benchmarks are tokenized into a verl validation
    dataset (one ``data_source`` per benchmark). ``RQValidatingTrainer._validate``
    (eval_trainer.py) reports per-benchmark accuracy; grading uses
    ``math_eval.grade_eval`` -- the training reward's ``answers_match`` WITHOUT its
    >200-char length guard (see docs/GRADING.md) -- and runs on the trainer's MAIN
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
    async_rollout: AsyncRolloutConfig = field(default_factory=AsyncRolloutConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RQEvolveConfig":
        return _dataclass_from_dict(cls, payload)


def load_raw_config(path: str | Path, _stack: tuple[Path, ...] = ()):
    """Load a possibly-derived YAML config as one merged OmegaConf tree.

    A top-level ``extends: base.yaml`` is resolved relative to the child file.
    This keeps experiment arms small and auditable: the structural-inspiration
    arm overrides only its feature flag and output directory instead of copying
    a 250-line trainer configuration that can drift away from its baseline.
    Existing standalone configs are unchanged.
    """
    path = Path(path).expanduser().resolve()
    if path in _stack:
        chain = " -> ".join(str(p) for p in (*_stack, path))
        raise ValueError(f"cyclic config extends chain: {chain}")
    raw = OmegaConf.load(path)
    parent = raw.get("extends") if hasattr(raw, "get") else None
    if not parent:
        return raw

    child = OmegaConf.create(OmegaConf.to_container(raw, resolve=False))
    del child["extends"]
    parent_path = Path(str(parent)).expanduser()
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    base = load_raw_config(parent_path, (*_stack, path))
    return OmegaConf.merge(base, child)


def load_config(path: str | Path) -> RQEvolveConfig:
    """Load YAML via OmegaConf, resolving an optional ``extends`` chain."""
    path = Path(path)
    raw = OmegaConf.to_container(load_raw_config(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return RQEvolveConfig.from_dict(raw)


def _load_minimal_yaml(path: Path) -> dict[str, Any]:
    """Parse the small YAML subset used by ``configs/rq_evolve_base.yaml``.

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
        elif item.name in (
            "frontier_s_hat_range",
            "frontier_p_hat_range",
        ) and isinstance(value, list):
            kwargs[item.name] = tuple(float(x) for x in value)
        elif item.name in (
            "target_modules",
            "manual_certified_seed_files",
        ) and isinstance(value, list):
            kwargs[item.name] = tuple(str(x) for x in value)
        else:
            kwargs[item.name] = value
    return cls(**kwargs)
