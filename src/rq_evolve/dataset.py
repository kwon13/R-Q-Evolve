import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .program import ProblemProgram
from .prompts import SOLVER_SYSTEM_PROMPT


STATIC_ID_ALIASES: dict[str, tuple[str, ...]] = {
    "sample_id": ("sample_id", "problem_id", "instance_id"),
    "independent_run_id": ("independent_run_id", "run_id"),
    "parent_id": ("parent_id", "parent_program_id"),
    "generator_id": (
        "generator_id",
        "generator_unit_id",
        "generator_pair_id",
        "program_id",
    ),
}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _first_nonempty(row: dict, aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        value = row.get(alias)
        if isinstance(value, bool) or value is None:
            continue
        canonical = str(value).strip()
        if canonical:
            return canonical
    return None


def _chat_prompt_token_count(tokenizer, problem: str) -> int:
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise ValueError(
            "the training tokenizer could not render the Solver chat template"
        ) from exc
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if (
        isinstance(token_ids, list)
        and token_ids
        and isinstance(token_ids[0], list)
    ):
        if len(token_ids) != 1:
            raise ValueError("chat template unexpectedly returned multiple prompts")
        token_ids = token_ids[0]
    if not isinstance(token_ids, (list, tuple)):
        raise ValueError("chat template did not return a token-id sequence")
    return len(token_ids)


def _reference_answer_token_count(tokenizer, answer: str) -> int:
    token_ids = tokenizer.encode(answer, add_special_tokens=False)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if not isinstance(token_ids, (list, tuple)):
        raise ValueError("tokenizer.encode did not return a token-id sequence")
    return len(token_ids)


def load_static_training_jsonl(
    path: str | Path,
    tokenizer,
    *,
    condition: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a fixed Solver-training JSONL and recompute its token accounting.

    This is intentionally stricter than the evolving-data path.  Every row needs
    stable run/parent/generator/sample identifiers, duplicate instances are
    rejected, and any supplied token count must match the actual training
    tokenizer.  ``problem_text`` and the expansion experiment's identifier
    aliases are normalized to the existing ``VerlDynamicDataset`` row shape.
    """

    if condition not in ("plain", "reasoning"):
        raise ValueError("static training condition must be 'plain' or 'reasoning'")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing static training JSONL: {source}")

    raw_bytes = source.read_bytes()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_ids: set[tuple[str, str]] = set()
    seen_instances: set[tuple[str, str]] = set()
    prompt_tokens = 0
    reference_answer_tokens = 0
    max_prompt_tokens = 0
    per_run: dict[str, dict[str, int]] = {}

    for line_number, raw_line in enumerate(
        raw_bytes.decode("utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(
                raw_line,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"line {line_number}: invalid JSON ({exc})")
            continue
        if not isinstance(value, dict):
            errors.append(f"line {line_number}: row must be a JSON object")
            continue
        row = dict(value)

        problem_value = row.get("problem")
        problem_text_value = row.get("problem_text")
        if (
            problem_value is not None
            and problem_text_value is not None
            and str(problem_value).strip() != str(problem_text_value).strip()
        ):
            errors.append(
                f"line {line_number}: problem and problem_text disagree"
            )
            continue
        problem = str(
            problem_value if problem_value is not None else problem_text_value or ""
        ).strip()
        answer_value = row.get("answer")
        answer = (
            ""
            if isinstance(answer_value, bool) or answer_value is None
            else str(answer_value).strip()
        )
        if not problem:
            errors.append(f"line {line_number}: problem must be non-empty")
            continue
        if not answer:
            errors.append(f"line {line_number}: answer must be non-empty")
            continue

        identifiers: dict[str, str] = {}
        missing_identifiers: list[str] = []
        for canonical, aliases in STATIC_ID_ALIASES.items():
            identifier = _first_nonempty(row, aliases)
            if identifier is None:
                missing_identifiers.append(canonical)
            else:
                identifiers[canonical] = identifier
        if missing_identifiers:
            errors.append(
                f"line {line_number}: missing stable identifier(s): "
                + ", ".join(missing_identifiers)
            )
            continue

        row_condition = row.get("condition")
        if row_condition is not None and str(row_condition).strip() != condition:
            errors.append(
                f"line {line_number}: condition={row_condition!r} does not match "
                f"configured static condition {condition!r}"
            )
            continue

        unique_id = (
            identifiers["independent_run_id"],
            identifiers["sample_id"],
        )
        if unique_id in seen_ids:
            errors.append(
                f"line {line_number}: duplicate (independent_run_id, sample_id) "
                f"{unique_id!r}"
            )
            continue
        instance_signature = (problem, answer)
        if instance_signature in seen_instances:
            errors.append(
                f"line {line_number}: duplicate (problem, answer) instance; "
                "repeat data only through the explicit static_epochs setting"
            )
            continue

        row_prompt_tokens = _chat_prompt_token_count(tokenizer, problem)
        row_reference_tokens = _reference_answer_token_count(tokenizer, answer)
        supplied_counts = {
            "prompt_token_count": row_prompt_tokens,
            "reference_answer_token_count": row_reference_tokens,
            # GRPO consumes the problem prompt and generates its own response;
            # the reference answer is reward metadata, not a training target.
            "token_count": row_prompt_tokens,
        }
        mismatch = False
        for field_name, computed_value in supplied_counts.items():
            if field_name not in row:
                continue
            supplied_value = row[field_name]
            if (
                isinstance(supplied_value, bool)
                or not isinstance(supplied_value, int)
                or supplied_value != computed_value
            ):
                errors.append(
                    f"line {line_number}: supplied {field_name}="
                    f"{supplied_value!r}, but the training tokenizer computes "
                    f"{computed_value}"
                )
                mismatch = True
        if mismatch:
            continue

        seen_ids.add(unique_id)
        seen_instances.add(instance_signature)
        normalized = {
            **row,
            **identifiers,
            "condition": condition,
            "problem": problem,
            "answer": answer,
            "prompt_token_count": row_prompt_tokens,
            "reference_answer_token_count": row_reference_tokens,
            "token_count": row_prompt_tokens,
        }
        rows.append(normalized)
        prompt_tokens += row_prompt_tokens
        reference_answer_tokens += row_reference_tokens
        max_prompt_tokens = max(max_prompt_tokens, row_prompt_tokens)
        run_summary = per_run.setdefault(
            identifiers["independent_run_id"],
            {"rows": 0, "prompt_tokens": 0, "reference_answer_tokens": 0},
        )
        run_summary["rows"] += 1
        run_summary["prompt_tokens"] += row_prompt_tokens
        run_summary["reference_answer_tokens"] += row_reference_tokens

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:20])
        remainder = (
            f"\n- ... and {len(errors) - 20} more error(s)"
            if len(errors) > 20
            else ""
        )
        raise ValueError(f"invalid static training JSONL {source}:\n{preview}{remainder}")
    if not rows:
        raise ValueError(f"static training JSONL contains no rows: {source}")

    report = {
        "schema_version": 1,
        "condition": condition,
        "source_path": str(source),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_rows": len(rows),
        "prompt_tokens": prompt_tokens,
        "max_prompt_tokens": max_prompt_tokens,
        "reference_answer_tokens": reference_answer_tokens,
        "token_count": prompt_tokens,
        "token_count_definition": (
            "Solver chat prompt tokens under the actual training tokenizer; "
            "reference answers are reward metadata and generated rollout tokens "
            "are measured only after training"
        ),
        "per_run": dict(sorted(per_run.items())),
    }
    return rows, report


def validate_static_training_schedule(
    report: dict[str, Any],
    *,
    batch_size: int,
    total_training_steps: int | None,
    trainer_total_epochs: int,
    static_epochs: int,
    expected_rows: int | None,
    expected_tokens: int | None,
    require_expected: bool,
    max_prompt_length: int | None = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Fail before Ray starts if fixed-data exposure would be implicit or partial."""

    rows = int(report["source_rows"])
    token_count = int(report["token_count"])
    issues: list[str] = []
    if batch_size < 1:
        issues.append("VERL generation batch size must be >= 1")
    elif rows < batch_size:
        issues.append(
            f"{rows} static rows are fewer than generation batch size {batch_size}; "
            "reduce the batch size or provide more independent rows"
        )
    elif rows % batch_size:
        issues.append(
            f"{rows} static rows are not divisible by generation batch size "
            f"{batch_size}; VERL drop_last would silently discard rows"
        )
    if static_epochs < 1:
        issues.append("training_data.static_epochs must be >= 1")
    if (
        max_prompt_length is not None
        and int(report.get("max_prompt_tokens", 0)) > int(max_prompt_length)
    ):
        issues.append(
            f"longest static prompt has {report['max_prompt_tokens']} tokens, "
            f"exceeding data.max_prompt_length={max_prompt_length}; truncation "
            "would change the fixed training condition"
        )
    if trainer_total_epochs != static_epochs:
        issues.append(
            f"trainer.total_epochs={trainer_total_epochs} must exactly equal "
            f"training_data.static_epochs={static_epochs}"
        )

    steps_per_epoch = rows // batch_size if batch_size > 0 else 0
    expected_steps = steps_per_epoch * static_epochs
    if total_training_steps is None:
        issues.append(
            "trainer.total_training_steps must be set explicitly for static "
            "training"
        )
    elif total_training_steps != expected_steps:
        issues.append(
            f"trainer.total_training_steps={total_training_steps}, but exact "
            f"static exposure requires {steps_per_epoch} step(s)/epoch x "
            f"{static_epochs} epoch(s) = {expected_steps}; partial or implicit "
            "extra passes are not allowed"
        )

    if require_expected and expected_rows is None:
        issues.append(
            "training_data.static_expected_rows is required for training "
            "(run --audit-static-data first)"
        )
    if require_expected and expected_tokens is None:
        issues.append(
            "training_data.static_expected_tokens is required for training "
            "(run --audit-static-data first)"
        )
    if expected_rows is not None and int(expected_rows) != rows:
        issues.append(
            f"static_expected_rows={expected_rows}, but the file contains {rows}"
        )
    if expected_tokens is not None and int(expected_tokens) != token_count:
        issues.append(
            f"static_expected_tokens={expected_tokens}, but the training "
            f"tokenizer computes {token_count}"
        )

    schedule = {
        **report,
        "batch_size": int(batch_size),
        "steps_per_epoch": steps_per_epoch,
        "static_epochs": int(static_epochs),
        "total_training_steps": (
            None
            if total_training_steps is None
            else int(total_training_steps)
        ),
        "row_exposures": rows * int(static_epochs),
        "prompt_token_exposures": int(report["prompt_tokens"]) * int(static_epochs),
        "reference_answer_token_exposures": int(
            report["reference_answer_tokens"]
        )
        * int(static_epochs),
        "token_exposures": token_count * int(static_epochs),
        "schedule_valid": not issues,
        "issues": issues,
    }
    if issues and raise_on_error:
        raise ValueError(
            "invalid static training schedule:\n"
            + "\n".join(f"- {issue}" for issue in issues)
        )
    return schedule
from .scoring import is_frontier, selection_priority


@dataclass
class DynamicProblemDataset:
    problems: list[dict] = field(default_factory=list)

    def update(self, problems: list[dict]) -> None:
        self.problems = list(problems)

    def snapshot(self) -> list[dict]:
        return list(self.problems)

    def __len__(self) -> int:
        return len(self.problems)


class VerlDynamicDataset:
    """A mutable dataset that emits the RLHF row shape expected by verl."""

    def __init__(
        self,
        dynamic_dataset: DynamicProblemDataset,
        tokenizer,
        *,
        max_prompt_length: int = 1024,
        truncation: str = "left",
        min_size: int = 1,
        data_source: str = "rq_evolved",
    ) -> None:
        self.dynamic_dataset = dynamic_dataset
        self.tokenizer = tokenizer
        self.max_prompt_length = int(max_prompt_length)
        self.truncation = truncation
        self.min_size = max(1, int(min_size))
        self.data_source = data_source
        # Where the padding wrap starts. verl's train dataloader is
        # drop_last=True with a fixed batch size, so when the replay set is
        # shorter than one batch the tail indices must wrap onto real rows --
        # and `item % len(rows)` alone always wraps onto rows 0, 1, ... which
        # are the highest-previous-R_Q elite's first instances. Measured: 15 rows
        # into a 32-row batch gave rows 0 and 1 three slots and every other row
        # two, every iteration, purely because 32 % 15 == 2. Rotating the start
        # spreads that surplus over the whole set instead of pinning it to one
        # elite. The sampler advances it once per epoch.
        self.pad_offset = 0

    def __len__(self) -> int:
        return max(len(self.dynamic_dataset.snapshot()), self.min_size)

    def __getitem__(self, item: int) -> dict:

        rows = self.dynamic_dataset.snapshot()
        if not rows:
            raise IndexError("VerlDynamicDataset is empty")
        row = rows[(item + self.pad_offset) % len(rows)]
        problem = row["problem"]
        answer = row["answer"]
        extra = {
            "index": item,
            "program_id": row.get("program_id"),
            "seed": row.get("seed"),
            "rq_score": row.get("rq_score"),
            "s_hat": row.get("s_hat"),
            "u_score": row.get("u_score"),
            "group_bin": row.get("group_bin"),
            "skill_bin": row.get("skill_bin"),
        }
        # Fixed expansion-study datasets carry explicit experimental units.
        # Preserve them in extra_info so rollout logs can be joined back to the
        # paired run/parent/generator/sample hierarchy.  Existing evolving rows
        # simply omit these fields and retain their prior output shape.
        for key in (
            "sample_id",
            "problem_id",
            "independent_run_id",
            "parent_id",
            "parent_program_id",
            "generator_id",
            "generator_unit_id",
            "generator_pair_id",
            "condition",
            "target_reasoning_move",
            "prompt_token_count",
            "reference_answer_token_count",
            "token_count",
        ):
            if key in row:
                extra[key] = row[key]

        messages = [
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        # verl 0.7.x AgentLoopWorker (SingleTurnAgentLoop) applies the chat
        # template itself, so the dataset only returns raw_prompt (chat msgs)
        # plus a dummy_tensor placeholder (DataProto requires a non-empty
        # tensor batch — see verl/utils/dataset/rl_dataset.py:RLHFDataset).
        # Tensor fields like input_ids must NOT be returned: the trainer's
        # _get_gen_batch leaves tensors in `batch` and unions the agent-loop
        # output into it; duplicate input_ids would trip a sanity assert.
        return {
            "raw_prompt": messages,
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "data_source": self.data_source,
            "reward_model": {"ground_truth": answer},
            "extra_info": extra,
            "index": item,
        }


def build_replay_training_examples(
    champions,
    *,
    replay,
    previous_rq,
    iteration: int,
    frontier_s_hat_range: tuple[float, float],
    training_budget: int | None = None,
    warmup: bool = False,
    allow_degenerate: bool = False,
) -> list[dict]:
    """Current re-scoring rollouts selected by the previous raw R_Q.

    There is no sampling pass. Each elite contributes exactly the instances the
    re-scoring already rolled out, so every instance trained on is an instance
    that was measured -- a tail instance the evaluation never saw cannot reach
    the batch, and nothing goes stale between scoring and the update.

    WHICH elites contribute is decided by their score as of the previous
    iteration. Ranking by the same rollouts that will be trained on conditions
    the sample on the selection event: the elite whose measurement noise
    happened to land high is the one kept, which biases the update even though
    each per-instance baseline is individually unbiased. The lag breaks that at
    first order and costs no extra rollouts. An elite with no previous score --
    inserted this iteration -- simply waits one iteration.
    """
    low, high = frontier_s_hat_range
    ranked: list[tuple[float, object]] = []
    for champion in champions:
        score = previous_rq.selection_score(champion.program_id, iteration)
        if warmup and score is None:
            # The one pass with nothing to lag against. Bootstrap scores every
            # seed under theta_0, which is exactly the weights the first update
            # starts from, so those rollouts are on-policy warm-up data -- and
            # there is no earlier measurement to select on because there is no
            # earlier iteration. Ranking falls back to the current score.
            score = float(getattr(champion, "rq_score", 0.0) or 0.0)
        if score is None:
            continue
        if not replay.has(champion.program_id):
            continue
        # The frontier band uses the CURRENT measurement: an elite selected on
        # its past score but degenerate right now would contribute a batch of
        # zero advantages.
        #
        # ``allow_degenerate`` is the caller saying the band admitted nobody at
        # all. A batch of zero advantages is a wasted step; an empty dataloader
        # is a dead run (IndexError: VerlDynamicDataset is empty). Take the
        # wasted step -- re-measurement runs next iteration and the band
        # normally refills.
        if not allow_degenerate and not is_frontier(
            float(getattr(champion, "s_hat", 0.0)), low, high
        ):
            continue
        ranked.append((score, champion))
    ranked.sort(key=lambda pair: pair[0], reverse=True)

    examples: list[dict] = []
    for score, champion in ranked:
        for group in replay.get(champion.program_id):
            if training_budget is not None and len(examples) >= training_budget:
                return examples
            examples.append(
                {
                    "problem": group.instance.problem,
                    "answer": group.instance.answer,
                    "program_id": champion.program_id,
                    "seed": group.instance.seed,
                    "rq_score": champion.rq_score,
                    "s_hat": champion.s_hat,
                    "u_score": champion.u_score,
                    "group_bin": int(getattr(champion, "niche_group", -1)),
                    "skill_bin": int(getattr(champion, "niche_skill", -1)),
                    "previous_rq": score,
                    "replay_rollouts": group.size,
                }
            )
    return examples


def build_training_examples(
    champions: list[ProblemProgram],
    *,
    instances_per_program: int,
    training_budget: int | None,
    frontier_s_hat_range: tuple[float, float],
    used_seeds: dict[str, set[int]] | None = None,
    strict_anti_reuse: bool = True,
    select_lowest_rq_first: bool = False,
    select_random_order: bool = False,
    select_random_seed: int = 0,
    select_ignores_uncertainty: bool = False,
    select_ignores_variance: bool = False,
) -> list[dict]:
    """Render champion programs into training problems (global R_Q priority).

    Frontier champions are sorted by ``rq_score`` descending, then each champion
    contributes up to ``instances_per_program`` generated instances before lower
    R_Q champions are considered. This makes ``training_budget`` bind globally
    by R_Q instead of by the MAP grid traversal order. Set
    ``select_lowest_rq_first=True`` to invert the order (ablation: drain the
    lowest-R_Q champions first), or ``select_random_order=True`` to drop the R_Q
    ordering entirely and drain frontier champions in a random (seeded) order
    (ablation: no priority signal). ``select_random_order`` takes precedence over
    ``select_lowest_rq_first`` when both are set. With
    ``training_budget=null`` every frontier champion still contributes up to
    ``instances_per_program`` instances, but the resulting rows are ordered by
    R_Q before the trainer's sampler shuffles them.

    Only frontier champions (low < s_hat < high) contribute; duplicate
    (problem, answer) instances are dropped; seeds advance monotonically under
    ``strict_anti_reuse``. This is framework-free — ``verl_adapter.py``
    tokenizes the returned dicts.
    """
    low, high = frontier_s_hat_range
    budget = training_budget or max(1, len(champions) * instances_per_program)
    used_seeds = used_seeds if used_seeds is not None else {}

    frontier_champions = [
        c
        for c in champions
        if is_frontier(float(getattr(c, "s_hat", 0.5)), low, high)
    ]

    if select_random_order:
        # Ablation: no R_Q ordering -- drain frontier champions in a random
        # (seeded) order. selection_priority / select_lowest_rq_first are ignored.
        ranked_champions = list(frontier_champions)
        random.Random(select_random_seed).shuffle(ranked_champions)
    else:
        ranked_champions = sorted(
            frontier_champions,
            key=lambda c: selection_priority(
                float(getattr(c, "s_hat", 0.5) or 0.5),
                float(getattr(c, "rq_score", 0.0) or 0.0),
                float(getattr(c, "u_score", 0.0) or 0.0),
                ignore_uncertainty=select_ignores_uncertainty,
                ignore_variance=select_ignores_variance,
            ),
            reverse=not select_lowest_rq_first,
        )

    examples: list[dict] = []
    emitted_per_program: dict[str, int] = {}
    emitted_signatures: set[tuple[str, str]] = set()

    def _try_emit(
        champ: ProblemProgram, group_bin: int, skill_bin: int
    ) -> tuple[bool, bool]:
        """Emit one instance. Returns (appended, advanced).

        ``advanced`` means a seed was consumed (execute may still have failed),
        used to tell a genuinely exhausted archive apart from transient seed
        failures across sweeps.
        """
        pid = champ.program_id
        if emitted_per_program.get(pid, 0) >= instances_per_program:
            return False, False
        seen = used_seeds.setdefault(pid, set())
        seed = 0
        if strict_anti_reuse:
            while seed in seen:
                seed += 1
        inst = champ.execute(seed=seed)
        seen.add(seed)
        if inst is None:
            return False, True
        signature = (inst.problem.strip(), inst.answer.strip())
        if signature in emitted_signatures:
            return False, True
        emitted_signatures.add(signature)
        examples.append(
            {
                "problem": inst.problem,
                "answer": inst.answer,
                "program_id": pid,
                "seed": inst.seed,
                "rq_score": champ.rq_score,
                "s_hat": champ.s_hat,
                "u_score": champ.u_score,
                "group_bin": group_bin,
                "skill_bin": skill_bin,
            }
        )
        emitted_per_program[pid] = emitted_per_program.get(pid, 0) + 1
        return True, True

    MAX_FAILED_ATTEMPTS = 2
    for champ in ranked_champions:
        group_bin = int(getattr(champ, "niche_group", -1))
        skill_bin = int(getattr(champ, "niche_skill", -1))
        failed_attempts = 0
        while len(examples) < budget:
            if emitted_per_program.get(champ.program_id, 0) >= instances_per_program:
                break
            appended, advanced = _try_emit(champ, group_bin, skill_bin)
            if appended:
                failed_attempts = 0
                continue
            if not advanced:
                break
            # Avoid spending indefinitely on generators whose next seeds all fail
            # or duplicate already-emitted problems.
            failed_attempts += 1
            if failed_attempts >= MAX_FAILED_ATTEMPTS:
                break
        if len(examples) >= budget:
            break

    return examples
