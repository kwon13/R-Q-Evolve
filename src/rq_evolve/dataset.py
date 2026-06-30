from dataclasses import dataclass, field

from .program import ProblemProgram
from .prompts import SOLVER_SYSTEM_PROMPT
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

    def __len__(self) -> int:
        return max(len(self.dynamic_dataset.snapshot()), self.min_size)

    def __getitem__(self, item: int) -> dict:
        import torch
        import verl.utils.torch_functional as verl_F

        rows = self.dynamic_dataset.snapshot()
        if rows:
            row = rows[item % len(rows)]
            problem = row["problem"]
            answer = row["answer"]
            extra = {
                "index": item,
                "program_id": row.get("program_id"),
                "seed": row.get("seed"),
                "rq_score": row.get("rq_score"),
                "p_hat": row.get("p_hat"),
                "h_score": row.get("h_score"),
                "h_bin": row.get("h_bin"),
                "d_bin": row.get("d_bin"),
            }
        else:
            problem = "What is 1 + 1?"
            answer = "2"
            extra = {"index": item, "fallback": True}

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


def build_training_examples(
    champions: list[ProblemProgram],
    *,
    instances_per_program: int,
    training_budget: int | None,
    frontier_p_hat_range: tuple[float, float],
    n_h_bins: int,
    n_div_bins: int,
    used_seeds: dict[str, set[int]] | None = None,
    strict_anti_reuse: bool = True,
    select_lowest_rq_first: bool = False,
    select_ignores_uncertainty: bool = False,
    select_ignores_variance: bool = False,
) -> list[dict]:
    """Render champion programs into training problems (global R_Q priority).

    Frontier champions are sorted by ``rq_score`` descending, then each champion
    contributes up to ``instances_per_program`` generated instances before lower
    R_Q champions are considered. This makes ``training_budget`` bind globally
    by R_Q instead of by the MAP grid traversal order. Set
    ``select_lowest_rq_first=True`` to invert the order (ablation: drain the
    lowest-R_Q champions first). With
    ``training_budget=null`` every frontier champion still contributes up to
    ``instances_per_program`` instances, but the resulting rows are ordered by
    R_Q before the trainer's sampler shuffles them.

    Only frontier champions (low < p_hat < high) contribute; duplicate
    (problem, answer) instances are dropped; seeds advance monotonically under
    ``strict_anti_reuse``. This is framework-free — ``verl_adapter.py``
    tokenizes the returned dicts.
    """
    low, high = frontier_p_hat_range
    budget = training_budget or max(1, len(champions) * instances_per_program)
    used_seeds = used_seeds if used_seeds is not None else {}

    ranked_champions = sorted(
        (
            c
            for c in champions
            if is_frontier(float(getattr(c, "p_hat", 0.5)), low, high)
        ),
        key=lambda c: selection_priority(
            float(getattr(c, "p_hat", 0.5) or 0.5),
            float(getattr(c, "rq_score", 0.0) or 0.0),
            float(getattr(c, "h_score", 0.0) or 0.0),
            ignore_uncertainty=select_ignores_uncertainty,
            ignore_variance=select_ignores_variance,
        ),
        reverse=not select_lowest_rq_first,
    )

    examples: list[dict] = []
    emitted_per_program: dict[str, int] = {}
    emitted_signatures: set[tuple[str, str]] = set()

    def _try_emit(champ: ProblemProgram, h_bin: int, d_bin: int) -> tuple[bool, bool]:
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
                "p_hat": champ.p_hat,
                "h_score": champ.h_score,
                "h_bin": h_bin,
                "d_bin": d_bin,
            }
        )
        emitted_per_program[pid] = emitted_per_program.get(pid, 0) + 1
        return True, True

    MAX_FAILED_ATTEMPTS = 2
    for champ in ranked_champions:
        h_bin = int(getattr(champ, "niche_h", -1))
        d_bin = int(getattr(champ, "niche_div", -1))
        failed_attempts = 0
        while len(examples) < budget:
            if emitted_per_program.get(champ.program_id, 0) >= instances_per_program:
                break
            appended, advanced = _try_emit(champ, h_bin, d_bin)
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


def _compute_position_id_with_mask(attention_mask):
    try:
        from verl.utils.model import compute_position_id_with_mask
    except ImportError:
        from verl.utils.model_utils import compute_position_id_with_mask

    return compute_position_id_with_mask(attention_mask)
