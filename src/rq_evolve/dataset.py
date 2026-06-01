import random
from dataclasses import dataclass, field

from .program import ProblemProgram
from .prompts import SOLVER_SYSTEM_PROMPT
from .scoring import is_frontier


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
        if getattr(self.tokenizer, "chat_template", None):
            raw_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            raw_prompt = (
                f"system: {SOLVER_SYSTEM_PROMPT}\n"
                f"user: {problem}\nassistant:"
            )

        model_inputs = self.tokenizer(
            raw_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_length=self.max_prompt_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = _compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "data_source": self.data_source,
            "reward_model": {"ground_truth": answer},
            "extra_info": extra,
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
) -> list[dict]:
    """Render champion programs into training problems (H-priority, D-uniform).

    Ports evo-sample's ``h_priority_d_uniform`` selection: repeated grid sweeps
    emit at most one instance per occupied niche per sweep, walking H bins from
    high to low and the D (concept) bins in a fresh random order each sweep.
    This spreads the training set evenly across the D axis instead of draining
    one champion fully before the next — which matters when ``training_budget``
    binds (smaller than frontier_champions x instances_per_program). With
    ``training_budget=null`` the budget never binds, so every frontier champion
    still contributes ``instances_per_program`` instances (same set as the
    legacy order, just interleaved).

    Only frontier champions (low < p_hat < high) contribute; duplicate
    (problem, answer) instances are dropped; seeds advance monotonically under
    ``strict_anti_reuse``. This is framework-free — ``verl_adapter.py``
    tokenizes the returned dicts.
    """
    low, high = frontier_p_hat_range
    budget = training_budget or max(1, len(champions) * instances_per_program)
    used_seeds = used_seeds if used_seeds is not None else {}

    # Reconstruct the occupied grid from champion niche coordinates.
    grid: dict[tuple[int, int], ProblemProgram] = {
        (c.niche_h, c.niche_div): c for c in champions
    }

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

    MAX_FAILED_SWEEPS = 2
    failed_sweeps = 0
    while len(examples) < budget:
        progress = False
        advanced = False
        for h_bin in range(n_h_bins - 1, -1, -1):       # high H (hard) first
            d_order = list(range(n_div_bins))
            random.shuffle(d_order)                       # D-uniform per sweep
            for d_bin in d_order:
                if len(examples) >= budget:
                    break
                champ = grid.get((h_bin, d_bin))
                if champ is None:
                    continue
                if not is_frontier(float(getattr(champ, "p_hat", 0.5)), low, high):
                    continue
                appended, stepped = _try_emit(champ, h_bin, d_bin)
                progress = progress or appended
                advanced = advanced or stepped
            else:
                continue
            break  # budget reached in the inner loop — propagate out
        if progress:
            failed_sweeps = 0
            continue
        # Nothing appended this sweep: either every frontier champion hit its
        # per-refresh cap (not advanced -> normal under-fill for small archives)
        # or only exec-failing seeds were consumed for MAX_FAILED_SWEEPS sweeps.
        failed_sweeps += 1
        if not advanced or failed_sweeps >= MAX_FAILED_SWEEPS:
            break

    return examples


def _compute_position_id_with_mask(attention_mask):
    try:
        from verl.utils.model import compute_position_id_with_mask
    except ImportError:
        from verl.utils.model_utils import compute_position_id_with_mask

    return compute_position_id_with_mask(attention_mask)
