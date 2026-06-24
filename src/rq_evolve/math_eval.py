"""Math benchmark validation datasets for in-trainer verl evaluation.

Ported from evo-sample/evaluation/math_benchmarks.py, trimmed for R-Q-Evolve:

  * Same R-Zero source mapping (OpenAI MATH-500 CSV; zwhe99 / HuggingFaceH4 /
    yentinglin for the rest; ×32 inflation for AMC/AIME).
  * GPT-judge is dropped.
  * Grading is NOT done here for the in-trainer path — benchmarks are emitted
    as a verl validation dataset (one ``data_source`` per benchmark). The agent
    loop's reward worker SKIPS grading these rows (each carries an
    ``extra_info[SKIP_WORKER_GRADE_KEY]`` sentinel); ``RQValidatingTrainer.
    _validate`` (eval_trainer.py) re-grades the decoded responses with
    ``grade_eval`` on the trainer's MAIN thread, where math_verify's SIGALRM
    timeout works. Per-benchmark accuracy is then reported via verl's own metric
    aggregation. (The old path graded on the worker thread, which stalled the
    GPU to 0% mid-eval — see eval_trainer.py.)
  * ``grade_eval`` is kept identical to the offline checkpoint grader
    (scripts/eval_vllm_math.py): brace-matched ``extract_boxed`` + \boxed-wrapped
    math_verify, NO length guard — so val-core matches eval_vllm_math.py exactly.
    The \boxed wrap is required (bare parse falsely rejects \dfrac<->\frac). NOT
    to be confused with ``reward.answers_match`` (the training reward grader,
    which keeps the length guard for reward-worker-thread safety).
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import _compute_position_id_with_mask
from .prompts import SOLVER_SYSTEM_PROMPT
from .reward import SKIP_WORKER_GRADE_KEY, answers_match, extract_boxed

logger = logging.getLogger(__name__)

# R-Zero's evaluation prompt (R-Zero/evaluation/generate.py).
MATH_EVAL_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

MATH500_CSV_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv"
)

# R-Zero ×32-inflates only AMC/AIME for pass@1-with-greedy stability.
INFLATE_X32 = {"amc23", "aime24", "aime25"}


@dataclass
class MathBenchmarkProblem:
    benchmark: str
    problem: str
    answer: str
    index: int


# ---------------------------------------------------------------------------
# Grading
#
# ``grade_eval`` is the single source of truth for the in-trainer val-core path
# (eval_trainer.RQValidatingTrainer._validate). It is kept byte-for-byte in sync
# with the offline checkpoint grader (scripts/eval_vllm_math.py -> evo-sample/
# evaluation.math_benchmarks.grade_math_response) so val-core matches the
# checkpoint number exactly. It is ``reward.answers_match`` WITHOUT the length
# guard:
#   * brace-matched ``extract_boxed`` (LAST \boxed) -- final answer, nesting-safe;
#   * ``verify(parse("\boxed{gt}"), parse("\boxed{pred}"))``. The \boxed wrap is
#     REQUIRED, not leniency: bare ``parse("\dfrac{1}{2}")`` fails math_verify's
#     LaTeX extractor and FALSELY rejects \dfrac<->\frac and fraction<->decimal
#     equivalences (verified: bare grades \dfrac{1}{2}==\frac{1}{2} as wrong).
# The length guard (reward.answers_match) is intentionally DROPPED here: eval
# runs on the trainer MAIN thread where math_verify's SIGALRM timeout already
# bounds each call, and the guard's normalized string-compare causes false
# negatives on long-but-correct answers. Do NOT swap in reward.answers_match.
# ---------------------------------------------------------------------------

def grade_eval(response: str, ground_truth: str) -> bool:
    """Measurement grader: extract_boxed + \\boxed-wrapped math_verify, no guard.
    Mirrors eval_vllm_math.py so val-core == the offline checkpoint number."""
    pred = extract_boxed(response)
    if pred is None:
        return False
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise ImportError(
            "math_verify is required for eval grading. "
            "Install math-verify + latex2sympy2_extended."
        ) from exc
    from .reward import _ensure_math_verify_thread_safe

    _ensure_math_verify_thread_safe()
    try:
        # verify(gold, target): ground truth first, prediction second. \boxed-wrap
        # both sides so math_verify's LaTeX extractor triggers on bare fragments.
        return bool(
            verify(
                parse("\\boxed{" + str(ground_truth) + "}"),
                parse("\\boxed{" + str(pred) + "}"),
            )
        )
    except Exception:
        return False


def grade(response: str, ground_truth: str, grader: str = "math_verify") -> bool:
    """Offline grading helper. ``math_verify`` = the eval grader (\\boxed wrap, no
    guard); ``sympy`` reuses the training-reward grader (reward.answers_match)."""
    if grader == "math_verify":
        return grade_eval(response, ground_truth)
    pred = extract_boxed(response)
    return pred is not None and answers_match(pred, ground_truth)


# ---------------------------------------------------------------------------
# Per-benchmark loaders (R-Zero source parity)
# ---------------------------------------------------------------------------

def _math500_cache_path() -> Path:
    base = os.environ.get(
        "RQ_MATH500_CACHE",
        os.path.expanduser("~/.cache/rq-evolve/math500"),
    )
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p / "math_500_test.csv"


def _load_math500_csv() -> list[dict[str, Any]]:
    import pandas as pd

    cache = _math500_cache_path()
    if not cache.exists():
        import requests

        logger.info("[math_eval] downloading MATH-500 CSV -> %s", cache)
        resp = requests.get(MATH500_CSV_URL, timeout=60)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    df = pd.read_csv(cache)
    return [row.to_dict() for _, row in df.iterrows()]


def _load_hf(hf_id: str, split: str | None = None, config_name: str | None = None):
    from datasets import load_dataset

    if config_name is not None:
        ds = load_dataset(hf_id, config_name)
        return ds[split or "train"]
    return load_dataset(hf_id, split=split)


def _load_benchmark_rows(name: str, inflate: bool = True) -> list[dict[str, str]]:
    if name == "math500":
        # The OpenAI simple-evals MATH-500 CSV "Answer" column holds the FULL
        # solution text with the result in \boxed{...}, NOT the bare answer.
        # Extract the boxed answer so grading compares answer-to-answer. Without
        # this the >200-char solution trips answers_match's length guard into a
        # normalized string-compare against the whole solution, and even a
        # perfect solver scores ~0 on math500.
        rows = [
            {
                "question": str(e["Question"]),
                "answer": extract_boxed(str(e["Answer"])) or str(e["Answer"]),
            }
            for e in _load_math500_csv()
        ]
    elif name == "amc23":
        ds = _load_hf("zwhe99/amc23", split="test")
        rows = [{"question": str(r["question"]), "answer": str(r["answer"])} for r in ds]
    elif name == "aime24":
        ds = _load_hf("HuggingFaceH4/aime_2024", split="train")
        rows = [{"question": str(r["problem"]), "answer": str(r["answer"])} for r in ds]
    elif name == "aime25":
        ds = _load_hf("yentinglin/aime_2025", split="train", config_name="default")
        rows = [{"question": str(r["problem"]), "answer": str(r["answer"])} for r in ds]
    elif name == "minerva_math":
        ds = _load_hf("zwhe99/simplerl-minerva-math", split="test")
        rows = [{"question": str(r["problem"]), "answer": str(r["answer"])} for r in ds]
    elif name == "olympiadbench":
        ds = _load_hf("zwhe99/simplerl-OlympiadBench", split="test")
        rows = []
        for r in ds:
            ans = r["final_answer"]
            if isinstance(ans, list):
                ans = ans[0] if ans else ""
            rows.append({"question": str(r["question"]), "answer": str(ans)})
    elif name == "gsm8k":
        ds = _load_hf("openai/gsm8k", split="test", config_name="main")
        rows = []
        for r in ds:
            ans = str(r["answer"])
            if "####" in ans:
                ans = ans.split("####")[-1].strip()
            rows.append({"question": str(r["question"]), "answer": ans})
    else:
        raise ValueError(f"unknown benchmark: {name!r}")

    if inflate and name in INFLATE_X32:
        rows = list(rows) * 32
    return rows


def load_math_benchmark(
    name: str,
    max_samples: int = -1,
    sample_seed: int = 42,
    inflate: bool = True,
) -> list[MathBenchmarkProblem]:
    rows = _load_benchmark_rows(name, inflate=inflate)
    if max_samples is not None and int(max_samples) > 0 and len(rows) > int(max_samples):
        rng = random.Random(sample_seed)
        rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), int(max_samples)))]

    problems = [
        MathBenchmarkProblem(benchmark=name, problem=str(q), answer=str(a), index=idx)
        for idx, item in enumerate(rows)
        if (q := item.get("question")) is not None and (a := item.get("answer")) is not None
    ]
    logger.info(
        "[math_eval] loaded %s: %d examples%s",
        name, len(problems), " (x32)" if (inflate and name in INFLATE_X32) else "",
    )
    return problems


# ---------------------------------------------------------------------------
# verl validation dataset
# ---------------------------------------------------------------------------

class MathBenchmarkDataset:
    """Emits the verl RLHF validation row shape, one ``data_source`` per file.

    Tokenization mirrors VerlDynamicDataset so train/val prompts are built the
    same way. ``reward_model.ground_truth`` carries the benchmark answer; verl's
    val_reward_fn scores it and groups metrics by ``data_source``.
    """

    def __init__(
        self,
        problems: list[MathBenchmarkProblem],
        tokenizer,
        *,
        max_prompt_length: int,
        data_source: str,
        system_prompt: str = MATH_EVAL_SYSTEM_PROMPT,
        truncation: str = "left",
    ) -> None:
        self.problems = list(problems)
        self.tokenizer = tokenizer
        self.max_prompt_length = int(max_prompt_length)
        self.data_source = data_source
        self.system_prompt = system_prompt or SOLVER_SYSTEM_PROMPT
        self.truncation = truncation

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, index: int) -> dict:
        import torch

        item = self.problems[index]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": item.problem},
        ]
        # verl 0.7.x AgentLoopWorker (SingleTurnAgentLoop) applies the chat
        # template itself, so the dataset only returns raw_prompt (chat msgs)
        # plus a dummy_tensor placeholder. Tensor fields like input_ids must
        # NOT be returned — _get_gen_batch leaves tensors in `batch` and the
        # agent-loop output's input_ids would then clash on union.
        return {
            "raw_prompt": messages,
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "data_source": self.data_source,
            "reward_model": {"ground_truth": item.answer},
            "extra_info": {
                "benchmark": self.data_source,
                "index": item.index,
                "problem": item.problem,
                # Tell reward.compute_score to skip grading on the agent loop's
                # (non-main) reward worker thread -- math_verify's SIGALRM timeout
                # can't fire there, so a pathological boxed answer would peg CPU
                # and stall vLLM (GPU 0%). RQValidatingTrainer._validate re-grades
                # these on the main thread instead. See eval_trainer.py.
                SKIP_WORKER_GRADE_KEY: True,
            },
            "index": item.index,
        }


def build_math_eval_val_dataset(math_eval_config, tokenizer, max_prompt_length: int):
    """Concatenate the configured benchmarks into one verl validation dataset.

    Returns None when disabled or when no benchmark loads — the caller then
    falls back to its default val dataset.
    """
    if not getattr(math_eval_config, "enabled", False):
        return None

    from torch.utils.data import ConcatDataset

    max_samples = int(getattr(math_eval_config, "max_samples_per_benchmark", -1))
    sample_seed = int(getattr(math_eval_config, "sample_seed", 42))
    # In-trainer periodic eval skips R-Zero's x32 AMC/AIME inflation by default:
    # at -1 samples the inflated set is ~4.6k prompts vs ~1.5k without. Grading
    # runs on the trainer's main thread now (RQValidatingTrainer._validate), so
    # this no longer risks the GPU-0% stall -- it's just an eval wall-time knob.
    # Offline final eval (load_math_benchmark with the default inflate=True) keeps
    # full R-Zero parity.
    inflate = bool(getattr(math_eval_config, "inflate_x32", False))

    datasets = []
    for name in math_eval_config.benchmarks:
        try:
            problems = load_math_benchmark(name, max_samples, sample_seed, inflate=inflate)
        except Exception as exc:
            logger.warning("[math_eval] skipping %s: %r", name, exc)
            continue
        if not problems:
            logger.warning("[math_eval] skipping %s: no examples", name)
            continue
        datasets.append(
            MathBenchmarkDataset(
                problems,
                tokenizer,
                max_prompt_length=max_prompt_length,
                data_source=name,
            )
        )

    if not datasets:
        logger.warning("[math_eval] no benchmarks loaded; validation disabled")
        return None
    total = sum(len(d) for d in datasets)
    logger.info(
        "[math_eval] validation dataset: %d benchmarks, %d problems",
        len(datasets), total,
    )
    return ConcatDataset(datasets)
