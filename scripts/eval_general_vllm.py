#!/usr/bin/env python
"""General-domain reasoning eval: MMLU-Pro, SuperGPQA, BBEH.

A port of R-Zero/evaluation/eval_{mmlupro,supergpqa,bbeh}.py into the shape the
rest of this repo already uses -- one benchmark per process, one GPU, writing
``summary.json`` + ``details.jsonl`` into ``--output_dir`` -- so the same
step x benchmark fan-out that drives the math eval drives this one.

The graders are carried over unchanged, including their quirks, because the
point is comparability with R-Zero's published numbers:

  * answer extraction is last-\\boxed, then "Final Answer:" / "The answer is:";
  * MMLU-Pro and SuperGPQA scan the extracted string for the FIRST option letter
    that appears anywhere in it, and fall back to a RANDOM letter when nothing
    parses. That random fallback is why ``--seed`` exists here: R-Zero's scripts
    leave it unseeded, so their multiple-choice numbers are not reproducible to
    the last digit. Seeding costs nothing and makes a re-run comparable to
    itself;
  * BBEH keeps its own fuzzy_match (bracket/quote/number normalisation).

Two deliberate departures from the originals:

  * ``tensor_parallel_size`` is an argument rather than a hardcoded 4, so a
    checkpoint can be evaluated on one GPU and the fan-out can run several
    checkpoints at once;
  * ``--max_samples`` subsamples deterministically. The full suite is ~42k
    questions (MMLU-Pro 12k + SuperGPQA 26k + BBEH 4.5k); across 8 checkpoints
    that is ~340k generations at 8192 max tokens, which is days of GPU time.
    Sampling is stratified by category so the per-category report stays
    meaningful, and the sampled indices are recorded in summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logger = logging.getLogger("eval_general")

# name -> (hf dataset id, split, category field, answer field)
BENCHMARKS: dict[str, dict[str, Any]] = {
    "mmlupro": {
        "hf_id": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "category_field": "category",
        "answer_field": "answer",
        "kind": "mcq",
        "categories": [
            "computer science", "math", "chemistry", "engineering", "law",
            "biology", "health", "physics", "business", "philosophy",
            "economics", "other", "psychology", "history",
        ],
    },
    "supergpqa": {
        "hf_id": "m-a-p/SuperGPQA",
        "split": "train",
        "category_field": "discipline",
        "answer_field": "answer_letter",
        "kind": "mcq",
        "categories": [
            "Engineering", "Medicine", "Science", "Philosophy",
            "Military Science", "Economics", "Management", "Sociology",
            "Literature and Arts", "History", "Agronomy", "Law", "Education",
        ],
    },
    "bbeh": {
        "hf_id": "MrLight/bbeh-eval",
        "split": "train",
        "category_field": "task",
        "answer_field": "answer",
        "kind": "free",
        "categories": None,  # discovered from the split
    },
}

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

MCQ_SUFFIX = (
    "\nPlease reason step by step, and put your final answer option within "
    "\\boxed{}. Only put the option letter in the box, e.g. \\boxed{A}. "
    "There is only one correct answer."
)
FREE_SUFFIX = (
    "\nPlease reason step by step, and put your final answer option within "
    "\\boxed{}."
)


# --------------------------------------------------------------------------
# answer extraction (verbatim from R-Zero/evaluation/eval_*.py)
# --------------------------------------------------------------------------


def extract_last_boxed(text: str) -> str | None:
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    return matches[-1].group(1) if matches else None


def extract_last_final_answer(text: str) -> str | None:
    m1 = list(re.finditer(r"Final Answer:((?:[^<]|<[^<])*?)\n", text))
    m2 = list(re.finditer(r"The answer is:((?:[^<]|<[^<])*?)\n", text))
    if m1:
        return m1[-1].group(1)
    if m2:
        return m2[-1].group(1)
    return None


def extract_solution(solution_str: str) -> str | None:
    if "<|im_start|>user" in solution_str:
        model_output = re.sub(
            r"^.*?<\|im_start\|>assistant",
            "<|im_start|>assistant",
            solution_str,
            flags=re.DOTALL,
            count=1,
        )
    elif "Assistant:" in solution_str:
        model_output = solution_str.split("Assistant:")[-1].strip()
    else:
        model_output = solution_str

    for stop_word in ("</s>", "<|im_end|>", "<|endoftext|>"):
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()

    boxed = extract_last_boxed(model_output)
    return boxed if boxed else extract_last_final_answer(model_output)


def form_options(options: list) -> str:
    out = "Options are:\n"
    for opt, letter in zip(options, OPTION_LETTERS):
        out += f"({letter}): {opt}\n"
    return out


def get_prediction(output: str, rng: random.Random) -> tuple[str, bool]:
    """(letter, parsed). ``parsed`` is False when the random fallback fired."""
    solution = extract_solution(output)
    if solution is None:
        return rng.choice(OPTION_LETTERS), False
    for option in OPTION_LETTERS:
        if option in solution:
            return option, True
    return rng.choice(OPTION_LETTERS), False


# --------------------------------------------------------------------------
# BBEH grading (verbatim from R-Zero/evaluation/eval_bbeh.py)
# --------------------------------------------------------------------------


def strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response


def extract_answer(sample: str | None) -> str:
    if sample is None:
        sample = ""
    answer = sample
    for prefix in (
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is ",
    ):
        if prefix in answer:
            answer = answer.split(prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
    if prediction == reference:
        return True
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True
    return False


def evaluate_correctness(sample: str | None, reference: str) -> bool:
    prediction = extract_answer((sample or "").strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[:-1] if prediction.endswith(".") else prediction
    return fuzzy_match(prediction, reference.strip().lower().replace(", ", ","))


# --------------------------------------------------------------------------


def stratified_sample(
    entries: list[dict], category_field: str, limit: int, seed: int
) -> list[dict]:
    """Take ``limit`` entries, spread proportionally over categories.

    Category-blind sampling would let one large category dominate and make the
    per-category report unreadable, and BBEH's tasks differ enough in
    difficulty that the mix has to be held fixed for two checkpoints to be
    comparable.
    """
    if limit <= 0 or limit >= len(entries):
        return entries
    by_cat: dict[str, list[dict]] = {}
    for entry in entries:
        by_cat.setdefault(str(entry.get(category_field)), []).append(entry)

    rng = random.Random(seed)
    picked: list[dict] = []
    total = len(entries)
    for category in sorted(by_cat):
        pool = by_cat[category]
        take = max(1, round(limit * len(pool) / total))
        take = min(take, len(pool))
        picked.extend(rng.sample(pool, take))
    # Rounding up per category can overshoot; trim deterministically.
    if len(picked) > limit:
        picked = rng.sample(picked, limit)
    return picked


def build_entries(spec: dict, name: str, args) -> list[dict]:
    import datasets

    logger.info("loading %s (%s)", name, spec["hf_id"])
    split = datasets.load_dataset(spec["hf_id"], split=spec["split"])
    entries = [dict(row) for row in split]

    categories = spec["categories"]
    if categories is None:
        categories = sorted({str(e[spec["category_field"]]) for e in entries})
    else:
        entries = [
            e for e in entries if str(e[spec["category_field"]]) in set(categories)
        ]
    logger.info("%s: %d entries over %d categories", name, len(entries), len(categories))

    if args.max_samples > 0:
        entries = stratified_sample(
            entries, spec["category_field"], args.max_samples, args.sample_seed
        )
        logger.info("%s: sampled down to %d entries", name, len(entries))
    return entries


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=12000)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 = the full split. Stratified by category when set.",
    )
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seeds the random-letter fallback R-Zero leaves unseeded",
    )
    p.add_argument("--no_tqdm", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # FlashInfer 0.6.x JIT-compiles CUDA 12-only sampling kernels and fails on a
    # host whose system nvcc is 11.8. Same guard the math eval carries.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    spec = BENCHMARKS[args.benchmark]
    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    entries = build_entries(spec, args.benchmark, args)

    suffix = MCQ_SUFFIX if spec["kind"] == "mcq" else FREE_SUFFIX
    prompts: list[str] = []
    for entry in entries:
        query = entry["question"] + "\n"
        if spec["kind"] == "mcq":
            query += form_options(entry["options"]) + "\n"
        content = query + suffix
        if tokenizer.chat_template:
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        else:
            # Base checkpoints have no chat template; R-Zero falls back to this
            # bare "user: " prefix and we keep it for parity.
            prompts.append("user: " + content)

    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=not args.no_tqdm)

    rng = random.Random(args.seed)
    per_category: dict[str, list[int]] = {}
    correct = 0
    unparsed = 0
    truncated = 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for entry, output in zip(entries, outputs):
            response = output.outputs[0].text
            category = str(entry[spec["category_field"]])
            reference = str(entry[spec["answer_field"]])

            if spec["kind"] == "mcq":
                prediction, parsed = get_prediction(response, rng)
                is_correct = prediction == reference
            else:
                prediction = extract_answer((extract_solution(response) or "").strip())
                parsed = extract_solution(response) is not None
                is_correct = evaluate_correctness(extract_solution(response), reference)

            bucket = per_category.setdefault(category, [0, 0])
            bucket[0 if is_correct else 1] += 1
            correct += int(is_correct)
            unparsed += int(not parsed)
            hit_cap = output.outputs[0].finish_reason == "length"
            truncated += int(hit_cap)

            f.write(
                json.dumps(
                    {
                        "benchmark": args.benchmark,
                        "category": category,
                        "question": entry["question"],
                        "reference": reference,
                        "prediction": prediction,
                        "parsed": parsed,
                        "score": int(is_correct),
                        "truncated": hit_cap,
                        "response": response,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    n = len(entries)
    macro = (
        sum(c / (c + w) for c, w in per_category.values() if c + w) / len(per_category)
        if per_category
        else 0.0
    )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "tokenizer": tokenizer_name,
        "grader": f"r-zero-{args.benchmark}",
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "n": 1,
        },
        "max_samples": args.max_samples,
        "sample_seed": args.sample_seed,
        "fallback_seed": args.seed,
        "benchmarks": {
            args.benchmark: {
                # Same key the math eval uses, so one aggregator reads both.
                "pass_at_1": (correct / n) if n else 0.0,
                "macro_avg": macro,
                "num_examples": n,
                # An unparsed multiple-choice answer still scores 1/10 of the
                # time through the random fallback; without this the accuracy
                # cannot be told apart from guessing.
                "unparsed": unparsed,
                "unparsed_rate": (unparsed / n) if n else 0.0,
                "truncated": truncated,
                "truncated_rate": (truncated / n) if n else 0.0,
                "per_category": {
                    c: {
                        "correct": v[0],
                        "total": v[0] + v[1],
                        "accuracy": v[0] / (v[0] + v[1]) if v[0] + v[1] else 0.0,
                    }
                    for c, v in sorted(per_category.items())
                },
            }
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        "%s  micro=%.2f%%  macro=%.2f%%  n=%d  unparsed=%.1f%%  truncated=%.1f%%",
        args.benchmark,
        100 * correct / n if n else 0.0,
        100 * macro,
        n,
        100 * unparsed / n if n else 0.0,
        100 * truncated / n if n else 0.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
