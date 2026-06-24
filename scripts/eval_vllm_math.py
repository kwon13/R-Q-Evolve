"""Offline checkpoint eval on the 6 R-Zero math benchmarks — R-Zero parity.

This is the R-Q-Evolve replacement for evo-sample/scripts/eval_vllm_math.py.
It is deliberately aligned with R-Zero/evaluation so the offline number matches
R-Zero's reported numbers as closely as possible:

  * Dataset loading: reuses ``rq_evolve.math_eval.load_math_benchmark`` (same
    R-Zero sources — OpenAI MATH-500 CSV, zwhe99 / HuggingFaceH4 / yentinglin).
  * Prompt: R-Zero system prompt + chat template (generate.py).
  * Generation: greedy (temperature 0.0, top_p 1.0, n 1), max_tokens 4096,
    stop_token_ids=[eos]  — matched to R-Zero/evaluation/generate.py.
  * Grading: an EXACT copy of R-Zero/evaluation/datasets_loader.py — first
    ``\\boxed`` via the line regex, then ``verify(parse(gt), parse(pred))`` on
    the RAW strings (NO ``\\boxed`` wrapping). This is NOT rq_evolve.math_eval
    .grade_eval (which \\boxed-wraps both sides); the offline number therefore
    follows R-Zero, while the in-trainer val-core grader is left untouched.
  * GPT-4o re-check: a port of R-Zero/evaluation/results_recheck.py — for every
    example scored < 0.5, ask gpt-4o (temperature 0.1) whether the model's full
    response matches the ground truth, and bump the score to 1 on a reply that
    contains "yes". The OpenAI key is loaded from R-Q-Evolve/.env.

Usage mirrors the old script's CLI so analysis/eval_steps_fanout.sh keeps working.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
from math_verify import parse, verify

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.math_eval import (  # noqa: E402
    MATH_EVAL_SYSTEM_PROMPT,
    MathBenchmarkProblem,
    load_math_benchmark,
)

logger = logging.getLogger("eval_vllm_math")


# ---------------------------------------------------------------------------
# .env loader (no python-dotenv dependency). Values are NEVER logged.
# ---------------------------------------------------------------------------

def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a KEY=VALUE .env file (does not overwrite)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# R-Zero EXACT grading (port of R-Zero/evaluation/datasets_loader.py)
# ---------------------------------------------------------------------------

# datasets_loader.py:9 — first \boxed, content up to a newline, via re.search.
ANSWER_PATTERN_BOXED = r"(?i)\\boxed\s*{([^\n]+)}"


def rzero_extract_answer(response: str) -> str | None:
    """datasets_loader.DatasetHandler.extract_answer — first regex match or None."""
    try:
        return re.search(ANSWER_PATTERN_BOXED, response).group(1)
    except Exception:
        return None


def rzero_compare_answer(response: str, answer: str) -> bool:
    """datasets_loader.DatasetHandler.compare_answer — byte-faithful to R-Zero.

    Note the original quirk: ``response_answer`` is ``str()``-cast (so a failed
    extraction becomes the literal "None") BEFORE the ``is None`` check, so that
    check never fires and a missed box falls through to ``verify(parse(gt),
    parse("None"))`` -> almost always False. Replicated exactly so the score
    matches R-Zero, including the edge case.
    """
    
    response_answer = rzero_extract_answer(response)
    answer = str(answer)
    response_answer = str(response_answer)
    return bool(verify(parse(answer), parse(response_answer)))


def rzero_score(response: str, answer: str) -> int:
    """1 if correct else 0 (datasets_loader.get_score per-example)."""
    return 1 if rzero_compare_answer(response, answer) else 0


# ---------------------------------------------------------------------------
# GPT-4o re-check (port of R-Zero/evaluation/results_recheck.py)
# ---------------------------------------------------------------------------

def _gpt_user_prompt(answer: str, response: str) -> str:
    """R-Zero results_recheck.py:process_example user content, verbatim.

    R-Zero calls process_example(results[i]['answer'], results[i]['response']):
    the ground-truth answer is passed as ``answer`` and the model's FULL response
    as ``response`` (the text then labels them the opposite way round — kept as-is
    for an exact match).
    """
    return (
        f"Hi, there is a answer: {answer}\n\n, and the ground truth answer is: "
        f"{response}\n\n, please check whether the answer is correct or not, and "
        f"return the **only** Yes or No."
    )


class GPTRechecker:
    """gpt-4o yes/no equivalence judge. Hardcoded model=gpt-4o, temperature=0.1.

    Results are cached by (ground_truth, response) so the ×32-inflated duplicates
    (identical under greedy decoding) cost a single API call, not 32.
    """

    def __init__(self, workers: int = 8) -> None:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found (loaded from R-Q-Evolve/.env). "
                "Set it in .env or pass --no_gpt_recheck."
            )
        # base_url honored if the .env defines one (azure/gateway); else default.
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.workers = max(1, int(workers))
        self._cache: dict[tuple[str, str], bool] = {}

    def _call_once(self, answer: str, response: str) -> str:
        """One gpt-4o call; returns the raw reply text, or 'No' on any failure
        (results_recheck.py returns 'No' in its except branch)."""
        try:
            messages = [
                {"role": "system", "content": "You are a math answer checker."},
                {"role": "user", "content": _gpt_user_prompt(answer, response)},
            ]
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1,
            )
            return resp.choices[0].message.content or "No"
        except Exception as exc:  # noqa: BLE001 — mirror R-Zero's swallow-and-No
            logger.warning("[gpt-recheck] call failed: %r", exc)
            return "No"

    def judge_yes(self, ground_truth: str, response: str) -> bool:
        """True iff gpt-4o reply contains 'yes' (R-Zero: `'yes' in text.lower()`)."""
        key = (str(ground_truth), str(response))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        text = self._call_once(str(ground_truth), str(response))
        verdict = "yes" in text.lower()
        self._cache[key] = verdict
        return verdict

    def recheck(self, rows: list[dict[str, Any]]) -> int:
        """Bump score 0 -> 1 for rows the judge approves. Returns #flips.

        R-Zero only rechecks examples with score < 0.5; here every row is
        score 0 or 1, so this targets exactly the score-0 rows. The judge sees
        the ground truth and the model's FULL response (R-Zero parity).
        """
        targets = [r for r in rows if r["score"] < 0.5]
        if not targets:
            return 0
        # Resolve unique (gt, response) pairs concurrently; cache fills in-thread.
        uniq = list({(r["answer"], r["response"]) for r in targets})

        def _resolve(pair: tuple[str, str]) -> tuple[tuple[str, str], bool]:
            gt, resp = pair
            return pair, self.judge_yes(gt, resp)

        verdicts: dict[tuple[str, str], bool] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for pair, verdict in pool.map(_resolve, uniq):
                verdicts[pair] = verdict

        flips = 0
        for r in targets:
            if verdicts.get((r["answer"], r["response"]), False):
                r["score"] = 1
                r["gpt_rechecked"] = True
                flips += 1
        return flips


# ---------------------------------------------------------------------------
# Prompt building (R-Zero generate.py)
# ---------------------------------------------------------------------------

def _build_prompt(tokenizer, problem: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            add_special_tokens=True,
        )
    # Base-model fallback (generate.py:33): system prompt repeated at the end.
    # return f"system: {system_prompt}\nuser: {problem}\n{system_prompt}"
    
    # Spiral
    return f"<|im_start|>user\n{system_prompt}\nQuestion: {problem}<|im_end|>\n<|im_start|>assistant\n"
    
    
def _parse_benchmark_name(spec: str) -> str:
    """Accept NAME=HF_ID[:SPLIT] / NAME:HF_ID[:SPLIT] / NAME — return NAME only.
    Sources are determined by name inside math_eval (R-Zero datasets_loader)."""
    if "=" in spec:
        return spec.split("=", 1)[0].strip()
    if ":" in spec:
        return spec.split(":", 1)[0].strip()
    return spec.strip()


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def _make_sampling_params(args, tokenizer):
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "n": args.n,
    }
    if getattr(tokenizer, "eos_token_id", None) is not None:
        kwargs["stop_token_ids"] = [int(tokenizer.eos_token_id)]
    return SamplingParams(**kwargs)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    names = [_parse_benchmark_name(s) for s in (args.benchmark or [])]
    if not names:
        names = ["math500", "gsm8k", "amc23", "aime24", "aime25", "minerva_math", "olympiadbench"]

    problems_by_benchmark: dict[str, list[MathBenchmarkProblem]] = {}
    for name in names:
        problems = load_math_benchmark(
            name=name,
            max_samples=int(args.max_samples),
            sample_seed=int(args.sample_seed),
            inflate=bool(args.inflate_x32),
        )
        if problems:
            problems_by_benchmark[name] = problems
        else:
            logger.warning("Skipping %s: no problems loaded", name)
    if not problems_by_benchmark:
        raise RuntimeError("No benchmark problems loaded.")

    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    sampling_params = _make_sampling_params(args, tokenizer)

    recheck = None
    if args.gpt_recheck:
        try:
            recheck = GPTRechecker(workers=args.gpt_workers)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[gpt-recheck] disabled: %r", exc)
            recheck = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "details.jsonl"

    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "tokenizer": tokenizer_name,
        "grader": "r-zero-exact (first-boxed regex + raw verify)",
        "gpt_recheck": bool(recheck is not None),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "n": args.n,
        },
        "benchmarks": {},
    }

    with details_path.open("w", encoding="utf-8") as details_f:
        for benchmark, problems in problems_by_benchmark.items():
            logger.info("Evaluating %s (%d problems)", benchmark, len(problems))
            prompts = [
                _build_prompt(tokenizer, item.problem, args.system_prompt)
                for item in problems
            ]
            start = time.time()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=not args.no_tqdm)

            rows: list[dict[str, Any]] = []
            for item, output in zip(problems, outputs):
                response = output.outputs[0].text  # n=1, greedy (generate.py)
                rows.append(
                    {
                        "benchmark": benchmark,
                        "index": item.index,
                        "problem": item.problem,
                        "answer": item.answer,
                        "response": response,
                        "score": rzero_score(response, item.answer),
                    }
                )

            n = len(rows)
            pre_correct = sum(r["score"] for r in rows)
            gpt_flips = recheck.recheck(rows) if recheck is not None else 0
            post_correct = sum(r["score"] for r in rows)

            for r in rows:
                details_f.write(json.dumps(r, ensure_ascii=False) + "\n")

            summary["benchmarks"][benchmark] = {
                "num_examples": n,
                "pass_at_1": (post_correct / n) if n else 0.0,
                "pass_at_1_pre_gpt": (pre_correct / n) if n else 0.0,
                "gpt_flips": gpt_flips,
                "elapsed_sec": round(time.time() - start, 1),
            }
            logger.info(
                "%s: pass@1=%.2f%% (pre-gpt %.2f%%, +%d flips) n=%d",
                benchmark,
                100.0 * summary["benchmarks"][benchmark]["pass_at_1"],
                100.0 * summary["benchmarks"][benchmark]["pass_at_1_pre_gpt"],
                gpt_flips,
                n,
            )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--config", default="")  # accepted for CLI compat; ignored
    p.add_argument("--output_dir", default="rq_output/vllm_math_eval")
    p.add_argument("--benchmark", action="append", help="NAME[=HF_ID[:SPLIT]]; repeatable")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument(
        "--inflate_x32",
        action="store_true",
        help="R-Zero ×32-inflates AMC/AIME. Identical pass@1 under greedy, 32× "
        "more generation; default OFF (same number, far cheaper).",
    )
    # Generation — defaults match R-Zero/evaluation/generate.py.
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--tensor_parallel_size", "--tp", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--system_prompt", default=MATH_EVAL_SYSTEM_PROMPT)
    p.add_argument("--no_tqdm", action="store_true")
    # GPT-4o re-check (R-Zero results_recheck.py). On by default for R-Zero parity.
    p.add_argument("--gpt_recheck", dest="gpt_recheck", action="store_true", default=True)
    p.add_argument("--no_gpt_recheck", dest="gpt_recheck", action="store_false")
    p.add_argument("--gpt_workers", type=int, default=8)
    p.add_argument("--log_level", default="INFO")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Load OpenAI/HF/W&B keys from R-Q-Evolve/.env (values never printed).
    _load_dotenv(ROOT / ".env")
    summary = evaluate(args)
    accs = [b["pass_at_1"] for b in summary["benchmarks"].values()]
    if accs:
        logger.info("AVG pass@1 = %.2f%%", 100.0 * sum(accs) / len(accs))


if __name__ == "__main__":
    main()
