#!/usr/bin/env python
"""Judge comparison: one or more rubrics x {policy, gpt} model.

Every arm runs over ONE frozen corpus, so all the numbers are on the same items.
Reports agreement with the declared labels -- split by whether those labels are
hand-written (seeds, where a disagreement is the judge's) or Evolver-written
(children, where it is the question the gate exists to answer) -- and agreement
between the two models on the same rubric.

An arm is named ``<rubric>-<model>``. ``<model>`` is ``gpt`` or ``policy``;
``<rubric>`` is a key of ``--rubrics``, which maps a short name to a file in the
prompt-template directory. The default compares the shipped rubric on both
models:

    python scripts/build_judge_corpus.py
    python scripts/compare_judges.py

To measure a candidate rubric against the shipped one, drop the file in
``prompt_templates/`` and name both:

    python scripts/compare_judges.py \
        --rubrics shipped=mutation_judge_system_prompt.txt,strict=my_strict.txt \
        --arms shipped-gpt,shipped-policy,strict-gpt,strict-policy
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.openai_evaluator import (  # noqa: E402
    OpenAIEvaluatorConfig,
    evaluate_messages_with_openai,
    load_project_dotenv,
)
from rq_evolve.prompts import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT_FILE,
    build_judge_messages,
    parse_judge_verdict,
)
from rq_evolve.vllm_runtime import configure_vllm_sampler_backend  # noqa: E402

DEFAULT_RUBRICS = f"shipped={JUDGE_SYSTEM_PROMPT_FILE}"


def run_gpt(corpus, rubric_file, args):
    cfg = OpenAIEvaluatorConfig(
        model=args.gpt_model,
        reasoning_effort=args.effort,
        timeout_s=240.0,
        max_output_tokens=args.max_tokens,
    )

    def one(row):
        try:
            return evaluate_messages_with_openai(
                build_judge_messages(row["problem"], row["answer"],
                                     rubric_file=rubric_file), cfg
            )
        except Exception as exc:  # a dead call must not read as a refusal
            return f"__ERROR__ {type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        return list(pool.map(one, corpus))


def run_policy(corpus, rubric_file, llm, tok, args):
    from vllm import SamplingParams

    prompts = [
        tok.apply_chat_template(
            build_judge_messages(r["problem"], r["answer"], rubric_file=rubric_file),
            tokenize=False, add_generation_prompt=True,
        )
        for r in corpus
    ]
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
    return [o.outputs[0].text for o in llm.generate(prompts, params)]


def score(corpus, texts):
    rows = []
    for row, text in zip(corpus, texts):
        verdict = parse_judge_verdict(text)
        rows.append({
            **{k: row[k] for k in ("source", "name", "declared_group", "declared_skill")},
            "judged_group": verdict.group,
            "judged_skill": verdict.skill,
            "group_ok": verdict.group == row["declared_group"],
            "skill_ok": verdict.skill == row["declared_skill"],
            "both_ok": (verdict.group == row["declared_group"]
                        and verdict.skill == row["declared_skill"]),
            "skill_none": verdict.skill is None,
            "errored": text.startswith("__ERROR__"),
            "failure_reason": verdict.failure_reason,
        })
    return rows


def block(rows, source):
    sub = [r for r in rows if source is None or r["source"] == source]
    if not sub:
        return None
    n = len(sub)
    return {
        "n": n,
        "group": sum(r["group_ok"] for r in sub) / n,
        "skill": sum(r["skill_ok"] for r in sub) / n,
        "both": sum(r["both_ok"] for r in sub) / n,
        "skill_none": sum(r["skill_none"] for r in sub) / n,
        "errors": sum(r["errored"] for r in sub),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="rq_output/judge_corpus.json")
    ap.add_argument("--arms", default="shipped-gpt,shipped-policy")
    ap.add_argument("--rubrics", default=DEFAULT_RUBRICS,
                    help="comma-separated name=filename pairs")
    ap.add_argument("--gpt-model", default="gpt-5.4-mini")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--policy-model", default="/data1/yhoon113/qwen3-8b-base")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--gpu-util", type=float, default=0.4)
    ap.add_argument("--max-model-len", type=int, default=12000)
    ap.add_argument("--out", default="rq_output/judge_comparison.json")
    args = ap.parse_args()

    load_project_dotenv(ROOT)
    corpus = json.loads(Path(args.corpus).read_text())
    rubrics = dict(
        pair.split("=", 1) for pair in args.rubrics.split(",") if "=" in pair
    )
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        name = arm.rsplit("-", 1)[0]
        if name not in rubrics:
            raise SystemExit(
                f"arm {arm!r} needs rubric {name!r}; --rubrics defines "
                f"{sorted(rubrics)}"
            )
    results: dict[str, list[dict]] = {}

    for arm in [a for a in arms if a.endswith("-gpt")]:
        rubric = arm.rsplit("-", 1)[0]
        print(f"[judge] {arm}: {len(corpus)} items via {args.gpt_model} "
              f"(effort={args.effort})", flush=True)
        results[arm] = score(corpus, run_gpt(corpus, rubrics[rubric], args))

    policy_arms = [a for a in arms if a.endswith("-policy")]
    if policy_arms:
        configure_vllm_sampler_backend("pytorch")
        from vllm import LLM

        llm = LLM(model=args.policy_model, tokenizer=args.policy_model,
                  dtype="bfloat16", trust_remote_code=True,
                  gpu_memory_utilization=args.gpu_util,
                  max_model_len=args.max_model_len, enforce_eager=True)
        tok = llm.get_tokenizer()
        for arm in policy_arms:
            rubric = arm.rsplit("-", 1)[0]
            print(f"[judge] {arm}: {len(corpus)} items via {args.policy_model}",
                  flush=True)
            results[arm] = score(corpus, run_policy(corpus, rubrics[rubric],
                                                    llm, tok, args))

    head = (f"\n{'arm':<16}{'items':>7}{'GROUP':>8}{'SKILL':>8}{'BOTH':>8}"
            f"{'SKILL=none':>12}{'err':>5}")
    for label, src in (("ALL", None), ("seeds (ground truth)", "seed"),
                       ("children (self-labelled)", "child")):
        print(f"\n=== {label} ===")
        print(head.strip("\n"))
        print("-" * 64)
        for arm in arms:
            b = block(results.get(arm, []), src)
            if not b:
                continue
            print(f"{arm:<16}{b['n']:>7}{b['group']:>8.0%}{b['skill']:>8.0%}"
                  f"{b['both']:>8.0%}{b['skill_none']:>12.0%}{b['errors']:>5}")

    # inter-model agreement on the same rubric
    print("\n=== inter-model agreement (same rubric, same item) ===")
    for rubric in rubrics:
        a, b = f"{rubric}-gpt", f"{rubric}-policy"
        if a not in results or b not in results:
            continue
        pairs = list(zip(results[a], results[b]))
        n = len(pairs)
        g = sum(x["judged_group"] == y["judged_group"] for x, y in pairs) / n
        s = sum(x["judged_skill"] == y["judged_skill"] for x, y in pairs) / n
        both = sum(x["judged_group"] == y["judged_group"]
                   and x["judged_skill"] == y["judged_skill"] for x, y in pairs) / n
        print(f"  {rubric:<9} GROUP {g:.0%}   SKILL {s:.0%}   BOTH {both:.0%}   (n={n})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
