#!/usr/bin/env python
"""Fast prompt-iteration probe: generate children, judge them, print a verdict.

Talks to a resident `vllm serve` instance so editing a template and re-running
costs seconds instead of the 31s engine boot plus 167s seed bootstrap that
scripts/sample_evolve_vllm.py pays. No R_Q, no archive, no solver rollouts --
this measures the mutation prompt only.

    vllm serve <model> --served-model-name qwen3-8b-base --port 8077
    python scripts/probe_mutation.py --n 8

What it checks per child, in the order the real pipeline does:
  parse -> lint -> execute seeds 0..4 -> label vocabulary -> operator contract
  -> the generator contract the prompt asks for (guards, assert, MAX_ATTEMPTS)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.code_utils import (  # noqa: E402
    extract_generator_code,
    lint_generator_source,
    lint_problem_instance,
)
from rq_evolve.ast_contract import check_generator_contract  # noqa: E402
from rq_evolve.concepts import GROUPS, SKILLS  # noqa: E402
from rq_evolve.evolution import RQEvolver  # noqa: E402
from rq_evolve.program import ProblemProgram  # noqa: E402
from rq_evolve.prompts import (  # noqa: E402
    MUTATION_OP,
    build_mutation_task,
    mutation_system_prompt,
)


def parents(seed_dir: Path) -> list[ProblemProgram]:
    out = []
    for path in sorted(seed_dir.glob("*.py")):
        program = ProblemProgram.from_file(path, generation=0)
        if program.declared_group() and program.declared_skill():
            out.append(program)
    return out


def contract_flags(source: str) -> dict:
    """Shape counters plus the real structural verdict.

    The counters stay because they are cheap descriptive statistics of what the
    model wrote. ``ast_contract`` is the gate the pipeline actually applies, so
    a probe run reports exactly what a training run would reject.
    """
    return {
        "MAX_ATTEMPTS": "MAX_ATTEMPTS" in source,
        "guards": len(re.findall(r"^\s+continue$", source, re.M)),
        "assert": bool(re.search(r"^\s+assert ", source, re.M)),
        "rng": "random.Random(seed)" in source,
        "sorted": bool(re.search(r"sorted\(", source)),
        "ast_contract": [str(f) for f in check_generator_contract(source)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8077/v1")
    ap.add_argument("--model", default="qwen3-8b-base")
    ap.add_argument("--n", type=int, default=8, help="children to generate")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--seed-dir", default="seed_programs")
    ap.add_argument("--out", default="rq_output/probe")
    ap.add_argument("--show-source", type=int, default=0, help="print N full sources")
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="none")
    pool = parents(Path(args.seed_dir))
    # Deterministic round-robin over the parents: the same command re-run after
    # a template edit hits the same parents, so a change in the output is a
    # change in the prompt and not a change in the sample.
    jobs = [pool[i % len(pool)] for i in range(args.n)]

    system = mutation_system_prompt()
    evolver = RQEvolver(archive=None, backend=None)
    rows, sigs = [], {}

    print(f"[probe] {args.n} children, temp={args.temperature}, model={args.model}")
    for i, parent in enumerate(jobs):
        op = MUTATION_OP
        task = build_mutation_task(parent, temperature=args.temperature,
                                   top_p=args.top_p)
        reply = client.chat.completions.create(
            model=args.model,
            messages=task.messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        text = reply.choices[0].message.content or ""
        source = extract_generator_code(text)
        row = {
            "i": i, "op": op,
            "parent": (parent.declared_group(), parent.declared_skill()),
            "finish": reply.choices[0].finish_reason,
            "tokens": reply.usage.completion_tokens,
            "raw": text,
        }
        if source is None:
            row["verdict"] = "no_code"
            rows.append(row)
            continue

        sigs.setdefault(hashlib.md5(source.encode()).hexdigest()[:8], []).append(i)
        child = ProblemProgram(source_code=source, metadata={"op": op})
        row["source"] = source
        row["labels"] = (child.declared_group(), child.declared_skill())
        row["contract"] = contract_flags(source)

        base = lint_generator_source(source)
        if base:
            row["verdict"] = "lint_failed"
            row["reason"] = "; ".join(base[:2])
            rows.append(row)
            continue

        inst, why = evolver.verify_program(child)
        if inst is None:
            row["verdict"] = "verify_failed"
            row["reason"] = why
            rows.append(row)
            continue

        # The operator contract is retired: nothing requires a particular
        # label move any more. What is worth flagging is a child that simply
        # repeats BOTH parent labels -- the mutation that did not happen.
        repeats_parent = (
            child.declared_group() == parent.declared_group()
            and child.declared_skill() == parent.declared_skill()
        )
        row["verdict"] = "ok"
        row["repeats_parent_labels"] = repeats_parent
        row["problem"] = " ".join(inst.problem.split())
        row["answer"] = inst.answer
        row["problem_chars"] = len(inst.problem)
        rows.append(row)

    # ---- report -----------------------------------------------------------
    print()
    for r in rows:
        mark = {"ok": "OK  ", "no_code": "CODE", "lint_failed": "LINT",
                "verify_failed": "EXEC", "contract_failed": "CTRT"}[r["verdict"]]
        lab = "/".join(str(x) for x in r.get("labels", ("?", "?")))
        print(f"[{r['i']}] {mark} {r['op']:<11} {r['parent'][0]}/{r['parent'][1]} -> {lab}")
        if r.get("reason"):
            print(f"      reason: {str(r['reason'])[:120]}")
        if r.get("problem"):
            print(f"      ({r['problem_chars']}자) {r['problem'][:130]}")
            print(f"      답: {r['answer'][:30]}")
        if "contract" in r:
            c = r["contract"]
            print(f"      계약: MAX_ATTEMPTS={'O' if c['MAX_ATTEMPTS'] else '-'} "
                  f"guards={c['guards']} assert={'O' if c['assert'] else '-'} "
                  f"rng={'O' if c['rng'] else '-'} sorted={'O' if c['sorted'] else '-'} "
                  f"| {r['tokens']} tok {r['finish']}")

    ok = [r for r in rows if r["verdict"] == "ok"]
    print(f"\n{'='*70}")
    print(f"통과 {len(ok)}/{len(rows)}", end="   ")
    for v in ("no_code", "lint_failed", "verify_failed", "contract_failed"):
        c = sum(1 for r in rows if r["verdict"] == v)
        if c:
            print(f"{v}={c}", end=" ")
    print()
    dups = {k: v for k, v in sigs.items() if len(v) > 1}
    if dups:
        print(f"중복 생성: {list(dups.values())}")
    if ok:
        agg = {k: sum(1 for r in ok if (r['contract'][k] if isinstance(r['contract'][k], bool) else r['contract'][k] > 0))
               for k in ("MAX_ATTEMPTS", "guards", "assert", "rng", "sorted")}
        print(f"통과분 계약 이행: " + "  ".join(f"{k}={v}/{len(ok)}" for k, v in agg.items()))
        print(f"문제문 길이: {[r['problem_chars'] for r in ok]}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    for r in rows[: args.show_source]:
        if r.get("source"):
            print(f"\n{'-'*70}\n[{r['i']}] source:\n{r['source']}")
    print(f"\n[probe] {out}/probe.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
