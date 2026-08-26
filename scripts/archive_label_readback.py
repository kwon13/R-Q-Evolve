#!/usr/bin/env python3
"""Do the archive's own (GROUP, SKILL) coordinates survive an independent reader?

A MAP cell is a claim about what a program exercises, and the archive takes that
claim from the program's self-declared labels. Coverage therefore measures
declarations, not behaviours -- if the declarations are wrong, a "new cell" is a
relabelling and the grid is telling the curriculum something false.

This asks a judge that sees ONLY the rendered problem and its reference answer
(never the source, never the declared labels) to assign the same two axes, and
reports the agreement. It reuses relabel_valid_rq_group_skill.py's SYSTEM_PROMPT
and its exact request shape, so the numbers line up with that pipeline's
(prompt_hash 8660d4a4b83b, reasoning_effort none). VALIDITY is not asked -- that
is the separate validity_recheck stage.
"""
import argparse, importlib.util, json, os, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rq_evolve.program import ProblemProgram  # noqa: E402
from rq_evolve.concepts import GROUPS, SKILLS  # noqa: E402


def _load_relabel(path: Path):
    spec = importlib.util.spec_from_file_location("_relabel", str(path))
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["_relabel"]
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    finally:
        sys.argv = argv
    return mod


def collect(archive: Path, seed: int) -> list[dict]:
    rows = []
    for c in json.load(open(archive))["champions"]:
        prog = ProblemProgram(source_code=c["source_code"], program_id=c["program_id"])
        inst = prog.execute(seed=seed)
        if inst is None:
            continue
        rows.append({
            "idx": len(rows),
            "program_id": c["program_id"],
            "source_id": c["program_id"],
            "question": inst.problem,
            "reference_answer": str(inst.answer),
            "input_fingerprint": "",
            "old_group": GROUPS[c["niche_group"]],
            "old_skill": SKILLS[c["niche_skill"]],
            "s_hat": c.get("s_hat"),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True, action="append",
                    help="archive*.json; repeat to compare several")
    ap.add_argument("--label", action="append", default=None)
    ap.add_argument("--relabel-script", type=Path,
                    default=Path("/data1/yhoon113/relabel_valid_rq_group_skill.py"))
    ap.add_argument("--env-file", type=Path, default=ROOT / ".env")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--effort", default="none")
    ap.add_argument("--max-completion-tokens", type=int, default=800)
    ap.add_argument("--api-url", default="https://api.openai.com/v1/chat/completions")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "rq_output/gate_experiment/archive_readback.json")
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args()

    R = _load_relabel(args.relabel_script)
    labels = args.label or [a.parent.parent.name for a in args.archive]
    all_rows, out_all = {}, {}

    for name, arch in zip(labels, args.archive):
        rows = collect(arch, args.seed)
        all_rows[name] = rows
        print(f"{name}: {len(rows)} champions from {arch}")

    total = sum(len(v) for v in all_rows.values())
    est = sum(R.estimate_tokens(v) for v in all_rows.values())
    print(f"\nprompt_hash={R.PROMPT_HASH}  model={args.model}  effort={args.effort}")
    print(f"API calls={total:,}  estimated input tokens~{est:,}  "
          f"output cap={total*args.max_completion_tokens:,}")
    if args.estimate_only:
        return

    judge = R.GroupSkillJudge(R.load_api_key(args.env_file), args.model,
                              args.effort, args.max_completion_tokens, args.api_url)

    for name, rows in all_rows.items():
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(judge, rows))
        out_all[name] = res
        ok = [r for r in res if not r.get("error")]
        g = sum(1 for r in ok if r["group"] == r["old_group"])
        s = sum(1 for r in ok if r["skill"] == r["old_skill"])
        both = sum(1 for r in ok if r["group"] == r["old_group"] and r["skill"] == r["old_skill"])
        n = max(len(ok), 1)
        print(f"\n=== {name} ===  judged {len(ok)}/{len(res)}"
              + (f"  errors {len(res)-len(ok)}" if len(res) != len(ok) else ""))
        print(f"  GROUP  {g}/{len(ok)} ({100*g/n:.0f}%)"
              f"   SKILL  {s}/{len(ok)} ({100*s/n:.0f}%)"
              f"   BOTH  {both}/{len(ok)} ({100*both/n:.0f}%)")
        mis = Counter((r["old_skill"], r["skill"]) for r in ok if r["skill"] != r["old_skill"])
        if mis:
            print("  SKILL 불일치 상위:")
            for (d, j), k in mis.most_common(8):
                print(f"    declared {d:20s} -> judged {str(j):20s} x{k}")
        per = Counter(r["old_skill"] for r in ok)
        agree = Counter(r["old_skill"] for r in ok if r["skill"] == r["old_skill"])
        print("  선언 SKILL별 일치율: " + "  ".join(
            f"{k[:6]}={agree[k]}/{per[k]}" for k in SKILLS if per[k]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"prompt_hash": R.PROMPT_HASH, "model": args.model, "effort": args.effort,
               "usage": dict(judge.usage), "results": out_all},
              open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"\nusage: {dict(judge.usage)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
