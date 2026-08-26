#!/usr/bin/env python
"""Score the seed corpus against the resident 4B and 8B servers.

    bash scripts/serve_calibration.sh        # once
    python scripts/calibrate_seeds.py        # after every seed edit

Reports what the ARCHIVE sees, not just a pass rate:

* per-seed s_hat over n FRESH seeds -- a program can be trivial on one seed and
  impossible on the next, and the fitness is a per-seed product, so that mixture
  scores R_Q = 0 even when the pooled rate looks healthy;
* the unbiased learnability m/(m-1) * s(1-s) and the resulting R_Q, using the
  same scoring module the run uses;
* `informative`: the fraction of seeds whose m rollouts disagreed. A seed whose
  rollouts all agree contributes exactly zero to the RLOO advantage, so this is
  the share of the measurement budget that produces gradient at all.

Target: s_hat near 0.5. R_Q = s(1-s)U peaks there, so a seed at 0.1 and a seed
at 0.9 are equally far from being useful curriculum.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.program import ProblemProgram  # noqa: E402
from rq_evolve.prompts import build_solver_messages  # noqa: E402
from rq_evolve.reward import answers_match, extract_boxed  # noqa: E402
from rq_evolve.scoring import unbiased_learnability  # noqa: E402
from rq_evolve.solver_trace import SOLVER_CHAT_BOUNDARY_STOPS  # noqa: E402

SERVERS = {"4B": ("http://127.0.0.1:8401/v1", "qwen3-4b-base"),
           "8B": ("http://127.0.0.1:8801/v1", "qwen3-8b-base")}


async def _one(client, model, problem, args):
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=build_solver_messages(problem),
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.tokens,
            # These are BASE checkpoints. Without the boundary stops a reply
            # runs past its answer into a hallucinated next turn and emits a
            # SECOND \boxed{}; extract_boxed returns the LAST box, so that one
            # overwrites the correct answer. The run's own scorer is unaffected
            # -- evolution.py regrades through sanitize_solver_trace, which cuts
            # the spilled turn -- but this script grades raw, so without this it
            # under-measures s_hat, and worst on exactly the hard seeds it
            # exists to calibrate. probe_seed_solvability.py:76 already does it.
            stop=list(SOLVER_CHAT_BOUNDARY_STOPS),
        )
        return r.choices[0].message.content or ""
    except Exception as exc:  # one bad request must not sink the sweep
        return f"<error: {exc}>"


async def score_model(tag: str, instances, args) -> dict:
    from openai import AsyncOpenAI

    base, model = SERVERS[tag]
    client = AsyncOpenAI(base_url=base, api_key="none", timeout=600.0)
    jobs = [
        _one(client, model, inst.problem, args)
        for _, _, inst in instances
        for _ in range(args.rollouts)
    ]
    texts = await asyncio.gather(*jobs)
    out: dict[str, list] = {}
    for i, (name, seed, inst) in enumerate(instances):
        chunk = texts[i * args.rollouts : (i + 1) * args.rollouts]
        hits = sum(
            1 for t in chunk
            if (pred := extract_boxed(t)) is not None and answers_match(pred, inst.answer)
        )
        out.setdefault(name, []).append(hits / args.rollouts)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", default="seed_programs")
    ap.add_argument("--eval-seeds", type=int, default=5, help="n: fresh seeds per program")
    ap.add_argument("--rollouts", type=int, default=4, help="m: rollouts per seed")
    # 5000 = the run's data.max_response_length. At 3000 a long seed is
    # truncated before its box and scores 0 for a reason that has nothing to
    # do with its difficulty -- the same trap this script's server wrapper
    # documents for generation_config's 2048 ceiling.
    ap.add_argument("--tokens", type=int, default=5000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--guess-seeds", type=int, default=300,
                    help="seeds used to estimate the modal-answer (guess) floor")
    ap.add_argument("--only", default=None, help="substring filter on the file name")
    ap.add_argument("--out", default="rq_output/seed_calibration.json")
    args = ap.parse_args()

    instances = []
    # Guess floor: the share of the seed space that yields the single most
    # common answer. Narrowing a one-parameter generator is the obvious way to
    # raise s_hat, and it raises this at the same time and on the SAME knob --
    # a seed whose s_hat merely matches its modal rate is being answered by
    # guessing, not by the SKILL it is supposed to demand. Read the two columns
    # together or calibration will manufacture exactly the wrong seeds.
    guess: dict[str, float] = {}
    for path in sorted(Path(args.seed_dir).glob("*.py")):
        if args.only and args.only not in path.name:
            continue
        program = ProblemProgram.from_file(path)
        answers = collections.Counter()
        for seed in range(args.guess_seeds):
            probe = program.execute(seed=seed)
            if probe is not None:
                answers[str(probe.answer)] += 1
        guess[path.name] = (
            answers.most_common(1)[0][1] / sum(answers.values()) if answers else 1.0
        )
        for seed in range(args.eval_seeds):
            inst = program.execute(seed=seed)
            if inst is None:
                print(f"{path.name}: EXECUTE FAILED at seed={seed}: "
                      f"{program.last_execution_error}")
                break
            instances.append((path.name, seed, inst))
    if not instances:
        print("no runnable seed programs"); return 1

    async def score_all():
        # One event loop for both servers. They are separate GPUs, so scoring
        # them one after the other left half the hardware idle and doubled the
        # turnaround of the edit-and-remeasure loop this script exists for.
        tags = list(SERVERS)
        outs = await asyncio.gather(
            *(score_model(tag, instances, args) for tag in tags)
        )
        return dict(zip(tags, outs))

    results = asyncio.run(score_all())

    print(f"\nn={args.eval_seeds} fresh seeds x m={args.rollouts} rollouts, "
          f"temperature={args.temperature}\n")
    head = (f"{'seed program':<52}"
            + "".join(f"{t+' s_hat':>11}{t+' R_Q':>10}{'info':>7}" for t in SERVERS)
            + f"{'guess':>8}  verdict")
    print(head); print("-" * len(head))
    rows = []
    for name in sorted({n for n, _, _ in instances}):
        line = f"{name.replace('seed_','').replace('.py',''):<52}"
        row = {"program": name}
        for tag in SERVERS:
            per_seed = results[tag].get(name, [])
            if not per_seed:
                line += f"{'-':>11}{'-':>10}{'-':>7}"; continue
            s = statistics.fmean(per_seed)
            rq = statistics.fmean(
                unbiased_learnability(x, args.rollouts) for x in per_seed
            )
            info = sum(1 for x in per_seed if 0.0 < x < 1.0) / len(per_seed)
            line += f"{s:>11.2f}{rq:>10.3f}{info:>7.0%}"
            row[tag] = {"s_hat": s, "learnability": rq, "informative": info,
                        "per_seed": per_seed}
        g = guess.get(name, 1.0)
        row["guess_floor"] = g
        line += f"{g:>8.0%}  "
        # The 4B server is the policy the run actually starts from, so its
        # s_hat is the one that decides whether this seed can ever carry
        # gradient. R_Q = s(1-s)U is zero at both ends; 0.30-0.70 is the band
        # where a seed is worth its rollouts.
        s4 = (row.get("4B") or {}).get("s_hat")
        if s4 is None:
            verdict = "no measurement"
        elif s4 <= 0.0:
            verdict = "DEAD (s=0): unsolvable, R_Q=0 from birth"
        elif s4 >= 1.0:
            verdict = "DEAD (s=1): trivial, R_Q=0 from birth"
        elif s4 < g + 0.10:
            verdict = f"AT GUESS FLOOR (s={s4:.2f} vs modal {g:.0%})"
        elif not 0.30 <= s4 <= 0.70:
            verdict = "off-band (want 0.30-0.70)"
        else:
            verdict = "ok"
        # Capability ordering. 8B is strictly the stronger solver, so a seed that
        # measures REASONING must be easier for it. When s_8B <= s_4B the seed is
        # ranking the two models the wrong way round, which no amount of
        # difficulty tuning fixes -- the score is coming from something other
        # than the SKILL the cell claims (answer-space luck, format, length).
        # Reported alongside rather than instead of the band verdict: a seed can
        # sit in the band and still be measuring nothing.
        s8 = (row.get("8B") or {}).get("s_hat")
        if s4 is not None and s8 is not None:
            row["capability_gap"] = s8 - s4
            if s8 <= s4:
                verdict += f"  [!] 8B <= 4B ({s8:.2f} vs {s4:.2f}): not measuring skill"
        row["verdict"] = verdict
        line += verdict
        print(line); rows.append(row)

    print()
    for tag in SERVERS:
        got = [r[tag]["s_hat"] for r in rows if tag in r]
        live = sum(1 for r in rows if tag in r and 0.0 < r[tag]["s_hat"] < 1.0)
        print(f"  {tag}: mean s_hat {statistics.fmean(got):.3f} | "
              f"{live}/{len(got)} in the frontier band (0 < s < 1) | "
              f"{sum(1 for r in rows if tag in r and r[tag]['s_hat'] == 0)} unsolved")
    print("\n  target: s_hat ~ 0.5, informative high. R_Q = s(1-s)U peaks at 0.5.")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
