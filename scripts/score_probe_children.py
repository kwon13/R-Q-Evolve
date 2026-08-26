#!/usr/bin/env python
"""Score the operator probe's SURVIVING children the way the archive scores them.

probe_operators_report.py stops at validity + novelty. Neither decides a child's
fate: the archive stores R_Q = L_hat * U_hat measured on ONE fresh instance at
m = group_size = 10 (evolution.py:1323, `compute_rq_program(stats[:1])`), and a
child is useful only if that single draw lands in 0 < s_hat < 1. This script
measures the quantities that draw is a sample OF, so an operator can be ranked
on the fate it produces rather than on the code it emits.

Three servers, three different jobs:

  --tier policy  (8601 step96 / 8701 step160)  the checkpoint that will actually
                 mutate and be trained. Its s_hat is the one that decides
                 admission and R_Q. This is the tier that ranks operators.
  --tier base4b  (8401 qwen3-4b-base)          drift-free difficulty anchor:
                 comparable across checkpoints, so a later re-run means the
                 same thing.
  --tier base8b  (8801 qwen3-8b-base)          capability ordering and the
                 answer-key check. s_8B <= s_4B means the problem is not
                 ranking solvers by reasoning; 8B consensus disagreeing with the
                 declared answer means the key is probably wrong.

Everything grades through the run's own path -- build_solver_messages,
SOLVER_CHAT_BOUNDARY_STOPS, sanitize_solver_trace, extract_boxed/answers_match,
scoring.score_seed/compute_rq_program -- so the numbers are commensurable with
the archive's, not a parallel definition of them.

U_hat: verl computes the exact length-normalized entropy from the actor's
logits (verl_backend._response_entropies). An OpenAI-protocol server cannot
return that, but at temperature 1.0 the sampled token's own surprisal is an
UNBIASED one-sample estimate of it: E_{y~p}[-log p(y)] = H(p). So
u_score = mean over response tokens of -logprob(sampled token). Top-p 0.95
truncation biases it slightly low, identically for every program; --u-const
turns it off (u := 1.0) to rank on learnability alone.

    python scripts/score_probe_children.py --tier policy --n-seeds 8 --rollouts 16
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.program import ProblemProgram              # noqa: E402
from rq_evolve.prompts import build_solver_messages       # noqa: E402
from rq_evolve.reward import answers_match, extract_boxed  # noqa: E402
from rq_evolve.scoring import compute_rq_program, score_seed, unbiased_learnability  # noqa: E402
from rq_evolve.solver_trace import SOLVER_CHAT_BOUNDARY_STOPS, sanitize_solver_trace  # noqa: E402

TIERS = {
    "policy":  ("http://127.0.0.1:8701/v1", "step160"),
    "policy96": ("http://127.0.0.1:8601/v1", "step96"),
    "base4b":  ("http://127.0.0.1:8401/v1", "qwen3-4b-base"),
    "base8b":  ("http://127.0.0.1:8801/v1", "qwen3-8b-base"),
}

# The archive's own draw: one fresh instance, group_size rollouts, admitted on
# 0 < s_hat < 1. Both are read from the live config rather than hardcoded.
ARCHIVE_M = 10


# --------------------------------------------------------------------------
# p_admit -- the statistic that actually predicts a child's fate.
#
# The archive does not see mean s_hat. It sees ONE Binomial(M, s_z) draw on one
# fresh seed z and keeps the child iff that draw is neither 0 nor M. So the
# survival probability of a child whose per-seed true rates are {s_z} is
#
#     p_admit = E_z [ 1 - s_z^M - (1-s_z)^M ]
#
# and it is NOT a function of mean s_hat: a program that is trivial on half its
# seeds and impossible on the other half has mean s_hat 0.5 and p_admit 0.
#
# Plugging s_hat_z straight in is badly biased at the ends -- a child measured
# 0/16 gets p_admit 0, while a true s of 0.05 survives 40% of the time. So the
# estimate integrates over the Beta(1+k, 1+m-k) posterior of s_z instead:
#     E[s^M] = prod_{i<M} (a+i)/(a+b+i).
# --------------------------------------------------------------------------
def _beta_moment(a: float, b: float, m: int) -> float:
    out = 1.0
    for i in range(m):
        out *= (a + i) / (a + b + i)
    return out


def p_admit_seed(k: int, m: int, archive_m: int = ARCHIVE_M) -> float:
    # Jeffreys Beta(0.5, 0.5), not the uniform Beta(1, 1). Measured on 24
    # children at n=4 x m=10, the flat prior put a FLOOR of 0.48 on p_admit for
    # a child every one of whose seeds came back 0/10 or 10/10 -- i.e. it
    # scored a program the archive would reject outright as a coin flip. The
    # Jeffreys prior keeps mass at the ends and drops that floor.
    a, b = 0.5 + k, 0.5 + (m - k)
    return 1.0 - _beta_moment(a, b, archive_m) - _beta_moment(b, a, archive_m)


def hyper_archive_draw(k: int, m: int, archive_m: int = ARCHIVE_M):
    """EXACT distribution of what the archive would store for this seed.

    Our m rollouts are exchangeable with the archive's, so the archive's draw is
    a size-``archive_m`` subsample of them: j correct with the hypergeometric
    weight C(k,j)C(m-k,archive_m-j)/C(m,archive_m). Returns
    (E[L_hat stored], P(stored L_hat == 0)) -- no Monte Carlo, no prior.
    """
    M = min(archive_m, m)
    tot = math.comb(m, M)
    e_l, p_dead = 0.0, 0.0
    for j in range(max(0, M - (m - k)), min(k, M) + 1):
        w = math.comb(k, j) * math.comb(m - k, M - j) / tot
        s = j / M
        e_l += w * unbiased_learnability(s, M)
        if j == 0 or j == M:
            p_dead += w
    return e_l, p_dead


# --------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------
async def _one(client, model, problem, args, sem):
    async with sem:
        for attempt in range(2):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=build_solver_messages(problem),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.tokens,
                    # Base checkpoints run past their answer into a hallucinated
                    # next turn and emit a SECOND \boxed{}; extract_boxed takes
                    # the LAST one. calibrate_seeds.py:61 documents the same trap.
                    stop=list(SOLVER_CHAT_BOUNDARY_STOPS),
                    logprobs=True,
                )
                ch = r.choices[0]
                text = ch.message.content or ""
                lps = []
                if ch.logprobs is not None and ch.logprobs.content:
                    lps = [t.logprob for t in ch.logprobs.content
                           if t.logprob is not None and math.isfinite(t.logprob)]
                # -mean logprob of the sampled tokens = unbiased estimate of the
                # length-normalized entropy at temperature 1.
                u = (-statistics.fmean(lps)) if lps else 0.0
                return text, u, True
            except Exception:
                if attempt == 0:
                    await asyncio.sleep(2.0)
        return "", 0.0, False


async def score_tier(tag, work, args):
    """work: list of (child_key, seed, instance). Returns {child_key: [SeedStat]}."""
    from openai import AsyncOpenAI

    base, model = TIERS[tag]
    client = AsyncOpenAI(base_url=base, api_key="none", timeout=1200.0,
                         max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    jobs = [_one(client, model, inst.problem, args, sem)
            for _, _, inst in work for _ in range(args.rollouts)]
    t0 = time.time()
    outs = await asyncio.gather(*jobs)
    dt = time.time() - t0

    per_child: dict = collections.defaultdict(list)
    preds: dict = collections.defaultdict(list)
    n_fail = 0
    for i, (key, seed, inst) in enumerate(work):
        chunk = outs[i * args.rollouts:(i + 1) * args.rollouts]
        flags, ents, answers = [], [], []
        for text, u, ok in chunk:
            if not ok:
                n_fail += 1
                continue  # a rejected rollout leaves the estimate, as in evolution.py
            cleaned = sanitize_solver_trace(text)
            pred = extract_boxed(cleaned)
            flags.append(bool(pred is not None and answers_match(pred, inst.answer)))
            ents.append(u)
            answers.append(pred)
        if not flags:
            continue
        per_child[key].append(score_seed(seed=seed, correct_flags=flags,
                                         rollout_entropies=ents))
        preds[key].append((answers, str(inst.answer)))
    return per_child, preds, dt, n_fail


# --------------------------------------------------------------------------
def load_children(path: Path, only_valid: bool):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out = []
    for i, r in enumerate(rows):
        if not r.get("code"):
            continue
        if only_valid and not r.get("_valid"):
            continue
        out.append({"key": f"{r['op']}:{r.get('i', i)}:{i}", "op": r["op"],
                    "code": r["code"], "row": r})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="rq_output/probe_operators_valid.jsonl")
    ap.add_argument("--tier", default="policy", choices=list(TIERS))
    ap.add_argument("--n-seeds", type=int, default=8,
                    help="n: FRESH instances per child (the archive draws 1; we "
                         "need the distribution that draw comes from)")
    ap.add_argument("--rollouts", type=int, default=16,
                    help="m: rollouts per instance. >= archive m=10 keeps the "
                         "per-seed rate estimate tighter than the archive's own")
    ap.add_argument("--seed-base", type=int, default=1000,
                    help="fresh-seed offset. NOT 0: the probe's validity check "
                         "already ran seeds 0-2, and a child that special-cases "
                         "the instance it was checked on must not be scored there")
    ap.add_argument("--tokens", type=int, default=5000,
                    help="= data.max_response_length in the run config")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--concurrency", type=int, default=192)
    ap.add_argument("--guess-seeds", type=int, default=200)
    ap.add_argument("--u-const", action="store_true",
                    help="u := 1.0; rank on learnability alone")
    ap.add_argument("--limit", type=int, default=0, help="first K children (timing)")
    ap.add_argument("--all-code", action="store_true",
                    help="score every child that produced CODE, not only the "
                         "ones probe_operators_report.py already ran")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = Path(args.out or f"rq_output/probe_operator_scores_{args.tier}.json")

    kids = load_children(ROOT / args.rows, not args.all_code)
    if args.limit:
        kids = kids[:args.limit]
    print(f"{len(kids)} children | tier={args.tier} {TIERS[args.tier][1]} | "
          f"n={args.n_seeds} x m={args.rollouts}", flush=True)

    # Instantiate on FRESH seeds; also the modal-answer (guess) floor.
    work, guess, execfail = [], {}, []
    for kid in kids:
        prog = ProblemProgram(source_code=kid["code"])
        kid["prog"] = prog
        got = 0
        for z in range(args.seed_base, args.seed_base + args.n_seeds * 3):
            if got >= args.n_seeds:
                break
            inst = prog.execute(seed=z)
            if inst is not None:
                work.append((kid["key"], z, inst))
                got += 1
        kid["n_instances"] = got
        if got == 0:
            execfail.append(kid["key"])
            continue
        ans = collections.Counter()
        for z in range(args.guess_seeds):
            p = prog.execute(seed=z)
            if p is not None:
                ans[str(p.answer)] += 1
        guess[kid["key"]] = (ans.most_common(1)[0][1] / sum(ans.values())) if ans else 1.0

    print(f"{len(work)} instances -> {len(work) * args.rollouts} rollouts; "
          f"{len(execfail)} children failed to execute on fresh seeds", flush=True)

    per_child, preds, dt, n_fail = asyncio.run(score_tier(args.tier, work, args))
    print(f"rollouts done in {dt/60:.1f} min ({len(work)*args.rollouts/max(dt,1e-9):.1f} "
          f"rollouts/s, {n_fail} failed)", flush=True)

    rows = []
    for kid in kids:
        stats = per_child.get(kid["key"], [])
        if not stats:
            rows.append({"key": kid["key"], "op": kid["op"], "scored": False})
            continue
        res = compute_rq_program(stats)
        # Fitness as the archive stores it: E over seeds of L*U, i.e. what the
        # archive's one-instance draw estimates. u_const drops the U factor.
        rq = (statistics.fmean(st.learnability for st in stats) if args.u_const
              else res.rq_score)
        s_vec = [st.s_hat for st in stats]
        m = args.rollouts
        p_adm = statistics.fmean(
            p_admit_seed(int(round(st.s_hat * st.num_rollouts)), st.num_rollouts)
            for st in stats)
        # What the archive would STORE, in expectation over which seed it drew
        # and which 10 rollouts it got: E[L_hat*U | admitted] * P(admitted).
        draws = [hyper_archive_draw(st.num_correct, st.num_rollouts) for st in stats]
        rq_exp = statistics.fmean(
            e_l * (1.0 if args.u_const else st.u_score)
            for (e_l, _), st in zip(draws, stats))
        p_adm_hyper = 1.0 - statistics.fmean(pd for _, pd in draws)
        # Answer-key check: does the solver's own consensus agree with the key?
        agree, decisive = 0, 0
        for answers, gold in preds.get(kid["key"], []):
            got = [a for a in answers if a is not None]
            if not got:
                continue
            top, cnt = collections.Counter(got).most_common(1)[0]
            if cnt / len(answers) >= 0.6:      # solver is confident about SOMETHING
                decisive += 1
                if answers_match(top, gold):
                    agree += 1
        rows.append({
            "key": kid["key"], "op": kid["op"], "scored": True,
            "n_instances": len(stats), "m": m,
            "s_hat_mean": res.s_hat, "s_hat_per_seed": s_vec,
            "s_hat_median": statistics.median(s_vec),
            "learnability": res.learnability, "u_score": res.u_score,
            "rq": rq, "rq_archive_expected": rq_exp,
            "dispersion": res.dispersion,
            "p_admit": p_adm, "p_admit_hyper": p_adm_hyper,
            "frac_seeds_live": sum(1 for s in s_vec if 0.0 < s < 1.0) / len(s_vec),
            "frac_seeds_zero": sum(1 for s in s_vec if s <= 0.0) / len(s_vec),
            "frac_seeds_one": sum(1 for s in s_vec if s >= 1.0) / len(s_vec),
            "guess_floor": guess.get(kid["key"], 1.0),
            "key_agreement": (agree / decisive) if decisive else None,
            "key_decisive_frac": decisive / len(stats),
        })

    out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))

    # Per-operator roll-up.
    hdr = (f"{'operator':<14}{'n':>4}{'exec':>6}{'s_hat':>8}{'live%':>7}"
           f"{'p_adm':>7}{'R_Q':>8}{'E[RQ]':>8}{'disp':>7}{'guess':>7}{'key%':>7}")
    print("\n" + hdr); print("-" * len(hdr))
    for op in sorted({k["op"] for k in kids}):
        g = [r for r in rows if r["op"] == op]
        sc = [r for r in g if r["scored"]]
        if not sc:
            print(f"{op:<14}{len(g):>4}{0:>6}"); continue
        ka = [r["key_agreement"] for r in sc if r["key_agreement"] is not None]
        print(f"{op:<14}{len(g):>4}{len(sc):>6}"
              f"{statistics.fmean(r['s_hat_mean'] for r in sc):>8.3f}"
              f"{100*statistics.fmean(r['frac_seeds_live'] for r in sc):>6.0f}%"
              f"{statistics.fmean(r['p_admit'] for r in sc):>7.2f}"
              f"{statistics.fmean(r['rq'] for r in sc):>8.4f}"
              f"{statistics.fmean(r['rq_archive_expected'] for r in sc):>8.4f}"
              f"{statistics.fmean(r['dispersion'] for r in sc):>7.3f}"
              f"{statistics.fmean(r['guess_floor'] for r in sc):>6.0%}"
              f"{(statistics.fmean(ka) if ka else float('nan')):>6.0%}")
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
