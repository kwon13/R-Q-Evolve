#!/usr/bin/env python
"""Paired comparison of the signed reward-entropy covariance across checkpoints.

Every run scored the SAME problems (same seed=0 instances, same prompt token
ids), so each program_id can be matched across checkpoints and the sign of
Cov(R,H) can be tracked rather than merely tabulated. This answers whether the
negative skew is present at RL initialisation or produced by training.

    python scripts/cov_sign_paired.py \
        --run base=analysis/.../cov_sign_union_base \
        --run step128=analysis/.../cov_sign_union_step128 \
        --run step256=analysis/.../cov_sign_union \
        --out-dir analysis/.../cov_sign_paired
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cov_sign_analyze import frac_neg, load, per_problem, wilson  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reference", default="", help="run name used as the paired baseline (default: first)")
    ap.add_argument("--target", default="", help="run name compared against the reference (default: last)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    runs: dict[str, Path] = {}
    for spec in args.run:
        name, _, d = spec.partition("=")
        runs[name] = Path(d)
    ref = args.reference or list(runs)[0]
    tgt = args.target or list(runs)[-1]

    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    raw = {n: load(d) for n, d in runs.items()}
    prob = {
        n: {"h_sum": per_problem(df, "h_sum"), "h_mean": per_problem(df, "h_mean")}
        for n, df in raw.items()
    }
    metas = {n: json.loads((d / "meta_generate.json").read_text()) for n, d in runs.items()}
    metas_e = {n: json.loads((d / "meta_entropy.json").read_text()) for n, d in runs.items()}

    from scipy.stats import spearmanr

    L: list[str] = []
    W = L.append
    W("# Signed reward-entropy covariance across the RL trajectory\n")
    W("| run | checkpoint | problems | rollouts | actor-vs-vLLM pearson |")
    W("|---|---|---|---|---|")
    for n in runs:
        W(f"| {n} | `{metas[n]['checkpoint']}` | {metas[n]['n_problems']} | "
          f"{metas[n]['n_rollouts_written']} | "
          f"{metas_e[n]['alignment_check']['pearson_actor_vs_vllm_surprisal']:.5f} |")
    W("")

    # -- problem-set identity ----------------------------------------------
    id_sets = {n: set(prob[n]["h_sum"]["program_id"]) for n in runs}
    common = set.intersection(*id_sets.values())
    W(f"* problem sets identical across runs: "
      f"**{all(s == id_sets[ref] for s in id_sets.values())}** ({len(common)} matched program_ids)")
    prompts = {}
    for n, d in runs.items():
        prompts[n] = {
            r["program_id"]: tuple(r["prompt_token_ids"])
            for r in json.loads((d / "instances.json").read_text())
        }
    same_prompt = all(
        prompts[n][pid] == prompts[ref][pid] for n in runs for pid in common
    )
    W(f"* prompt token ids identical across runs: **{same_prompt}** "
      "(paired on the input, so any difference is the policy)\n")

    # -- 1. what the policy does to the problem set -------------------------
    W("## 1. Problem-set state per checkpoint\n")
    W("| run | s=0 | s=1 | degenerate | thin (min(n+,n-)<2) | non-degenerate | "
      "mean s_hat | extract-fail rollouts | truncated rollouts |")
    W("|---|---|---|---|---|---|---|---|---|")
    for n in runs:
        d = prob[n]["h_sum"]
        df = raw[n]
        W(f"| {n} | {int((d['s_hat']==0).sum())} | {int((d['s_hat']==1).sum())} | "
          f"{int(d['degenerate'].sum())} | {int(d['thin_class'].sum())} | "
          f"{int((~d['degenerate']).sum())} | {d['s_hat'].mean():.3f} | "
          f"{(~df['extracted']).mean():.3f} | {(df['finish_reason']=='length').mean():.3f} |")
    W("")

    # -- 2. sign distribution side by side ----------------------------------
    W("## 2. Sign distribution per checkpoint\n")
    for var in ("h_sum", "h_mean"):
        W(f"### {var}\n")
        W("| run | n non-degenerate | frac C<0 (all) | n significant | frac C<0 (significant) | 95% CI |")
        W("|---|---|---|---|---|---|")
        for n in runs:
            d = prob[n][var]
            nd = d[~d["degenerate"]]
            sig = d[d["significant"]]
            k = int((sig["C"] < 0).sum())
            lo, hi = wilson(k, len(sig))
            W(f"| {n} | {len(nd)} | {frac_neg(nd['C']):.3f} | {len(sig)} | "
              f"{(k/len(sig) if len(sig) else float('nan')):.3f} | [{lo:.3f}, {hi:.3f}] |")
        W("")

    # -- 3. paired sign flips ----------------------------------------------
    names = list(runs)
    legs = [(names[i], names[i + 1]) for i in range(len(names) - 1)]
    if (ref, tgt) not in legs:
        legs.append((ref, tgt))
    W("## 3. Paired sign tables\n")
    W("Restricted to problems non-degenerate in BOTH runs of a pair (elsewhere C "
      "is identically 0 and its sign is not defined). Consecutive legs first, "
      "then the endpoints.\n")
    for ref, tgt in legs:
      W(f"### {ref} -> {tgt}\n")
      for var in ("h_sum", "h_mean"):
          a = prob[ref][var].set_index("program_id")
          b = prob[tgt][var].set_index("program_id")
          both = [p for p in common if not a.loc[p, "degenerate"] and not b.loc[p, "degenerate"]]
          if not both:
              W(f"* {var}: no problem is non-degenerate in both runs.\n")
              continue
          an = a.loc[both, "C"] < 0
          bn = b.loc[both, "C"] < 0
          nn = int((an & bn).sum())
          np_ = int((an & ~bn).sum())
          pn = int((~an & bn).sum())
          pp = int((~an & ~bn).sum())
          W(f"**{var}** (n = {len(both)})\n")
          W(f"| {ref} \\\\ {tgt} | C<0 | C>=0 | total |")
          W("|---|---|---|---|")
          W(f"| C<0 | {nn} | {np_} | {nn+np_} |")
          W(f"| C>=0 | {pn} | {pp} | {pn+pp} |")
          W(f"| total | {nn+pn} | {np_+pp} | {len(both)} |")
          agree = (nn + pp) / len(both)
          W(f"\n* sign agreement {agree:.3f}; "
            f"{ref} negative {(nn+np_)/len(both):.3f} vs {tgt} negative {(nn+pn)/len(both):.3f}")
          rs = spearmanr(a.loc[both, "C"], b.loc[both, "C"])
          W(f"* spearman(C_{ref}, C_{tgt}) = {rs.statistic:.3f} (p={rs.pvalue:.3g})\n")

    ref, tgt = names[0], names[-1]

    # -- 4. entropy level ---------------------------------------------------
    W("## 4. Entropy level and spread\n")
    W("`h_bar` is the per-problem mean over its G=32 rollouts (this is the U in "
      "the production fitness); `sd_h` is the within-problem spread that the gap "
      "signal has to beat.\n")
    W("| run | mean h_bar (sum) | median h_bar (sum) | mean sd_h (sum) | "
      "mean h_bar (mean) | mean sd_h (mean) | mean tokens |")
    W("|---|---|---|---|---|---|---|")
    for n in runs:
        s_, m_ = prob[n]["h_sum"], prob[n]["h_mean"]
        W(f"| {n} | {s_['h_bar'].mean():.1f} | {s_['h_bar'].median():.1f} | "
          f"{s_['sd_h'].mean():.1f} | {m_['h_bar'].mean():.4f} | "
          f"{m_['sd_h'].mean():.4f} | {raw[n]['tokens'].mean():.0f} |")
    W("")

    # -- 5. what |Cov| would select ----------------------------------------
    W("## 5. What |Cov| fitness would select, per checkpoint\n")
    W("| run | variant | frac C<0 in top-10 | top-25 | spearman(rq_now, \\|C\\|) |")
    W("|---|---|---|---|---|")
    for n in runs:
        for var in ("h_sum", "h_mean"):
            d = prob[n][var]
            t10 = d.nlargest(min(10, len(d)), "absC")
            t25 = d.nlargest(min(25, len(d)), "absC")
            sp = spearmanr(d["rq_now"], d["absC"])
            W(f"| {n} | {var} | {frac_neg(t10['C']):.3f} | {frac_neg(t25['C']):.3f} | {sp.statistic:.3f} |")
    W("")

    # -- 6. verdict ---------------------------------------------------------
    W("## 6. Verdict against the pre-registered criterion\n")
    W("Criterion on the RL-init run: significant-problem negative fraction >= 0.75 "
      "-> the skew is present at initialisation (|Cov| fitness rejected across the "
      "whole run); <= 0.55 -> the skew is a product of training (the shift itself is "
      "the result); in between -> undetermined from two checkpoints.\n")
    for var in ("h_sum", "h_mean"):
        d = prob[ref][var]
        sig = d[d["significant"]]
        k, n_ = int((sig["C"] < 0).sum()), len(sig)
        if n_ == 0:
            W(f"* {ref} / {var}: no significant problem -- not evaluable.")
            continue
        frac = k / n_
        lo, hi = wilson(k, n_)
        if frac >= 0.75:
            v = "SKEW PRESENT AT INITIALISATION"
        elif frac <= 0.55:
            v = "SKEW IS A PRODUCT OF TRAINING"
        else:
            v = "UNDETERMINED from these checkpoints"
        W(f"* {ref} / {var}: {k}/{n_} = {frac:.3f} negative, 95% CI [{lo:.3f}, {hi:.3f}] -> **{v}**")
    W("")

    figures(prob, raw, runs, ref, tgt, out / "figures")
    (out / "paired_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[paired] report -> {out/'paired_report.md'}")
    return 0


def figures(prob, raw, runs, ref, tgt, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

    # C_ref vs C_tgt, per variant
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ax, var in zip(axes, ("h_sum", "h_mean")):
        a = prob[ref][var].set_index("program_id")
        b = prob[tgt][var].set_index("program_id")
        idx = [p for p in a.index if p in b.index
               and not a.loc[p, "degenerate"] and not b.loc[p, "degenerate"]]
        ax.scatter(a.loc[idx, "C"], b.loc[idx, "C"], s=24, c="#5b8c85",
                   edgecolor="k", linewidth=0.3)
        ax.axhline(0, color="k", lw=1)
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel(f"Cov(R, {var}) @ {ref}")
        ax.set_ylabel(f"Cov(R, {var}) @ {tgt}")
        ax.set_title(f"{var}, n={len(idx)}", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "paired_C.png")
    plt.close(fig)

    # s_hat shift
    a = prob[ref]["h_sum"].set_index("program_id")
    b = prob[tgt]["h_sum"].set_index("program_id")
    idx = [p for p in a.index if p in b.index]
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.scatter(a.loc[idx, "s_hat"], b.loc[idx, "s_hat"], s=24, c="#c1666b",
               edgecolor="k", linewidth=0.3, alpha=0.75)
    ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls=":")
    ax.set_xlabel(f"$\\hat{{s}}$ @ {ref}")
    ax.set_ylabel(f"$\\hat{{s}}$ @ {tgt}")
    ax.set_title("pass rate shift over training", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "paired_s_hat.png")
    plt.close(fig)

    # entropy level distributions
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for ax, var in zip(axes, ("h_sum", "h_mean")):
        data = [prob[n][var]["h_bar"].to_numpy() for n in runs]
        ax.boxplot(data, tick_labels=list(runs), showfliers=False)
        ax.set_ylabel(f"per-problem mean {var}")
        ax.set_title(f"entropy level, {var}", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "entropy_levels.png")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
