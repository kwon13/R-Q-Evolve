#!/usr/bin/env python
"""Phase 3: signed reward-entropy covariance statistics over the stored rollouts.

Answers one question: if R_Q fitness were redefined as 4|Cov(R,H)|, would the
sign of Cov be balanced, or is the population one-sided? Cui et al.
(arXiv:2505.22617) predicts correct rollouts carry LOWER entropy, i.e. Cov<0
dominant, in which case |Cov| maximisation is -Cov maximisation: an entropy
collapse accelerator.

Per problem and per entropy variant h in {H_sum, H_mean}:

    C     = (1/G) sum_i (r_i - s_hat)(h_i - h_bar)       signed covariance
    gap   = h_bar_+ - h_bar_-                            (only when 0 < s < 1)
    SE    = sigma_pooled * sqrt(1/n+ + 1/n-)             pooled within-class
    rho   = C / (sqrt(s(1-s)) * sd(h))                   point-biserial

C is computed in the covariance form on purpose: it is automatically 0 at
s_hat in {0,1}, where the factorised s(1-s)(h_+ - h_-) form is undefined.

    python scripts/cov_sign_analyze.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(ROOT / "analysis/rq_evolve_base_8b/cov_sign"))
    p.add_argument("--top-k", type=int, nargs="+", default=[10, 25])
    p.add_argument("--label", default="", help="title suffix for figures/report")
    return p.parse_args()


def load(out_dir: Path) -> pd.DataFrame:
    inst = {r["program_id"]: r for r in json.loads((out_dir / "instances.json").read_text())}
    rolls = [json.loads(l) for l in (out_dir / "rollouts.jsonl").read_text().split("\n") if l.strip()]
    ents = {
        (e["program_id"], e["rollout_idx"]): e
        for e in (
            json.loads(l) for l in (out_dir / "entropy.jsonl").read_text().split("\n") if l.strip()
        )
    }
    rows = []
    for r in rolls:
        e = ents[(r["program_id"], r["rollout_idx"])]
        meta = inst[r["program_id"]]
        rows.append(
            {
                "program_id": r["program_id"],
                "rollout_idx": r["rollout_idx"],
                "correct": bool(r["correct"]),
                "extracted": r["predicted_answer"] is not None,
                "tokens": int(r["response_tokens"]),
                "finish_reason": r["finish_reason"],
                "h_sum": float(e["h_sum"]),
                "h_mean": float(e["h_mean"]),
                "group": meta["group"],
                "skill": meta["skill"],
                "archived_p_hat": meta["archived_p_hat"],
                "archived_h_score": meta["archived_h_score"],
                "archived_rq_score": meta["archived_rq_score"],
            }
        )
    return pd.DataFrame(rows)


def per_problem(df: pd.DataFrame, hcol: str) -> pd.DataFrame:
    out = []
    for pid, g in df.groupby("program_id", sort=False):
        r = g["correct"].to_numpy(dtype=float)
        h = g[hcol].to_numpy(dtype=float)
        L = g["tokens"].to_numpy(dtype=float)
        G = len(r)
        s = float(r.mean())
        n_p, n_m = int(r.sum()), int(G - r.sum())
        # Population covariance (divide by G): identically 0 when s in {0,1}.
        C = float(((r - s) * (h - h.mean())).mean())
        sd_h = float(h.std())  # population sd, consistent with C
        rho = (
            float(C / (math.sqrt(s * (1 - s)) * sd_h))
            if 0.0 < s < 1.0 and sd_h > 0
            else float("nan")
        )
        hp = float(h[r == 1].mean()) if n_p else float("nan")
        hm = float(h[r == 0].mean()) if n_m else float("nan")
        gap = hp - hm if (n_p and n_m) else float("nan")
        if n_p >= 1 and n_m >= 1 and (n_p + n_m) > 2:
            ss = ((h[r == 1] - hp) ** 2).sum() + ((h[r == 0] - hm) ** 2).sum()
            sigma = math.sqrt(ss / (n_p + n_m - 2))
            se = sigma * math.sqrt(1.0 / n_p + 1.0 / n_m)
        else:
            se = float("nan")
        out.append(
            {
                "program_id": pid,
                "group": g["group"].iloc[0],
                "skill": g["skill"].iloc[0],
                "G": G,
                "s_hat": s,
                "n_plus": n_p,
                "n_minus": n_m,
                "degenerate": s in (0.0, 1.0),
                "thin_class": min(n_p, n_m) < 2,
                "C": C,
                "h_bar": float(h.mean()),
                "sd_h": sd_h,
                "gap": gap,
                "se_gap": se,
                "significant": bool(np.isfinite(gap) and np.isfinite(se) and se > 0 and abs(gap) > 2 * se),
                "rho": rho,
                "L_plus": float(L[r == 1].mean()) if n_p else float("nan"),
                "L_minus": float(L[r == 0].mean()) if n_m else float("nan"),
                "rq_now": s * (1 - s) * float(h.mean()),
                "archived_p_hat": g["archived_p_hat"].iloc[0],
                "archived_rq_score": g["archived_rq_score"].iloc[0],
                "truncated_frac": float((g["finish_reason"] == "length").mean()),
                # A rollout with no \boxed{} is graded r=0. If the policy simply
                # will not emit the format, those zeros are a formatting artefact,
                # not a failure to solve -- and they enter C exactly like real ones.
                "extract_fail_frac": float((~g["extracted"]).mean()),
            }
        )
    d = pd.DataFrame(out)
    d["L_diff"] = d["L_minus"] - d["L_plus"]  # positive = failures are longer
    d["absC"] = d["C"].abs()
    return d


def frac_neg(x: pd.Series) -> float:
    x = x[np.isfinite(x)]
    return float((x < 0).mean()) if len(x) else float("nan")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - hw), min(1.0, c + hw))


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load(out_dir)
    variants = {"h_sum": per_problem(df, "h_sum"), "h_mean": per_problem(df, "h_mean")}
    for name, d in variants.items():
        d.to_csv(out_dir / f"per_problem_{name}.csv", index=False)

    ds, dm = variants["h_sum"], variants["h_mean"]
    merged = ds.merge(dm, on="program_id", suffixes=("_sum", "_mean"))
    merged["sign_agree"] = np.sign(merged["C_sum"]) == np.sign(merged["C_mean"])

    from scipy.stats import spearmanr

    lines: list[str] = []
    W = lines.append
    suffix = f" -- {args.label}" if args.label else ""
    W(f"# Signed reward-entropy covariance{suffix}\n")
    meta_g = json.loads((out_dir / "meta_generate.json").read_text())
    meta_e = json.loads((out_dir / "meta_entropy.json").read_text())
    W(f"* checkpoint: `{meta_g['checkpoint']}`")
    W(f"* archive: `{meta_g['archive']}`")
    W(f"* problems: {len(ds)} (seed={meta_g['instance_seed']} representative instance), G={meta_g['g']}")
    W(f"* sampling: {json.dumps(meta_g['sampling'])}")
    W(f"* entropy: verl `entropy_from_logits` on actor logits / T={meta_e['temperature']}, "
      f"summed over response tokens; float32 accumulation")
    W(f"* actor-vs-vLLM surprisal check: pearson="
      f"{meta_e['alignment_check']['pearson_actor_vs_vllm_surprisal']:.5f}, "
      f"median |rel diff|={meta_e['alignment_check']['median_abs_rel_diff']:.4f}\n")

    # -- 1. degenerate / thin problems -------------------------------------
    W("## 1. Degenerate and thin problems\n")
    W(f"* s_hat in {{0,1}} (C identically 0): **{int(ds['degenerate'].sum())} / {len(ds)}** "
      f"(s=0: {int((ds['s_hat']==0).sum())}, s=1: {int((ds['s_hat']==1).sum())})")
    W(f"* min(n+, n-) < 2: **{int(ds['thin_class'].sum())} / {len(ds)}** (flagged, not excluded)")
    W(f"* both classes >= 2: {int((~ds['thin_class']).sum())} / {len(ds)}")
    W(f"* mean fraction of rollouts hitting the {meta_g['sampling']['max_tokens']}-token cap: "
      f"{ds['truncated_frac'].mean():.3f}\n")

    # -- 1b. answer-format hygiene -----------------------------------------
    W("## 1b. Answer-format hygiene\n")
    W(f"* rollouts with no `\\boxed{{}}` (graded r=0 by construction): "
      f"**{df[~df['extracted']].shape[0]} / {len(df)}** "
      f"({(~df['extracted']).mean():.3f})")
    dirty = ds[ds["extract_fail_frac"] > 0.10]
    W(f"* problems where >10% of rollouts fail extraction: **{len(dirty)} / {len(ds)}** (flagged)")
    W(f"* rollouts hitting the {meta_g['sampling']['max_tokens']}-token cap: "
      f"{(df['finish_reason'] == 'length').mean():.3f} of all rollouts; "
      f"truncation clips the trajectory sum, so H_sum is a lower bound on those.")
    if len(dirty):
        W("")
        W("Sensitivity -- section 2 recomputed with those problems dropped:\n")
        W("| variant | n non-degenerate | frac C<0 (all) | n significant | frac C<0 (significant) |")
        W("|---|---|---|---|---|")
        keep = set(ds.loc[ds["extract_fail_frac"] <= 0.10, "program_id"])
        for name, d in variants.items():
            dd = d[d["program_id"].isin(keep)]
            nd2 = dd[~dd["degenerate"]]
            sig2 = dd[dd["significant"]]
            k2 = int((sig2["C"] < 0).sum())
            W(f"| {name} | {len(nd2)} | {frac_neg(nd2['C']):.3f} | {len(sig2)} | "
              f"{(k2/len(sig2) if len(sig2) else float('nan')):.3f} |")
    W("")

    # -- 2. sign distribution ----------------------------------------------
    W("## 2. Signed covariance sign distribution\n")
    W("| variant | n non-degenerate | frac C<0 (all) | n significant | frac C<0 (significant) | 95% CI (sig) |")
    W("|---|---|---|---|---|---|")
    for name, d in variants.items():
        nd = d[~d["degenerate"]]
        sig = d[d["significant"]]
        k = int((sig["C"] < 0).sum())
        lo, hi = wilson(k, len(sig))
        W(f"| {name} | {len(nd)} | {frac_neg(nd['C']):.3f} | {len(sig)} | "
          f"{(k/len(sig) if len(sig) else float('nan')):.3f} | [{lo:.3f}, {hi:.3f}] |")
    W("")
    for name, d in variants.items():
        nd = d[~d["degenerate"]]
        W(f"* {name}: mean C = {nd['C'].mean():.4g}, median C = {nd['C'].median():.4g}, "
          f"mean rho = {nd['rho'].mean():.3f}, median rho = {nd['rho'].median():.3f}")
    W("")

    # -- 3. sum vs mean sign agreement -------------------------------------
    W("## 3. H_sum vs H_mean sign agreement\n")
    nd = merged[~merged["degenerate_sum"]]
    W(f"* sign agreement over non-degenerate problems: **{nd['sign_agree'].mean():.3f}** "
      f"({int(nd['sign_agree'].sum())}/{len(nd)})")
    flips = nd[~nd["sign_agree"]]
    W(f"* length normalisation flips the sign on {len(flips)} problems")
    q = nd["L_diff_sum"].abs().quantile([1 / 3, 2 / 3]).to_list()
    if len(nd) >= 6 and q[0] < q[1]:
        bucket = pd.cut(nd["L_diff_sum"].abs(), [-np.inf, q[0], q[1], np.inf],
                        labels=["low |L-  - L+|", "mid", "high |L- - L+|"])
        W("")
        W("| |L- - L+| tercile | n | sign agreement |")
        W("|---|---|---|")
        for b, gg in nd.groupby(bucket, observed=True):
            W(f"| {b} | {len(gg)} | {gg['sign_agree'].mean():.3f} |")
    both = nd[["C_sum", "C_mean"]].dropna()
    if len(both) > 2:
        rs = spearmanr(both["C_sum"], both["C_mean"])
        W(f"\n* spearman(C_sum, C_mean) = {rs.statistic:.3f} (p={rs.pvalue:.3g})")
    W("")

    # -- 4. top-|C| ---------------------------------------------------------
    W("## 4. What |Cov| fitness would actually select\n")
    W("| variant | k | frac C<0 in top-k by |C| | note |")
    W("|---|---|---|---|")
    for name, d in variants.items():
        for k in args.top_k:
            top = d.nlargest(min(k, len(d)), "absC")
            note = "covers nearly the whole set" if k >= 0.8 * len(d) else ""
            W(f"| {name} | {k} | {frac_neg(top['C']):.3f} | {note} |")
    W("")

    # -- 5. rank correlation with current fitness --------------------------
    W("## 5. Current fitness s(1-s)*U vs |C|\n")
    W("`rq_now` = s(1-s) * mean_i h_i, recomputed on these G=32 rollouts. The h_sum row is "
      "the production fitness (verl sums entropy over the trajectory); the h_mean row is the "
      "length-normalised hypothetical. `archived R_Q` is the value stored at insert time from "
      "G=10 rollouts of an older policy, so it is stale by construction.\n")
    W("| variant | spearman(rq_now, \\|C\\|) | p | spearman(archived R_Q, \\|C\\|) | p |")
    W("|---|---|---|---|---|")
    for name, d in variants.items():
        a = spearmanr(d["rq_now"], d["absC"])
        b = spearmanr(d["archived_rq_score"], d["absC"])
        W(f"| {name} | {a.statistic:.3f} | {a.pvalue:.3g} | {b.statistic:.3f} | {b.pvalue:.3g} |")
    W("")

    # -- 6. length asymmetry -----------------------------------------------
    W("## 6. Length asymmetry\n")
    ok = ds.dropna(subset=["L_plus", "L_minus"])
    W(f"* mean tokens: correct {ok['L_plus'].mean():.0f}, incorrect {ok['L_minus'].mean():.0f} "
      f"(L- - L+ = {ok['L_diff'].mean():+.0f} on average, "
      f"{int((ok['L_diff']>0).sum())}/{len(ok)} problems have longer failures)")
    for name, d in variants.items():
        dd = d[~d["degenerate"]].dropna(subset=["L_diff"])
        if len(dd) > 2:
            rr = spearmanr(dd["L_diff"], dd["C"])
            W(f"* {name}: spearman(L- - L+, C) = {rr.statistic:.3f} (p={rr.pvalue:.3g})")
    W("")

    # -- 7. per descriptor cell --------------------------------------------
    W("## 7. Sign distribution by descriptor cell (GROUP x SKILL)\n")
    W("| GROUP | SKILL | n | n degenerate | frac C<0 (sum) | frac C<0 (mean) | n significant (sum) |")
    W("|---|---|---|---|---|---|---|")
    for (grp, skl), gg in ds.groupby(["group", "skill"], sort=True):
        mm = dm[dm["program_id"].isin(gg["program_id"])]
        ndg = gg[~gg["degenerate"]]
        ndm = mm[~mm["degenerate"]]
        W(f"| {grp} | {skl} | {len(gg)} | {int(gg['degenerate'].sum())} | "
          f"{frac_neg(ndg['C']):.2f} | {frac_neg(ndm['C']):.2f} | {int(gg['significant'].sum())} |")
    W("")
    W("### By GROUP\n")
    W("| GROUP | n | frac C<0 (sum) | frac C<0 (mean) |")
    W("|---|---|---|---|")
    for grp, gg in ds.groupby("group", sort=True):
        mm = dm[dm["program_id"].isin(gg["program_id"])]
        W(f"| {grp} | {len(gg)} | {frac_neg(gg[~gg['degenerate']]['C']):.2f} | "
          f"{frac_neg(mm[~mm['degenerate']]['C']):.2f} |")
    W("")
    W("### By SKILL\n")
    W("| SKILL | n | frac C<0 (sum) | frac C<0 (mean) |")
    W("|---|---|---|---|")
    for skl, gg in ds.groupby("skill", sort=True):
        mm = dm[dm["program_id"].isin(gg["program_id"])]
        W(f"| {skl} | {len(gg)} | {frac_neg(gg[~gg['degenerate']]['C']):.2f} | "
          f"{frac_neg(mm[~mm['degenerate']]['C']):.2f} |")
    W("")

    W("## 8. Verdict against the pre-registered criterion\n")
    W("Criterion: if the negative fraction among problems with a SIGNIFICANT gap is "
      ">= ~0.75-0.80, the population is one-sided and |Cov| is a collapse accelerator "
      "(signed/clipped variant needed). Balanced signs make the "
      "'bidirectional leverage' reading defensible.\n")
    for name, d in variants.items():
        sig = d[d["significant"]]
        k = int((sig["C"] < 0).sum())
        n = len(sig)
        if n == 0:
            W(f"* {name}: no problem clears |gap| > 2 SE -- criterion not evaluable.")
            continue
        lo, hi = wilson(k, n)
        frac = k / n
        if lo >= 0.75:
            verdict = "ONE-SIDED (CI lower bound above the 0.75 threshold)"
        elif hi < 0.75:
            verdict = "NOT one-sided at the 0.75 threshold (CI entirely below)"
        else:
            verdict = "INCONCLUSIVE (CI straddles 0.75)"
        W(f"* {name}: {k}/{n} = {frac:.3f} negative, 95% CI [{lo:.3f}, {hi:.3f}] -> **{verdict}**")
    nd_all = merged[~merged["degenerate_sum"]]
    W(f"\n* sum/mean sign agreement = {nd_all['sign_agree'].mean():.3f}; below ~0.9 makes the "
      "choice of entropy variant a prerequisite question to the fitness redefinition.")
    W("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    merged.to_csv(out_dir / "per_problem_merged.csv", index=False)

    make_figures(ds, dm, fig_dir, args.label)
    print("\n".join(lines))
    print(f"\n[analyze] report -> {out_dir/'report.md'}")
    return 0


def make_figures(ds: pd.DataFrame, dm: pd.DataFrame, fig_dir: Path, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
    tag = f" ({label})" if label else ""

    # histograms of signed C
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for ax, (name, d) in zip(axes, [("H_sum", ds), ("H_mean", dm)]):
        nd = d[~d["degenerate"]]
        sig = nd[nd["significant"]]
        bins = np.histogram_bin_edges(nd["C"], bins=15)
        ax.hist(nd["C"], bins=bins, color="#9db4c0", edgecolor="white", label="all non-degenerate")
        ax.hist(sig["C"], bins=bins, color="#c1666b", edgecolor="white", label="significant gap")
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel(f"signed C = Cov(R, {name})")
        ax.set_ylabel("problems")
        ax.set_title(f"{name}: {frac_neg(nd['C']):.0%} negative{tag}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "hist_signed_C.png")
    plt.close(fig)

    # C vs s_hat and C vs length asymmetry
    for name, d in [("H_sum", ds), ("H_mean", dm)]:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
        for ax, xcol, xlab in [
            (axes[0], "s_hat", r"$\hat{s}$ (pass rate at G=32)"),
            (axes[1], "L_diff", r"$\bar{L}_- - \bar{L}_+$ (tokens)"),
        ]:
            sig = d["significant"]
            ax.scatter(d.loc[~sig, xcol], d.loc[~sig, "C"], s=26, c="#9db4c0",
                       edgecolor="k", linewidth=0.3, label="not significant")
            ax.scatter(d.loc[sig, xcol], d.loc[sig, "C"], s=32, c="#c1666b",
                       edgecolor="k", linewidth=0.3, label="significant")
            ax.axhline(0, color="k", lw=1)
            if xcol == "L_diff":
                ax.axvline(0, color="k", lw=0.6, ls=":")
            ax.set_xlabel(xlab)
            ax.set_ylabel(f"Cov(R, {name})")
            ax.legend(fontsize=7)
        fig.suptitle(f"signed covariance, {name}{tag}", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / f"scatter_{name}.png")
        plt.close(fig)

    # H_sum vs H_mean, coloured by length asymmetry: the sign flips are not
    # random, they concentrate where failures run much longer than successes.
    nd = ds[~ds["degenerate"]].merge(
        dm[["program_id", "C"]].rename(columns={"C": "C_mean"}), on="program_id"
    )
    if len(nd):
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        v = float(np.nanmax(np.abs(nd["L_diff"]))) or 1.0
        sc = ax.scatter(
            nd["C"], nd["C_mean"], c=nd["L_diff"], cmap="coolwarm",
            vmin=-v, vmax=v, s=34, edgecolor="k", linewidth=0.3,
        )
        ax.axhline(0, color="k", lw=1)
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel(r"Cov(R, $H_{sum}$)")
        ax.set_ylabel(r"Cov(R, $H_{mean}$)")
        agree = (np.sign(nd["C"]) == np.sign(nd["C_mean"])).mean()
        ax.set_title(f"sign agreement {agree:.0%}{tag}", fontsize=10)
        fig.colorbar(sc, ax=ax, label=r"$\bar{L}_- - \bar{L}_+$ (tokens)")
        fig.tight_layout()
        fig.savefig(fig_dir / "scatter_sum_vs_mean.png")
        plt.close(fig)

    # rank agreement between current fitness and |C|
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for ax, (name, d) in zip(axes, [("H_sum", ds), ("H_mean", dm)]):
        ax.scatter(d["rq_now"].rank(), d["absC"].rank(), s=28, c="#5b8c85",
                   edgecolor="k", linewidth=0.3)
        ax.set_xlabel(r"rank of $\hat{s}(1-\hat{s})\hat{U}$")
        ax.set_ylabel("rank of |C|")
        ax.set_title(f"{name}{tag}")
    fig.tight_layout()
    fig.savefig(fig_dir / "rank_fitness_vs_absC.png")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
