"""Claim-evidence figures from archived R_Q-Evolve snapshots."""
import argparse, glob, json, re, io, pickle, zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from viz_common import (
    contiguous_ranges,
    family_history,
    load_snapshots,
    operator_of,
)

parser = argparse.ArgumentParser()
parser.add_argument("--rq-output", type=Path, required=True)
parser.add_argument("--analysis-output", type=Path, required=True)
parser.add_argument("--peak-step", type=int, default=None)
parser.add_argument("--peak-outer-iteration", type=int, default=None)
parser.add_argument("--max-outer-iteration", type=int, default=None)
args = parser.parse_args()
BASE = args.rq_output / "rq_archive"
OUT = args.analysis_output
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"seed":"#888681", "in_breadth":"#eb6834", "in_depth":"#2a78d6"}
LABELS = {"seed":"seed", "in_breadth":"breadth", "in_depth":"depth"}
def checkpoint_outer_iteration(rq_output, global_step):
    p = rq_output / f"global_step_{global_step}" / "data.pt"
    if not p.exists():
        return None
    class U(pickle.Unpickler):
        def persistent_load(self, pid):
            return None
    try:
        with zipfile.ZipFile(p) as z:
            payload = U(io.BytesIO(z.read("data/data.pkl"))).load()
        return payload["_sampler_iter_state"]["sampler_state"].get("active_iteration")
    except Exception:
        return None

def read_eval_scores(rq_output):
    scores_file = rq_output / "scores.md"
    headers, rows = [], {}
    if scores_file.exists():
        in_final = False
        for line in scores_file.read_text().splitlines():
            if line.startswith("## pass@1 (final"):
                in_final = True
                continue
            if in_final and line.startswith("## pass@1 (pre-GPT"):
                break
            if in_final and line.startswith("| step |"):
                headers = [x.strip() for x in line.strip().strip("|").split("|")]
            elif in_final and headers and re.match(r"^\|\s*\d+\s*\|", line):
                cells = [x.strip() for x in line.strip().strip("|").split("|")]
                try:
                    rows[int(cells[0])] = {h: float(v) for h, v in zip(headers[1:], cells[1:])}
                except ValueError:
                    pass
    if not rows:
        for p in rq_output.glob("global_step_*/eval/*/summary.json"):
            m = re.search(r"global_step_(\d+)", str(p))
            if not m:
                continue
            data = json.loads(p.read_text())
            for name, rec in data.get("benchmarks", {}).items():
                rows.setdefault(int(m.group(1)), {})[name] = 100 * rec["pass_at_1"]
    for step, vals in rows.items():
        if "AVG" not in vals:
            vals["AVG"] = sum(vals.values()) / len(vals) if vals else 0.0
    return dict(sorted(rows.items()))

eval_scores = read_eval_scores(args.rq_output)
PEAK_GLOBAL_STEP = args.peak_step or (max(eval_scores, key=lambda s: eval_scores[s].get("AVG", -1)) if eval_scores else None)
PEAK_OUTER_ITER = args.peak_outer_iteration if args.peak_outer_iteration is not None else (
    checkpoint_outer_iteration(args.rq_output, PEAK_GLOBAL_STEP) if PEAK_GLOBAL_STEP is not None else None
)

snapshots = load_snapshots(BASE, args.max_outer_iteration)
presence, first = family_history(snapshots, BASE / "evolution_log.jsonl")
final_rows = []
for it, data in snapshots:
    for c in data["champions"]:
        if it == snapshots[-1][0]:
            final_rows.append(c)

types = sorted(presence, key=lambda t: (first[t][0], t))
last_it = snapshots[-1][0]

plt.rcParams.update({
    "font.family":"DejaVu Sans", "figure.facecolor":"white", "axes.facecolor":"white",
    "axes.edgecolor":"#d9d8d4", "axes.grid":True, "grid.color":"#e9e8e5",
    "grid.linewidth":0.6, "axes.axisbelow":True,
})

# Figure 1: family lifecycle and archive retention.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios":[1.8, 1]})
for row, typ in enumerate(types):
    its = presence[typ]
    op = first[typ][1]
    for start, end in contiguous_ranges(its):
        ax1.plot([start, end], [row, row], color=COLORS[op], lw=5,
                 solid_capstyle="round", alpha=.82)
    ax1.scatter([first[typ][0]], [row], color=COLORS[op], s=32, zorder=3)
    if last_it in its:
        ax1.scatter([last_it], [row], facecolors="white", edgecolors="#111111", s=56, linewidths=1.3, zorder=4)
ax1.set_yticks(range(len(types)))
ax1.set_yticklabels([t.replace(".", " · ") for t in types], fontsize=7)
ax1.set_xlabel("outer iteration")
ax1.set_title("A · Family lifecycle in the archive", loc="left")
ax1.set_xlim(0, last_it + 2)
if PEAK_OUTER_ITER is not None:
    ax1.axvline(PEAK_OUTER_ITER, color="#111111", ls="--", lw=1.1, alpha=.8)
    ax1.text(PEAK_OUTER_ITER + .6, len(types) - .8, f"global step {PEAK_GLOBAL_STEP}\nouter iter {PEAK_OUTER_ITER}", fontsize=8, va="top", color="#333333")
elif PEAK_GLOBAL_STEP is not None:
    ax1.text(.99, .02, f"peak: global step {PEAK_GLOBAL_STEP}\nouter iteration unavailable", transform=ax1.transAxes, ha="right", va="bottom", fontsize=8, color="#55524e")
ax1.text(.01, .985, "line segments = observed champion presence; open circle = retained at final snapshot", transform=ax1.transAxes, fontsize=8, color="#55524e", va="top")
for op in ["seed", "in_breadth", "in_depth"]:
    ax1.plot([], [], color=COLORS[op], lw=5, label=LABELS[op])
ax1.legend(frameon=False, loc="lower right")

# Retention bars based on first-introduced operator.
for op in ["seed", "in_breadth", "in_depth"]:
    introduced = {t for t, (_, o) in first.items() if o == op}
    retained = {t for t in introduced if last_it in presence[t]}
    retention = len(retained) / len(introduced) if introduced else 0.0
    ax2.barh(LABELS[op], retention, color=COLORS[op], alpha=.88)
    ax2.text(retention + .025, LABELS[op], f"{len(retained)}/{len(introduced)}", va="center", fontsize=10)
ax2.set_xlim(0, 1.15)
ax2.set_xlabel("fraction retained at final snapshot")
ax2.set_title("B · Retention by introducing operator", loc="left")
ax2.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
fig.suptitle(f"Problem-family lifecycle and archive retention · {args.rq_output.name}", x=.05, y=.99, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .95])
fig.savefig(OUT / "family_lifecycle_retention.png", dpi=220)
fig.savefig(OUT / "family_lifecycle_retention.pdf")
plt.close(fig)

# Figure 2: final curriculum and training trajectory.
logs = [json.loads(x) for x in (BASE / "evolution_log.jsonl").read_text().splitlines()]
if args.max_outer_iteration is not None:
    logs = [
        row for row in logs
        if int(row.get("metrics", {}).get("outer_iteration", row.get("iteration", -1)))
        <= args.max_outer_iteration
    ]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios":[1.05, 1.2, 1.35]})
ax = axes[0]
for c in final_rows:
    op = operator_of(c)
    p, h, rq = c.get("p_hat", 0), c.get("h_score", 0), c.get("rq_score", 0)
    ax.scatter(p, h, s=80 + 2600*rq, color=COLORS[op], edgecolor="white", linewidth=.8, alpha=.9)
ax.axvspan(0, 1, color="#f1f0ed", zorder=0)
ax.set_xlim(-.03, 1.03); ax.set_xlabel("Solver pass rate $\\hat p$"); ax.set_ylabel("uncertainty $H$")
ax.set_title("A · Final archive curriculum", loc="left")
ax.text(.03, .96, "size = $R_Q$", transform=ax.transAxes, va="top", fontsize=8, color="#55524e")
for op in ["seed", "in_breadth", "in_depth"]:
    ax.scatter([], [], color=COLORS[op], s=55, label=LABELS[op])
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
its = [r["metrics"]["outer_iteration"] for r in logs]
ax.plot(its, [r["metrics"]["dataset_size"] for r in logs], color="#222222", lw=2, label="dataset")
ax.plot(its, [r["metrics"]["num_champions"] for r in logs], color="#eb6834", lw=2, label="champions")
if PEAK_OUTER_ITER is not None:
    ax.axvline(PEAK_OUTER_ITER, color="#111111", ls="--", lw=1.1, label=f"step {PEAK_GLOBAL_STEP} peak")
ax.set_xlabel("outer iteration"); ax.set_ylabel("count")
ax.set_title("B · Curriculum/archive size", loc="left")
ax.legend(frameon=False, fontsize=8)

ax = axes[2]
steps = list(eval_scores)
preferred = ["AVG", "math500", "olympiadbench", "aime25"]
plot_names = [n for n in preferred if any(n in eval_scores[s] for s in steps)]
for name in plot_names:
    vals = [eval_scores[s].get(name, float("nan")) for s in steps]
    ax.plot(steps, vals, marker="o", lw=2.6 if name == "AVG" else 1.2, label=name)
if PEAK_GLOBAL_STEP is not None:
    peak_y = eval_scores[PEAK_GLOBAL_STEP].get("AVG", 0.0)
    ax.axvline(PEAK_GLOBAL_STEP, color="#99958f", ls="--", lw=1)
    label = f"peak: step {PEAK_GLOBAL_STEP}"
    if PEAK_OUTER_ITER is not None:
        label += f"\nouter iter {PEAK_OUTER_ITER}"
    else:
        label += "\nouter iter unavailable"
    ax.annotate(label, (PEAK_GLOBAL_STEP, peak_y), xytext=(PEAK_GLOBAL_STEP + 5, peak_y + 4), arrowprops={"arrowstyle":"->", "color":"#55524e"}, fontsize=8)
ax.set_xlabel("global step"); ax.set_ylabel("pass@1 (%)"); ax.set_ylim(0, 100)
ax.set_title("C · Downstream Solver performance", loc="left")
ax.legend(frameon=False, fontsize=7, ncol=2)
fig.suptitle(f"R_Q curriculum and downstream Solver outcome · {args.rq_output.name}", x=.05, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .93])
fig.savefig(OUT / "rq_curriculum_solver_outcome.png", dpi=220)
fig.savefig(OUT / "rq_curriculum_solver_outcome.pdf")
plt.close(fig)
print("saved", OUT / "family_lifecycle_retention.png")
print("saved", OUT / "rq_curriculum_solver_outcome.png")
