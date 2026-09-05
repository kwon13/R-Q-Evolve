#!/usr/bin/env python3
"""Compare frozen Omni-MATH-2 runs without dropping any of the 35 MAP cells.

Journal files use the last record per id. Final reports require successful,
hash-matched predictions and judgements for every eligible, gradable manifest
row, including rows whose problem type abstains. --allow-partial is diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DOMAINS = (
    "Algebra", "Geometry", "Number Theory", "Discrete Mathematics",
    "Applied Mathematics", "Calculus", "Precalculus",
)
TYPES = ("decision", "search", "counting", "optimization", "function")
CELLS = tuple((domain, kind) for domain in DOMAINS for kind in TYPES)
DOMAIN_LOOKUP = {value.lower().replace(" ", "_"): value for value in DOMAINS}
COLORS = {"rzero": "#677787", "rq": "#128783"}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path, *, journal: bool = False) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["id"])
            if key in rows and not journal:
                raise ValueError(f"Duplicate id {key} in {path}:{lineno}")
            rows[key] = row
    return rows


def normalize_domains(values: list[str]) -> list[str]:
    result = []
    for value in values:
        normalized = DOMAIN_LOOKUP.get(value.lower().replace(" ", "_"))
        if normalized is None:
            raise ValueError(f"Unknown domain {value!r}")
        if normalized not in result:
            result.append(normalized)
    return result


def load_manifest(path: Path) -> dict[str, dict]:
    manifest = read_jsonl(path)
    for key, row in manifest.items():
        actual = hashlib.sha256(row["problem"].encode("utf-8")).hexdigest()
        if actual != row["problem_sha256"]:
            raise ValueError(f"Problem hash mismatch for id {key}")
        row["domains"] = normalize_domains(row["domains"])
        if row.get("problem_type") not in (*TYPES, None):
            raise ValueError(f"Unknown problem type for id {key}")
        for flag in ("eligible", "gradable"):
            if not isinstance(row.get(flag), bool):
                raise ValueError(f"Manifest {key}: {flag} must be boolean")
    return manifest


def load_run(root: Path, manifest: dict[str, dict]) -> tuple[dict[str, bool], dict]:
    predictions = read_jsonl(root / "predictions.jsonl", journal=True)
    judgements = read_jsonl(root / "judgements.jsonl", journal=True)
    unknown = (set(predictions) | set(judgements)) - set(manifest)
    if unknown:
        raise ValueError(f"Unknown ids in {root}: {sorted(unknown)[:5]}")
    config_path = root / "inference_config.json"
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.exists() else None
    if config_hash is not None:
        # Validate every saved prediction, including intentionally ungradable
        # rows. Shared display settings alone do not prove run provenance.
        for key, prediction in predictions.items():
            if prediction.get("inference_config_sha256") != config_hash:
                raise ValueError(f"Prediction inference config hash mismatch in {root}, id {key}")
    successful: dict[str, bool] = {}
    failures: dict[str, str] = {}
    truncated = 0
    judge_protocols = set()
    missing_judge_provenance = 0
    for key, row in manifest.items():
        if not (row["eligible"] and row["gradable"]):
            continue
        prediction = predictions.get(key)
        judgement = judgements.get(key)
        if prediction is None:
            failures[key] = "missing_prediction"
            continue
        response_hash = hashlib.sha256(prediction["response"].encode("utf-8")).hexdigest()
        if response_hash != prediction.get("response_sha256"):
            raise ValueError(f"Corrupt prediction hash in {root}, id {key}")
        if prediction.get("problem_sha256", row["problem_sha256"]) != row["problem_sha256"]:
            raise ValueError(f"Prediction problem hash mismatch in {root}, id {key}")
        truncated += prediction.get("finish_reason") == "length"
        if judgement is None:
            failures[key] = "missing_judgement"
        elif judgement.get("response_sha256") != response_hash:
            failures[key] = "stale_judgement_hash"
        elif judgement.get("status") != "ok":
            failures[key] = "judge_" + str(judgement.get("status"))
        elif not isinstance(judgement.get("correct"), bool):
            failures[key] = "invalid_correct_field"
        else:
            if "judge_settings" in judgement and "input_sha256" in judgement:
                # Exactly matches judge_omni_math2.Judge.identities. This also
                # protects against a reference-answer/QA change with the same id.
                cache_key = sha256(canonical([row["problem"], row["answer"], prediction["response"], judgement["judge_settings"]]))
                fingerprint = sha256(canonical([cache_key, row.get("eligible"), row.get("gradable"), row.get("qa_flags")]))
                if judgement["input_sha256"] != fingerprint:
                    failures[key] = "stale_judgement_input"
                    continue
                judge_protocols.add(canonical(judgement["judge_settings"]))
            else:
                missing_judge_provenance += 1
            successful[key] = judgement["correct"]
    counts: dict[str, int] = {}
    for reason in failures.values():
        counts[reason] = counts.get(reason, 0) + 1
    return successful, {
        "run": root.name, "successful": len(successful), "failure_counts": counts,
        "failure_ids": failures, "length_limited_predictions": truncated,
        "judge_protocols": [json.loads(p) for p in sorted(judge_protocols)],
        "missing_judge_provenance": missing_judge_provenance,
        "inference_config_sha256": config_hash,
    }


def verify_protocol_pair(left_root: Path, right_root: Path, manifest_hash: str, audits: list[dict]) -> dict:
    configs = []
    fields = ("sampling", "eos_token_id", "system_prompt", "chat_template_sha256", "prompt_probe_sha256", "packages")
    for root in (left_root, right_root):
        config = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))
        if config.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"Inference manifest hash mismatch in {root}")
        if any(field not in config for field in fields):
            raise ValueError(f"Incomplete inference protocol in {root}")
        configs.append(config)
    for field in fields:
        if configs[0][field] != configs[1][field]:
            raise ValueError(f"Inference protocol mismatch between runs: {field}")
    for audit in audits:
        if audit["missing_judge_provenance"] or len(audit["judge_protocols"]) != 1:
            raise ValueError(f"Missing or mixed judge protocol in {audit['run']}")
    if audits[0]["judge_protocols"] != audits[1]["judge_protocols"]:
        raise ValueError("Judge settings mismatch between compared runs")
    return {
        "generation_settings_equal": True,
        "manifest_hashes_equal": True,
        "rendered_prompt_probe_equal": True,
        "judge_settings_equal": True,
        "shared_generation_settings": {field: configs[0][field] for field in fields},
        "shared_judge_settings": audits[0]["judge_protocols"][0],
    }


def wilson_interval(successes: float, n: int, z: float = 1.959963984540054) -> tuple:
    if n == 0:
        return None, None
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, center - radius), min(1.0, center + radius)


def finite_interval(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"low": None, "high": None, "valid_replicates": 0}
    low, high = np.quantile(finite, [0.025, 0.975])
    return {"low": float(low), "high": float(high), "valid_replicates": len(finite)}


def ranking_metrics(scores: np.ndarray) -> dict[str, float]:
    """Fixed observed support; do not call this a 35-cell metric if support <35."""
    tail_size = max(1, math.ceil(0.2 * len(scores)))
    return {
        "equal_cell_macro": float(np.mean(scores)),
        "lower_20pct_cell_mean": float(np.mean(np.sort(scores)[:tail_size])),
        "observed_minimum": float(np.min(scores)),
        "observed_maximum": float(np.max(scores)),
    }


def compare(
    manifest: dict[str, dict], left: dict[str, bool], right: dict[str, bool],
    *, bootstrap: int = 10000, seed: int = 271828, allow_partial: bool = False,
) -> tuple[list[dict], dict]:
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    required = {key for key, row in manifest.items() if row["eligible"] and row["gradable"]}
    paired = required & left.keys() & right.keys()
    if paired != required and not allow_partial:
        raise ValueError(
            f"Incomplete paired evaluation: {len(paired)}/{len(required)} eligible and "
            "gradable rows. Refusing final figures; use --allow-partial for diagnostics."
        )
    if not paired:
        raise ValueError("No successfully judged paired rows")
    # Original manifest order is preserved; no model-dependent row selection.
    ids = [key for key in manifest if key in paired]
    hashes = list(dict.fromkeys(manifest[key]["problem_sha256"] for key in ids))
    cluster_index = {value: index for index, value in enumerate(hashes)}
    count = np.zeros((len(hashes), 35), dtype=float)
    sums_left = np.zeros_like(count)
    sums_right = np.zeros_like(count)
    cell_index = {value: index for index, value in enumerate(CELLS)}
    mapped = 0
    for key in ids:
        row = manifest[key]
        if row["problem_type"] is None or not row["domains"]:
            continue
        mapped += 1
        cluster = cluster_index[row["problem_sha256"]]
        for domain in row["domains"]:
            cell = cell_index[(domain, row["problem_type"])]
            count[cluster, cell] += 1
            sums_left[cluster, cell] += left[key]
            sums_right[cluster, cell] += right[key]
    n = count.sum(axis=0)
    left_sum = sums_left.sum(axis=0)
    right_sum = sums_right.sum(axis=0)
    support = n > 0
    left_acc = np.divide(left_sum, n, out=np.full(35, np.nan), where=support)
    right_acc = np.divide(right_sum, n, out=np.full(35, np.nan), where=support)
    delta = right_acc - left_acc
    unique_n = (count > 0).sum(axis=0)
    unique_left = np.divide(sums_left, count, out=np.zeros_like(count), where=count > 0).sum(axis=0)
    unique_right = np.divide(sums_right, count, out=np.zeros_like(count), where=count > 0).sum(axis=0)
    unique_left = np.divide(unique_left, unique_n, out=np.full(35, np.nan), where=unique_n > 0)
    unique_right = np.divide(unique_right, unique_n, out=np.full(35, np.nan), where=unique_n > 0)

    rng = np.random.default_rng(seed)
    bootstrap_left = np.full((bootstrap, 35), np.nan)
    bootstrap_right = np.full_like(bootstrap_left, np.nan)
    # Jointly resample exact-problem clusters, preserving both model outcomes,
    # duplicate rows, and all overlapping domain memberships of each problem.
    for start in range(0, bootstrap, 128):
        stop = min(start + 128, bootstrap)
        weights = rng.multinomial(len(hashes), np.full(len(hashes), 1 / len(hashes)), size=stop-start)
        denominator = weights @ count
        bootstrap_left[start:stop] = np.divide(
            weights @ sums_left, denominator,
            out=np.full_like(denominator, np.nan), where=denominator > 0,
        )
        bootstrap_right[start:stop] = np.divide(
            weights @ sums_right, denominator,
            out=np.full_like(denominator, np.nan), where=denominator > 0,
        )
    bootstrap_delta = bootstrap_right - bootstrap_left
    rows = []
    for i, (domain, kind) in enumerate(CELLS):
        low_left, high_left = wilson_interval(left_sum[i], int(n[i]))
        low_right, high_right = wilson_interval(right_sum[i], int(n[i]))
        interval = finite_interval(bootstrap_delta[:, i])
        rows.append({
            "domain": domain, "problem_type": kind, "n": int(n[i]),
            "unique_problems": int(unique_n[i]),
            "rzero_correct": int(left_sum[i]), "rq_correct": int(right_sum[i]),
            "rzero_acc": float(left_acc[i]) if support[i] else None,
            "rq_acc": float(right_acc[i]) if support[i] else None,
            "delta_pp": float(delta[i] * 100) if support[i] else None,
            "rzero_wilson_low": low_left, "rzero_wilson_high": high_left,
            "rq_wilson_low": low_right, "rq_wilson_high": high_right,
            "delta_ci_low_pp": interval["low"] * 100 if interval["low"] is not None else None,
            "delta_ci_high_pp": interval["high"] * 100 if interval["high"] is not None else None,
            "delta_ci_valid_replicates": interval["valid_replicates"],
            "deduplicated_rzero_acc": float(unique_left[i]) if support[i] else None,
            "deduplicated_rq_acc": float(unique_right[i]) if support[i] else None,
            "deduplicated_delta_pp": float((unique_right[i] - unique_left[i]) * 100) if support[i] else None,
            "small_n_warning": bool(n[i] < 20) if support[i] else None,
        })

    supported_n = int(support.sum())
    metrics = {}
    deduplicated_metrics = {}
    extrema = {}
    if supported_n:
        point_left = ranking_metrics(left_acc[support])
        point_right = ranking_metrics(right_acc[support])
        dedup_left = ranking_metrics(unique_left[support])
        dedup_right = ranking_metrics(unique_right[support])
        deduplicated_metrics = {
            key: {"rzero": dedup_left[key], "rq": dedup_right[key], "delta_pp": 100*(dedup_right[key]-dedup_left[key])}
            for key in dedup_left
        }
        joint_valid = np.all(np.isfinite(bootstrap_left[:, support]), axis=1)
        bootstrap_metrics = {
            key: [] for key in ("equal_cell_macro", "lower_20pct_cell_mean", "observed_minimum", "observed_maximum")
        }
        for a, b in zip(bootstrap_left[joint_valid][:, support], bootstrap_right[joint_valid][:, support]):
            a_metrics, b_metrics = ranking_metrics(a), ranking_metrics(b)
            for key in bootstrap_metrics:
                bootstrap_metrics[key].append(b_metrics[key] - a_metrics[key])
        for key in point_left:
            interval = finite_interval(np.asarray(bootstrap_metrics[key]))
            metrics[key] = {
                "rzero": point_left[key], "rq": point_right[key],
                "delta_pp": 100 * (point_right[key] - point_left[key]),
                "conditional_delta_ci_low_pp": None if interval["low"] is None else 100*interval["low"],
                "conditional_delta_ci_high_pp": None if interval["high"] is None else 100*interval["high"],
                "conditional_ci_valid_replicates": interval["valid_replicates"],
            }
        for model, score in (("rzero", left_acc), ("rq", right_acc)):
            extrema[model] = {}
            for label, value in (("observed_lowest", np.nanmin(score)), ("observed_highest", np.nanmax(score))):
                extrema[model][label] = [
                    {"domain": row["domain"], "problem_type": row["problem_type"], "n": row["n"], "accuracy": float(value)}
                    for i, row in enumerate(rows) if support[i] and np.isclose(score[i], value, rtol=0, atol=1e-12)
                ]
    summary = {
        "diagnostic_partial": len(paired) != len(required),
        "eligible_gradable_rows": len(required), "paired_rows": len(ids),
        "unique_problem_clusters": len(hashes), "mapped_rows": mapped,
        "unmapped_rows": len(ids)-mapped, "overlapping_cell_memberships": int(n.sum()),
        "observed_cells": supported_n, "total_cells": 35,
        "full_35_cell_ranking_observable": supported_n == 35 and paired == required,
        "metrics_scope": "all_35_cells" if supported_n == 35 else f"observed_{supported_n}_cells_only",
        "lower_tail_cell_count": math.ceil(0.2*supported_n),
        "overall_rzero_acc": sum(left[key] for key in ids)/len(ids),
        "overall_rq_acc": sum(right[key] for key in ids)/len(ids),
        "improved_cells": int(np.sum(delta > 1e-12)),
        "declined_cells": int(np.sum(delta < -1e-12)),
        "tied_cells": int(np.sum(support & (np.abs(delta) <= 1e-12))),
        "bootstrap_replicates": bootstrap, "bootstrap_seed": seed,
        "metrics": metrics, "deduplicated_metrics": deduplicated_metrics, "extrema": extrema,
        "limitations": [
            "Observed extrema are sample extrema, not established population worst/best cells.",
            "No minimum-n selection: all35 coordinates appear; n=0 is N/A, never zero accuracy.",
            "Domain memberships overlap; cell observations and comparisons are not independent.",
            "Frozen problem-type labels are provisional and may abstain; unlabeled rows are not invented.",
            "Wilson intervals are pointwise row-binomial intervals; exact duplicates may violate row independence.",
            "Delta intervals jointly resample paired exact-problem clusters; intervals are pointwise, not simultaneous.",
            "Cell bootstrap quantiles exclude replicates where that cell is empty (valid replicate counts are provided).",
            "Aggregate bootstrap intervals condition on every originally observed cell being represented; small-cell support loss limits their interpretation.",
            "Extremum and lower-tail membership is recomputed within every retained joint bootstrap replicate.",
            "Deduplicated sensitivity averages duplicate-row outcomes within exact statements, then weights statements equally within each cell.",
            "A confidence interval covering zero is not evidence of no regression; no multiplicity-adjusted significance claim is made.",
        ],
    }
    return rows, summary


def plot_figures(rows: list[dict], summary: dict, output: Path, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    diagnostic = "DIAGNOSTIC — INCOMPLETE PAIRED EVALUATION\n" if summary["diagnostic_partial"] else ""
    provenance = f"{summary['paired_rows']:,} paired scorable rows; {summary['mapped_rows']:,} mapped · overlapping domains · frozen provisional types"
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 7.2), gridspec_kw={"width_ratios": [1, 1, 1]})
    for panel, (field, title) in enumerate((("rzero_acc", "R-Zero accuracy (%)"), ("rq_acc", "RQ-Evolve accuracy (%)"), ("delta_pp", "RQ-Evolve − R-Zero (pp)"))):
        ax = axes[panel]
        values = np.array([np.nan if row[field] is None else row[field]*(100 if field != "delta_pp" else 1) for row in rows]).reshape(7, 5)
        if field == "delta_pp":
            finite_values = values[np.isfinite(values)]
            limit = max(10.0, float(np.max(np.abs(finite_values))) if len(finite_values) else 10.0)
            cmap = plt.cm.RdBu.with_extremes(bad="#eceff1")
            norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        else:
            cmap = plt.cm.Blues.with_extremes(bad="#eceff1")
            norm = colors.Normalize(0, 100)
        im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        for i, row in enumerate(rows):
            y, x = divmod(i, 5)
            value = values[y, x]
            if not np.isfinite(value):
                text = "N/A\nn=0"
                color = "#646b73"
            else:
                text = (f"{value:+.1f}" if field == "delta_pp" else f"{value:.1f}") + f"\nn={row['n']}"
                shade = norm(value)
                color = "white" if (field == "delta_pp" and (shade < 0.16 or shade > 0.84)) or (field != "delta_pp" and shade > 0.68) else "#18242e"
            ax.text(x, y, text, color=color, ha="center", va="center", fontsize=10)
        ax.set_xticks(range(5), [t.title() for t in TYPES], rotation=35, ha="right")
        ax.set_yticks(range(7), DOMAINS if panel == 0 else [])
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        ax.set_title(title, fontsize=12, pad=12)
        fig.colorbar(im, ax=ax, shrink=.78, fraction=.046, pad=.025)
    fig.suptitle(f"{diagnostic}Omni-MATH-2 · {label}\nAll 35 domain × problem-type cells", fontsize=16, y=.98)
    fig.text(.5, .022, provenance + "\nObserved sample scores; tiny cells have large uncertainty (see interval plot). N/A ≠ 0%.", ha="center", fontsize=9, color="#505761")
    fig.tight_layout(rect=(0, .105, 1, .89))
    save_figure(fig, output / "all35_performance_map")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.3, 14.0))
    for i, row in enumerate(rows):
        if row["n"] == 0:
            ax.text(50, i, "N/A — no scorable mapped problem", ha="center", va="center", color="#828991", fontsize=9)
            continue
        a, b = 100*row["rzero_acc"], 100*row["rq_acc"]
        ax.plot([a, b], [i, i], color="#ccd2d7", linewidth=2, zorder=1)
        for model, value, offset, marker in (("rzero", a, -.13, "o"), ("rq", b, .13, "s")):
            lower, upper = row[f"{model}_wilson_low"]*100, row[f"{model}_wilson_high"]*100
            ax.errorbar(value, i+offset, xerr=[[max(0, value-lower)], [max(0, upper-value)]], fmt=marker, markersize=4.5, elinewidth=1.05, capsize=2, color=COLORS[model], label={"rzero":"R-Zero", "rq":"RQ-Evolve"}[model] if i == next(j for j,r in enumerate(rows) if r["n"]) else None, zorder=2)
    labels = [f"{r['domain']} × {r['problem_type'].title()}   (n={r['n']})" for r in rows]
    ax.set_yticks(range(35), labels, fontsize=9)
    ax.set_ylim(34.6, -.7)
    ax.set_xlim(-2, 102)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xlabel("Accuracy (%) · pointwise Wilson 95% intervals")
    ax.xaxis.grid(True, alpha=.2)
    for y in (4.5, 9.5, 14.5, 19.5, 24.5, 29.5):
        ax.axhline(y, color="#e1e5e9", linewidth=.8)
    ax.legend(loc="lower right", frameon=True)
    ax.set_title(f"{diagnostic}Omni-MATH-2 · {label}\nAll 35 cells, fixed taxonomy order; no minimum-n filter", fontsize=15, pad=17)
    fig.text(.5, .014, "Observed weakest/strongest cells are uncertain, especially at small n. Intervals are not simultaneous.\nWilson intervals assume independent rows; paired problem-cluster delta intervals and dedup sensitivity are in the CSV.", ha="center", color="#505761", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, 1))
    save_figure(fig, output / "all35_accuracy_intervals")
    plt.close(fig)


def save_figure(fig: Any, basename: Path) -> None:
    for extension in ("svg", "pdf", "png"):
        fig.savefig(basename.with_suffix("." + extension), bbox_inches="tight", dpi=180)


def write_report(rows: list[dict], summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "cell_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--comparisons", nargs="+", default=["primary=rzero_256:rq_256", "secondary=rzero_96:rq_224"])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.input_dir / "manifest.jsonl")
    manifest_hash = hashlib.sha256((args.input_dir / "manifest.jsonl").read_bytes()).hexdigest()
    destination = args.output_dir or args.input_dir / ("diagnostic_figures" if args.allow_partial else "figures")
    prepared = []
    # Validate every requested run before writing any final chart.
    for spec in args.comparisons:
        name, run_pair = spec.split("=", 1)
        left_name, right_name = run_pair.split(":", 1)
        left, left_audit = load_run(args.input_dir / "runs" / left_name, manifest)
        right, right_audit = load_run(args.input_dir / "runs" / right_name, manifest)
        protocol_audit = None
        if not args.allow_partial:
            protocol_audit = verify_protocol_pair(
                args.input_dir / "runs" / left_name, args.input_dir / "runs" / right_name,
                manifest_hash, [left_audit, right_audit],
            )
        rows, summary = compare(manifest, left, right, bootstrap=args.bootstrap, seed=args.seed, allow_partial=args.allow_partial)
        summary["comparison"] = name
        summary["run_names"] = {"rzero": left_name, "rq": right_name}
        summary["run_audits"] = [left_audit, right_audit]
        summary["manifest_sha256"] = manifest_hash
        summary["protocol_audit"] = protocol_audit
        # Explicit diagnostic request remains labelled even when data are complete.
        summary["diagnostic_partial"] = summary["diagnostic_partial"] or args.allow_partial
        prepared.append((name, rows, summary))
    for name, rows, summary in prepared:
        output = destination / name
        left_step = summary['run_names']['rzero'].rsplit("_", 1)[-1]
        right_step = summary['run_names']['rq'].rsplit("_", 1)[-1]
        label = f"R-Zero step {left_step} vs RQ-Evolve step {right_step}"
        write_report(rows, summary, output)
        plot_figures(rows, summary, output, label)
        print(json.dumps({"comparison": name, "output": str(output), "paired_rows": summary["paired_rows"], "observed_cells": summary["observed_cells"]}))


if __name__ == "__main__":
    main()
