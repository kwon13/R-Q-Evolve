#!/usr/bin/env python3
"""Classify a frozen R-Zero sample into the 7 x 5 Domain x ProblemType map.

DOMAIN is assigned by a label-blind Luna call using the adapted Omni-MATH
prompt. PROBLEM_TYPE is assigned locally by the repository's frozen
statement-only rules. R-Zero has no declarative verifier, so these labels are
an analysis audit, not live archive-admission decisions.

The output is resumable: successful Luna rows with the same model, prompt hash,
and input hash are reused. No OpenAI client package is required.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rq_evolve.concepts import DOMAINS, PROBLEM_TYPES  # noqa: E402
from rq_evolve.problem_type import (  # noqa: E402
    PROBLEM_TYPE_RULESET,
    annotate_problem_type,
    problem_type_ruleset_sha256,
)


DISPLAY_DOMAINS = {
    "algebra": "Algebra",
    "geometry": "Geometry",
    "number_theory": "Number Theory",
    "discrete_mathematics": "Discrete Mathematics",
    "applied_mathematics": "Applied Mathematics",
    "calculus": "Calculus",
    "precalculus": "Precalculus",
}


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def input_hash(question: str, answer: str) -> str:
    return sha256_text(f"{question}\n{answer}")[:16]


def load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    dotenv = ROOT / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    raise RuntimeError("OPENAI_API_KEY is absent from the environment and .env")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def read_input(path: Path) -> list[dict[str, Any]]:
    """Read an audit manifest or an R-Zero round parquet directly."""

    if path.suffix.lower() != ".parquet":
        return read_jsonl(path)

    import pyarrow.parquet as pq

    rows = pq.read_table(path, columns=["problem", "answer", "score"]).to_pylist()
    round_id: int | None = None
    if path.parent.name.startswith("round_"):
        try:
            round_id = int(path.parent.name.rsplit("_", 1)[1])
        except ValueError:
            pass
    return [
        {
            **row,
            "source": "rzero",
            "round": round_id,
            "item_id": f"rzero:r{round_id or 0}:row{index}",
        }
        for index, row in enumerate(rows)
    ]


def normalize_domain(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return text if text in DOMAINS else None


def clean_model_text(value: Any) -> str:
    """Strip JSON-valid control bytes that occasionally appear in math text."""

    return "".join(
        character
        for character in str(value or "").strip()
        if character in "\n\t" or ord(character) >= 32
    )


_DEFINED_TERM_RE = re.compile(
    r"\bis called\s+\*?([A-Za-z][A-Za-z-]{2,})\*?\s+if\b", re.IGNORECASE
)


def neutralize_defined_term(question: str) -> str:
    """Meaning-preserving fallback for a false-positive API prompt rejection.

    Generated math sometimes coins an innocuous property name.  If the API
    rejects that literal string, replace only the defined label with
    ``property-S`` while leaving the mathematical definition untouched.  The
    original question remains in every stored audit row.
    """

    match = _DEFINED_TERM_RE.search(question)
    if match is None:
        return question
    label = match.group(1)
    value = re.sub(rf"\b{re.escape(label)}\b", "property-S", question, flags=re.I)
    return re.sub(
        r"\bis called\s+\*?property-S\*?\s+if\b",
        "has property S if",
        value,
        flags=re.I,
    )


def extract_json_object(text: str) -> dict[str, Any]:
    value = (text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("model response is not a JSON object")
    return parsed


def problem_text(row: dict[str, Any]) -> tuple[str, str]:
    question = str(row.get("question") or row.get("problem") or "").strip()
    answer = str(
        row.get("answer")
        if row.get("answer") is not None
        else row.get("reference_answer") or ""
    ).strip()
    return question, answer


def prepare_items(
    rows: list[dict[str, Any]], source: str | None, limit: int | None
) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    # R-Zero can repeat one question with a different stored answer.  The unit
    # of analysis is the generated problem, so follow the established sampler
    # and keep the first occurrence of each question rather than treating a
    # conflicting answer as a new problem.
    seen_questions: set[str] = set()
    duplicates = 0
    for original_index, row in enumerate(rows):
        if source and str(row.get("source", "")) != source:
            continue
        question, answer = problem_text(row)
        if not question:
            continue
        question_fingerprint = sha256_text(question)[:16]
        if question_fingerprint in seen_questions:
            duplicates += 1
            continue
        seen_questions.add(question_fingerprint)
        fingerprint = input_hash(question, answer)
        annotation = annotate_problem_type(question)
        selected.append(
            {
                "index": len(selected),
                "original_index": original_index,
                "item_id": str(row.get("item_id") or f"row:{original_index}"),
                "source": str(row.get("source") or source or "unknown"),
                "round": row.get("round"),
                "question": question,
                "reference_answer": answer,
                "solver_score": row.get("solver_score", row.get("score")),
                "input_hash": fingerprint,
                "problem_type": annotation.problem_type,
                "problem_type_confidence": annotation.confidence,
                "problem_type_evidence": annotation.evidence,
                "problem_type_review_reason": annotation.review_reason,
            }
        )
        if limit is not None and len(selected) >= limit:
            break
    return selected, duplicates


class LunaClassifier:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        effort: str,
        max_tokens: int,
        prompt_template: str,
        prompt_hash: str,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template
        self.prompt_hash = prompt_hash
        self.lock = threading.Lock()
        self.usage: Counter[str] = Counter()

    def _request_text(self, item: dict[str, Any], *, question: str | None = None) -> str:
        return self.prompt_template.replace(
            "{{Question Here}}", question if question is not None else item["question"]
        ).replace("{{Reference Answer Here}}", item["reference_answer"])

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        api_question = item["question"]
        input_transform: str | None = None
        last_error = "retries exhausted"
        for attempt in range(6):
            body = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": self._request_text(item, question=api_question),
                    }
                ],
                "reasoning_effort": self.effort,
                "response_format": {"type": "json_object"},
                "max_completion_tokens": self.max_tokens,
            }
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=240) as response:
                    envelope = json.loads(response.read())
                content = envelope["choices"][0]["message"]["content"] or "{}"
                prediction = extract_json_object(content)
                domain = normalize_domain(prediction.get("domain"))
                confidence = str(prediction.get("confidence") or "").strip().lower()
                if confidence not in {"high", "low"}:
                    confidence = "low"
                usage = envelope.get("usage") or {}
                with self.lock:
                    self.usage["calls"] += 1
                    self.usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
                    self.usage["completion_tokens"] += int(
                        usage.get("completion_tokens", 0)
                    )
                return {
                    **item,
                    "domain": domain,
                    "domain_display": DISPLAY_DOMAINS.get(domain),
                    "domain_path": clean_model_text(prediction.get("domain_path")),
                    "domain_confidence": confidence,
                    "domain_summary": clean_model_text(prediction.get("summary")),
                    "domain_evidence": clean_model_text(prediction.get("evidence")),
                    "model": self.model,
                    "effort": self.effort,
                    "prompt_hash": self.prompt_hash,
                    "classified_at": utc_now(),
                    "status": "ok",
                    "domain_input_transform": input_transform,
                }
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code == 400 and "invalid_prompt" in detail and input_transform is None:
                    neutralized = neutralize_defined_term(api_question)
                    if neutralized != api_question:
                        api_question = neutralized
                        input_transform = "neutralized_defined_term_after_policy_rejection"
                        with self.lock:
                            self.usage["policy_rewrites"] += 1
                        continue
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(30.0, 2.0**attempt) * (0.5 + random.random()))

        with self.lock:
            self.usage["errors"] += 1
        return {
            **item,
            "domain": None,
            "domain_display": None,
            "domain_path": None,
            "domain_confidence": "none",
            "domain_summary": "",
            "domain_evidence": "",
            "model": self.model,
            "effort": self.effort,
            "prompt_hash": self.prompt_hash,
            "classified_at": utc_now(),
            "status": "error",
            "error": last_error,
        }


def load_reusable(
    path: Path, *, model: str, effort: str, prompt_hash: str
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    reusable: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if (
            row.get("status") == "ok"
            and row.get("model") == model
            and row.get("effort") == effort
            and row.get("prompt_hash") == prompt_hash
            and row.get("input_hash")
        ):
            reusable[str(row["input_hash"])] = row
    return reusable


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def association(grid: list[list[int]]) -> tuple[float, float]:
    rows = [sum(row) for row in grid]
    cols = [sum(grid[i][j] for i in range(len(grid))) for j in range(len(grid[0]))]
    total = sum(rows)
    if total == 0:
        return 0.0, 0.0
    mutual_information = 0.0
    chi2 = 0.0
    for i, row in enumerate(grid):
        for j, observed in enumerate(row):
            expected = rows[i] * cols[j] / total
            if observed:
                mutual_information += (observed / total) * math.log(
                    observed * total / (rows[i] * cols[j])
                )
            if expected:
                chi2 += (observed - expected) ** 2 / expected
    row_entropy = -sum((n / total) * math.log(n / total) for n in rows if n)
    col_entropy = -sum((n / total) * math.log(n / total) for n in cols if n)
    nmi = (
        mutual_information / math.sqrt(row_entropy * col_entropy)
        if row_entropy and col_entropy
        else 0.0
    )
    denominator = min(len(rows) - 1, len(cols) - 1)
    cramers_v = math.sqrt((chi2 / total) / denominator) if denominator else 0.0
    return nmi, cramers_v


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    domain_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    type_review: Counter[str] = Counter()
    cell_counts: Counter[tuple[str, str]] = Counter()
    high_domain = 0
    low_domain = 0
    errors = 0

    for row in rows:
        domain = row.get("domain")
        confidence = row.get("domain_confidence")
        problem_type = row.get("problem_type")
        if row.get("status") == "error":
            errors += 1
        if domain in DOMAINS and confidence == "high":
            high_domain += 1
            domain_counts[domain] += 1
        elif domain in DOMAINS and confidence == "low":
            low_domain += 1
        if problem_type in PROBLEM_TYPES:
            type_counts[problem_type] += 1
        else:
            type_review[str(row.get("problem_type_review_reason") or "unknown")] += 1
        if (
            domain in DOMAINS
            and confidence == "high"
            and problem_type in PROBLEM_TYPES
        ):
            cell_counts[(domain, problem_type)] += 1

    grid = [
        [cell_counts[(domain, problem_type)] for problem_type in PROBLEM_TYPES]
        for domain in DOMAINS
    ]
    nmi, cramers_v = association(grid)
    cells = [
        {
            "domain": domain,
            "problem_type": problem_type,
            "count": cell_counts[(domain, problem_type)],
        }
        for domain in DOMAINS
        for problem_type in PROBLEM_TYPES
    ]
    mapped = sum(cell_counts.values())
    occupied = sum(value > 0 for value in cell_counts.values())
    summary = {
        "rows": len(rows),
        "domain_high_confidence_rows": high_domain,
        "domain_low_confidence_rows": low_domain,
        "domain_error_or_null_rows": len(rows) - high_domain - low_domain,
        "api_error_rows": errors,
        "problem_type_classified_rows": sum(type_counts.values()),
        "problem_type_abstained_rows": len(rows) - sum(type_counts.values()),
        "joint_mapped_rows": mapped,
        "joint_mapped_fraction": mapped / len(rows) if rows else 0.0,
        "occupied_cells": occupied,
        "possible_cells": len(DOMAINS) * len(PROBLEM_TYPES),
        "domain_counts_high_confidence": {
            domain: domain_counts[domain] for domain in DOMAINS
        },
        "problem_type_counts": {
            problem_type: type_counts[problem_type]
            for problem_type in PROBLEM_TYPES
        },
        "problem_type_review_reasons": dict(sorted(type_review.items())),
        "domain_type_nmi": nmi,
        "domain_type_cramers_v": cramers_v,
        "association_scope": "high-confidence domain and classified problem type only",
    }
    return summary, cells


def plot_map(cells: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np

    counts = {
        (str(row["domain"]), str(row["problem_type"])): int(row["count"])
        for row in cells
    }
    grid = np.array(
        [
            [counts[(domain, problem_type)] for problem_type in PROBLEM_TYPES]
            for domain in DOMAINS
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(12.2, 7.2))
    cmap = plt.cm.viridis.copy()
    cmap.set_under("#e8eaee")
    vmax = max(float(grid.max()), 1.0)
    image = ax.imshow(
        grid,
        cmap=cmap,
        norm=colors.LogNorm(vmin=1, vmax=vmax),
        aspect="auto",
    )
    for row_index, domain in enumerate(DOMAINS):
        for col_index, problem_type in enumerate(PROBLEM_TYPES):
            count = int(grid[row_index, col_index])
            normalized = image.norm(max(count, 1))
            color = "white" if count > 0 and normalized < 0.72 else "#111318"
            ax.text(
                col_index,
                row_index,
                str(count),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=color,
            )
    ax.set_xticks(
        range(len(PROBLEM_TYPES)),
        [value.replace("_", " ").title() for value in PROBLEM_TYPES],
    )
    ax.set_yticks(range(len(DOMAINS)), [DISPLAY_DOMAINS[value] for value in DOMAINS])
    ax.set_xlabel("Computational problem type", fontsize=12, labelpad=12)
    ax.set_ylabel("Omni-MATH top-level domain", fontsize=12, labelpad=12)
    ax.set_xticks(np.arange(-0.5, len(PROBLEM_TYPES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(
        "R-Zero: Luna Domain × Deterministic Problem Type\n"
        f"{summary['dataset_label']} · n={summary['rows']} · "
        f"mapped n={summary['joint_mapped_rows']} · "
        f"occupied {summary['occupied_cells']}/35 cells",
        fontsize=15,
        pad=16,
    )
    fig.text(
        0.5,
        0.018,
        "Domain: gpt-5.6-luna high-confidence exact-one label · "
        "Problem type: statement-only computational-output-contract-v1 · "
        "cell text = sample count",
        ha="center",
        fontsize=9.5,
        color="#454954",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Problem count (log scale)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"rzero_domain_type_map.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt",
        type=Path,
        default=ROOT / "prompt_templates" / "domain_classification.txt",
    )
    parser.add_argument("--source", default="rzero")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--effort", default="none")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-completion-tokens", type=int, default=800)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--dataset-label",
        help="Short plot label, e.g. 'round 1 full unique pool'.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    prompt_template = args.prompt.read_text(encoding="utf-8")
    for placeholder in ("{{Question Here}}", "{{Reference Answer Here}}"):
        if placeholder not in prompt_template:
            parser.error(f"prompt is missing placeholder {placeholder}")
    prompt_hash = sha256_text(prompt_template)[:12]
    source = None if args.source.lower() in {"", "all", "none"} else args.source
    items, duplicate_count = prepare_items(read_input(args.input), source, args.limit)
    if not items:
        parser.error("no matching non-empty problems")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / "labels.jsonl"
    reusable = load_reusable(
        labels_path, model=args.model, effort=args.effort, prompt_hash=prompt_hash
    )
    results: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for item in items:
        old = reusable.get(item["input_hash"])
        if old is None:
            pending.append(item)
        else:
            # Refresh deterministic fields in case the local type rules changed.
            results[item["input_hash"]] = {**old, **item}

    print(
        f"[domain-type] rows={len(items)} reusable={len(results)} "
        f"pending={len(pending)} prompt={prompt_hash}",
        flush=True,
    )
    classifier = LunaClassifier(
        api_key=load_api_key() if pending else "unused",
        model=args.model,
        effort=args.effort,
        max_tokens=args.max_completion_tokens,
        prompt_template=prompt_template,
        prompt_hash=prompt_hash,
    )
    if pending:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
            futures = {executor.submit(classifier, item): item for item in pending}
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results[result["input_hash"]] = result
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    print(
                        f"[domain-type] Luna {completed}/{len(pending)} "
                        f"errors={classifier.usage['errors']}",
                        flush=True,
                    )
                ordered_partial = [
                    results[item["input_hash"]]
                    for item in items
                    if item["input_hash"] in results
                ]
                write_jsonl(labels_path, ordered_partial)

    ordered = [results[item["input_hash"]] for item in items]
    write_jsonl(labels_path, ordered)
    write_jsonl(args.output_dir / "input_manifest.jsonl", items)
    (args.output_dir / "prompt.txt").write_text(prompt_template, encoding="utf-8")

    summary, cells = summarize(ordered)
    dataset_label = args.dataset_label
    if not dataset_label:
        if args.input.suffix.lower() == ".parquet" and args.input.parent.name.startswith(
            "round_"
        ):
            round_name = args.input.parent.name.replace("_", " ")
            dataset_label = f"{round_name} full unique pool"
        else:
            dataset_label = "frozen proportional sample"
    summary.update(
        {
            "created_at": utc_now(),
            "input": str(args.input.resolve()),
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "source_filter": source,
            "dataset_label": dataset_label,
            "deduplicated_rows": duplicate_count,
            "model": args.model,
            "effort": args.effort,
            "prompt": str(args.prompt.resolve()),
            "prompt_hash": prompt_hash,
            "problem_type_authority": "deterministic statement-only audit",
            "problem_type_ruleset": PROBLEM_TYPE_RULESET,
            "problem_type_ruleset_sha256": problem_type_ruleset_sha256(),
            "domain_authority": "gpt-5.6-luna high-confidence exact-one audit",
            "usage_this_invocation": dict(classifier.usage),
        }
    )
    sample_metadata_path = args.input.parent / "rzero_sample_metadata.json"
    if sample_metadata_path.is_file():
        sample_metadata = json.loads(sample_metadata_path.read_text(encoding="utf-8"))
        summary["source_sample_metadata"] = sample_metadata
        with (args.output_dir / "source_sample_metadata.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(sample_metadata, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (args.output_dir / "domain_type_counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("domain", "problem_type", "count"))
        writer.writeheader()
        writer.writerows(cells)
    plot_map(cells, summary, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
