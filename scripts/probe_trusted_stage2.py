#!/usr/bin/env python3
"""Run the production two-stage prompt/assembler contract without training.

This is a prompt-path smoke test, not an optimizer step.  It calls the same
Stage-1/Stage-2 builders used by evolution, compiles the Stage-2 reply through
the trusted assembler, runs the production multi-seed verifier, and records
diversity plus parent/child similarity metrics.

Example::

    python scripts/probe_trusted_stage2.py \
      --base-url <OpenAI-compatible-endpoint>/v1 \
      --model <model-name> --repeats 3
"""

from __future__ import annotations

import argparse
import ast
import collections
import difflib
import hashlib
import json
import random
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.archive import MAPElitesArchive  # noqa: E402
from rq_evolve.code_utils import (  # noqa: E402
    TRUSTED_ASSEMBLER_VERSION,
    compile_stage2_reply,
    extract_problem_statement_template,
)
from rq_evolve.config import EvolutionConfig  # noqa: E402
from rq_evolve.constancy import canonical_template  # noqa: E402
from rq_evolve.evolution import RQEvolver  # noqa: E402
from rq_evolve.program import ProblemProgram  # noqa: E402
from rq_evolve.prompts import (  # noqa: E402
    MUTATION_OP,
    build_family_task,
    build_generator_task,
    parse_family_plan,
)
from rq_evolve.structural_fingerprint import (  # noqa: E402
    exact_top_level_build_instance,
)


_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-z]+(?:_[a-z0-9]+)*|\d+", re.I)
_PLACEHOLDER = re.compile(r"\[\[[a-z][a-z0-9_]*\]\]", re.I)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


class _NoBackend:
    """Verification does not call a rollout backend."""


def _normalize(text: str) -> str:
    return _SPACE.sub(" ", str(text or "")).strip().lower()


def _family_shape(text: str) -> str:
    text = _PLACEHOLDER.sub("[[parameter]]", _normalize(text))
    return _NUMBER.sub("N", text)


def _ratio(left, right) -> float:
    return difflib.SequenceMatcher(
        None, left, right, autojunk=False
    ).ratio()


def _token_jaccard(left: str, right: str) -> float:
    a, b = set(_WORD.findall(_family_shape(left))), set(_WORD.findall(_family_shape(right)))
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _entry_skeleton(source: str) -> tuple[str, ...] | None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    builder, claimed = exact_top_level_build_instance(tree)
    if claimed and builder is None:
        return None
    entry = builder
    if entry is None:
        candidates = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "generate"
        ]
        if len(candidates) != 1:
            return None
        entry = candidates[0]
    return tuple(type(node).__name__ for node in ast.walk(entry))


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "ge_0_85_rate": sum(value >= 0.85 for value in values) / len(values),
    }


def _pairwise_family_similarity(families: list[str]) -> dict:
    shaped = [_family_shape(family) for family in families]
    values = [
        _ratio(shaped[i], shaped[j])
        for i in range(len(shaped))
        for j in range(i)
    ]
    result = _summary(values)
    result["pairs"] = len(values)
    return result


def _ask(client, *, model: str, messages: list[dict], temperature: float,
         top_p: float, max_tokens: int, seed: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )
    return response.choices[0].message.content or ""


def _load_parent_sources(archive_dir: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    for path in archive_dir.glob("archive_iter*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        champions = payload.get("champions", {})
        rows = champions.values() if isinstance(champions, dict) else champions
        for row in rows:
            if row.get("program_id") and row.get("source_code"):
                sources[str(row["program_id"])] = str(row["source_code"])
    return sources


def _baseline_summary(log_path: Path, archive_dir: Path) -> dict:
    parents = _load_parent_sources(archive_dir)
    rows: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        for report in payload.get("reports", []):
            source = report.get("source_code")
            parent_source = parents.get(str(report.get("parent_id", "")))
            if not source or not parent_source:
                continue
            child_family = extract_problem_statement_template(source)
            parent_family = extract_problem_statement_template(parent_source)
            child_skeleton = _entry_skeleton(source)
            parent_skeleton = _entry_skeleton(parent_source)
            rows.append(
                {
                    "source": source,
                    "family": child_family,
                    "source_parent_similarity": _ratio(source, parent_source),
                    "family_parent_similarity": (
                        _ratio(_family_shape(child_family), _family_shape(parent_family))
                        if child_family and parent_family
                        else None
                    ),
                    "skeleton_parent_similarity": (
                        _ratio(child_skeleton, parent_skeleton)
                        if child_skeleton and parent_skeleton
                        else None
                    ),
                }
            )
    families = [_normalize(row["family"]) for row in rows if row["family"]]
    return {
        "candidate_rows": len(rows),
        "unique_source_rate": (
            len({hashlib.sha256(row["source"].encode()).hexdigest() for row in rows})
            / len(rows)
            if rows
            else 0.0
        ),
        "unique_family_rate": len(set(families)) / len(families) if families else 0.0,
        "source_parent_similarity": _summary(
            [row["source_parent_similarity"] for row in rows]
        ),
        "family_parent_similarity": _summary(
            [
                row["family_parent_similarity"]
                for row in rows
                if row["family_parent_similarity"] is not None
            ]
        ),
        "skeleton_parent_similarity": _summary(
            [
                row["skeleton_parent_similarity"]
                for row in rows
                if row["skeleton_parent_similarity"] is not None
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    # Keep both explicit: a prompt audit must never start or assume a local
    # shared-GPU endpoint merely because the caller omitted an argument.
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed-dir", type=Path, default=ROOT / "seed_programs_domain_type")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--verify-seeds", type=int, default=5)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "analysis" / "trusted_stage2_probe",
    )
    parser.add_argument("--baseline-log", type=Path)
    parser.add_argument("--baseline-archive", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="none", timeout=900)
    parents = [ProblemProgram.from_file(path) for path in sorted(args.seed_dir.glob("*.py"))]
    if not parents:
        raise SystemExit(f"no seed programs in {args.seed_dir}")
    jobs = [
        (index, parent, args.seed + index)
        for index, parent in enumerate(parents * args.repeats)
    ]

    def stage_one(job):
        index, parent, draw_seed = job
        task = build_family_task(
            parent,
            temperature=0.4,
            top_p=0.95,
            max_output_tokens=1024,
            rotate_shots=True,
            rng=random.Random(draw_seed),
        )
        reply = _ask(
            client,
            model=args.model,
            messages=task.messages,
            temperature=0.4,
            top_p=0.95,
            max_tokens=1024,
            seed=draw_seed,
        )
        return index, parent, draw_seed, task, reply, parse_family_plan(reply)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        stage_one_rows = list(pool.map(stage_one, jobs))

    def stage_two(row):
        index, parent, draw_seed, family_task, family_reply, plan = row
        record = {
            "index": index,
            "parent_id": parent.program_id,
            "parent_source": parent.source_code,
            "family_reply": family_reply,
            "plan": plan,
        }
        if plan is None:
            record.update(status="stage1_parse_failed")
            return record
        task = build_generator_task(
            parent,
            plan,
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=1536,
            provenance={"family_plan": dict(plan)},
        )
        reply = _ask(
            client,
            model=args.model,
            messages=task.messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=1536,
            seed=draw_seed,
        )
        source, compile_error = compile_stage2_reply(reply, plan["CHILD FAMILY"])
        record["stage2_reply"] = reply
        record["compile_error"] = compile_error
        if source is None:
            record.update(
                status=("stage2_invalid" if "INVALID:" in str(compile_error) else "stage2_compile_failed")
            )
            return record
        child = ProblemProgram(
            source_code=source,
            parent_id=parent.program_id,
            generation=parent.generation + 1,
            metadata={
                "op": MUTATION_OP,
                "generator_contract": {"version": TRUSTED_ASSEMBLER_VERSION},
                "family_plan": dict(plan),
            },
        )
        record["source"] = source
        record["child_id"] = child.program_id
        record["child"] = child
        record["status"] = "compiled"
        return record

    parsed_rows = [row for row in stage_one_rows if row[-1] is not None]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        compiled_by_index = {
            row["index"]: row for row in pool.map(stage_two, parsed_rows)
        }
    rows = []
    for row in stage_one_rows:
        index, parent, draw_seed, family_task, family_reply, plan = row
        if plan is None:
            rows.append(
                {
                    "index": index,
                    "parent_id": parent.program_id,
                    "parent_source": parent.source_code,
                    "family_reply": family_reply,
                    "plan": None,
                    "status": "stage1_parse_failed",
                }
            )
        else:
            rows.append(compiled_by_index[index])

    evolver = RQEvolver(
        archive=MAPElitesArchive(),
        backend=_NoBackend(),
        evolution_config=EvolutionConfig(
            verify_seeds=args.verify_seeds,
            ast_contract="enforce",
        ),
    )
    for row in rows:
        child = row.pop("child", None)
        if child is None:
            continue
        instance, reason = evolver.verify_program(child)
        row["verify_error"] = reason
        if instance is None:
            row["status"] = "verify_failed"
            continue
        row["status"] = "verified"
        row["domain"] = child.get_domain()
        row["problem_type"] = child.get_problem_type()
        row["rendered_problem"] = instance.problem
        row["rendered_answer"] = instance.answer

    for row in rows:
        plan = row.get("plan")
        parent_source = row.pop("parent_source")
        if not plan:
            continue
        parent_family = extract_problem_statement_template(parent_source)
        if not parent_family:
            parent_instance = ProblemProgram(parent_source).execute(0)
            parent_family = parent_instance.problem if parent_instance else ""
        child_family = plan["CHILD FAMILY"]
        row["family_parent_similarity"] = _ratio(
            _family_shape(child_family), _family_shape(parent_family)
        )
        row["family_parent_token_jaccard"] = _token_jaccard(
            child_family, parent_family
        )
        if row.get("source"):
            row["source_parent_similarity"] = _ratio(
                row["source"], parent_source
            )
            child_skeleton = _entry_skeleton(row["source"])
            parent_skeleton = _entry_skeleton(parent_source)
            row["skeleton_parent_similarity"] = (
                _ratio(child_skeleton, parent_skeleton)
                if child_skeleton and parent_skeleton
                else None
            )

    status_counts = collections.Counter(row["status"] for row in rows)
    planned = [row for row in rows if row.get("plan")]
    compiled = [row for row in rows if row.get("source")]
    verified = [row for row in rows if row["status"] == "verified"]
    families = [row["plan"]["CHILD FAMILY"] for row in planned]
    exact_families = {_normalize(family) for family in families}
    shaped_families = {_family_shape(family) for family in families}
    summary = {
        "model": args.model,
        "jobs": len(rows),
        "status_counts": dict(status_counts),
        "stage1_parse_rate": len(planned) / len(rows),
        "stage2_compile_rate_given_plan": len(compiled) / max(1, len(planned)),
        "full_verify_rate": len(verified) / len(rows),
        "exact_unique_family_rate": len(exact_families) / max(1, len(families)),
        "structural_unique_family_rate": len(shaped_families) / max(1, len(families)),
        "unique_compiled_source_rate": (
            len({row["child_id"] for row in compiled}) / len(compiled)
            if compiled
            else 0.0
        ),
        "pairwise_family_similarity": _pairwise_family_similarity(families),
        "family_parent_similarity": _summary(
            [row["family_parent_similarity"] for row in planned]
        ),
        "family_parent_token_jaccard": _summary(
            [row["family_parent_token_jaccard"] for row in planned]
        ),
        "source_parent_similarity": _summary(
            [row["source_parent_similarity"] for row in compiled]
        ),
        "skeleton_parent_similarity": _summary(
            [
                row["skeleton_parent_similarity"]
                for row in compiled
                if row.get("skeleton_parent_similarity") is not None
            ]
        ),
        "verified_cells": sorted(
            {
                f"{row['domain']}x{row['problem_type']}"
                for row in verified
            }
        ),
        "verified_canonical_templates": len(
            {canonical_template(row["rendered_problem"]) for row in verified}
        ),
    }
    if args.baseline_log and args.baseline_archive:
        summary["legacy_baseline"] = _baseline_summary(
            args.baseline_log, args.baseline_archive
        )

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out / "probe.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    summary_path = args.out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"rows: {jsonl_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
