"""Fixed Seed-ID benchmark utilities for the Evolved Performance Score.

The benchmark holds the *generator family* fixed (the repository seed
programs) and renders a deterministic set of instances from high, fixed seeds.
Every model checkpoint is graded on the exact same JSONL.  Evolution progress
is reconstructed separately from the run logs so model performance and the
curriculum's inner/outer iterations can share one plot without conflating their
units.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .program import ProblemProgram


BENCHMARK_NAME = "evolved_performance_seed_id_v1"
SCHEMA_VERSION = 1
_CHECKPOINT_RE = re.compile(r"global_step_(\d+)")


def normalize_problem(text: str) -> str:
    """Canonicalize harmless whitespace for duplicate/leakage detection."""

    return " ".join(str(text).split())


def instance_sha256(problem: str, answer: str) -> str:
    payload = json.dumps(
        [normalize_problem(problem), str(answer).strip()],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def benchmark_sha256(rows: Iterable[dict[str, Any]]) -> str:
    """Hash only immutable benchmark content, never paths or timestamps."""

    canonical = [
        {
            "sample_id": str(row["sample_id"]),
            "program_name": str(row["program_name"]),
            "program_id": str(row["program_id"]),
            "program_sha256": str(row["program_sha256"]),
            "seed": int(row["seed"]),
            "problem": str(row["problem"]),
            "answer": str(row["answer"]),
            "instance_sha256": str(row["instance_sha256"]),
        }
        for row in rows
    ]
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_seed_id_rows(
    seed_dir: str | Path,
    *,
    examples_per_program: int = 40,
    seed_start: int = 1_000_000,
    max_seed_scan: int = 100_000,
    benchmark_name: str = BENCHMARK_NAME,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render a balanced, duplicate-free benchmark from every seed program."""

    seed_dir = Path(seed_dir).expanduser().resolve()
    if examples_per_program < 1:
        raise ValueError("examples_per_program must be >= 1")
    if max_seed_scan < examples_per_program:
        raise ValueError("max_seed_scan must be >= examples_per_program")
    paths = sorted(seed_dir.glob("*.py"))
    if not paths:
        raise FileNotFoundError(f"no seed programs under {seed_dir}")

    rows: list[dict[str, Any]] = []
    programs: list[dict[str, Any]] = []
    global_signatures: set[str] = set()
    for path in paths:
        source_bytes = path.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        program = ProblemProgram.from_file(path)
        group = program.declared_group()
        skill = program.declared_skill()
        selected = 0
        scanned = 0
        local_signatures: set[str] = set()
        for seed in range(seed_start, seed_start + max_seed_scan):
            scanned += 1
            instance = program.execute(seed)
            if instance is None:
                continue
            signature = instance_sha256(instance.problem, instance.answer)
            if signature in local_signatures or signature in global_signatures:
                continue
            local_signatures.add(signature)
            global_signatures.add(signature)
            sample_id = f"{path.stem}:{selected:03d}"
            rows.append(
                {
                    "benchmark": benchmark_name,
                    "sample_id": sample_id,
                    "program_name": path.stem,
                    "program_id": program.program_id,
                    "program_sha256": source_hash,
                    "group": group,
                    "skill": skill,
                    "seed": int(seed),
                    "problem": instance.problem,
                    "answer": instance.answer,
                    "instance_sha256": signature,
                }
            )
            selected += 1
            if selected >= examples_per_program:
                break
        if selected != examples_per_program:
            raise RuntimeError(
                f"{path.name} produced only {selected} globally unique instances "
                f"after scanning {scanned} seeds from {seed_start}; lower "
                "--examples-per-program or increase --max-seed-scan"
            )
        programs.append(
            {
                "program_name": path.stem,
                "program_id": program.program_id,
                "program_sha256": source_hash,
                "group": group,
                "skill": skill,
                "num_examples": selected,
                "seeds_scanned": scanned,
            }
        )

    for index, row in enumerate(rows):
        row["index"] = index
    return rows, programs


def write_benchmark(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    programs: list[dict[str, Any]],
    *,
    seed_start: int,
    examples_per_program: int,
    created_at: str,
    benchmark_name: str = BENCHMARK_NAME,
) -> tuple[Path, Path, dict[str, Any]]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "benchmark.jsonl"
    manifest_path = output_dir / "manifest.json"
    digest = benchmark_sha256(rows)
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark_name,
        "created_at": created_at,
        "benchmark_sha256": digest,
        "num_examples": len(rows),
        "num_programs": len(programs),
        "examples_per_program": int(examples_per_program),
        "seed_start": int(seed_start),
        "selection": (
            "first globally unique (problem, answer) instances from sequential "
            "seeds; balanced equally across seed programs"
        ),
        "programs": programs,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return jsonl_path, manifest_path, manifest


def load_benchmark(
    jsonl_path: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    jsonl_path = Path(jsonl_path).expanduser().resolve()
    if manifest_path is None:
        manifest_path = jsonl_path.with_name("manifest.json")
    manifest_path = Path(manifest_path).expanduser().resolve()
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"empty benchmark: {jsonl_path}")
    expected = str(manifest.get("benchmark_sha256", ""))
    actual = benchmark_sha256(rows)
    if actual != expected:
        raise ValueError(
            f"benchmark hash mismatch: manifest={expected or '<missing>'}, "
            f"actual={actual}"
        )
    if int(manifest.get("num_examples", -1)) != len(rows):
        raise ValueError(
            f"manifest num_examples={manifest.get('num_examples')} but JSONL "
            f"contains {len(rows)} rows"
        )
    seen_ids: set[str] = set()
    seen_instances: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        signature = str(row.get("instance_sha256", ""))
        computed = instance_sha256(row.get("problem", ""), row.get("answer", ""))
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"missing or duplicate sample_id: {sample_id!r}")
        if signature != computed or signature in seen_instances:
            raise ValueError(
                f"invalid or duplicate instance signature for {sample_id}: "
                f"stored={signature}, computed={computed}"
            )
        seen_ids.add(sample_id)
        seen_instances.add(signature)
    return rows, manifest


def audit_known_seed_overlap(
    rows: list[dict[str, Any]],
    seed_dir: str | Path,
    used_seeds_json: str | Path,
) -> dict[str, Any]:
    """Conservatively audit exact overlap with consumed original-program seeds.

    This can reconstruct instances produced by the original seed programs.  It
    cannot prove that a mutated generator never emitted the same statement, so
    the report states that limitation explicitly.
    """

    used_path = Path(used_seeds_json).expanduser().resolve()
    payload = json.loads(used_path.read_text(encoding="utf-8"))
    used = payload.get("used_seeds") or {}
    benchmark_by_program: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        benchmark_by_program[str(row["program_id"])].add(
            str(row["instance_sha256"])
        )

    per_program: dict[str, Any] = {}
    all_overlaps: set[str] = set()
    for path in sorted(Path(seed_dir).expanduser().resolve().glob("*.py")):
        program = ProblemProgram.from_file(path)
        consumed = sorted({int(seed) for seed in used.get(program.program_id, [])})
        training_signatures: set[str] = set()
        failures = 0
        for seed in consumed:
            instance = program.execute(seed)
            if instance is None:
                failures += 1
                continue
            training_signatures.add(
                instance_sha256(instance.problem, instance.answer)
            )
        overlap = training_signatures & benchmark_by_program.get(
            program.program_id, set()
        )
        all_overlaps.update(overlap)
        per_program[path.stem] = {
            "program_id": program.program_id,
            "consumed_seed_count": len(consumed),
            "reconstructed_unique_instances": len(training_signatures),
            "execution_failures": failures,
            "benchmark_examples": len(
                benchmark_by_program.get(program.program_id, set())
            ),
            "exact_overlap_count": len(overlap),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "audit": "known_original_seed_program_training_overlap",
        "used_seeds_json": str(used_path),
        "benchmark_examples": len(rows),
        "exact_overlap_count": len(all_overlaps),
        "exact_overlap_rate": len(all_overlaps) / len(rows) if rows else 0.0,
        "per_program": per_program,
        "limitation": (
            "Exact overlap from original seed programs only; semantically "
            "equivalent problems and collisions emitted by mutated programs "
            "cannot be reconstructed from rq_used_seeds.json."
        ),
    }


def summarize_scored_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize zero scored rows")
    grouped: dict[str, list[int]] = defaultdict(list)
    concepts: dict[str, tuple[str, str]] = {}
    for row in rows:
        name = str(row["program_name"])
        grouped[name].append(1 if row["correct"] else 0)
        concepts[name] = (
            str(row.get("group") or "unknown"),
            str(row.get("skill") or "unknown"),
        )
    per_program = {
        name: {
            "correct": sum(scores),
            "num_examples": len(scores),
            "accuracy": sum(scores) / len(scores),
            "group": concepts[name][0],
            "skill": concepts[name][1],
        }
        for name, scores in sorted(grouped.items())
    }
    concept_grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        concept = (
            str(row.get("group") or "unknown"),
            str(row.get("skill") or "unknown"),
        )
        concept_grouped[concept].append(1 if row["correct"] else 0)
    per_concept = {
        f"{group}/{skill}": {
            "correct": sum(scores),
            "num_examples": len(scores),
            "accuracy": sum(scores) / len(scores),
            "group": group,
            "skill": skill,
        }
        for (group, skill), scores in sorted(concept_grouped.items())
    }
    total_correct = sum(item["correct"] for item in per_program.values())
    total = sum(item["num_examples"] for item in per_program.values())
    macro = sum(item["accuracy"] for item in per_program.values()) / len(
        per_program
    )
    return {
        "correct": total_correct,
        "num_examples": total,
        "micro_accuracy": total_correct / total,
        "macro_accuracy": macro,
        "score_percent": 100.0 * macro,
        "per_program": per_program,
        "per_concept": per_concept,
    }


def build_concept_change_rows(
    checkpoint_results: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare per-seed-program accuracy between adjacent checkpoints."""

    ordered = sorted(
        checkpoint_results, key=lambda item: int(item.get("global_step", 0))
    )
    changes: list[dict[str, Any]] = []
    for previous, current in zip(ordered, ordered[1:]):
        # A combined benchmark can contain multiple generators for the same
        # GROUP/SKILL pair. Prefer the aggregate 80-item concept scores in that
        # case, while retaining backward compatibility with older summaries.
        use_concepts = bool(previous.get("per_concept")) and bool(
            current.get("per_concept")
        )
        score_key = "per_concept" if use_concepts else "per_program"
        previous_programs = previous.get(score_key) or {}
        current_programs = current.get(score_key) or {}
        if set(previous_programs) != set(current_programs):
            raise ValueError(
                "checkpoint per_program sets differ: "
                f"step {previous.get('global_step')} has "
                f"{sorted(previous_programs)}, step {current.get('global_step')} "
                f"has {sorted(current_programs)}"
            )
        concepts: list[dict[str, Any]] = []
        for program_name in sorted(current_programs):
            before = previous_programs[program_name]
            after = current_programs[program_name]
            before_accuracy = float(before.get("accuracy", 0.0) or 0.0)
            after_accuracy = float(after.get("accuracy", 0.0) or 0.0)
            group = str(after.get("group") or before.get("group") or "unknown")
            skill = str(after.get("skill") or before.get("skill") or "unknown")
            concepts.append(
                {
                    "program_name": program_name,
                    "group": group,
                    "skill": skill,
                    "previous_accuracy_percent": 100.0 * before_accuracy,
                    "accuracy_percent": 100.0 * after_accuracy,
                    "delta_pp": 100.0 * (after_accuracy - before_accuracy),
                }
            )
        concepts.sort(key=lambda item: (-item["delta_pp"], item["program_name"]))
        changes.append(
            {
                "previous_step": int(previous.get("global_step", 0)),
                "checkpoint_step": int(current.get("global_step", 0)),
                "eps_delta_pp": float(current.get("score_percent", 0.0))
                - float(previous.get("score_percent", 0.0)),
                "concepts": concepts,
                "improved": [item for item in concepts if item["delta_pp"] > 0.0],
                "declined": [item for item in concepts if item["delta_pp"] < 0.0],
                "unchanged": [item for item in concepts if item["delta_pp"] == 0.0],
            }
        )
    return changes


def write_concept_change_report(
    output_dir: str | Path, rows: list[dict[str, Any]]
) -> tuple[Path, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "concept_score_changes.json"
    markdown_path = output_dir / "concept_score_changes.md"
    json_path.write_text(
        json.dumps({"checkpoint_concept_changes": rows}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    def format_concepts(items: list[dict[str, Any]], sign: str) -> str:
        if not items:
            return "-"
        return ", ".join(
            f"{item['group']}/{item['skill']} ({sign}{abs(item['delta_pp']):.1f}%p)"
            for item in items
        )

    lines = [
        "# Evolved Performance Concept Score Changes",
        "",
        (
            "Accuracy-point changes are measured against the immediately "
            "previous saved model on the same fixed benchmark rows."
        ),
        "",
        "| checkpoint | EPS delta | improved concepts | declined concepts |",
        "|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['previous_step']} → {row['checkpoint_step']} | "
            f"{row['eps_delta_pp']:+.2f}%p | "
            f"{format_concepts(row['improved'], '+')} | "
            f"{format_concepts(row['declined'], '-')} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


@dataclass(frozen=True)
class EvolutionEvent:
    outer_iteration: int
    global_step: int
    attempted: int
    inserted: int
    cumulative_inner: int
    cumulative_inserted: int


@dataclass(frozen=True)
class HighRQCandidate:
    """One generated program that was actually inserted into the archive."""

    outer_iteration: int
    global_step: int
    child_id: str
    rq_score: float
    s_hat: float
    u_score: float
    op: str
    source_code: str | None


def _jsonl_objects(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            if isinstance(value, dict):
                yield value


def _checkpoint_step(value: Any) -> int | None:
    match = _CHECKPOINT_RE.search(str(value or ""))
    return int(match.group(1)) if match else None


def load_evolution_events(run_dir: str | Path) -> list[EvolutionEvent]:
    """Join outer metrics to their actual trainer global-step start times."""

    run_dir = Path(run_dir).expanduser().resolve()
    archive_dir = run_dir / "rq_archive"
    metrics_by_iteration: dict[int, dict[str, Any]] = {}
    for row in _jsonl_objects(archive_dir / "evolution_log.jsonl"):
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        iteration = int(metrics.get("outer_iteration", row.get("iteration", -1)))
        if iteration >= 0:
            metrics_by_iteration[iteration] = metrics

    start_step_by_iteration: dict[int, int] = {}
    for row in _jsonl_objects(archive_dir / "rollout_metrics.jsonl"):
        if "iteration" not in row:
            continue
        iteration = int(row["iteration"])
        step = row.get("global_step")
        if not isinstance(step, int):
            step = _checkpoint_step(row.get("source_checkpoint"))
        if iteration >= 0 and step is not None:
            start_step_by_iteration[iteration] = int(step)

    cumulative_inner = 0
    cumulative_inserted = 0
    events: list[EvolutionEvent] = []
    for iteration in sorted(metrics_by_iteration):
        metrics = metrics_by_iteration[iteration]
        attempted = int(metrics.get("attempted", 0) or 0)
        inserted = int(metrics.get("inserted", 0) or 0)
        cumulative_inner += attempted
        cumulative_inserted += inserted
        if iteration not in start_step_by_iteration:
            continue
        events.append(
            EvolutionEvent(
                outer_iteration=iteration,
                global_step=start_step_by_iteration[iteration],
                attempted=attempted,
                inserted=inserted,
                cumulative_inner=cumulative_inner,
                cumulative_inserted=cumulative_inserted,
            )
        )
    return events


def evolution_state_at_step(
    events: list[EvolutionEvent], step: int
) -> tuple[int | None, int, int]:
    """Return (active outer, cumulative inner, cumulative insertions)."""

    active: EvolutionEvent | None = None
    for event in events:
        if event.global_step <= step:
            active = event
        else:
            break
    if active is None:
        return None, 0, 0
    return (
        active.outer_iteration,
        active.cumulative_inner,
        active.cumulative_inserted,
    )


def load_inserted_rq_candidates(
    run_dir: str | Path,
    events: list[EvolutionEvent] | None = None,
) -> list[HighRQCandidate]:
    """Load every successfully inserted candidate and its emergence step."""

    run_dir = Path(run_dir).expanduser().resolve()
    events = load_evolution_events(run_dir) if events is None else events
    step_by_outer = {event.outer_iteration: event.global_step for event in events}
    candidates: list[HighRQCandidate] = []
    for row in _jsonl_objects(run_dir / "rq_archive" / "evolution_log.jsonl"):
        iteration = int(row.get("iteration", -1))
        if iteration not in step_by_outer:
            continue
        for report in row.get("reports") or []:
            if not isinstance(report, dict) or report.get("status") != "inserted":
                continue
            child_id = str(report.get("child_id") or "")
            if not child_id:
                continue
            candidates.append(
                HighRQCandidate(
                    outer_iteration=iteration,
                    global_step=step_by_outer[iteration],
                    child_id=child_id,
                    rq_score=float(report.get("rq_score", 0.0) or 0.0),
                    # Reports written before the notation matched the paper
                    # spell these "p_hat"/"uncertainty". This reads completed
                    # runs off disk, so both spellings have to work or every
                    # EPS number from an older run silently becomes 0.0.
                    s_hat=float(
                        report.get("s_hat", report.get("p_hat", 0.0)) or 0.0
                    ),
                    u_score=float(
                        report.get("u_score", report.get("uncertainty", 0.0)) or 0.0
                    ),
                    op=str(report.get("op") or "unknown"),
                    source_code=(
                        str(report["source_code"])
                        if report.get("source_code")
                        else None
                    ),
                )
            )
    return sorted(
        candidates,
        key=lambda item: (
            item.outer_iteration,
            item.global_step,
            -item.rq_score,
            item.child_id,
        ),
    )


def select_prominent_high_rq(
    candidates: list[HighRQCandidate],
    *,
    quantile: float = 0.90,
    max_count: int = 8,
) -> list[HighRQCandidate]:
    """Select unusually high-R_Q evolution events, independent of checkpoints.

    First retain only the highest inserted candidate in each outer iteration.
    Then keep the upper ``quantile`` tail across those outer maxima.  ``max_count``
    caps visual clutter by retaining the largest values before restoring temporal
    order.  This deliberately does not create one item per checkpoint interval.
    """

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if max_count < 0:
        raise ValueError("max_count must be >= 0")
    if not candidates or max_count == 0:
        return []

    by_outer: dict[int, HighRQCandidate] = {}
    for candidate in candidates:
        current = by_outer.get(candidate.outer_iteration)
        if current is None or candidate.rq_score > current.rq_score:
            by_outer[candidate.outer_iteration] = candidate

    outer_maxima = list(by_outer.values())
    scores = sorted(candidate.rq_score for candidate in outer_maxima)
    threshold_index = max(0, math.ceil(quantile * len(scores)) - 1)
    threshold = scores[threshold_index]
    selected = [
        candidate for candidate in outer_maxima if candidate.rq_score >= threshold
    ]
    if len(selected) > max_count:
        selected = sorted(
            selected,
            key=lambda candidate: (-candidate.rq_score, candidate.outer_iteration),
        )[:max_count]
    return sorted(
        selected,
        key=lambda candidate: (
            candidate.global_step,
            candidate.outer_iteration,
            candidate.child_id,
        ),
    )


def select_interval_high_rq(
    candidates: list[HighRQCandidate],
    checkpoint_steps: Iterable[int],
    events: list[EvolutionEvent],
) -> list[tuple[int, HighRQCandidate]]:
    """Pick the highest-R_Q insertion added between saved-model states.

    Windows are defined by active *outer iteration*, not merely by comparing
    global-step numbers.  This handles an evolution phase that starts on the
    same integer step at which a checkpoint was saved: the checkpoint's active
    outer state determines which side of the window owns that candidate.
    """

    selected: list[tuple[int, HighRQCandidate]] = []
    previous_outer: int | None = None
    for step in sorted({int(value) for value in checkpoint_steps}):
        active_outer, _, _ = evolution_state_at_step(events, step)
        if active_outer is None:
            previous_outer = None
            continue
        lower = -1 if previous_outer is None else previous_outer
        eligible = [
            candidate
            for candidate in candidates
            if lower < candidate.outer_iteration <= active_outer
        ]
        if eligible:
            selected.append((step, max(eligible, key=lambda item: item.rq_score)))
        previous_outer = active_outer
    return selected


def _archive_payloads(
    archive_dir: Path, wanted_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """Recover full sources for selected IDs without retaining every champion."""

    if not wanted_ids:
        return {}
    numbered: list[tuple[int, Path]] = []
    for path in archive_dir.glob("archive_iter*.json"):
        match = re.fullmatch(r"archive_iter(\d+)\.json", path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    paths = [path for _, path in sorted(numbered)]
    final = archive_dir / "archive.json"
    if final.is_file():
        paths.append(final)
    found: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for champion in payload.get("champions") or []:
            child_id = str(champion.get("program_id") or "")
            if child_id in wanted_ids:
                found[child_id] = champion
    return found


def _short_problem(text: str, max_chars: int = 140) -> str:
    text = normalize_problem(text)
    if len(text) <= max_chars:
        return text
    shortened = text[: max_chars - 1].rsplit(" ", 1)[0].rstrip()
    return (shortened or text[: max_chars - 1]).rstrip() + "…"


def build_high_rq_interval_rows(
    run_dir: str | Path,
    checkpoint_steps: Iterable[int],
    *,
    render_seed: int = 0,
) -> list[dict[str, Any]]:
    """Materialize the top inserted problem between each checkpoint pair."""

    run_dir = Path(run_dir).expanduser().resolve()
    events = load_evolution_events(run_dir)
    candidates = load_inserted_rq_candidates(run_dir, events)
    selected = select_interval_high_rq(candidates, checkpoint_steps, events)
    payloads = _archive_payloads(
        run_dir / "rq_archive",
        {candidate.child_id for _, candidate in selected},
    )

    rows: list[dict[str, Any]] = []
    for checkpoint_step, candidate in selected:
        payload = payloads.get(candidate.child_id) or {}
        metadata = payload.get("metadata") or {}
        source = payload.get("source_code") or candidate.source_code
        group = str(metadata.get("group") or "unknown")
        skill = str(metadata.get("skill") or "unknown")
        problem = ""
        answer = ""
        render_error: str | None = None
        if source:
            program = ProblemProgram(
                source_code=str(source),
                program_id=candidate.child_id,
                metadata=dict(metadata),
            )
            group = str(program.get_group() or group)
            skill = str(program.get_skill() or skill)
            instance = program.execute(render_seed)
            if instance is not None:
                problem = instance.problem
                answer = instance.answer
            else:
                render_error = program.last_execution_error or "execution failed"
        else:
            render_error = "source code unavailable"
        rows.append(
            {
                "checkpoint_step": int(checkpoint_step),
                "emerged_global_step": candidate.global_step,
                "outer_iteration": candidate.outer_iteration,
                "child_id": candidate.child_id,
                "op": candidate.op,
                "group": group,
                "skill": skill,
                "rq_score": candidate.rq_score,
                "s_hat": candidate.s_hat,
                "u_score": candidate.u_score,
                "render_seed": int(render_seed),
                "problem": problem,
                "problem_short": _short_problem(problem) if problem else "",
                "answer": answer,
                "render_error": render_error,
            }
        )
    return rows


def build_prominent_high_rq_rows(
    run_dir: str | Path,
    *,
    quantile: float = 0.90,
    max_count: int = 8,
) -> list[dict[str, Any]]:
    """Describe only prominent high-R_Q events using concept labels.

    Unlike :func:`build_high_rq_interval_rows`, this selection is made over the
    evolution timeline itself.  It neither depends on checkpoint boundaries nor
    renders a full problem statement.
    """

    run_dir = Path(run_dir).expanduser().resolve()
    events = load_evolution_events(run_dir)
    candidates = load_inserted_rq_candidates(run_dir, events)
    selected = select_prominent_high_rq(
        candidates,
        quantile=quantile,
        max_count=max_count,
    )
    payloads = _archive_payloads(
        run_dir / "rq_archive",
        {candidate.child_id for candidate in selected},
    )
    event_by_outer = {event.outer_iteration: event for event in events}

    rows: list[dict[str, Any]] = []
    for candidate in selected:
        payload = payloads.get(candidate.child_id) or {}
        metadata = payload.get("metadata") or {}
        source = payload.get("source_code") or candidate.source_code
        group = str(metadata.get("group") or "unknown")
        skill = str(metadata.get("skill") or "unknown")
        if source:
            program = ProblemProgram(
                source_code=str(source),
                program_id=candidate.child_id,
                metadata=dict(metadata),
            )
            group = str(program.get_group() or group)
            skill = str(program.get_skill() or skill)
        event = event_by_outer.get(candidate.outer_iteration)
        rows.append(
            {
                "emerged_global_step": candidate.global_step,
                "outer_iteration": candidate.outer_iteration,
                "cumulative_inner": event.cumulative_inner if event else None,
                "group": group,
                "skill": skill,
                "rq_score": candidate.rq_score,
                "s_hat": candidate.s_hat,
                "u_score": candidate.u_score,
                "op": candidate.op,
            }
        )
    return rows


def write_prominent_high_rq_report(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    quantile: float,
) -> tuple[Path, Path]:
    """Write the concise concept-only report used by the trajectory plot."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "high_rq_problems.json"
    markdown_path = output_dir / "high_rq_problems.md"
    json_path.write_text(
        json.dumps(
            {
                "selection": {
                    "scope": "per-outer-iteration maxima",
                    "quantile": quantile,
                    "checkpoint_independent": True,
                },
                "prominent_high_rq_events": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Prominent High-R_Q Evolution Events",
        "",
        (
            f"These are the top {(1.0 - quantile) * 100:.0f}% of per-outer "
            "R_Q maxima (subject to the plot annotation cap). Selection is "
            "independent of saved-model checkpoints; only GROUP/SKILL labels "
            "are shown."
        ),
        "",
        "| emerged step | outer | group / skill | R_Q | s_hat | H |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['emerged_global_step']} | {row['outer_iteration']} | "
            f"{row['group']} / {row['skill']} | {row['rq_score']:.2f} | "
            f"{row['s_hat']:.3f} | {row['u_score']:.2f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def write_high_rq_interval_report(
    output_dir: str | Path, rows: list[dict[str, Any]]
) -> tuple[Path, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "high_rq_problems.json"
    markdown_path = output_dir / "high_rq_problems.md"
    json_path.write_text(
        json.dumps({"interval_high_rq_problems": rows}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Highest-R_Q Inserted Problem per Saved-Model Interval",
        "",
        (
            "Each row is the highest-R_Q generated program that was actually "
            "inserted between the previous saved model's active outer iteration "
            "and this checkpoint. Problem text is deterministically rendered at "
            "seed 0 for inspection."
        ),
        "",
        "| checkpoint | emerged | outer | group / skill | R_Q | s_hat | H | child |",
        "|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint_step']} | {row['emerged_global_step']} | "
            f"{row['outer_iteration']} | {row['group']} / {row['skill']} | "
            f"{row['rq_score']:.2f} | {row['s_hat']:.3f} | "
            f"{row['u_score']:.2f} | `{row['child_id']}` |"
        )
    lines.extend(["", "## Rendered problems", ""])
    for row in rows:
        statement = (
            normalize_problem(row["problem"])
            if row["problem"]
            else f"[unavailable: {row['render_error']}]"
        )
        lines.extend(
            [
                f"### Checkpoint {row['checkpoint_step']} — O{row['outer_iteration']}",
                "",
                f"> {statement}",
                ">",
                f"Ground truth at seed {row['render_seed']}: `{row['answer']}`",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return json_path, markdown_path
