"""Visualise an R-Q-Evolve MAP-Elites archive as a mutation-lineage tree.

The archive directory of a run holds one snapshot per outer iteration
(``archive_iter<N>.json``) plus the final state (``archive.json``). Each
snapshot stores only the *current* champions, so no single file contains the
whole evolutionary history: a program that was a champion at iteration 12 and
got displaced at 13 exists only in the earlier snapshots. This script unions
every snapshot, reconnects ``parent_id -> program_id`` into a forest rooted at
the seed programs, and renders it.

Because ``p_hat`` is re-estimated as the solver improves, each node carries a
per-iteration trajectory rather than one number; the terminal view compresses it
into a sparkline and the HTML view shows the full series.

Usage:
  python scripts/viz_archive_tree.py [ARCHIVE_DIR]
  python scripts/viz_archive_tree.py [ARCHIVE_DIR] --html tree.html
  python scripts/viz_archive_tree.py [ARCHIVE_DIR] --sort rq --no-color

  ARCHIVE_DIR default: rq_output/rq_evolve_4b_nr20/rq_archive
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "rq_output" / "rq_evolve_4b_nr20" / "rq_archive"

SPARK_CHARS = "▁▂▃▄▅▆▇█"

# Seeds carry no ``op`` in metadata; generated children carry the operator that
# produced them.
OP_SEED = "seed"
OP_COLORS = {
    OP_SEED: "\033[37m",
    "in_depth": "\033[36m",
    "in_breadth": "\033[35m",
}
OP_SHORT = {OP_SEED: "seed", "in_depth": "depth", "in_breadth": "breadth"}

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


@dataclass
class Node:
    """One program that was a MAP-Elites champion at least once."""

    program_id: str
    parent_id: str
    generation: int
    op: str
    concept_group: str
    concept_type: str
    niche_h: int = -1
    niche_div: int = -1
    source_code: str = ""
    # (iteration, p_hat, rq_score, h_score), one entry per snapshot it survived.
    history: list[tuple[int, float, float, float]] = field(default_factory=list)
    alive: bool = False  # still a champion in the final archive.json
    orphan: bool = False  # parent_id set but that parent never appears
    # program_id is md5(source_code)[:12], so byte-identical output from a
    # DIFFERENT parent collapses onto the same node: the real structure is a DAG,
    # not a tree. Every (generation, parent_id) pair ever seen is kept here and
    # the renderers draw only the first; this set is what makes the rest visible
    # instead of silently dropped.
    lineages: set[tuple[int, str]] = field(default_factory=set)
    children: list["Node"] = field(default_factory=list)

    @property
    def first_iter(self) -> int:
        return self.history[0][0] if self.history else -1

    @property
    def last_iter(self) -> int:
        return self.history[-1][0] if self.history else -1

    @property
    def rederivations(self) -> int:
        """How many extra (generation, parent) pairs this source appeared under."""
        return max(0, len(self.lineages) - 1)

    @property
    def lifespan(self) -> int:
        return len(self.history)

    @property
    def p_hat(self) -> float:
        return self.history[-1][1] if self.history else 0.0

    @property
    def rq(self) -> float:
        return self.history[-1][2] if self.history else 0.0

    @property
    def best_rq(self) -> float:
        return max((h[2] for h in self.history), default=0.0)

    def sparkline(self) -> str:
        """p_hat over time; the axis MAP-Elites is actually selecting on."""
        return "".join(
            SPARK_CHARS[min(len(SPARK_CHARS) - 1, max(0, int(p * len(SPARK_CHARS))))]
            for _, p, _, _ in self.history
        )


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _snapshot_paths(archive_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in archive_dir.glob("archive_iter*.json"):
        match = re.fullmatch(r"archive_iter(\d+)\.json", path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def _read_champions(path: Path) -> tuple[list[dict], dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[viz] skipping unreadable {path.name}: {exc}", file=sys.stderr)
        return [], {}
    return data.get("champions", []), data.get("meta", {})


def _make_node(champion: dict) -> Node:
    metadata = champion.get("metadata") or {}
    return Node(
        program_id=champion["program_id"],
        parent_id=champion.get("parent_id") or "",
        generation=int(champion.get("generation", 0)),
        op=metadata.get("op") or OP_SEED,
        concept_group=metadata.get("concept_group") or "?",
        concept_type=metadata.get("concept_type") or "?",
        niche_h=int(champion.get("niche_h", -1)),
        niche_div=int(champion.get("niche_div", -1)),
        source_code=champion.get("source_code", ""),
    )


def load_nodes(archive_dir: Path) -> tuple[dict[str, Node], dict, list[int]]:
    """Union every snapshot into one ``program_id -> Node`` map."""
    snapshots = _snapshot_paths(archive_dir)
    if not snapshots:
        raise SystemExit(f"[viz] no archive_iter*.json under {archive_dir}")

    nodes: dict[str, Node] = {}
    meta: dict = {}
    for iteration, path in snapshots:
        champions, snapshot_meta = _read_champions(path)
        if snapshot_meta:
            meta = snapshot_meta
        for champion in champions:
            node = nodes.get(champion["program_id"])
            if node is None:
                node = _make_node(champion)
                nodes[node.program_id] = node
            # Niche assignment can move when the H-axis re-bins on re-eval, so
            # the newest snapshot wins.
            node.niche_h = int(champion.get("niche_h", node.niche_h))
            node.niche_div = int(champion.get("niche_div", node.niche_div))
            node.lineages.add(
                (int(champion.get("generation", 0)), champion.get("parent_id") or "")
            )
            node.history.append(
                (
                    iteration,
                    float(champion.get("p_hat", 0.0)),
                    float(champion.get("rq_score", 0.0)),
                    float(champion.get("h_score", 0.0)),
                )
            )

    final_path = archive_dir / "archive.json"
    if final_path.is_file():
        champions, final_meta = _read_champions(final_path)
        if final_meta:
            meta = final_meta
        tail = snapshots[-1][0] + 1
        for champion in champions:
            node = nodes.get(champion["program_id"])
            if node is None:
                # Final-only champion (inserted after the last numbered
                # snapshot); give it a single trailing point so it still renders.
                node = _make_node(champion)
                node.history.append(
                    (
                        tail,
                        float(champion.get("p_hat", 0.0)),
                        float(champion.get("rq_score", 0.0)),
                        float(champion.get("h_score", 0.0)),
                    )
                )
                nodes[node.program_id] = node
            node.alive = True

    return nodes, meta, [i for i, _ in snapshots]


def build_forest(nodes: dict[str, Node], sort_key: str) -> list[Node]:
    """Link children to parents and return the roots, sorted for display."""
    roots: list[Node] = []
    for node in nodes.values():
        parent = nodes.get(node.parent_id) if node.parent_id else None
        if parent is not None and parent is not node:
            parent.children.append(node)
        else:
            node.orphan = bool(node.parent_id) and parent is None
            roots.append(node)

    def key(node: Node):
        if sort_key == "rq":
            return (-node.best_rq, node.program_id)
        if sort_key == "id":
            return (node.program_id,)
        if sort_key == "group":
            return (node.concept_group, node.first_iter, node.program_id)
        return (node.first_iter, node.program_id)  # "iter" (default)

    for node in nodes.values():
        node.children.sort(key=key)
    roots.sort(key=key)
    return roots


# --------------------------------------------------------------------------
# terminal rendering
# --------------------------------------------------------------------------


def _flatten(roots: list[Node]) -> list[tuple[str, Node]]:
    """Depth-first walk returning (plain tree glyphs + id, node) pairs."""
    rows: list[tuple[str, Node]] = []

    def walk(node: Node, prefix: str, connector: str, child_prefix: str) -> None:
        rows.append((f"{prefix}{connector}{node.program_id}", node))
        base = prefix + child_prefix
        for index, child in enumerate(node.children):
            is_last = index == len(node.children) - 1
            walk(
                child,
                base,
                "└─ " if is_last else "├─ ",
                "   " if is_last else "│  ",
            )

    for root in roots:
        walk(root, "", "", "")
    return rows


def render_tree(roots: list[Node], *, color: bool, show_spark: bool) -> str:
    rows = _flatten(roots)
    if not rows:
        return "(empty archive)"
    width = max(len(glyphs) for glyphs, _ in rows)

    def paint(text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if color else text

    lines: list[str] = []
    header = (
        f"{'lineage':<{width}}  {'gen':>3} {'op':<7} {'concept_group':<14} "
        f"{'p̂':>5} {'R_Q':>7} {'best':>7} {'niche':>7} {'iters':>10}  "
        f"{'concept_type':<34}"
    )
    if show_spark:
        header += "  p̂ over iterations"
    lines.append(paint(header, DIM))
    lines.append(paint("─" * min(len(header), 160), DIM))

    for glyphs, node in rows:
        tree_cell = glyphs.ljust(width)
        if color:
            # Colour only the id at the tail of the glyph run, so padding stays
            # width-correct.
            tint = GREEN if node.alive else DIM
            tree_cell = tree_cell.replace(
                node.program_id, f"{tint}{node.program_id}{RESET}", 1
            )
        span = f"{node.first_iter}-{node.last_iter}"
        status = f"{span:>7}{'*' if node.alive else ' '}"
        op_cell = OP_SHORT.get(node.op, node.op).ljust(7)
        if color:
            op_cell = f"{OP_COLORS.get(node.op, '')}{op_cell}{RESET}"
        rq_cell = f"{node.rq:7.4f}"
        if color and node.rq >= 0.05:
            rq_cell = f"{YELLOW}{rq_cell}{RESET}"
        line = (
            f"{tree_cell}  {node.generation:>3} {op_cell} "
            f"{node.concept_group[:14]:<14} "
            f"{node.p_hat:5.2f} {rq_cell} {node.best_rq:7.4f} "
            f"{f'({node.niche_h},{node.niche_div})':>7} {status:>10}  "
            f"{node_label(node, 'type', 34):<34}"
        )
        if node.rederivations:
            line += paint(f" ↺{node.rederivations}", YELLOW)
        if show_spark:
            line += f"  {node.sparkline()}"
        if node.orphan:
            line += paint("  [orphan]", YELLOW)
        lines.append(line)
    return "\n".join(lines)


def render_summary(
    nodes: dict[str, Node],
    roots: list[Node],
    meta: dict,
    iterations: list[int],
    archive_dir: Path,
    *,
    color: bool,
) -> str:
    def paint(text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if color else text

    alive = [n for n in nodes.values() if n.alive]
    by_gen: dict[int, int] = {}
    by_op: dict[str, int] = {}
    by_group: dict[str, int] = {}
    for node in nodes.values():
        by_gen[node.generation] = by_gen.get(node.generation, 0) + 1
        by_op[node.op] = by_op.get(node.op, 0) + 1
        by_group[node.concept_group] = by_group.get(node.concept_group, 0) + 1

    surviving_roots = sum(
        1 for r in roots if _subtree_has_alive(r)
    )
    stats = meta.get("stats", {}) or {}
    top = sorted(nodes.values(), key=lambda n: -n.best_rq)[:5]

    out = [
        "",
        paint("── summary " + "─" * 60, DIM),
        f"archive        {archive_dir}",
        f"snapshots      {len(iterations)} (iter {min(iterations)}..{max(iterations)})",
        f"programs       {len(nodes)} ever champion, {len(alive)} alive in final archive",
        f"roots          {len(roots)} seeds, {surviving_roots} with surviving descendants",
        f"depth          generation {max(by_gen) if by_gen else 0} deepest",
        "by generation  "
        + "  ".join(f"g{g}:{c}" for g, c in sorted(by_gen.items())),
        "by operator    "
        + "  ".join(f"{OP_SHORT.get(o, o)}:{c}" for o, c in sorted(by_op.items())),
        "by group       "
        + "  ".join(f"{g}:{c}" for g, c in sorted(by_group.items())),
    ]
    if stats:
        out.append(
            f"final MAP      coverage {stats.get('coverage', 0):.1%} "
            f"({stats.get('num_champions', 0)}/{stats.get('total_niches', 0)} niches)  "
            f"mean R_Q {stats.get('mean_rq', 0):.4f}  max {stats.get('max_rq', 0):.4f}  "
            f"insert/replace {stats.get('total_insertions', 0)}/"
            f"{stats.get('total_replacements', 0)}"
        )
    out.append("top R_Q        " + ", ".join(f"{n.program_id}:{n.best_rq:.4f}" for n in top))

    rederived = [n for n in nodes.values() if n.rederivations]
    if rederived:
        extra = sum(n.rederivations for n in rederived)
        out.append(
            f"re-derived     {len(rederived)} programs appeared under {extra} other "
            f"(generation, parent) pairs — program_id is a source hash, so the "
            f"lineage is a DAG; only the first edge is drawn (marked ↺)"
        )

    accepted, attempted = _acceptance(archive_dir)
    if attempted:
        out.append(
            f"acceptance     {accepted}/{attempted} candidates inserted "
            f"({accepted / attempted:.1%}) — from evolution_log.jsonl"
        )
    seen_log = _candidate_ids(archive_dir)
    if seen_log:
        out.append(
            f"NOT SHOWN      {len(seen_log - set(nodes)):,} candidates from "
            f"evolution_log.jsonl never entered the archive (rejected before "
            f"insertion); their reports carry no parent_id, so they cannot be "
            f"placed in a lineage"
        )
    out.append("")
    out.append(
        paint(
            "legend  * = alive in final archive   sparkline = p̂ per iteration "
            "(▁ 0.0 → █ 1.0)",
            DIM,
        )
    )
    return "\n".join(out)


def _subtree_has_alive(node: Node) -> bool:
    if node.alive:
        return True
    return any(_subtree_has_alive(child) for child in node.children)


def _candidate_ids(archive_dir: Path) -> set[str]:
    """Every child_id the evolution log mentions, archived or rejected."""
    path = archive_dir / "evolution_log.jsonl"
    if not path.is_file():
        return set()
    ids: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            for report in json.loads(line).get("reports", []) or []:
                if report.get("child_id"):
                    ids.add(report["child_id"])
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return set()
    return ids


def _acceptance(archive_dir: Path) -> tuple[int, int]:
    """Candidate accept counts from evolution_log.jsonl, if present."""
    path = archive_dir / "evolution_log.jsonl"
    if not path.is_file():
        return 0, 0
    accepted = attempted = 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            metrics = json.loads(line).get("metrics", {}) or {}
            attempted += int(metrics.get("attempted", 0))
            accepted += int(metrics.get("inserted", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0, 0
    return accepted, attempted


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------

HTML_CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --line: #e5e7eb;
  --card: #f8f9fb; --seed: #64748b; --depth: #0891b2; --breadth: #a855f7;
  --alive: #16a34a; --accent: #d97706;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a; --fg: #e6e8eb; --muted: #9aa3af; --line: #2a2e35;
    --card: #1b1e24; --seed: #94a3b8; --depth: #22d3ee; --breadth: #c084fc;
    --alive: #4ade80; --accent: #fbbf24;
  }
}
:root[data-theme="light"] {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --line: #e5e7eb;
  --card: #f8f9fb; --seed: #64748b; --depth: #0891b2; --breadth: #a855f7;
  --alive: #16a34a; --accent: #d97706;
}
:root[data-theme="dark"] {
  --bg: #14161a; --fg: #e6e8eb; --muted: #9aa3af; --line: #2a2e35;
  --card: #1b1e24; --seed: #94a3b8; --depth: #22d3ee; --breadth: #c084fc;
  --alive: #4ade80; --accent: #fbbf24;
}
body { background: var(--bg); color: var(--fg); margin: 0; padding: 24px;
  font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif; }
h1 { font-size: 18px; margin: 0 0 4px; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 16px; }
.stats { display: flex; flex-wrap: wrap; gap: 8px 20px; padding: 12px 14px;
  background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  margin-bottom: 16px; font-size: 13px; }
.stats b { font-variant-numeric: tabular-nums; }
.toolbar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
button, select { font: inherit; padding: 5px 10px; border-radius: 6px;
  border: 1px solid var(--line); background: var(--card); color: var(--fg);
  cursor: pointer; }
.tree { overflow-x: auto; }
.node { border-left: 1px solid var(--line); margin-left: 9px; padding-left: 13px; }
.node.root { border-left: none; margin-left: 0; padding-left: 0; }
summary { list-style: none; cursor: pointer; padding: 3px 0;
  display: flex; align-items: baseline; gap: 8px; white-space: nowrap; }
summary::-webkit-details-marker { display: none; }
summary::before { content: "▸"; color: var(--muted); width: 10px;
  display: inline-block; flex: none; }
details[open] > summary::before { content: "▾"; }
details.leaf > summary::before { content: "·"; }
.pid { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12.5px; }
.pid.alive { color: var(--alive); font-weight: 600; }
.badge { font-size: 11px; padding: 1px 6px; border-radius: 4px;
  border: 1px solid currentColor; flex: none; }
.op-seed { color: var(--seed); } .op-in_depth { color: var(--depth); }
.op-in_breadth { color: var(--breadth); }
.grp { color: var(--muted); font-size: 12px; }
.m { font-variant-numeric: tabular-nums; font-size: 12px; color: var(--muted); }
.m b { color: var(--fg); font-weight: 600; }
.m.hot b { color: var(--accent); }
.spark { font-family: ui-monospace, monospace; letter-spacing: -1px;
  color: var(--muted); }
.detail { margin: 6px 0 10px 18px; padding: 10px 12px; background: var(--card);
  border: 1px solid var(--line); border-radius: 6px; }
.detail dl { display: grid; grid-template-columns: max-content 1fr;
  gap: 2px 14px; margin: 0 0 8px; font-size: 12.5px; }
.detail dt { color: var(--muted); }
.detail dd { margin: 0; font-variant-numeric: tabular-nums; }
pre { overflow-x: auto; margin: 0; padding: 10px; background: var(--bg);
  border: 1px solid var(--line); border-radius: 6px; font-size: 12px;
  line-height: 1.45; }
.dim { opacity: 0.45; }
"""

HTML_JS = """
const setAll = (open) =>
  document.querySelectorAll('details.n').forEach((d) => { d.open = open; });
document.getElementById('expand').onclick = () => setAll(true);
document.getElementById('collapse').onclick = () => setAll(false);
// One combined predicate so the group filter and the survivor filter compose
// instead of each clobbering the other's .dim toggles.
const groupSel = document.getElementById('group');
const aliveBox = document.getElementById('alive');
const applyFilter = () => {
  const want = groupSel.value;
  const onlyAlive = aliveBox.checked;
  document.querySelectorAll('.node').forEach((n) => {
    const hide =
      (want !== '*' && n.dataset.group !== want) ||
      (onlyAlive && n.dataset.alive !== '1');
    n.classList.toggle('dim', hide);
  });
};
groupSel.onchange = applyFilter;
aliveBox.onchange = applyFilter;
"""


def _node_html(node: Node, *, root: bool, with_source: bool) -> str:
    esc = html.escape
    classes = "node root" if root else "node"
    leaf = " leaf" if not node.children else ""
    pid_class = "pid alive" if node.alive else "pid"
    hot = " hot" if node.best_rq >= 0.05 else ""
    span = f"{node.first_iter}–{node.last_iter}"

    parts = [
        f'<div class="{classes}" data-group="{esc(node.concept_group)}" '
        f'data-alive="{1 if node.alive else 0}">',
        f'<details class="n{leaf}" open><summary>',
        f'<span class="{pid_class}">'
        f"{esc(node_label(node, 'type', 60))}</span>",
        f'<span class="badge op-{esc(node.op)}">{esc(OP_SHORT.get(node.op, node.op))}</span>',
        f'<span class="grp">{esc(node.concept_group)}</span>',
        f'<span class="grp">{esc(node.program_id)}</span>',
        f'<span class="m">p̂ <b>{node.p_hat:.2f}</b></span>',
        f'<span class="m{hot}">R_Q <b>{node.rq:.4f}</b></span>',
        f'<span class="m">gen <b>{node.generation}</b></span>',
        f'<span class="m">iter <b>{esc(span)}</b></span>',
        f'<span class="spark">{esc(node.sparkline())}</span>',
        "</summary>",
        '<div class="detail"><dl>',
        f"<dt>concept</dt><dd>{esc(node.concept_type)}</dd>",
        f"<dt>niche (h, div)</dt><dd>({node.niche_h}, {node.niche_div})</dd>",
        f"<dt>parent</dt><dd>{esc(node.parent_id or '— seed —')}</dd>",
        f"<dt>lifespan</dt><dd>{node.lifespan} snapshots"
        f"{' (alive)' if node.alive else ' (displaced)'}</dd>",
        f"<dt>best R_Q</dt><dd>{node.best_rq:.4f}</dd>",
        "<dt>p̂ series</dt><dd>"
        + esc(", ".join(f"{i}:{p:.2f}" for i, p, _, _ in node.history))
        + "</dd>",
        "</dl>",
    ]
    if with_source and node.source_code:
        parts.append(f"<pre>{esc(node.source_code)}</pre>")
    parts.append("</div>")
    for child in node.children:
        parts.append(_node_html(child, root=False, with_source=with_source))
    parts.append("</details></div>")
    return "".join(parts)


def render_html(
    nodes: dict[str, Node],
    roots: list[Node],
    meta: dict,
    iterations: list[int],
    archive_dir: Path,
    *,
    with_source: bool,
) -> str:
    esc = html.escape
    stats = meta.get("stats", {}) or {}
    alive = sum(1 for n in nodes.values() if n.alive)
    groups = sorted({n.concept_group for n in nodes.values()})
    options = "".join(f'<option value="{esc(g)}">{esc(g)}</option>' for g in groups)

    stat_items = [
        ("snapshots", f"{len(iterations)}"),
        ("ever champion", f"{len(nodes)}"),
        ("alive", f"{alive}"),
        ("seeds (roots)", f"{len(roots)}"),
        ("max generation", f"{max((n.generation for n in nodes.values()), default=0)}"),
        ("coverage", f"{stats.get('coverage', 0):.1%}"),
        ("mean R_Q", f"{stats.get('mean_rq', 0):.4f}"),
        ("insert/replace",
         f"{stats.get('total_insertions', 0)}/{stats.get('total_replacements', 0)}"),
    ]
    stats_html = "".join(
        f"<span>{esc(k)} <b>{esc(v)}</b></span>" for k, v in stat_items
    )
    body = "".join(_node_html(r, root=True, with_source=with_source) for r in roots)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>R-Q-Evolve lineage — {esc(archive_dir.parent.name)}</title>
<style>{HTML_CSS}</style></head><body>
<h1>MAP-Elites mutation lineage</h1>
<div class="sub">{esc(str(archive_dir))}</div>
<div class="stats">{stats_html}</div>
<div class="toolbar">
  <button id="expand">expand all</button>
  <button id="collapse">collapse all</button>
  <select id="group"><option value="*">all concept groups</option>{options}</select>
  <label><input type="checkbox" id="alive"> only surviving</label>
</div>
<div class="tree">{body}</div>
<script>{HTML_JS}</script>
</body></html>"""


# --------------------------------------------------------------------------
# SVG rendering (drawn node-link tree; no third-party dependency)
# --------------------------------------------------------------------------

# Sequential blue ramp, light -> dark, used for p_hat magnitude. A single hue is
# the correct encoding for a continuous magnitude; the categorical concept group
# rides on the text tag instead, so identity is never colour-alone.
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
        "grid": "#e6e5e1", "edge": "#b9b8b3", "ring": "#0b0b0b",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
        "grid": "#33332f", "edge": "#5a5a54", "ring": "#ffffff",
    },
}

GROUP_TAG = {
    "number_theory": "num", "combinatorics": "comb", "sequence": "seq",
    "algebra": "alg", "geometry": "geo", "inequality": "ineq",
}

# One marker per CONCEPT_GROUP. Six groups, six shapes, fixed assignment -- the
# concept group is a categorical identity, so it gets its own channel instead of
# competing with p_hat for the fill.
GROUP_SHAPE = {
    "number_theory": "circle",
    "combinatorics": "square",
    "sequence": "triangle",
    "algebra": "diamond",
    "geometry": "hexagon",
    "inequality": "down_triangle",
}
SHAPE_ORDER = ("number_theory", "combinatorics", "sequence",
               "algebra", "geometry", "inequality")

# Circumradius multipliers that make every shape cover the same AREA as a circle
# of radius r, so "size = best R_Q" stays comparable across groups. A square of
# side r would read far smaller than a circle of radius r.
_SHAPE_K = {
    "circle": 1.0,
    "square": 1.2533,         # half-diagonal; side = r*sqrt(pi)
    "diamond": 1.2533,
    "triangle": 1.5551,       # equilateral, area = 3*sqrt(3)/4 * R^2
    "down_triangle": 1.5551,
    "hexagon": 1.0996,        # regular hexagon, area = 3*sqrt(3)/2 * R^2
}

ROW_H = 26
MARGIN_L = 34
MARGIN_T = 148
ROOT_GAP = 0.85  # blank rows inserted between separate seed lineages
SPARK_W = 40
CHAR_W = 5.55  # measured advance of the 9.5px label font
LEGEND_LABEL_W = 112  # room for the "CONCEPT_GROUP" row title
LEGEND_SLOT_W = 190   # per-group slot; fits "number_theory (12)" in both fonts
# Label inset from the node centre. Must clear the widest marker (triangle,
# circumradius 1.5551*r) plus its survivor halo, or big nodes sit on their text.
LABEL_DX0 = 26


def _metrics(label_chars: int) -> tuple[int, int, int, float]:
    """Column geometry derived from how wide the name label is allowed to be.

    Returns ``(col_w, label_dx, trunk_dx, label_px)``. Elbow connectors turn at
    ``trunk_dx``; it has to clear the parent's own label run or every edge is
    drawn straight through the text it belongs to.
    """
    label_px = label_chars * CHAR_W
    label_dx = int(LABEL_DX0 + label_px + 12 + SPARK_W + 8 + 30)
    trunk_dx = label_dx + 10
    return trunk_dx + 18, label_dx, trunk_dx, label_px


def _shape_points(shape: str, x: float, y: float, r: float) -> list[tuple[float, float]]:
    """Polygon vertices for an area-normalised marker (empty for a circle)."""
    import math

    k = _SHAPE_K.get(shape, 1.0)
    if shape == "circle":
        return []
    if shape == "square":
        h = r * k / (2 ** 0.5)
        return [(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)]
    if shape == "diamond":
        d = r * k
        return [(x, y - d), (x + d, y), (x, y + d), (x - d, y)]
    sides, rot = (3, -90.0) if shape == "triangle" else (
        (3, 90.0) if shape == "down_triangle" else (6, 0.0)
    )
    radius = r * k
    return [
        (
            x + radius * math.cos(math.radians(rot + i * 360.0 / sides)),
            y + radius * math.sin(math.radians(rot + i * 360.0 / sides)),
        )
        for i in range(sides)
    ]


def _outer_radius(shape: str, r: float) -> float:
    """Circumradius of the marker -- the survivor halo has to sit outside it."""
    return r * _SHAPE_K.get(shape, 1.0)


def node_label(node: Node, mode: str, max_chars: int) -> str:
    """Display name for a node.

    ``type`` drops the group prefix from CONCEPT_TYPE (the marker shape already
    carries the group), so ``number_theory.crt_count`` shows as ``crt_count``.
    """
    if mode == "id":
        return node.program_id[:8]
    if mode == "group":
        text = node.concept_group
    else:
        text = (node.concept_type or "?").split(".", 1)[-1]
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def _placed(ordered: list[Node]) -> dict[str, Node]:
    """program_id -> node, restricted to nodes the layout actually positioned."""
    return {n.program_id: n for n in ordered}


def _layout(roots: list[Node]) -> tuple[list[Node], float, int]:
    """Tidy layout: leaves stack down, parents centre on their children."""
    ordered: list[Node] = []
    cursor = [0.0]
    max_depth = [0]

    def place(node: Node, depth: int) -> None:
        node._depth = depth  # type: ignore[attr-defined]
        max_depth[0] = max(max_depth[0], depth)
        ordered.append(node)
        if not node.children:
            node._row = cursor[0]  # type: ignore[attr-defined]
            cursor[0] += 1.0
        else:
            for child in node.children:
                child._tree_parent = node.program_id  # type: ignore[attr-defined]
                place(child, depth + 1)
            first = node.children[0]._row  # type: ignore[attr-defined]
            last = node.children[-1]._row  # type: ignore[attr-defined]
            node._row = (first + last) / 2.0  # type: ignore[attr-defined]

    for root in roots:
        place(root, 0)
        cursor[0] += ROOT_GAP
    return ordered, cursor[0], max_depth[0]


def _ramp(theme: str) -> list[str]:
    """Ramp oriented so the low end recedes toward the surface in either mode.

    On light, that is light->dark as published. On dark, the published dark end
    (#0d366b) sits almost on the dark surface, so the whole high-p̂ half of the
    scale collapses into the background and the encoding stops reading; the
    dark mode therefore steps the same hue in the opposite direction rather
    than flipping the surface only.
    """
    return BLUE_RAMP if theme == "light" else BLUE_RAMP[::-1]


def _blue(p_hat: float, theme: str = "light") -> str:
    ramp = _ramp(theme)
    idx = int(max(0.0, min(1.0, p_hat)) * (len(ramp) - 1) + 0.5)
    return ramp[idx]


def _radius(best_rq: float, scale: float) -> float:
    # Area-proportional (sqrt) so a 4x R_Q reads as 4x area, not 4x width.
    return 4.0 + 8.5 * (best_rq / scale) ** 0.5 if scale > 0 else 4.0


def _spark_path(node: Node, x: float, y: float, w: float, h: float) -> str:
    pts = node.history
    if len(pts) < 2:
        return ""
    step = w / (len(pts) - 1)
    coords = " ".join(
        f"{x + i * step:.1f},{y + h - max(0.0, min(1.0, p)) * h:.1f}"
        for i, (_, p, _, _) in enumerate(pts)
    )
    return f'<polyline points="{coords}" fill="none" stroke-width="1"/>'


def render_svg(
    nodes: dict[str, Node],
    roots: list[Node],
    meta: dict,
    iterations: list[int],
    archive_dir: Path,
    *,
    theme: str = "light",
    label_mode: str = "type",
    label_chars: int = 34,
    show_converged: bool = False,
) -> str:
    esc = html.escape
    palette = THEMES[theme]
    ordered, total_rows, max_depth = _layout(roots)
    scale = max((n.best_rq for n in nodes.values()), default=1.0) or 1.0
    col_w, label_dx, trunk_dx, _ = _metrics(label_chars)

    width = MARGIN_L + (max_depth + 1) * col_w + 40
    height = MARGIN_T + int(total_rows * ROW_H) + 46

    def cx(node: Node) -> float:
        return MARGIN_L + node._depth * col_w  # type: ignore[attr-defined]

    def cy(node: Node) -> float:
        return MARGIN_T + node._row * ROW_H  # type: ignore[attr-defined]

    def marker(shape: str, x: float, y: float, r: float, attrs: str) -> str:
        pts = _shape_points(shape, x, y, r)
        if not pts:
            return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}"{attrs}/>'
        coords = " ".join(f"{px_:.1f},{py_:.1f}" for px_, py_ in pts)
        return f'<polygon points="{coords}"{attrs}/>'

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="ui-sans-serif, system-ui, -apple-system, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="{palette["surface"]}"/>',
    ]

    # --- title block ------------------------------------------------------
    stats = meta.get("stats", {}) or {}
    alive = sum(1 for n in nodes.values() if n.alive)
    subtitle = (
        f"{len(nodes)} programs ever champion · {alive} alive · {len(roots)} seed roots · "
        f"{len(iterations)} iterations · coverage {stats.get('coverage', 0):.0%}"
    )
    out += [
        f'<text x="{MARGIN_L}" y="34" font-size="19" font-weight="600" '
        f'fill="{palette["ink"]}">MAP-Elites mutation lineage</text>',
        f'<text x="{MARGIN_L}" y="54" font-size="12" fill="{palette["ink2"]}">'
        f"{esc(archive_dir.parent.name)} — {esc(subtitle)}</text>",
    ]

    # --- legend row 1: encodings -----------------------------------------
    lx, ly = MARGIN_L, 78
    out.append(f'<g font-size="11" fill="{palette["ink2"]}">')
    out.append(f'<text x="{lx}" y="{ly + 4}">p̂ (fill)</text>')
    for i, hexv in enumerate(_ramp(theme)):
        out.append(
            f'<rect x="{lx + 48 + i * 11}" y="{ly - 5}" width="11" height="11" '
            f'fill="{hexv}"/>'
        )
    out.append(f'<text x="{lx + 48 + len(BLUE_RAMP) * 11 + 6}" y="{ly + 4}">0 → 1</text>')
    lx2 = lx + 48 + len(BLUE_RAMP) * 11 + 48
    out += [
        f'<circle cx="{lx2 + 6}" cy="{ly}" r="7" fill="none" '
        f'stroke="{palette["ink2"]}" stroke-width="1"/>',
        f'<text x="{lx2 + 18}" y="{ly + 4}">size = best R_Q</text>',
        f'<circle cx="{lx2 + 128}" cy="{ly}" r="4" fill="{palette["edge"]}"/>',
        f'<circle cx="{lx2 + 128}" cy="{ly}" r="7.2" fill="none" '
        f'stroke="{palette["ring"]}" stroke-width="1.6"/>',
        f'<text x="{lx2 + 140}" y="{ly + 4}">ring = alive in final archive</text>',
        f'<line x1="{lx2 + 300}" y1="{ly}" x2="{lx2 + 330}" y2="{ly}" '
        f'stroke="{palette["edge"]}" stroke-width="1.6"/>',
        f'<text x="{lx2 + 336}" y="{ly + 4}">in_depth</text>',
        f'<line x1="{lx2 + 400}" y1="{ly}" x2="{lx2 + 430}" y2="{ly}" '
        f'stroke="{palette["edge"]}" stroke-width="1.6" stroke-dasharray="4 3"/>',
        f'<text x="{lx2 + 436}" y="{ly + 4}">in_breadth</text>',
        "</g>",
    ]

    # --- legend row 2: one marker per concept group -----------------------
    ly2 = 104
    out.append(f'<g font-size="11" fill="{palette["ink2"]}">')
    out.append(f'<text x="{lx}" y="{ly2 + 4}">CONCEPT_GROUP</text>')
    gx = lx + LEGEND_LABEL_W
    counts = {g: 0 for g in SHAPE_ORDER}
    for node in nodes.values():
        if node.concept_group in counts:
            counts[node.concept_group] += 1
    for group in SHAPE_ORDER:
        shape = GROUP_SHAPE[group]
        out.append(
            marker(shape, gx, ly2, 5.5,
                   f' fill="{palette["ink2"]}" opacity="0.75"')
        )
        text = f"{group} ({counts[group]})"
        out.append(f'<text x="{gx + 12}" y="{ly2 + 4}">{esc(text)}</text>')
        gx += LEGEND_SLOT_W
    out.append("</g>")

    # --- generation columns ----------------------------------------------
    out.append(f'<g font-size="11" fill="{palette["ink2"]}">')
    for depth in range(max_depth + 1):
        x = MARGIN_L + depth * col_w
        out.append(
            f'<line x1="{x}" y1="{MARGIN_T - 12}" x2="{x}" '
            f'y2="{height - 24}" stroke="{palette["grid"]}" stroke-width="1"/>'
        )
        out.append(f'<text x="{x - 8}" y="{MARGIN_T - 20}">gen {depth}</text>')
    out.append("</g>")

    # --- edges (drawn first so nodes sit on top) --------------------------
    out.append(f'<g fill="none" stroke="{palette["edge"]}" stroke-width="1.6">')
    for node in ordered:
        for child in node.children:
            trunk = cx(node) + trunk_dx
            dash = ' stroke-dasharray="4 3"' if child.op == "in_breadth" else ""
            out.append(
                f'<path d="M{trunk:.1f},{cy(node):.1f} V{cy(child):.1f} '
                f'H{cx(child):.1f}"{dash}/>'
            )
        if node.children:
            # Stub from just past the parent's label to its trunk, drawn once.
            out.append(
                f'<path d="M{cx(node) + label_dx:.1f},{cy(node):.1f} '
                f'H{cx(node) + trunk_dx:.1f}"/>'
            )
    out.append("</g>")

    # --- alternate-parent overlay ----------------------------------------
    if show_converged:
        accent = "#eb6834" if theme == "light" else "#d95926"
        drawn = _placed(ordered)
        for node in ordered:
            for parent_id in sorted(_parent_ids(node)):
                if parent_id == getattr(node, "_tree_parent", None):
                    continue
                parent = drawn.get(parent_id)
                if parent is None:
                    continue
                x1, y1, x2, y2 = cx(parent), cy(parent), cx(node), cy(node)
                # Bow away from the trunks so the curve stays readable even when
                # the alternate parent sits to the RIGHT (a backward edge).
                bow = 26 + abs(y2 - y1) * 0.18
                out.append(
                    f'<path d="M{x1:.1f},{y1:.1f} '
                    f'Q{(x1 + x2) / 2:.1f},{(y1 + y2) / 2 - bow:.1f} '
                    f'{x2:.1f},{y2:.1f}" fill="none" stroke="{accent}" '
                    f'stroke-width="1.1" opacity="0.55"/>'
                )

    # --- nodes ------------------------------------------------------------
    for node in ordered:
        x, y = cx(node), cy(node)
        r = _radius(node.best_rq, scale)
        shape = GROUP_SHAPE.get(node.concept_group, "circle")
        opacity = "" if node.alive else ' opacity="0.75"'
        out.append(
            marker(
                shape, x, y, r,
                f' fill="{_blue(node.p_hat, theme)}" stroke="{palette["grid"]}"'
                f' stroke-width="1"{opacity}',
            )
        )
        if node.alive:
            # Detached halo, not a stroke on the mark: at the light end of the
            # ramp a stroke in the ink colour is the same value as the fill in
            # one mode or the other, and the survivor flag silently vanishes.
            out.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" '
                f'r="{_outer_radius(shape, r) + 3.2:.1f}" '
                f'fill="none" stroke="{palette["ring"]}" stroke-width="1.6"/>'
            )
        tx = x + LABEL_DX0
        out.append(
            f'<text x="{tx:.1f}" y="{y + 3.5:.1f}" font-size="9.5" '
            f'fill="{palette["ink"]}">'
            f"{esc(node_label(node, label_mode, label_chars))}</text>"
        )
        sx = x + label_dx - SPARK_W - 38
        spark = _spark_path(node, sx, y - 6, SPARK_W, 12)
        if spark:
            out.append(f'<g stroke="{palette["ink2"]}" opacity="0.8">{spark}</g>')
        out.append(
            f'<text x="{x + label_dx - 30:.1f}" y="{y + 3.5:.1f}" font-size="9.5" '
            f'fill="{palette["ink2"]}">{node.p_hat:.2f}</text>'
        )

    out.append(
        f'<text x="{MARGIN_L}" y="{height - 8}" font-size="10" '
        f'fill="{palette["ink2"]}">R_Q = p̂(1-p̂)·H peaks at p̂ = 0.5 — the '
        f"darkest and lightest fills are both dead niches. Label = CONCEPT_TYPE "
        f"with its group prefix dropped; sparkline = p̂ per iteration while the "
        f"program held its niche. Only archived champions appear; program_id is a "
        f"source hash, so re-derivations from other parents collapse onto one "
f"node.</text>"
    )
    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------------------
# PNG rendering (same layout, rasterised with Pillow if it is installed)
# --------------------------------------------------------------------------

FONT_CANDIDATES = {
    "sans": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ),
    "bold": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
    "mono": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ),
}


def _load_font(kind: str, size: int):
    from PIL import ImageFont

    for path in FONT_CANDIDATES[kind]:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _dashed_line(draw, p1, p2, fill, width, dash=8, gap=6) -> None:
    """PIL has no dash support; walk the segment emitting on/off runs."""
    (x1, y1), (x2, y2) = p1, p2
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if length < 1e-6:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        draw.line(
            [(x1 + ux * pos, y1 + uy * pos), (x1 + ux * end, y1 + uy * end)],
            fill=fill, width=width,
        )
        pos = end + gap


def render_png(
    nodes: dict[str, Node],
    roots: list[Node],
    meta: dict,
    iterations: list[int],
    archive_dir: Path,
    out_path: Path,
    *,
    theme: str = "light",
    label_mode: str = "type",
    label_chars: int = 34,
    show_converged: bool = False,
    supersample: int = 2,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "[viz] --png needs Pillow (pip install pillow), or use --svg "
            "which has no dependencies."
        ) from exc

    palette = THEMES[theme]
    ordered, total_rows, max_depth = _layout(roots)
    scale = max((n.best_rq for n in nodes.values()), default=1.0) or 1.0
    col_w, label_dx, trunk_dx, _ = _metrics(label_chars)

    width = MARGIN_L + (max_depth + 1) * col_w + 40
    height = MARGIN_T + int(total_rows * ROW_H) + 46
    s = max(1, int(supersample))

    image = Image.new("RGB", (width * s, height * s), palette["surface"])
    draw = ImageDraw.Draw(image)
    f_title = _load_font("bold", 19 * s)
    f_sub = _load_font("sans", 12 * s)
    f_small = _load_font("sans", 11 * s)
    f_label = _load_font("sans", 10 * s)

    def px(v: float) -> float:
        return v * s

    def cx(node: Node) -> float:
        return px(MARGIN_L + node._depth * col_w)  # type: ignore[attr-defined]

    def cy(node: Node) -> float:
        return px(MARGIN_T + node._row * ROW_H)  # type: ignore[attr-defined]

    def marker(shape, x, y, r, fill=None, outline=None, w=1):
        """Draw an area-normalised group marker in device pixels."""
        pts = _shape_points(shape, x, y, r)
        if pts:
            draw.polygon(pts, fill=fill, outline=outline, width=w)
        else:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=fill,
                         outline=outline, width=w)

    # --- title ------------------------------------------------------------
    stats = meta.get("stats", {}) or {}
    alive = sum(1 for n in nodes.values() if n.alive)
    draw.text(
        (px(MARGIN_L), px(18)), "MAP-Elites mutation lineage",
        font=f_title, fill=palette["ink"],
    )
    draw.text(
        (px(MARGIN_L), px(45)),
        f"{archive_dir.parent.name} — {len(nodes)} programs ever champion · "
        f"{alive} alive · {len(roots)} seed roots · {len(iterations)} iterations "
        f"· coverage {stats.get('coverage', 0):.0%}",
        font=f_sub, fill=palette["ink2"],
    )

    # --- legend row 1: encodings -----------------------------------------
    ly = px(78)
    draw.text((px(MARGIN_L), ly - px(6)), "p̂ (fill)", font=f_small,
              fill=palette["ink2"])
    for i, hexv in enumerate(_ramp(theme)):
        x0 = px(MARGIN_L + 48 + i * 11)
        draw.rectangle([x0, ly - px(5), x0 + px(10), ly + px(6)], fill=hexv)
    lx2 = MARGIN_L + 48 + len(BLUE_RAMP) * 11
    draw.text((px(lx2 + 6), ly - px(6)), "0 → 1", font=f_small, fill=palette["ink2"])
    lx2 += 48
    draw.ellipse(
        [px(lx2) - px(7), ly - px(7), px(lx2) + px(7), ly + px(7)],
        outline=palette["ink2"], width=max(1, s),
    )
    draw.text((px(lx2 + 14), ly - px(6)), "size = best R_Q", font=f_small,
              fill=palette["ink2"])
    draw.ellipse(
        [px(lx2 + 126), ly - px(4), px(lx2 + 134), ly + px(4)],
        fill=palette["edge"],
    )
    draw.ellipse(
        [px(lx2 + 123), ly - px(7), px(lx2 + 137), ly + px(7)],
        outline=palette["ring"], width=max(1, 2 * s),
    )
    draw.text((px(lx2 + 142), ly - px(6)), "ring = alive in final archive",
              font=f_small, fill=palette["ink2"])
    draw.line([px(lx2 + 300), ly, px(lx2 + 330), ly], fill=palette["edge"],
              width=max(1, 2 * s))
    draw.text((px(lx2 + 336), ly - px(6)), "in_depth", font=f_small,
              fill=palette["ink2"])
    _dashed_line(draw, (px(lx2 + 400), ly), (px(lx2 + 430), ly),
                 palette["edge"], max(1, 2 * s), dash=4 * s, gap=3 * s)
    draw.text((px(lx2 + 436), ly - px(6)), "in_breadth", font=f_small,
              fill=palette["ink2"])

    # --- legend row 2: one marker per concept group -----------------------
    ly2 = px(104)
    draw.text((px(MARGIN_L), ly2 - px(6)), "CONCEPT_GROUP", font=f_small,
              fill=palette["ink2"])
    gx = MARGIN_L + LEGEND_LABEL_W
    counts = {g: 0 for g in SHAPE_ORDER}
    for node in nodes.values():
        if node.concept_group in counts:
            counts[node.concept_group] += 1
    for group in SHAPE_ORDER:
        marker(GROUP_SHAPE[group], px(gx), ly2, px(5.5), fill=palette["ink2"])
        text = f"{group} ({counts[group]})"
        draw.text((px(gx + 12), ly2 - px(6)), text, font=f_small,
                  fill=palette["ink2"])
        gx += LEGEND_SLOT_W

    # --- generation columns ----------------------------------------------
    for depth in range(max_depth + 1):
        x = px(MARGIN_L + depth * col_w)
        draw.line([x, px(MARGIN_T - 12), x, px(height - 24)],
                  fill=palette["grid"], width=max(1, s))
        draw.text((x - px(8), px(MARGIN_T - 30)), f"gen {depth}",
                  font=f_small, fill=palette["ink2"])

    # --- edges ------------------------------------------------------------
    for node in ordered:
        if not node.children:
            continue
        trunk = cx(node) + px(trunk_dx)
        draw.line(
            [(cx(node) + px(label_dx), cy(node)), (trunk, cy(node))],
            fill=palette["edge"], width=max(1, 2 * s),
        )
        for child in node.children:
            y1, x2, y2 = cy(node), cx(child), cy(child)
            for a, b in (((trunk, y1), (trunk, y2)), ((trunk, y2), (x2, y2))):
                if child.op == "in_breadth":
                    _dashed_line(draw, a, b, palette["edge"], max(1, 2 * s),
                                 dash=4 * s, gap=3 * s)
                else:
                    draw.line([a, b], fill=palette["edge"], width=max(1, 2 * s))

    # --- alternate-parent overlay ----------------------------------------
    if show_converged:
        accent = "#eb6834" if theme == "light" else "#d95926"
        drawn = _placed(ordered)
        for node in ordered:
            for parent_id in sorted(_parent_ids(node)):
                if parent_id == getattr(node, "_tree_parent", None):
                    continue
                parent = drawn.get(parent_id)
                if parent is None:
                    continue
                x1, y1, x2, y2 = cx(parent), cy(parent), cx(node), cy(node)
                bow = px(26 + abs(y2 - y1) / s * 0.18)
                qx, qy = (x1 + x2) / 2, (y1 + y2) / 2 - bow
                pts = []
                for i in range(29):
                    t = i / 28
                    u = 1 - t
                    pts.append((u * u * x1 + 2 * u * t * qx + t * t * x2,
                                u * u * y1 + 2 * u * t * qy + t * t * y2))
                draw.line(pts, fill=accent, width=max(1, s), joint="curve")

    # --- nodes ------------------------------------------------------------
    for node in ordered:
        x, y = cx(node), cy(node)
        r = _radius(node.best_rq, scale)
        shape = GROUP_SHAPE.get(node.concept_group, "circle")
        marker(shape, x, y, px(r), fill=_blue(node.p_hat, theme),
               outline=palette["grid"], w=max(1, s))
        if node.alive:
            h = px(_outer_radius(shape, r) + 3.2)
            draw.ellipse(
                [x - h, y - h, x + h, y + h], outline=palette["ring"],
                width=max(1, 2 * s),
            )
        draw.text((x + px(LABEL_DX0), y - px(6)),
                  node_label(node, label_mode, label_chars),
                  font=f_label, fill=palette["ink"])
        pts = node.history
        if len(pts) > 1:
            sx = x + px(label_dx - SPARK_W - 38)
            sy, sh = y - px(6), px(12)
            step = px(SPARK_W) / (len(pts) - 1)
            draw.line(
                [
                    (sx + i * step, sy + sh - max(0.0, min(1.0, p)) * sh)
                    for i, (_, p, _, _) in enumerate(pts)
                ],
                fill=palette["ink2"], width=max(1, s),
            )
        draw.text((x + px(label_dx - 30), y - px(6)), f"{node.p_hat:.2f}",
                  font=f_label, fill=palette["ink2"])

    draw.text(
        (px(MARGIN_L), px(height - 18)),
        "R_Q = p̂(1-p̂)·H peaks at p̂ = 0.5 — the darkest and lightest fills are "
        "both dead niches. Label = CONCEPT_TYPE with its group prefix dropped; "
        "sparkline = p̂ per iteration while the program held its niche. Only "
        "archived champions appear; program_id is a source hash, so "
        "re-derivations from other parents collapse onto one node.",
        font=f_label, fill=palette["ink2"],
    )

    if s > 1:
        image = image.resize((width, height), Image.LANCZOS)
    image.save(out_path)


# --------------------------------------------------------------------------
# convergence view: where the lineage stops being a tree
# --------------------------------------------------------------------------


def _parent_ids(node: Node) -> set[str]:
    return {p for _, p in node.lineages if p}


def find_cycles(nodes: dict[str, Node]) -> list[list[str]]:
    """Cycles in the full parent->child graph, deduplicated by rotation.

    The lineage is not merely a DAG: a mutation can emit source byte-identical
    to one of its own ancestors, and since program_id is md5(source) that closes
    a loop. Colour-marking DFS; every back edge to a grey node is one cycle.
    """
    adj: dict[str, set[str]] = {k: set() for k in nodes}
    for child in nodes.values():
        for parent in _parent_ids(child):
            if parent in adj:
                adj[parent].add(child.program_id)

    WHITE, GREY, BLACK = 0, 1, 2
    color = dict.fromkeys(nodes, WHITE)
    found: list[list[str]] = []
    seen: set[frozenset[str]] = set()

    def visit(start: str) -> None:
        # Explicit stack: a deep lineage would blow the recursion limit.
        stack: list[tuple[str, object]] = [(start, iter(sorted(adj[start])))]
        path = [start]
        color[start] = GREY
        while stack:
            node_id, children = stack[-1]
            nxt = next(children, None)  # type: ignore[arg-type]
            if nxt is None:
                color[node_id] = BLACK
                stack.pop()
                path.pop()
                continue
            if color[nxt] == GREY:
                cycle = path[path.index(nxt):] + [nxt]
                key = frozenset(cycle)
                if key not in seen:
                    seen.add(key)
                    found.append(cycle)
            elif color[nxt] == WHITE:
                color[nxt] = GREY
                path.append(nxt)
                stack.append((nxt, iter(sorted(adj[nxt]))))

    for node_id in sorted(nodes):
        if color[node_id] == WHITE:
            visit(node_id)
    return sorted(found, key=lambda c: (-len(c), c[0]))


def find_attractors(nodes: dict[str, Node]) -> list[tuple[Node, list[Node]]]:
    """Programs re-derived from two or more distinct parents, busiest first."""
    out: list[tuple[Node, list[Node]]] = []
    for node in nodes.values():
        parents = [nodes[p] for p in sorted(_parent_ids(node)) if p in nodes]
        if len(parents) >= 2:
            out.append((node, parents))
    return sorted(out, key=lambda pair: (-len(pair[1]), pair[0].program_id))


class _Canvas:
    """Minimal shared drawing surface so both views render once, not twice."""

    def __init__(self, width: int, height: int, palette: dict) -> None:
        self.width, self.height, self.palette = width, height, palette


class _SvgCanvas(_Canvas):
    def __init__(self, width: int, height: int, palette: dict) -> None:
        super().__init__(width, height, palette)
        self._out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            f'font-family="ui-sans-serif, system-ui, -apple-system, sans-serif">',
            f'<rect width="{width}" height="{height}" fill="{palette["surface"]}"/>',
        ]

    @staticmethod
    def _stroke(color: str, width: float, dash, opacity: float) -> str:
        bits = f' stroke="{color}" stroke-width="{width}" fill="none"'
        if dash:
            bits += f' stroke-dasharray="{dash}"'
        if opacity < 1:
            bits += f' opacity="{opacity}"'
        return bits

    def line(self, x1, y1, x2, y2, color, width=1.0, dash=None, opacity=1.0):
        self._out.append(
            f'<path d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}"'
            + self._stroke(color, width, dash, opacity) + "/>"
        )

    def curve(self, x1, y1, cx, cy, x2, y2, color, width=1.0, dash=None, opacity=1.0):
        self._out.append(
            f'<path d="M{x1:.1f},{y1:.1f} Q{cx:.1f},{cy:.1f} {x2:.1f},{y2:.1f}"'
            + self._stroke(color, width, dash, opacity) + "/>"
        )

    def marker(self, shape, x, y, r, fill, outline=None, width=1.0):
        stroke = f' stroke="{outline}" stroke-width="{width}"' if outline else ""
        pts = _shape_points(shape, x, y, r)
        if pts:
            coords = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
            self._out.append(f'<polygon points="{coords}" fill="{fill}"{stroke}/>')
        else:
            self._out.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}"{stroke}/>'
            )

    def ring(self, x, y, r, color, width=1.6):
        self._out.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="none" '
            f'stroke="{color}" stroke-width="{width}"/>'
        )

    def text(self, x, y, s, size=10, fill=None, anchor="start", mono=False, bold=False):
        extra = ""
        if anchor != "start":
            extra += f' text-anchor="{"end" if anchor == "end" else "middle"}"'
        if mono:
            extra += ' font-family="ui-monospace, SFMono-Regular, Menlo, monospace"'
        if bold:
            extra += ' font-weight="600"'
        self._out.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'fill="{fill or self.palette["ink"]}"{extra}>{html.escape(s)}</text>'
        )

    def result(self) -> str:
        return "\n".join(self._out + ["</svg>"])


class _PilCanvas(_Canvas):
    """Same API on Pillow. Curves are sampled; Pillow has no bezier or dash."""

    def __init__(self, width: int, height: int, palette: dict, supersample: int = 2):
        super().__init__(width, height, palette)
        from PIL import Image, ImageDraw

        self.s = max(1, int(supersample))
        self._image = Image.new(
            "RGB", (width * self.s, height * self.s), palette["surface"]
        )
        self._draw = ImageDraw.Draw(self._image)
        self._fonts: dict[tuple[str, int], object] = {}

    def _font(self, kind: str, size: int):
        key = (kind, size)
        if key not in self._fonts:
            self._fonts[key] = _load_font(kind, max(1, int(size * self.s)))
        return self._fonts[key]

    def _w(self, width: float) -> int:
        return max(1, int(round(width * self.s)))

    def line(self, x1, y1, x2, y2, color, width=1.0, dash=None, opacity=1.0):
        s = self.s
        a, b = (x1 * s, y1 * s), (x2 * s, y2 * s)
        if dash:
            _dashed_line(self._draw, a, b, color, self._w(width),
                         dash=4 * s, gap=3 * s)
        else:
            self._draw.line([a, b], fill=color, width=self._w(width))

    def curve(self, x1, y1, cx, cy, x2, y2, color, width=1.0, dash=None, opacity=1.0):
        s, steps = self.s, 28
        pts = []
        for i in range(steps + 1):
            t = i / steps
            u = 1 - t
            pts.append((
                (u * u * x1 + 2 * u * t * cx + t * t * x2) * s,
                (u * u * y1 + 2 * u * t * cy + t * t * y2) * s,
            ))
        if dash:
            for i in range(0, len(pts) - 1, 2):
                self._draw.line([pts[i], pts[i + 1]], fill=color, width=self._w(width))
        else:
            self._draw.line(pts, fill=color, width=self._w(width), joint="curve")

    def marker(self, shape, x, y, r, fill, outline=None, width=1.0):
        s = self.s
        pts = _shape_points(shape, x * s, y * s, r * s)
        if pts:
            self._draw.polygon(pts, fill=fill, outline=outline, width=self._w(width))
        else:
            self._draw.ellipse(
                [(x - r) * s, (y - r) * s, (x + r) * s, (y + r) * s],
                fill=fill, outline=outline, width=self._w(width),
            )

    def ring(self, x, y, r, color, width=1.6):
        s = self.s
        self._draw.ellipse(
            [(x - r) * s, (y - r) * s, (x + r) * s, (y + r) * s],
            outline=color, width=self._w(width),
        )

    def text(self, x, y, s_, size=10, fill=None, anchor="start", mono=False, bold=False):
        font = self._font("mono" if mono else ("bold" if bold else "sans"), size)
        px_ = x * self.s
        if anchor != "start":
            span = self._draw.textlength(s_, font=font)
            px_ -= span if anchor == "end" else span / 2
        # SVG y is a baseline; Pillow y is the top of the box.
        self._draw.text(
            (px_, y * self.s - size * self.s * 0.8), s_, font=font,
            fill=fill or self.palette["ink"],
        )

    def save(self, path: Path) -> None:
        from PIL import Image

        image = self._image
        if self.s > 1:
            image = image.resize((self.width, self.height), Image.LANCZOS)
        image.save(path)


CONV_ROW_H = 20
CONV_SRC_X = 344      # right edge of the source-program column
CONV_DST_X = 560      # attractor marker column
CONV_TOP = 96


def draw_convergence(
    canvas,
    nodes: dict[str, Node],
    archive_dir: Path,
    attractors: list[tuple[Node, list[Node]]],
    cycles: list[list[str]],
) -> None:
    """Fan diagram: every alternate parent, plus the cycles it closes."""
    palette = canvas.palette
    accent = "#eb6834" if palette["surface"] == "#fcfcfb" else "#d95926"

    canvas.text(34, 34, "Lineage convergence", size=19, bold=True)
    canvas.text(
        34, 54,
        f"{archive_dir.parent.name} — {len(attractors)} programs re-derived from "
        f"2+ distinct parents; {len(cycles)} cycles. program_id is md5(source), "
        f"so identical output from a different parent is the SAME node.",
        size=12, fill=palette["ink2"],
    )
    canvas.text(
        34, CONV_TOP - 22, "PARENT (the program that was mutated)", size=10,
        fill=palette["ink2"],
    )
    canvas.text(
        CONV_DST_X + 22, CONV_TOP - 22, "CONVERGED ON", size=10, fill=palette["ink2"],
    )

    in_cycle = {pid for cycle in cycles for pid in cycle}
    y = CONV_TOP
    for target, parents in attractors:
        block_top = y
        for parent in parents:
            gens = sorted(g for g, p in target.lineages if p == parent.program_id)
            canvas.text(
                34, y + 3.5, node_label(parent, "type", 30), size=9.5,
                fill=palette["ink"],
            )
            canvas.text(
                CONV_SRC_X - 70, y + 3.5, parent.program_id[:8], size=9,
                anchor="end", fill=palette["ink2"], mono=True,
            )
            canvas.text(
                CONV_SRC_X, y + 3.5,
                "gen " + ",".join(str(g) for g in gens), size=9,
                anchor="end", fill=palette["ink2"],
            )
            canvas.marker(
                GROUP_SHAPE.get(parent.concept_group, "circle"),
                CONV_SRC_X + 12, y, 4.0, _blue(parent.p_hat, _theme_of(palette)),
                outline=palette["grid"],
            )
            y += CONV_ROW_H
        block_mid = (block_top + y - CONV_ROW_H) / 2

        # One curve per parent, bowing right into the shared target.
        for i, parent in enumerate(parents):
            py = block_top + i * CONV_ROW_H
            cycled = (
                parent.program_id in in_cycle and target.program_id in in_cycle
            )
            canvas.curve(
                CONV_SRC_X + 20, py,
                (CONV_SRC_X + CONV_DST_X) / 2, py,
                CONV_DST_X - 16, block_mid,
                accent if cycled else palette["edge"],
                width=1.4 if cycled else 1.0,
                opacity=0.9 if cycled else 0.55,
            )

        r = 5.0 + 5.0 * (len(parents) / max(len(a[1]) for a in attractors)) ** 0.5
        shape = GROUP_SHAPE.get(target.concept_group, "circle")
        canvas.marker(
            shape, CONV_DST_X, block_mid, r,
            _blue(target.p_hat, _theme_of(palette)), outline=palette["grid"],
        )
        if target.alive:
            canvas.ring(CONV_DST_X, block_mid, _outer_radius(shape, r) + 3.2,
                        palette["ring"])
        canvas.text(
            CONV_DST_X + 22, block_mid + 3.5, node_label(target, "type", 40),
            size=10.5, fill=palette["ink"],
        )
        canvas.text(
            CONV_DST_X + 22, block_mid + 16, target.program_id[:8], size=9,
            fill=palette["ink2"], mono=True,
        )
        canvas.text(
            CONV_DST_X + 80, block_mid + 16,
            f"← {len(parents)} parents · gen "
            + "–".join(
                str(v) for v in
                sorted({g for g, p in target.lineages if p})[:: max(1, len(
                    {g for g, p in target.lineages if p}) - 1)]
            ),
            size=9, fill=palette["ink2"],
        )
        y += CONV_ROW_H  # gap between attractor blocks

    # --- cycles band ------------------------------------------------------
    y += 14
    canvas.line(34, y, canvas.width - 34, y, palette["grid"], width=1)
    y += 26
    canvas.text(34, y, f"CYCLES ({len(cycles)})", size=11, fill=palette["ink2"])
    canvas.text(
        170, y,
        "a mutation emitted source byte-identical to one of its own ancestors",
        size=10, fill=palette["ink2"],
    )
    y += 22
    for cycle in cycles:
        x = 46
        for i, pid in enumerate(cycle):
            node = nodes.get(pid)
            if node is None:
                continue
            canvas.marker(
                GROUP_SHAPE.get(node.concept_group, "circle"), x, y, 4.0,
                _blue(node.p_hat, _theme_of(palette)), outline=palette["grid"],
            )
            label = node_label(node, "type", 26)
            canvas.text(x + 10, y + 3.5, label, size=9.5, fill=palette["ink"])
            step = 16 + len(label) * 5.4 + 22
            if i < len(cycle) - 1:
                canvas.text(x + step - 14, y + 3.5, "→", size=11, fill=accent,
                            anchor="middle")
            x += step
        canvas.text(x - 6, y + 3.5, " ⟲", size=10, fill=accent)
        y += CONV_ROW_H

    canvas.text(
        34, canvas.height - 10,
        "Orange = an edge that participates in a cycle. Marker shape = "
        "CONCEPT_GROUP, fill = p̂, ring = alive in the final archive.",
        size=10, fill=palette["ink2"],
    )


def _theme_of(palette: dict) -> str:
    return "light" if palette["surface"] == "#fcfcfb" else "dark"


def convergence_size(
    attractors: list[tuple[Node, list[Node]]], cycles: list[list[str]]
) -> tuple[int, int]:
    rows = sum(len(p) + 1 for _, p in attractors)
    height = CONV_TOP + rows * CONV_ROW_H + 60 + (len(cycles) + 1) * CONV_ROW_H + 40
    longest = max((len(c) for c in cycles), default=1)
    width = max(1240, 60 + longest * 220)
    return width, int(height)


def render_convergence(
    nodes: dict[str, Node],
    archive_dir: Path,
    out_path: Path,
    *,
    theme: str = "light",
) -> tuple[int, int]:
    """Write the convergence view; format follows the file extension."""
    palette = THEMES[theme]
    attractors = find_attractors(nodes)
    cycles = find_cycles(nodes)
    if not attractors and not cycles:
        raise SystemExit(
            "[viz] this archive's lineage is a clean tree -- no convergence to draw."
        )
    width, height = convergence_size(attractors, cycles)

    if out_path.suffix.lower() == ".png":
        canvas = _PilCanvas(width, height, palette)
        draw_convergence(canvas, nodes, archive_dir, attractors, cycles)
        canvas.save(out_path)
    else:
        canvas = _SvgCanvas(width, height, palette)
        draw_convergence(canvas, nodes, archive_dir, attractors, cycles)
        out_path.write_text(canvas.result(), encoding="utf-8")
    return len(attractors), len(cycles)


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an R-Q-Evolve archive directory as a lineage tree."
    )
    parser.add_argument(
        "archive_dir", nargs="?", default=str(DEFAULT_DIR),
        help=f"archive directory (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--svg", metavar="PATH",
        help="write a drawn node-link lineage tree as SVG (no dependencies)",
    )
    parser.add_argument(
        "--png", metavar="PATH",
        help="write the same lineage tree as PNG (needs Pillow)",
    )
    parser.add_argument(
        "--theme", choices=("light", "dark"), default="light",
        help="image colour scheme (default: light)",
    )
    parser.add_argument(
        "--label", choices=("type", "group", "id"), default="type",
        help="image node label: CONCEPT_TYPE (default), CONCEPT_GROUP, or program id",
    )
    parser.add_argument(
        "--show-converged", action="store_true",
        help="overlay the alternate-parent edges the tree drops (re-derivations)",
    )
    parser.add_argument(
        "--convergence", metavar="PATH",
        help="write the convergence view (attractors + cycles); .svg or .png",
    )
    parser.add_argument(
        "--label-chars", type=int, default=34, metavar="N",
        help="truncate labels past N characters; column width follows (default: 34)",
    )
    parser.add_argument("--html", metavar="PATH", help="also write a standalone HTML tree")
    parser.add_argument(
        "--no-source", action="store_true",
        help="omit generator source code from the HTML (much smaller file)",
    )
    parser.add_argument(
        "--sort", choices=("iter", "rq", "id", "group"), default="iter",
        help="sibling ordering (default: iter = discovery order)",
    )
    parser.add_argument("--no-spark", action="store_true", help="hide the p̂ sparkline")
    parser.add_argument(
        "--quiet", action="store_true",
        help="with --svg, skip the terminal tree and print only the output path",
    )
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    parser.add_argument(
        "--alive-only", action="store_true",
        help="prune lineages with no descendant in the final archive",
    )
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir).expanduser().resolve()
    if not archive_dir.is_dir():
        raise SystemExit(f"[viz] not a directory: {archive_dir}")

    nodes, meta, iterations = load_nodes(archive_dir)
    roots = build_forest(nodes, args.sort)

    if args.alive_only:
        _prune_dead(roots)
        roots = [r for r in roots if _subtree_has_alive(r)]

    color = not args.no_color and sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    if not ((args.svg or args.png or args.convergence) and args.quiet):
        print(render_tree(roots, color=color, show_spark=not args.no_spark))
        print(
            render_summary(
                nodes, roots, meta, iterations, archive_dir, color=color
            )
        )

    if args.svg:
        out = Path(args.svg).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            render_svg(
                nodes, roots, meta, iterations, archive_dir, theme=args.theme,
                label_mode=args.label, label_chars=max(6, args.label_chars),
                show_converged=args.show_converged,
            ),
            encoding="utf-8",
        )
        print(f"\n[viz] wrote {out} ({out.stat().st_size / 1024:.0f} KB)")

    if args.png:
        out = Path(args.png).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        render_png(
            nodes, roots, meta, iterations, archive_dir, out, theme=args.theme,
            label_mode=args.label, label_chars=max(6, args.label_chars),
            show_converged=args.show_converged,
        )
        print(f"[viz] wrote {out} ({out.stat().st_size / 1024:.0f} KB)")

    if args.html:
        out = Path(args.html).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            render_html(
                nodes, roots, meta, iterations, archive_dir,
                with_source=not args.no_source,
            ),
            encoding="utf-8",
        )
        size_kb = out.stat().st_size / 1024
        print(f"\n[viz] wrote {out} ({size_kb:.0f} KB)")

    if args.convergence:
        out = Path(args.convergence).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        n_attr, n_cyc = render_convergence(
            nodes, archive_dir, out, theme=args.theme
        )
        print(
            f"[viz] wrote {out} ({out.stat().st_size / 1024:.0f} KB) — "
            f"{n_attr} attractors, {n_cyc} cycles"
        )


def _prune_dead(roots: list[Node]) -> None:
    for node in roots:
        node.children = [c for c in node.children if _subtree_has_alive(c)]
        _prune_dead(node.children)


if __name__ == "__main__":
    main()
