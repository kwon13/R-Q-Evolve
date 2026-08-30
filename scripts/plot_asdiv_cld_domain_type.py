#!/usr/bin/env python3
"""Plot Domain x Problem Type maps with ASDiv-style lexical diversity.

The cell value is Corpus Lexicon Diversity (CLD), computed by treating each
Domain x Problem Type cell as its own corpus.  For every problem i, lexical
diversity is

    LD_i = 1 - max_{j != i} (BLEU(i, j) + BLEU(j, i)) / 2

and cell CLD is mean(LD_i).  BLEU uses uniform 1--4 gram weights and no
smoothing.  The preprocessing follows Miao et al. (ACL 2020): tokenize/POS,
lemmatize, remove stop words, and normalize named entities and quantities.
NLTK is used as a documented local substitute for the paper's unavailable
CoreNLP preprocessing code, so outputs are labelled "ASDiv-style" rather than
claimed as bit-identical reproductions of the paper's reported values.

For R-Q, one row is one training problem exposure
``(iteration, program_id, instance_seed)``.  Its verified family template is
used because ASDiv normalization intentionally treats instances that differ
only in parameter values as identical lexical patterns.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rq_evolve.concepts import DOMAINS, PROBLEM_TYPES  # noqa: E402


PAPER_URL = "https://aclanthology.org/2020.acl-main.92/"
METRIC_NAME = "asdiv-style-cell-cld-v1"
DISPLAY_DOMAINS = {
    "algebra": "Algebra",
    "geometry": "Geometry",
    "number_theory": "Number Theory",
    "discrete_mathematics": "Discrete Mathematics",
    "applied_mathematics": "Applied Mathematics",
    "calculus": "Calculus",
    "precalculus": "Precalculus",
}


@dataclass(frozen=True)
class ProblemRecord:
    item_id: str
    domain: str
    problem_type: str
    text: str


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_PLACEHOLDER_RE = re.compile(r"\[\[[^\]]+\]\]")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])[-+]?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)"
    r"(?:\.\d+)?(?:/\d+)?(?![A-Za-z])"
)
_LATEX_COMMAND_RE = re.compile(r"\\([A-Za-z]+)\*?")
_HAS_WORD_RE = re.compile(r"[A-Za-z0-9]")

_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion",
}


class AsdivNormalizer:
    """Deterministic, batched approximation of ASDiv text preprocessing."""

    def __init__(self, nltk_data: Path) -> None:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        self.nltk = nltk
        self.nltk_data = nltk_data.expanduser().resolve()
        nltk.data.path.insert(0, str(self.nltk_data))
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.cache: dict[str, tuple[str, ...]] = {}

    @staticmethod
    def _wordnet_pos(tag: str) -> str:
        if tag.startswith("J"):
            return "a"
        if tag.startswith("V"):
            return "v"
        if tag.startswith("R"):
            return "r"
        return "n"

    @staticmethod
    def _prepare_text(text: str) -> str:
        value = unicodedata.normalize("NFKC", html.unescape(str(text)))
        value = _PLACEHOLDER_RE.sub(" QUANTITYTOKEN ", value)
        value = _NUMBER_RE.sub(" QUANTITYTOKEN ", value)
        # Preserve semantic LaTeX command names (sqrt, frac, log, ...), while
        # removing the markup backslash that would otherwise become a token.
        value = _LATEX_COMMAND_RE.sub(r" \1 ", value)
        return value

    def normalize_many(self, texts: Sequence[str]) -> dict[str, tuple[str, ...]]:
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if not missing:
            return {text: self.cache[text] for text in texts}

        prepared = [self._prepare_text(text) for text in missing]
        tokenized = [self.nltk.word_tokenize(text) for text in prepared]
        tagged = self.nltk.pos_tag_sents(tokenized)
        chunked = self.nltk.ne_chunk_sents(tagged, binary=False)

        for raw_text, tree in zip(missing, chunked):
            flattened: list[tuple[str, str]] = []
            for node in tree:
                if isinstance(node, self.nltk.Tree):
                    flattened.append(("ENTITYTOKEN", "NN"))
                else:
                    token, tag = node
                    flattened.append((str(token), str(tag)))

            normalized: list[str] = []
            for token, tag in flattened:
                lower = token.lower()
                if lower == "quantitytoken" or lower in _NUMBER_WORDS:
                    normalized.append("quantity")
                    continue
                if lower == "entitytoken":
                    normalized.append("entity")
                    continue
                if lower in self.stop_words or not _HAS_WORD_RE.search(lower):
                    continue
                lemma = self.lemmatizer.lemmatize(lower, self._wordnet_pos(tag))
                if lemma and lemma not in self.stop_words:
                    normalized.append(lemma)

            # BLEU-4 is undefined for empty input.  Retain a stable marker so
            # fail-closed normalization remains visible rather than dropping a
            # problem silently.
            self.cache[raw_text] = tuple(normalized or ("emptytext",))
        return {text: self.cache[text] for text in texts}

    def contract(self) -> dict[str, Any]:
        return {
            "metric": METRIC_NAME,
            "paper": PAPER_URL,
            "paper_preprocessing": (
                "CoreNLP tokenize/POS; NLTK lemmatize; stop-word removal; "
                "named-entity and quantity normalization"
            ),
            "local_preprocessing": (
                "NLTK tokenize/POS/NER and WordNet lemmatize; NLTK English "
                "stop-word removal; regex numeric/placeholder plus NLTK NER normalization"
            ),
            "reproduction_scope": "ASDiv-style; not bit-identical CoreNLP reproduction",
            "bleu": {
                "maximum_ngram": 4,
                "weights": [0.25, 0.25, 0.25, 0.25],
                "smoothing": "none",
                "direction": "mean of both directions",
                "nearest_neighbor_scope": "within Domain x Problem Type cell",
            },
            "nltk_version": self.nltk.__version__,
            "nltk_data": str(self.nltk_data),
        }


def ngram_counter(tokens: tuple[str, ...], order: int) -> Counter[tuple[str, ...]]:
    return Counter(zip(*(tokens[offset:] for offset in range(order))))


def directional_bleu(
    hypothesis_index: int,
    reference_index: int,
    sequences: list[tuple[str, ...]],
    ngrams: list[list[Counter[tuple[str, ...]]]],
) -> float:
    hypothesis = sequences[hypothesis_index]
    reference = sequences[reference_index]
    precisions: list[float] = []
    for order in range(4):
        hypothesis_counts = ngrams[hypothesis_index][order]
        denominator = sum(hypothesis_counts.values())
        if denominator == 0:
            return 0.0
        reference_counts = ngrams[reference_index][order]
        numerator = sum(
            min(count, reference_counts.get(gram, 0))
            for gram, count in hypothesis_counts.items()
        )
        if numerator == 0:
            return 0.0
        precisions.append(numerator / denominator)

    hypothesis_length = len(hypothesis)
    reference_length = len(reference)
    brevity_penalty = (
        1.0
        if hypothesis_length > reference_length
        else math.exp(1.0 - reference_length / hypothesis_length)
    )
    return brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / 4)


def symmetric_bleu(
    left: int,
    right: int,
    sequences: list[tuple[str, ...]],
    ngrams: list[list[Counter[tuple[str, ...]]]],
) -> float:
    return (
        directional_bleu(left, right, sequences, ngrams)
        + directional_bleu(right, left, sequences, ngrams)
    ) / 2


def cell_cld(token_rows: Sequence[tuple[str, ...]]) -> dict[str, Any]:
    total = len(token_rows)
    frequencies: Counter[tuple[str, ...]] = Counter(token_rows)
    if total < 2:
        return {
            "count": total,
            "unique_normalized_texts": len(frequencies),
            "exact_duplicate_rows": 0,
            "exact_duplicate_fraction": 0.0,
            "cld": None,
        }

    sequences = list(frequencies)
    ngrams = [
        [ngram_counter(sequence, order) for order in range(1, 5)]
        for sequence in sequences
    ]
    max_similarity = [1.0 if frequencies[sequence] > 1 else 0.0 for sequence in sequences]

    # With unsmoothed BLEU-4, a positive score requires a shared 4-gram.
    # The inverted index therefore removes provably-zero comparisons without
    # approximating the nearest-neighbor result.
    inverted: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, per_order in enumerate(ngrams):
        for gram in per_order[3]:
            inverted[gram].append(index)

    comparisons = 0
    for left, per_order in enumerate(ngrams):
        candidates: set[int] = set()
        for gram in per_order[3]:
            candidates.update(inverted[gram])
        for right in candidates:
            if right <= left:
                continue
            score = symmetric_bleu(left, right, sequences, ngrams)
            comparisons += 1
            if score > max_similarity[left]:
                max_similarity[left] = score
            if score > max_similarity[right]:
                max_similarity[right] = score

    weighted_ld_sum = sum(
        frequencies[sequence] * (1.0 - max_similarity[index])
        for index, sequence in enumerate(sequences)
    )
    duplicate_rows = sum(
        count for count in frequencies.values() if count > 1
    )
    return {
        "count": total,
        "unique_normalized_texts": len(frequencies),
        "exact_duplicate_rows": duplicate_rows,
        "exact_duplicate_fraction": duplicate_rows / total,
        "positive_bleu_pair_comparisons": comparisons,
        "cld": weighted_ld_sum / total,
    }


def score_records(
    records: Sequence[ProblemRecord], normalizer: AsdivNormalizer
) -> list[dict[str, Any]]:
    normalized = normalizer.normalize_many([record.text for record in records])
    grouped: defaultdict[tuple[str, str], list[tuple[str, ...]]] = defaultdict(list)
    for record in records:
        grouped[(record.domain, record.problem_type)].append(normalized[record.text])

    cells: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for problem_type in PROBLEM_TYPES:
            result = cell_cld(grouped[(domain, problem_type)])
            cells.append({"domain": domain, "problem_type": problem_type, **result})
    return cells


def plot_map(
    *,
    cells: list[dict[str, Any]],
    title_prefix: str,
    dataset_label: str,
    output_stem: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {
        (str(row["domain"]), str(row["problem_type"])): row for row in cells
    }
    grid = np.array(
        [
            [
                float(lookup[(domain, problem_type)]["cld"])
                if lookup[(domain, problem_type)]["cld"] is not None
                else np.nan
                for problem_type in PROBLEM_TYPES
            ]
            for domain in DOMAINS
        ],
        dtype=float,
    )
    defined = int(np.isfinite(grid).sum())
    mapped = sum(int(row["count"]) for row in cells)
    fig, ax = plt.subplots(figsize=(12.2, 7.2))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#f2f3f5")
    image = ax.imshow(
        np.ma.masked_invalid(grid),
        cmap=cmap,
        norm=colors.Normalize(vmin=0.0, vmax=1.0),
        aspect="auto",
    )
    for row_index, domain in enumerate(DOMAINS):
        for col_index, problem_type in enumerate(PROBLEM_TYPES):
            value = grid[row_index, col_index]
            if math.isnan(value):
                text = "—"
                color = "#686d76"
            else:
                text = f"{value:.3f}"
                color = "white" if value < 0.58 else "#111318"
            ax.text(
                col_index,
                row_index,
                text,
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
    ax.set_ylabel("Mathematical domain", fontsize=12, labelpad=12)
    ax.set_xticks(np.arange(-0.5, len(PROBLEM_TYPES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(
        f"{title_prefix}: ASDiv-style Lexical Diversity Fitness\n"
        f"{dataset_label} · mapped n={mapped} · CLD defined {defined}/35 cells (n≥2)",
        fontsize=15,
        pad=16,
    )
    fig.text(
        0.5,
        0.018,
        "Cell fitness = mean LD · LD = 1 − max symmetric BLEU-4 within cell · "
        "quantity/name normalized · higher = more lexically diverse",
        ha="center",
        fontsize=9.3,
        color="#454954",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Cell CLD fitness", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    *,
    records: Sequence[ProblemRecord],
    cells: list[dict[str, Any]],
    normalizer: AsdivNormalizer,
    title_prefix: str,
    dataset_label: str,
    output_dir: Path,
    stem: str,
    source_metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}_cells.csv"
    fieldnames = [
        "domain",
        "problem_type",
        "count",
        "unique_normalized_texts",
        "exact_duplicate_rows",
        "exact_duplicate_fraction",
        "positive_bleu_pair_comparisons",
        "cld",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cells)

    defined = [float(row["cld"]) for row in cells if row["cld"] is not None]
    weighted_denominator = sum(
        int(row["count"]) for row in cells if row["cld"] is not None
    )
    weighted_cld = (
        sum(float(row["cld"]) * int(row["count"]) for row in cells if row["cld"] is not None)
        / weighted_denominator
        if weighted_denominator
        else None
    )
    summary = {
        "dataset_label": dataset_label,
        "mapped_problem_rows": len(records),
        "defined_cells_n_ge_2": len(defined),
        "possible_cells": len(DOMAINS) * len(PROBLEM_TYPES),
        "macro_cell_cld": sum(defined) / len(defined) if defined else None,
        "problem_weighted_cell_cld": weighted_cld,
        "metric_contract": normalizer.contract(),
        "source": source_metadata,
    }
    (output_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_map(
        cells=cells,
        title_prefix=title_prefix,
        dataset_label=dataset_label,
        output_stem=output_dir / stem,
    )
    print(
        json.dumps(
            {
                "dataset": dataset_label,
                "mapped": len(records),
                "defined_cells": len(defined),
                "macro_cell_cld": summary["macro_cell_cld"],
                "problem_weighted_cell_cld": weighted_cld,
                "output": str(output_dir / stem),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def extract_family_template(champion: dict[str, Any]) -> str:
    source = str(champion.get("source_code") or "")
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            else:
                continue
            if any(
                isinstance(target, ast.Name) and target.id == "FAMILY_TEMPLATE"
                for target in targets
            ):
                literal = ast.literal_eval(value)
                if isinstance(literal, str) and literal.strip():
                    return literal.strip()
    except (SyntaxError, ValueError):
        pass

    metadata = champion.get("metadata") or {}
    family_plan = metadata.get("family_plan") or {}
    for value in (family_plan.get("CHILD FAMILY"), metadata.get("_template_text_5")):
        if value and str(value).strip():
            return str(value).strip()
    raise ValueError(f"no family template for champion {champion.get('program_id')}")


def load_rq_records(run_dir: Path, end_round: int) -> list[ProblemRecord]:
    archive_dir = run_dir / "rq_archive"
    snapshots: dict[int, dict[str, tuple[str, str, str]]] = {}
    manual_seeds: dict[str, tuple[str, str, str]] = {}
    for round_number in range(1, end_round + 1):
        path = archive_dir / f"archive_iter{round_number}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload["meta"]
        domains = list(meta["domain_labels"])
        problem_types = list(meta["problem_type_labels"])
        state: dict[str, tuple[str, str, str]] = {}
        for champion in payload.get("champions") or []:
            value = (
                domains[int(champion["niche_domain"])],
                problem_types[int(champion["niche_problem_type"])],
                extract_family_template(champion),
            )
            program_id = str(champion["program_id"])
            state[program_id] = value
            if not str(champion.get("parent_id") or ""):
                manual_seeds[program_id] = value
        snapshots[round_number] = state

    candidates: dict[tuple[int, str], tuple[str, str, str]] = {}
    evolution_path = archive_dir / "evolution_log.jsonl"
    for row in read_jsonl(evolution_path):
        iteration = int(row.get("iteration", -1))
        if iteration > end_round:
            break
        for report in row.get("reports") or []:
            if not isinstance(report, dict) or not report.get("child_id"):
                continue
            decision = report.get("archive_decision") or {}
            labels = decision.get("placement_labels")
            family = report.get("child_family")
            if not isinstance(labels, list) or len(labels) != 2 or not family:
                continue
            key = (iteration, str(report["child_id"]))
            value = (str(labels[0]), str(labels[1]), str(family).strip())
            previous = candidates.get(key)
            if previous is not None and previous != value:
                raise ValueError(f"conflicting candidate metadata for {key}")
            candidates[key] = value

    records: list[ProblemRecord] = []
    seen: set[tuple[int, str, int]] = set()
    samples_path = archive_dir / "rollout_samples.jsonl"
    for row in read_jsonl(samples_path):
        raw_iteration = int(row["iteration"])
        display_round = 0 if raw_iteration == -1 else raw_iteration
        if display_round > end_round:
            continue
        program_id = str(row["program_id"])
        instance_seed = int(row["instance_seed"])
        key = (raw_iteration, program_id, instance_seed)
        if key in seen:
            continue
        seen.add(key)
        if raw_iteration >= 1 and (raw_iteration, program_id) in candidates:
            domain, problem_type, family = candidates[(raw_iteration, program_id)]
        elif raw_iteration >= 2 and program_id in snapshots[raw_iteration - 1]:
            domain, problem_type, family = snapshots[raw_iteration - 1][program_id]
        elif program_id in manual_seeds:
            domain, problem_type, family = manual_seeds[program_id]
        else:
            raise ValueError(f"could not resolve R-Q descriptor/template for {key}")
        records.append(
            ProblemRecord(
                item_id=f"rq:r{display_round}:{program_id}:s{instance_seed}",
                domain=domain,
                problem_type=problem_type,
                text=family,
            )
        )
    return records


def load_rzero_records(analysis_dir: Path) -> list[ProblemRecord]:
    records: list[ProblemRecord] = []
    for row in read_jsonl(analysis_dir / "labels.jsonl"):
        domain = row.get("domain")
        problem_type = row.get("problem_type")
        if (
            row.get("status") == "ok"
            and row.get("domain_confidence") == "high"
            and domain in DOMAINS
            and problem_type in PROBLEM_TYPES
        ):
            records.append(
                ProblemRecord(
                    item_id=str(row.get("item_id") or row.get("input_hash")),
                    domain=str(domain),
                    problem_type=str(problem_type),
                    text=str(row["question"]),
                )
            )
    return records


def run_rq(args: argparse.Namespace, normalizer: AsdivNormalizer) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    all_records = load_rq_records(run_dir, max(args.checkpoints))
    for checkpoint in args.checkpoints:
        marker = re.compile(r"^rq:r(\d+):")
        records = [
            record
            for record in all_records
            if int(marker.match(record.item_id).group(1)) <= checkpoint
        ]
        cells = score_records(records, normalizer)
        output_dir = (
            ROOT
            / "analysis"
            / run_dir.name
            / "problem_domain_type"
            / f"rounds_000_{checkpoint:03d}"
        )
        write_outputs(
            records=records,
            cells=cells,
            normalizer=normalizer,
            title_prefix="R-Q Evolve",
            dataset_label=f"cumulative rounds 0–{checkpoint}",
            output_dir=output_dir,
            stem=f"rq_problem_domain_type_cld_rounds_000_{checkpoint:03d}",
            source_metadata={
                "run_dir": str(run_dir),
                "checkpoint": checkpoint,
                "count_unit": "unique (raw_iteration, program_id, instance_seed)",
                "text_unit": "verified family template; exposure multiplicity retained",
                "rollout_samples_sha256": sha256_file(
                    run_dir / "rq_archive" / "rollout_samples.jsonl"
                ),
                "evolution_log_sha256": sha256_file(
                    run_dir / "rq_archive" / "evolution_log.jsonl"
                ),
            },
        )


def run_rzero(args: argparse.Namespace, normalizer: AsdivNormalizer) -> None:
    for analysis_dir in args.analysis_dirs:
        analysis_dir = analysis_dir.expanduser().resolve()
        summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
        records = load_rzero_records(analysis_dir)
        cells = score_records(records, normalizer)
        write_outputs(
            records=records,
            cells=cells,
            normalizer=normalizer,
            title_prefix="R-Zero",
            dataset_label=str(summary["dataset_label"]),
            output_dir=analysis_dir,
            stem="rzero_domain_type_cld_map",
            source_metadata={
                "analysis_dir": str(analysis_dir),
                "input": summary.get("input"),
                "input_sha256": summary.get("input_sha256"),
                "domain_model": summary.get("model"),
                "domain_prompt_hash": summary.get("prompt_hash"),
                "problem_type_ruleset_sha256": summary.get(
                    "problem_type_ruleset_sha256"
                ),
                "labels_sha256": sha256_file(analysis_dir / "labels.jsonl"),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nltk-data",
        type=Path,
        default=ROOT / "analysis" / "asdiv_cld" / "nltk_data",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rq_parser = subparsers.add_parser("rq", help="plot cumulative R-Q checkpoints")
    rq_parser.add_argument("--run-dir", required=True, type=Path)
    rq_parser.add_argument("--checkpoints", required=True, nargs="+", type=int)

    rzero_parser = subparsers.add_parser("rzero", help="plot R-Zero audit directories")
    rzero_parser.add_argument("analysis_dirs", nargs="+", type=Path)

    args = parser.parse_args()
    normalizer = AsdivNormalizer(args.nltk_data)
    if args.command == "rq":
        run_rq(args, normalizer)
    else:
        run_rzero(args, normalizer)


if __name__ == "__main__":
    main()
