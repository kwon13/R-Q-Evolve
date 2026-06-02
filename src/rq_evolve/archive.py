import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from .code_utils import lint_problem_instance
from .concepts import DEFAULT_CONCEPT_TYPES, CONCEPT_GROUPS
from .program import ProblemProgram


@dataclass
class Niche:
    h_bin: int
    div_bin: int
    champion: ProblemProgram | None = None
    champion_rq: float = -1.0
    selection_count: int = 0
    update_count: int = 0
    history: list[dict] = field(default_factory=list)


class MAPElitesArchive:
    """Two-dimensional MAP-Elites grid.

    H-axis: uncertainty bins.
    D-axis: diversity bins, usually controlled ``CONCEPT_GROUP`` labels.
    """

    def __init__(
        self,
        n_h_bins: int = 6,
        n_div_bins: int = 6,
        h_range: tuple[float, float] = (0.0, 6.0),
        diversity_axis: str = "concept_group",
        epsilon: float = 0.3,
        ucb_c: float = 1.0,
        selection_strategy: str = "ucb",
    ) -> None:
        if diversity_axis not in {"concept_group", "concept_type", "hash"}:
            raise ValueError(f"unknown diversity_axis: {diversity_axis}")
        if selection_strategy not in {"ucb", "random"}:
            raise ValueError(f"unknown selection_strategy: {selection_strategy}")
        self.diversity_axis = diversity_axis
        if diversity_axis == "concept_group":
            n_div_bins = len(CONCEPT_GROUPS)
        elif diversity_axis == "concept_type":
            n_div_bins = len(DEFAULT_CONCEPT_TYPES)

        self.n_h_bins = int(n_h_bins)
        self.n_div_bins = int(n_div_bins)
        self.h_range = (float(h_range[0]), float(h_range[1]))
        self.epsilon = float(epsilon)
        self.ucb_c = float(ucb_c)
        self.selection_strategy = selection_strategy
        self.total_insertions = 0
        self.total_replacements = 0
        self.total_selections = 0
        self.grid: dict[tuple[int, int], Niche] = {
            (h, d): Niche(h_bin=h, div_bin=d)
            for h in range(self.n_h_bins)
            for d in range(self.n_div_bins)
        }

    def h_to_bin(self, h_value: float) -> int:
        low, high = self.h_range
        clipped = min(max(float(h_value), low), high)
        width = (high - low) / self.n_h_bins
        return min(int((clipped - low) / width), self.n_h_bins - 1)

    def program_to_div_bin(self, program: ProblemProgram, problem_text: str = "") -> int:
        if self.diversity_axis == "concept_group":
            group = program.get_concept_group()
            if group not in CONCEPT_GROUPS:
                raise ValueError(f"invalid concept group: {group!r}")
            return CONCEPT_GROUPS.index(group)
        if self.diversity_axis == "concept_type":
            concept_type = program.get_concept_type()
            if concept_type not in DEFAULT_CONCEPT_TYPES:
                return _stable_hash(concept_type or problem_text) % self.n_div_bins
            return DEFAULT_CONCEPT_TYPES.index(concept_type)
        return _stable_hash(problem_text or program.program_id) % self.n_div_bins

    def try_insert(
        self,
        program: ProblemProgram,
        h_value: float,
        problem_text: str,
        rq_score: float,
    ) -> bool:
        h_bin = self.h_to_bin(h_value)
        d_bin = self.program_to_div_bin(program, problem_text)

        program.niche_h = h_bin
        program.niche_div = d_bin
        program.h_score = float(h_value)
        program.rq_score = float(rq_score)
        program.fitness = float(rq_score)

        # --- Safety gates (ported from evo-sample), applied at EVERY archive
        # entry, including champion re-evaluation. ---
        # 1. Strict seed-variation: block near-constant / thin-rewording
        #    generators before they pollute the archive and mutation chain.
        if not self._passes_seed_variation(program):
            program.metadata["archive_status"] = "seed_variation_rejected"
            return False
        # 2. Behavior duplicate: digit-identical clone already in the grid.
        dup = self._find_duplicate_behavior(program)
        if dup is not None:
            program.metadata["archive_status"] = "duplicate_behavior_rejected"
            program.metadata["duplicate_of"] = dup.program_id
            return False
        # 3. Template duplicate: same problem skeleton, different numbers.
        tdup = self._find_duplicate_template(program)
        if tdup is not None:
            program.metadata["archive_status"] = "duplicate_template_rejected"
            program.metadata["duplicate_of"] = tdup.program_id
            return False

        niche = self.grid[(h_bin, d_bin)]
        if niche.champion is None or rq_score > niche.champion_rq:
            event = "inserted" if niche.champion is None else "replaced"
            if niche.champion is not None:
                self.total_replacements += 1
            niche.champion = program
            niche.champion_rq = float(rq_score)
            program.metadata["archive_status"] = "champion"
            niche.update_count += 1
            niche.history.append(
                {
                    "event": event,
                    "program_id": program.program_id,
                    "rq_score": float(rq_score),
                }
            )
            self.total_insertions += 1
            # Archive-global uniqueness: one generator occupies one cell only.
            self._purge_program_from_other_cells(
                program.program_id, keep_cell=(h_bin, d_bin)
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Safety gates (ported from evo-sample map_elites / mutation)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_sig_text(text: str) -> str:
        """Whitespace-collapsed, lowercased text for exact-clone signatures."""
        return " ".join(str(text or "").split()).lower()

    @staticmethod
    def _template_normalize_text(text: str) -> str:
        """Numeric-free skeleton: every int / decimal literal becomes 'N'."""
        skeleton = re.sub(r"-?\d+(?:\.\d+)?", "N", str(text or ""))
        return " ".join(skeleton.split()).lower()

    def program_behavior_signature(
        self, program: ProblemProgram, n_seeds: int = 5
    ) -> str | None:
        """Hash of the (problem, answer) sequence across seeds, id-independent.

        Catches near-clone generators that emit the identical problem/answer
        sequence for the verification seeds but land in different cells under
        noisy H/D estimates. Cached on program metadata.
        """
        cache_key = f"_behavior_sig_{n_seeds}"
        cached = (program.metadata or {}).get(cache_key)
        if cached:
            return str(cached)
        pairs = []
        for seed in range(n_seeds):
            inst = program.execute(seed=seed)
            if inst is None:
                return None
            pairs.append(
                (
                    self._normalize_sig_text(inst.problem),
                    self._normalize_sig_text(inst.answer),
                )
            )
        signature = hashlib.sha256(
            json.dumps(pairs, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        program.metadata[cache_key] = signature
        return signature

    def program_template_signature(
        self, program: ProblemProgram, n_seeds: int = 5
    ) -> str | None:
        """Hash of the numeric-free problem skeleton across seeds.

        Unlike the behavior signature (exact pairs -> digit-identical clones),
        this replaces every number with 'N', catching generators that emit the
        same sentence template with different sampled numbers.
        """
        cache_key = f"_template_sig_{n_seeds}"
        cached = (program.metadata or {}).get(cache_key)
        if cached:
            return str(cached)
        templates = []
        for seed in range(n_seeds):
            inst = program.execute(seed=seed)
            if inst is None:
                return None
            templates.append(self._template_normalize_text(inst.problem))
        signature = hashlib.sha256(
            json.dumps(sorted(set(templates)), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        program.metadata[cache_key] = signature
        return signature

    def _find_duplicate(self, program, signature_fn) -> ProblemProgram | None:
        signature = signature_fn(program)
        if not signature:
            return None
        for niche in self.grid.values():
            existing = niche.champion
            if existing is None or existing.program_id == program.program_id:
                continue
            if signature_fn(existing) == signature:
                return existing
        return None

    def _find_duplicate_behavior(self, program: ProblemProgram) -> ProblemProgram | None:
        return self._find_duplicate(program, self.program_behavior_signature)

    def _find_duplicate_template(self, program: ProblemProgram) -> ProblemProgram | None:
        return self._find_duplicate(program, self.program_template_signature)

    def _purge_program_from_other_cells(
        self, program_id: str, keep_cell: tuple[int, int]
    ) -> int:
        """Archive-global uniqueness: a generator is champion in one cell only.

        Clears ``program_id`` from every cell except ``keep_cell`` (e.g. an
        older champion seat left behind after H/D rebinning). Returns the count
        purged (logging only).
        """
        purged = 0
        for key, niche in self.grid.items():
            if key == keep_cell:
                continue
            if niche.champion is not None and niche.champion.program_id == program_id:
                niche.champion = None
                niche.champion_rq = -1.0
                purged += 1
        return purged

    def _passes_seed_variation(self, program: ProblemProgram, n_seeds: int = 5) -> bool:
        """Strict seed-variation gate (evo-sample champion_passes_validity).

        Reject unless ALL n_seeds produce a valid instance AND the problems are
        all distinct AND the answers are all distinct. Near-constant or
        thin-rewording generators are blocked. ``lint_problem_instance`` stands
        in for evo-sample's domain-specific ``looks_broken``. Cached by
        rq_score, so champion re-evaluation (which changes rq_score) re-runs it.
        """
        rq_now = float(getattr(program, "rq_score", 0.0) or 0.0)
        cache = (program.metadata or {}).get("validity_check")
        if cache is not None and cache.get("rq_score_at_check") == rq_now:
            return bool(cache.get("passed"))

        problems: list[str] = []
        answers: list[str] = []
        n_broken = 0
        for seed in range(n_seeds):
            inst = program.execute(seed=seed)
            if inst is None:
                n_broken += 1
                continue
            answer = (inst.answer or "").strip()
            if not answer or lint_problem_instance(inst):
                n_broken += 1
                continue
            problems.append((inst.problem or "").strip())
            answers.append(answer)

        n_total = n_seeds
        if len(answers) == n_total:
            seed_invariant = (
                len(set(problems)) < n_total or len(set(answers)) < n_total
            )
        else:
            seed_invariant = len(answers) >= 2 and len(set(answers)) == 1
        passed = n_broken == 0 and not seed_invariant and len(answers) == n_total

        program.metadata["validity_check"] = {
            "passed": passed,
            "n_distinct_problems": len(set(problems)),
            "n_distinct_answers": len(set(answers)),
            "n_valid": len(answers),
            "n_total": n_total,
            "rq_score_at_check": rq_now,
        }
        return passed

    def champions(self) -> list[ProblemProgram]:
        return [n.champion for n in self.grid.values() if n.champion is not None]

    @staticmethod
    def _is_learnable(program: ProblemProgram | None) -> bool:
        """RQ>0: a learnable parent. Too-easy generators (p_hat=1.0 -> RQ=0)
        stay in the archive but are not selected for mutation."""
        return program is not None and (program.rq_score or 0.0) > 0.0

    def sample_parent(self) -> ProblemProgram | None:
        occupied = [(key, n) for key, n in self.grid.items() if n.champion is not None]
        if not occupied:
            return None
        # Prefer learnable (RQ>0) champions as mutation parents; fall back to all
        # occupied niches when none are learnable (e.g. early bootstrap).
        pool = [(key, n) for key, n in occupied if self._is_learnable(n.champion)] or occupied

        if self.selection_strategy == "random":
            key, niche = random.choice(pool)
        elif random.random() < self.epsilon:
            key, niche = random.choice(pool)
        else:
            key, niche = self._sample_ucb(pool)
        niche.selection_count += 1
        self.total_selections += 1
        assert niche.champion is not None
        return niche.champion

    def sample_two_parents(self) -> tuple[ProblemProgram | None, ProblemProgram | None]:
        champions = self.champions()
        if len(champions) < 2:
            return None, None
        # Both parents from learnable (RQ>0) champions when at least two exist;
        # otherwise fall back to the full set so crossover is not starved.
        learnable = [c for c in champions if self._is_learnable(c)]
        pool = learnable if len(learnable) >= 2 else champions
        first = self.sample_parent()
        if first is None:
            return None, None
        remaining = [p for p in pool if p.program_id != first.program_id]
        if not remaining:
            return None, None
        return first, random.choice(remaining)

    def _sample_ucb(self, occupied: list[tuple[tuple[int, int], Niche]]):
        rqs = [n.champion_rq for _, n in occupied]
        sorted_rqs = sorted(set(rqs))
        denom = max(len(sorted_rqs) - 1, 1)
        total = self.total_selections + 1

        best_score = -math.inf
        best = occupied[0]
        for item in occupied:
            _, niche = item
            rank = sorted_rqs.index(niche.champion_rq) / denom
            if niche.selection_count <= 0:
                exploration = math.inf
            else:
                exploration = self.ucb_c * math.sqrt(
                    math.log(total + 1) / niche.selection_count
                )
            score = rank + exploration
            if score > best_score:
                best_score = score
                best = item
        return best

    def stats(self) -> dict[str, float | int]:
        champions = self.champions()
        rqs = [p.rq_score for p in champions]
        total = self.n_h_bins * self.n_div_bins
        return {
            "num_champions": len(champions),
            "total_niches": total,
            "coverage": len(champions) / total if total else 0.0,
            "mean_rq": sum(rqs) / len(rqs) if rqs else 0.0,
            "max_rq": max(rqs) if rqs else 0.0,
            "total_insertions": self.total_insertions,
            "total_replacements": self.total_replacements,
            "total_selections": self.total_selections,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "n_h_bins": self.n_h_bins,
                "n_div_bins": self.n_div_bins,
                "h_range": self.h_range,
                "diversity_axis": self.diversity_axis,
                "epsilon": self.epsilon,
                "ucb_c": self.ucb_c,
                "selection_strategy": self.selection_strategy,
                "stats": self.stats(),
            },
            "champions": [p.to_dict() for p in self.champions()],
        }
        (path / "archive.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> int:
        """Restore champions written by :meth:`save`.

        Returns the number of champions placed. Every niche is cleared first so
        the restored grid reflects exactly the saved state. Champions are placed
        directly at their saved ``(niche_h, niche_div)`` when those coordinates
        fit the current grid (no try_insert, so validity/RQ gates are not
        re-applied — the saved state is reproduced as-is). If the grid shape
        changed since the snapshot, those champions are re-binned via
        ``try_insert`` from their stored ``h_score``.
        """
        path = Path(path)
        archive_file = path / "archive.json" if path.is_dir() else path
        if not archive_file.exists():
            raise FileNotFoundError(f"no archive snapshot at {archive_file}")
        payload = json.loads(archive_file.read_text(encoding="utf-8"))

        meta = payload.get("meta", {})
        if (
            meta.get("n_h_bins") not in (None, self.n_h_bins)
            or meta.get("n_div_bins") not in (None, self.n_div_bins)
        ):
            print(
                f"[archive.load] grid shape changed "
                f"({meta.get('n_h_bins')}x{meta.get('n_div_bins')} -> "
                f"{self.n_h_bins}x{self.n_div_bins}); re-binning champions"
            )

        for niche in self.grid.values():
            niche.champion = None
            niche.champion_rq = -1.0

        placed = 0
        for champ_dict in payload.get("champions", []):
            program = ProblemProgram.from_dict(champ_dict)
            h_bin, d_bin = program.niche_h, program.niche_div
            coords_ok = (
                0 <= h_bin < self.n_h_bins
                and 0 <= d_bin < self.n_div_bins
                and (h_bin, d_bin) in self.grid
                and meta.get("n_h_bins") == self.n_h_bins
                and meta.get("n_div_bins") == self.n_div_bins
            )
            if coords_ok:
                niche = self.grid[(h_bin, d_bin)]
                niche.champion = program
                niche.champion_rq = float(program.rq_score)
                niche.update_count += 1
                placed += 1
            elif self.try_insert(
                program=program,
                h_value=program.h_score,
                problem_text="",
                rq_score=program.rq_score,
            ):
                placed += 1
        return placed


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest(), 16)
