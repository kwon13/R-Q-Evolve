import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from .code_utils import lint_problem_instance
from .concepts import GROUPS, SKILLS, axis_index
from .program import ProblemProgram
from .scoring import selection_priority


@dataclass
class Niche:
    group_bin: int
    skill_bin: int
    champion: ProblemProgram | None = None
    champion_rq: float = -1.0
    selection_count: int = 0
    update_count: int = 0
    history: list[dict] = field(default_factory=list)


class MAPElitesArchive:
    """GROUP x SKILL MAP-Elites grid.

    Both axes are behavioural descriptors read off a program's own declared
    labels: GROUP is the mathematical domain, SKILL the reasoning the visible
    problem forces. The grid is therefore a fixed ``len(GROUPS) x len(SKILLS)``
    and needs no bin-count configuration -- a cell is a (domain, reasoning)
    pair, and the two mutation operators each move exactly one coordinate.

    Uncertainty is NOT an axis. H stays in the fitness, ``R_Q = p(1-p)H``, which
    decides who holds a cell and which champion is sampled as a parent. Binning
    on H as well would have meant one grid coordinate the mutation operators
    cannot aim at: a child lands in an H bin only as a side effect of how hard
    it turned out to be.
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        ucb_c: float = 1.0,
        selection_strategy: str = "random",
        select_ignores_uncertainty: bool = False,
        select_ignores_variance: bool = False,
    ) -> None:
        if selection_strategy not in {"ucb", "random"}:
            raise ValueError(f"unknown selection_strategy: {selection_strategy}")
        if select_ignores_uncertainty and select_ignores_variance:
            raise ValueError(
                "select_ignores_uncertainty and select_ignores_variance are "
                "mutually exclusive: enabling both drops every R_Q factor and "
                "leaves a constant (signal-free) selection priority."
            )
        self.n_group_bins = len(GROUPS)
        self.n_skill_bins = len(SKILLS)
        self.epsilon = float(epsilon)
        self.ucb_c = float(ucb_c)
        self.selection_strategy = selection_strategy
        # Ablations: rank mutation parents by s(1-s) (ignore_uncertainty) or by H
        # (ignore_variance) instead of s(1-s)*H. Neither changes what is stored
        # or binned -- champion_rq and the cell labels stay real.
        self.select_ignores_uncertainty = bool(select_ignores_uncertainty)
        self.select_ignores_variance = bool(select_ignores_variance)
        self.total_insertions = 0
        self.total_replacements = 0
        self.total_selections = 0
        self.grid: dict[tuple[int, int], Niche] = {
            (g, s): Niche(group_bin=g, skill_bin=s)
            for g in range(self.n_group_bins)
            for s in range(self.n_skill_bins)
        }

    def program_to_cell(self, program: ProblemProgram) -> tuple[int, int] | None:
        """Grid coordinate from the program's own labels, None if unlabelled.

        None rather than a hashed fallback: a program whose GROUP or SKILL is
        outside the vocabulary has no meaningful niche, and folding those into
        one bin would make a single cell the contest for every mislabelled
        generator.
        """
        group_bin = axis_index("group", program.get_group())
        skill_bin = axis_index("skill", program.get_skill())
        if group_bin is None or skill_bin is None:
            return None
        return (group_bin, skill_bin)

    def cell_labels(self, cell: tuple[int, int]) -> tuple[str, str]:
        return (GROUPS[cell[0]], SKILLS[cell[1]])

    def try_insert(
        self,
        program: ProblemProgram,
        h_value: float,
        problem_text: str,
        rq_score: float,
    ) -> bool:
        cell = self.program_to_cell(program)
        if cell is None:
            program.metadata["archive_status"] = "unlabelled_rejected"
            return False
        group_bin, skill_bin = cell

        program.niche_group = group_bin
        program.niche_skill = skill_bin
        # H is no longer a coordinate, but it is still the uncertainty factor of
        # R_Q, so it is recorded on the program exactly as before.
        program.h_score = float(h_value)
        program.rq_score = float(rq_score)
        program.fitness = float(rq_score)

        # --- Safety gates (ported from evo-sample), applied at EVERY archive
        # entry, including champion re-evaluation. ---
        # 0. The live MAP is the frontier archive: problems at p=0 or p=1 have
        #    R_Q=0 and must not occupy a niche.
        if float(rq_score) <= 0.0:
            program.metadata["archive_status"] = "rq_zero_rejected"
            return False
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

        niche = self.grid[cell]
        # Champion competition ranks by selection priority: real R_Q in
        # production, s(1-s) under the select_ignores_uncertainty ablation
        # (program.rq_score / p_hat are already set above). The stored
        # champion_rq stays the real R_Q, so the MAP still logs true scores --
        # only the winner choice is H-blind under the ablation.
        new_priority = self._select_priority(program)
        if niche.champion is None or new_priority > self._select_priority(niche.champion):
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
                program.program_id, keep_cell=cell
            )
            return True
        return False

    def target_cell(self, program: ProblemProgram) -> tuple[int, int] | None:
        """Return the cell a program would target without mutating the MAP.

        None when the program carries no usable GROUP/SKILL pair, matching what
        :meth:`try_insert` would do with it.
        """
        return self.program_to_cell(program)

    def remove_program(self, program_id: str) -> list[tuple[int, int]]:
        """Remove one champion identity from the single live MAP archive."""
        removed: list[tuple[int, int]] = []
        for key, niche in self.grid.items():
            if (
                niche.champion is not None
                and niche.champion.program_id == program_id
            ):
                niche.champion = None
                niche.champion_rq = -1.0
                removed.append(key)
        return removed

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
        """Require valid execution and visible variation across verification seeds.

        Constant answers can be mathematically legitimate (for example an
        invariant or feasibility family), so answer diversity is diagnostic
        rather than a validity requirement. Hidden seed-dependent answers behind
        one unchanged problem are rejected because the visible instances must
        themselves vary.
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
            problems.append(" ".join((inst.problem or "").split()))
            answers.append(answer)

        n_total = n_seeds
        passed = (
            n_broken == 0
            and len(answers) == n_total
            and (n_total <= 1 or len(set(problems)) >= 2)
        )

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

    def _select_priority(self, program: ProblemProgram | None) -> float:
        """Selection priority for a champion: full R_Q, or s(1-s) under the
        select_ignores_uncertainty ablation, or H under select_ignores_variance."""
        if program is None:
            return 0.0
        return selection_priority(
            float(getattr(program, "p_hat", 0.0) or 0.0),
            float(getattr(program, "rq_score", 0.0) or 0.0),
            float(getattr(program, "h_score", 0.0) or 0.0),
            ignore_uncertainty=self.select_ignores_uncertainty,
            ignore_variance=self.select_ignores_variance,
        )

    def _is_learnable(self, program: ProblemProgram | None) -> bool:
        """Priority>0: a learnable parent. Too-easy generators (p_hat=1.0 ->
        priority 0) stay in the archive but are not selected for mutation.
        Under the ablation the priority is s(1-s), so p_hat in (0,1) qualifies."""
        return program is not None and self._select_priority(program) > 0.0

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

    def _sample_ucb(self, occupied: list[tuple[tuple[int, int], Niche]]):
        # Exploitation term ranks by selection priority: real R_Q in production,
        # s(1-s) under the select_ignores_uncertainty ablation. champion_rq stays
        # the real stored R_Q; we recompute priority from each champion.
        priorities = {
            key: self._select_priority(n.champion) for key, n in occupied
        }
        sorted_rqs = sorted(set(priorities.values()))
        denom = max(len(sorted_rqs) - 1, 1)
        total = self.total_selections + 1

        best_score = -math.inf
        best = occupied[0]
        for item in occupied:
            key, niche = item
            rank = sorted_rqs.index(priorities[key]) / denom
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
        total = self.n_group_bins * self.n_skill_bins
        # Per-axis coverage separates "we only ever work in two domains" from
        # "we only ever exercise two reasoning moves" -- the two failure modes
        # the operators are meant to fix, and they look identical in the single
        # coverage number.
        groups_hit = {p.get_group() for p in champions if p.get_group()}
        skills_hit = {p.get_skill() for p in champions if p.get_skill()}
        return {
            "num_champions": len(champions),
            "total_niches": total,
            "coverage": len(champions) / total if total else 0.0,
            "group_coverage": len(groups_hit) / self.n_group_bins,
            "skill_coverage": len(skills_hit) / self.n_skill_bins,
            "mean_rq": sum(rqs) / len(rqs) if rqs else 0.0,
            "max_rq": max(rqs) if rqs else 0.0,
            "total_insertions": self.total_insertions,
            "total_replacements": self.total_replacements,
            "total_selections": self.total_selections,
        }

    def to_payload(self) -> dict:
        """In-memory snapshot of the archive (same structure written to
        ``archive.json``). Used to embed the live MAP into the verl ``data.pt``
        checkpoint so the grid is restored atomically with the weights."""
        return {
            "meta": {
                "axes": ["group", "skill"],
                "group_labels": list(GROUPS),
                "skill_labels": list(SKILLS),
                "epsilon": self.epsilon,
                "ucb_c": self.ucb_c,
                "selection_strategy": self.selection_strategy,
                "stats": self.stats(),
            },
            "champions": [p.to_dict() for p in self.champions()],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "archive.json").write_text(
            json.dumps(self.to_payload(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_payload(self, payload: dict) -> int:
        """Restore champions from an in-memory payload (see :meth:`to_payload`).

        Shares all placement/re-binning logic with :meth:`load`; the file-based
        ``load`` is a thin reader that delegates here.
        """
        meta = payload.get("meta", {})
        pre_migration = meta.get("axes") != ["group", "skill"]
        if pre_migration and payload.get("champions"):
            print(
                "[archive.load] snapshot predates the GROUP x SKILL grid "
                f"(axes={meta.get('axes') or 'h x diversity'}). Its champions "
                "carry no SKILL label, so they cannot be placed on the skill "
                "axis and are dropped; the run bootstraps from seed_programs."
            )

        for niche in self.grid.values():
            niche.champion = None
            niche.champion_rq = -1.0

        placed = 0
        unlabelled = 0
        for champ_dict in payload.get("champions", []):
            program = ProblemProgram.from_dict(champ_dict)
            cell = self.program_to_cell(program)
            if cell is None:
                unlabelled += 1
                continue
            # The saved coordinates are re-derived rather than trusted: both
            # axes are pure functions of the program's own labels, so a stored
            # coordinate can only ever agree or be stale.
            if (
                program.niche_group,
                program.niche_skill,
            ) != cell:
                program.niche_group, program.niche_skill = cell
            niche = self.grid[cell]
            incumbent = niche.champion
            if incumbent is not None and self._select_priority(
                incumbent
            ) >= self._select_priority(program):
                continue
            niche.champion = program
            niche.champion_rq = float(program.rq_score)
            niche.update_count += 1
            placed += 1
        if unlabelled:
            print(
                f"[archive.load] dropped {unlabelled} champion(s) without a "
                "usable GROUP/SKILL pair"
            )
        return placed

    def load(self, path: str | Path) -> int:
        """Restore champions written by :meth:`save`.

        Returns the number of champions placed. Every niche is cleared first so
        the restored grid reflects exactly the saved state. Each champion's cell
        is re-derived from its own GROUP/SKILL labels rather than trusted from
        the snapshot, because both coordinates are pure functions of those
        labels. Validity/RQ gates are not re-applied -- the saved state is
        reproduced as-is -- and a champion with no usable label pair is dropped.
        """
        path = Path(path)
        archive_file = path / "archive.json" if path.is_dir() else path
        if not archive_file.exists():
            raise FileNotFoundError(f"no archive snapshot at {archive_file}")
        payload = json.loads(archive_file.read_text(encoding="utf-8"))
        return self.load_payload(payload)


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest(), 16)
