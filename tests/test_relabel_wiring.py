"""The SKILL relabeller's link to the backend's logprobs.

The relabeller asks eight binary questions per child (one per skill) with
``max_tokens=1`` and the answer pinned to the YES/NO token ids, then picks the
argmax of the offset-corrected P(YES). That last part only works if the sampled
token's logprob actually comes back from the backend. When it does not, the
picker silently drops to reading YES/NO out of the decoded text -- which is the
greedy variant, measured at 0.641 against the logprob path's 0.812. A silent
12-point regression is exactly the kind of thing a test has to hold, because
nothing else about the run looks different.
"""

import math

import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.config import EvolutionConfig
from rq_evolve.evolution import RQEvolver, _relabel_p_yes
from rq_evolve.program import ProblemInstance, ProblemProgram
from rq_evolve.prompts import GROUPS, MUTATION_OP, SKILLS, MutationTask
from rq_evolve.verl_backend import VerlPolicyBackend

YES_ID, NO_ID = 101, 202


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"YES": [YES_ID], "NO": [NO_ID]}[text]


class _Backend:
    """Answers each (child, skill) call with a preset P(YES)."""

    def __init__(self, p_by_skill: dict[str, float], *, logprobs: bool = True) -> None:
        self.p_by_skill = p_by_skill
        self.tokenizer = _Tokenizer()
        self.max_model_len = None
        self.seen: list[MutationTask] = []
        self._emit_logprobs = logprobs
        self.last_mutation_logprobs: list[tuple[int, float] | None] = []

    def mutate(self, tasks):
        self.seen.extend(tasks)
        replies, pairs = [], []
        for task in tasks:
            # The skill under test is the last block of the user turn.
            skill = next(s for s in SKILLS if f"\n{s}:" in task.prompt)
            p = self.p_by_skill[skill]
            # Greedy with two allowed tokens: the model emits whichever side wins.
            if p >= 0.5:
                replies.append("YES")
                pairs.append((YES_ID, math.log(p)))
            else:
                replies.append("NO")
                pairs.append((NO_ID, math.log(1.0 - p)))
        self.last_mutation_logprobs = pairs if self._emit_logprobs else []
        return replies


def _child(declared: str) -> ProblemProgram:
    return ProblemProgram(
        source_code=(
            "def generate(seed):\n"
            '    return f"What is 1 + {seed}?", str(1 + seed)\n\n\n'
            'GROUP = "algebra"\n'
            f'SKILL = "{declared}"\n'
        ),
        metadata={
            "op": MUTATION_OP,
            "skill": declared,
            # Existing tests isolate the skill relabeller. mutate_both has its
            # own regression test below and must probe both axes.
            "mutation_strategy": "mutate_skill",
        },
    )


def _entry(declared: str) -> dict:
    return {
        "task": type("T", (), {"op": MUTATION_OP, "parent": None})(),
        "child": _child(declared),
        "inst": ProblemInstance(
            problem="What is 1 + 1?", answer="2", program_id="p", seed=0
        ),
    }


def _evolver(backend) -> RQEvolver:
    return RQEvolver(
        archive=MAPElitesArchive(),
        backend=backend,
        evolution_config=EvolutionConfig(relabel_skill=True),
    )


# --- the conversion itself ----------------------------------------------


def test_a_yes_token_reports_its_own_probability():
    assert _relabel_p_yes((YES_ID, math.log(0.8)), None, YES_ID) == pytest.approx(0.8)


def test_a_no_token_reports_the_complement():
    """Two allowed tokens: the losing side's mass is fixed by the winner's."""
    assert _relabel_p_yes((NO_ID, math.log(0.7)), None, YES_ID) == pytest.approx(0.3)


def test_text_answers_only_when_there_is_no_logprob():
    assert _relabel_p_yes(None, "YES", YES_ID) == 1.0
    assert _relabel_p_yes(None, "NO", YES_ID) == 0.0
    assert _relabel_p_yes(None, "maybe", YES_ID) is None


# --- the wiring ----------------------------------------------------------


def test_the_relabeller_reads_the_backend_logprobs():
    """Not the text. A run with the text path scores 0.641, not 0.812."""
    peak = "induction"
    p = {s: 0.9 if s == peak else 0.2 for s in SKILLS}
    backend = _Backend(p)
    evolver = _evolver(backend)
    entries = [_entry("counting")]

    evolver._apply_relabel(entries)

    child = entries[0]["child"]
    assert child.metadata["skill"] == peak
    assert child.metadata["skill_declared"] == "counting"
    assert child.metadata["skill_margin"] > 0
    assert len(backend.seen) == len(SKILLS), "one binary call per skill"


def test_every_call_pins_the_answer_to_the_two_tokens():
    backend = _Backend({s: 0.5 for s in SKILLS})
    _evolver(backend)._apply_relabel([_entry("counting")])

    assert backend.seen, "the relabeller ran"
    for task in backend.seen:
        assert task.max_output_tokens == 1
        assert task.temperature == 0.0
        assert task.logprobs == 1
        assert task.allowed_token_ids == [YES_ID, NO_ID]


def test_the_picker_separates_two_skills_the_text_path_ties():
    """Both sides say YES; only the logprobs tell them apart."""
    p = {s: 0.51 for s in SKILLS}
    p["contradiction"] = 0.95
    backend = _Backend(p)
    evolver = _evolver(backend)
    entries = [_entry("counting")]

    evolver._apply_relabel(entries)
    assert entries[0]["child"].metadata["skill"] == "contradiction"


def test_a_backend_without_logprobs_still_labels_from_the_text():
    p = {s: 0.2 for s in SKILLS}
    p["casework"] = 0.9
    backend = _Backend(p, logprobs=False)
    evolver = _evolver(backend)
    entries = [_entry("counting")]

    evolver._apply_relabel(entries)
    # One YES among seven NOs, so even the degraded path lands on it.
    assert entries[0]["child"].metadata["skill"] == "casework"


def test_relabelling_off_leaves_the_declared_skill_alone():
    backend = _Backend({s: 0.9 for s in SKILLS})
    evolver = RQEvolver(
        archive=MAPElitesArchive(),
        backend=backend,
        evolution_config=EvolutionConfig(relabel_skill=False),
    )
    entries = [_entry("counting")]

    evolver._apply_relabel(entries)
    assert entries[0]["child"].metadata["skill"] == "counting"
    assert backend.seen == [], "no calls when the feature is off"


def test_the_child_keeps_its_identity_when_the_cell_moves():
    """program_id is md5(source_code); relabelling must not rewrite the source."""
    p = {s: 0.9 if s == "invariant" else 0.1 for s in SKILLS}
    entries = [_entry("counting")]
    before_id = entries[0]["child"].program_id
    before_src = entries[0]["child"].source_code

    _evolver(_Backend(p))._apply_relabel(entries)

    child = entries[0]["child"]
    assert child.metadata["skill"] == "invariant"
    assert child.program_id == before_id
    assert child.source_code == before_src


def test_the_run_reports_how_many_labels_moved():
    p = {s: 0.9 if s == "invariant" else 0.1 for s in SKILLS}
    evolver = _evolver(_Backend(p))
    evolver._apply_relabel([_entry("counting"), _entry("invariant")])

    stats = evolver._relabel_stats
    assert stats["relabel_children"] == 2
    assert stats["relabel_changed"] == 1
    assert stats["relabel_agreed"] == 1


def test_mutate_both_blindly_relabels_group_and_skill():
    """The no-target structural-inspiration arm frees both descriptor axes."""

    class _BothAxesBackend(_Backend):
        def __init__(self):
            super().__init__({s: 0.95 if s == "contradiction" else 0.1 for s in SKILLS})
            self.p_by_group = {g: 0.95 if g == "sequence" else 0.1 for g in GROUPS}

        def mutate(self, tasks):
            self.seen.extend(tasks)
            replies, pairs = [], []
            for task in tasks:
                labels = SKILLS if "CANDIDATE SKILL" in task.prompt else GROUPS
                table = self.p_by_skill if labels is SKILLS else self.p_by_group
                label = next(x for x in labels if f"\n{x}:" in task.prompt)
                p = table[label]
                if p >= 0.5:
                    replies.append("YES")
                    pairs.append((YES_ID, math.log(p)))
                else:
                    replies.append("NO")
                    pairs.append((NO_ID, math.log(1.0 - p)))
            self.last_mutation_logprobs = pairs
            return replies

    backend = _BothAxesBackend()
    entry = _entry("counting")
    entry["child"].metadata["mutation_strategy"] = "mutate_both"
    entry["child"].metadata["group"] = "algebra"

    _evolver(backend)._apply_relabel([entry])

    child = entry["child"]
    assert child.metadata["skill"] == "contradiction"
    assert child.metadata["group"] == "sequence"
    assert len(backend.seen) == len(SKILLS) + len(GROUPS)


# --- the backend contract the wiring depends on --------------------------


def _task(logprobs, allowed):
    return MutationTask(
        op="relabel",
        prompt="q",
        parent=None,
        stage="relabel",
        max_output_tokens=1,
        temperature=0.0,
        top_p=1.0,
        logprobs=logprobs,
        allowed_token_ids=allowed,
    )


def test_a_mixed_logprobs_batch_is_refused():
    """meta_info is per batch, so a mixed batch would silently drop rows."""
    backend = VerlPolicyBackend()
    with pytest.raises(ValueError, match="every task in the batch"):
        backend.mutate([_task(1, [YES_ID, NO_ID]), _task(None, [YES_ID, NO_ID])])


def test_two_different_allowed_sets_are_refused():
    backend = VerlPolicyBackend()
    with pytest.raises(ValueError, match="logprobs/allowed_token_ids"):
        backend.mutate([_task(1, [YES_ID, NO_ID]), _task(1, [YES_ID])])


def test_rows_map_back_to_their_tasks_in_order():
    """_generate_with_batch unpads before returning, so row k is task k."""
    backend = VerlPolicyBackend()
    output = type(
        "O",
        (),
        {
            "batch": {
                "responses": [[YES_ID], [NO_ID], [YES_ID]],
                "rollout_log_probs": [[-0.1], [-0.2], [-0.3]],
            }
        },
    )()
    pairs = backend._first_token_logprobs(output, 3)
    assert pairs == [(YES_ID, -0.1), (NO_ID, -0.2), (YES_ID, -0.3)]


def test_a_backend_that_reports_no_logprobs_yields_all_none():
    backend = VerlPolicyBackend()
    output = type("O", (), {"batch": {"responses": [[YES_ID]]}})()
    assert backend._first_token_logprobs(output, 1) == [None]


def test_raw_logprobs_is_refused_rather_than_silently_biased():
    """exp(logprob) is the two-way probability ONLY under processed_logprobs.

    Under raw_logprobs vLLM takes the logprobs before allowed_token_ids masks
    the vocabulary (sampler.py:80-81), so P(YES) + P(NO) < 1 and `1 - p` is not
    P(NO). The run would still finish -- with every archive coordinate wrong.
    """
    backend = VerlPolicyBackend()
    backend._logprobs_mode = "raw_logprobs"
    with pytest.raises(ValueError, match="processed_logprobs"):
        backend.mutate([_task(1, [YES_ID, NO_ID])])


def test_processed_logprobs_passes_the_check():
    backend = VerlPolicyBackend()
    backend._logprobs_mode = "processed_logprobs"
    backend._require_two_way_logprobs()  # does not raise


def test_the_check_defaults_to_verls_own_default():
    """An unbound backend must not fail closed on a mode nobody set."""
    VerlPolicyBackend()._require_two_way_logprobs()
