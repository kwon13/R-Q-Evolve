"""Local policy DOMAIN labeling and archive provenance contracts."""

import hashlib
import math
import re

import pytest

from rq_evolve.archive import (
    ArchiveSchemaError,
    CONFIRMED_ARCHIVE_SCHEMA,
    GENERATED_DOMAIN_AUTHORITY,
    MAPElitesArchive,
)
from rq_evolve.code_utils import TRUSTED_ASSEMBLER_VERSION, compile_stage2_reply
from rq_evolve.concepts import DOMAINS
from rq_evolve.config import EvolutionConfig
from rq_evolve.evolution import RQEvolver
from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import MUTATION_OP, MutationTask


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"YES": [14004], "NO": [8996]}[text]


class _BinaryBackend:
    def __init__(
        self,
        probabilities,
        *,
        missing=None,
        tokenizer=True,
        unexpected_token=None,
    ):
        self.tokenizer = _Tokenizer() if tokenizer else None
        self.probabilities = dict(probabilities)
        self.missing = missing
        self.unexpected_token = unexpected_token
        self.last_mutation_logprobs = []
        self.tasks = []

    def mutate(self, tasks):
        self.tasks = list(tasks)
        replies = []
        pairs = []
        for task in tasks:
            user = task.messages[-1]["content"]
            domain = next(
                domain
                for domain in DOMAINS
                if re.search(rf"(?m)^{re.escape(domain)}:", user)
            )
            if domain == self.missing:
                replies.append("YES")
                pairs.append(None)
                continue
            probability = float(self.probabilities[domain])
            if self.unexpected_token is not None:
                replies.append("YES")
                pairs.append((self.unexpected_token, math.log(max(probability, 1e-6))))
            elif probability >= 0.5:
                replies.append("YES")
                pairs.append((14004, math.log(probability)))
            else:
                replies.append("NO")
                pairs.append((8996, math.log(1.0 - probability)))
        self.last_mutation_logprobs = pairs
        return replies


def _reply():
    core = """
def build_instance(rng):
    n = rng.randint(3, 20)
    answer = n * (n + 1) // 2
    check = sum(range(1, n + 1))
    parameters = {"n": n}
    return parameters, answer, check
""".strip()
    fence = "`" * 3
    return f"MODE: expression\nCORE:\n{fence}python\n{core}\n{fence}"


def _entry(evolver):
    family = "Let n = [[n]]. Compute the sum of the integers from 1 through n."
    source, reason = compile_stage2_reply(_reply(), family)
    assert source is not None, reason
    assert "DOMAIN" not in source
    child = ProblemProgram(
        source_code=source,
        generation=1,
        parent_id="parent-program",
        metadata={
            "op": MUTATION_OP,
            "generator_contract": {
                "version": TRUSTED_ASSEMBLER_VERSION,
                "family_sha256": hashlib.sha256(family.encode()).hexdigest(),
            },
        },
    )
    instance, reason = evolver.verify_program(child)
    assert instance is not None, reason
    assert instance.domain is None
    task = MutationTask(
        op=MUTATION_OP,
        prompt="",
        parent=ProblemProgram('DOMAIN = "algebra"\ndef generate(seed):\n    pass\n'),
        stage="generator",
        provenance={"family_plan": {"CHILD FAMILY": family}},
    )
    return {"task": task, "child": child, "inst": instance}


def _evolver(backend, *, min_probability=0.55, min_margin=0.5):
    return RQEvolver(
        archive=MAPElitesArchive(
            require_domain_labeling=True,
            domain_labeling_min_probability=min_probability,
            domain_labeling_min_logit_margin=min_margin,
        ),
        backend=backend,
        evolution_config=EvolutionConfig(
            verify_seeds=3,
            ast_contract="enforce",
            independent_domain_labeling=True,
            domain_labeling_min_probability=min_probability,
            domain_labeling_min_logit_margin=min_margin,
        ),
    )


def _probabilities(winner, top=0.9, runner=0.2):
    values = {domain: 0.05 for domain in DOMAINS}
    values[winner] = top
    values["geometry" if winner != "geometry" else "algebra"] = runner
    return values


def test_high_confidence_readback_assigns_domain_and_unlocks_archive():
    backend = _BinaryBackend(_probabilities("geometry"))
    evolver = _evolver(backend)
    entry = _entry(evolver)
    child = entry["child"]

    assert evolver.archive.program_to_cell(child) is None
    evolver._apply_domain_labeling([entry])

    assert "report" not in entry
    assert len(backend.tasks) == len(DOMAINS)
    labeling = child.metadata["domain_labeling"]
    assert labeling["passed"] is True
    assert labeling["predicted"] == "geometry"
    assert child.get_domain() == "geometry"
    assert entry["inst"].domain == "geometry"
    assert child.metadata["descriptor_contract"]["domain_authority"] == (
        GENERATED_DOMAIN_AUTHORITY
    )
    assert evolver.archive.try_insert(child, u_value=0.2, rq_score=0.1)
    payload = evolver.archive.to_payload()
    assert payload["meta"]["schema"] == CONFIRMED_ARCHIVE_SCHEMA
    assert payload["meta"]["domain_labeling_required"] is True

    restored = MAPElitesArchive(
        require_domain_labeling=True,
        domain_labeling_min_probability=0.55,
        domain_labeling_min_logit_margin=0.5,
    )
    assert restored.load_payload(payload) == 1
    assert restored.program_to_cell(restored.champions()[0]) is not None
    with pytest.raises(ArchiveSchemaError, match="incompatible archive snapshot"):
        MAPElitesArchive().load_payload(payload)


def test_low_margin_and_incomplete_readbacks_fail_closed_with_score_audit():
    close = _probabilities("algebra", top=0.60, runner=0.58)
    evolver = _evolver(_BinaryBackend(close))
    entry = _entry(evolver)
    evolver._apply_domain_labeling([entry])
    assert entry["report"].status == "domain_labeling_failed"
    assert entry["report"].reason.startswith("top-vs-runner logit margin")
    assert set(entry["report"].domain_labeling["probabilities"]) == set(DOMAINS)

    backend = _BinaryBackend(_probabilities("algebra"), missing="calculus")
    evolver = _evolver(backend)
    entry = _entry(evolver)
    evolver._apply_domain_labeling([entry])
    assert entry["report"].status == "domain_labeling_failed"
    assert "calculus" in entry["report"].reason


@pytest.mark.parametrize(
    "backend",
    [
        _BinaryBackend(_probabilities("algebra"), tokenizer=False),
        _BinaryBackend(_probabilities("algebra"), missing="algebra"),
        _BinaryBackend(_probabilities("algebra"), unexpected_token=12345),
    ],
)
def test_text_only_or_invalid_token_verdicts_never_fabricate_confidence(backend):
    evolver = _evolver(backend, min_probability=0.99, min_margin=10.0)
    entry = _entry(evolver)
    child = entry["child"]
    evolver._apply_domain_labeling([entry])
    assert entry["report"].status == "domain_labeling_failed"
    assert child.get_domain() is None


def test_archive_recomputes_probabilities_and_pins_thresholds_on_resume():
    evolver = _evolver(_BinaryBackend(_probabilities("algebra")))
    entry = _entry(evolver)
    evolver._apply_domain_labeling([entry])
    assert evolver.archive.try_insert(entry["child"], u_value=0.2, rq_score=0.1)
    payload = evolver.archive.to_payload()

    labeling = payload["champions"][0]["metadata"]["domain_labeling"]
    labeling["probabilities"]["geometry"] = 0.99
    payload["champions"][0]["metadata"]["descriptor_contract"][
        "domain_labeling"
    ] = dict(labeling)
    with pytest.raises(ArchiveSchemaError, match="no valid DOMAIN/PROBLEM_TYPE"):
        MAPElitesArchive(
            require_domain_labeling=True,
            domain_labeling_min_probability=0.55,
            domain_labeling_min_logit_margin=0.5,
        ).load_payload(payload)

    clean = _evolver(_BinaryBackend(_probabilities("algebra")))
    entry = _entry(clean)
    clean._apply_domain_labeling([entry])
    assert clean.archive.try_insert(entry["child"], u_value=0.2, rq_score=0.1)
    with pytest.raises(ArchiveSchemaError, match="min_probability"):
        MAPElitesArchive(
            require_domain_labeling=True,
            domain_labeling_min_probability=0.0,
            domain_labeling_min_logit_margin=0.0,
        ).load_payload(clean.archive.to_payload())


def test_only_loaded_seed_file_can_receive_manual_domain_authority(tmp_path):
    evolver = _evolver(_BinaryBackend(_probabilities("algebra")))
    pending = _entry(evolver)["child"]
    seed_source = 'DOMAIN = "algebra"\n' + pending.source_code

    forged = ProblemProgram(
        source_code=seed_source,
        generation=0,
        metadata={"source_file": "seed.py"},
    )
    instance, reason = evolver.verify_program(forged)
    assert instance is None
    assert "load_seed_programs" in reason

    seed_path = tmp_path / "seed.py"
    seed_path.write_text(seed_source, encoding="utf-8")
    loaded = evolver.load_seed_programs(tmp_path)
    assert len(loaded) == 1
    manual = loaded[0]
    assert evolver.archive.try_insert(manual, u_value=0.2, rq_score=0.1)
