import hashlib
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from .code_utils import (
    extract_problem_template,
    strip_label_declarations,
    strip_module_docstring,
)
from .concepts import DOMAINS, GROUPS, SKILLS
from .program import ProblemProgram

SOLVER_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


# One mutation operator. The pair that forced a label change on one axis --
# in_depth held GROUP and moved SKILL, in_breadth the mirror -- is retired.
# Ordering the model to land on a SKILL it had not yet solved for made the
# label a target instead of a description, and the child was written to satisfy
# the order: the archived GROUP/SKILL then disagreed with what the visible
# problem actually demanded, which is the one error the MAP cannot survive.
# Stage 1 receives no descriptor or destination. Stage 2 labels the already
# fixed child with one DOMAIN, and deterministic code derives PROBLEM_TYPE from
# the visible request and verifier contract.
MUTATION_OP = "mutate"


@dataclass(slots=True)
class MutationTask:
    op: str
    prompt: str
    parent: ProblemProgram
    # When set, the backend renders this full chat conversation as the prompt
    # (multi-turn self-fix) instead of wrapping ``prompt`` as a single user msg.
    messages: list[dict] | None = None
    stage: str = "code"
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    # Relabelling probe. With logprobs on and the answer restricted to the YES
    # and NO token ids, one greedy token plus its logprob determines the whole
    # two-way distribution: the sampled side is exp(logprob) and the other is
    # its complement. Nothing else in the mutation path sets these.
    logprobs: int | None = None
    allowed_token_ids: list[int] | None = None
    # Audit-only context carried from stage 1 through stage 2 and into every
    # candidate report. Backends ignore it, and prompt builders render only the
    # explicitly supplied inspiration template -- never this metadata.
    provenance: dict = field(default_factory=dict)
    # Ephemeral audit object. It is never rendered or serialized; keeping the
    # frozen batch donor alive lets the copy gate compare against it even if a
    # later archive insertion evicts that donor from the live MAP.
    inspiration_donor: ProblemProgram | None = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_TEMPLATE_DIR = PROJECT_ROOT / "prompt_templates"
PROMPT_TEMPLATE_DIR = Path(
    os.environ.get("RQ_EVOLVE_PROMPT_DIR", DEFAULT_PROMPT_TEMPLATE_DIR)
)
SHOT_TEMPLATE_DIR = Path(
    os.environ.get("RQ_EVOLVE_SHOT_DIR", PROMPT_TEMPLATE_DIR / "shots")
)


def _load_template(filename: str) -> str:
    path = PROMPT_TEMPLATE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"missing prompt template: {path}")
    return path.read_text(encoding="utf-8")


def _template_identifiers(template: str) -> set[str]:
    """Names referenced by a ``$``-placeholder template.

    ``Template.get_identifiers`` only exists on Python 3.11+, and the training
    env runs 3.10, so walk the class's own pattern instead.
    """
    identifiers: set[str] = set()
    for match in Template.pattern.finditer(template):
        name = match.group("named") or match.group("braced")
        if name is not None:
            identifiers.add(name)
    return identifiers


def _render_template(template: str, context: dict[str, str]) -> str:
    """Substitute every placeholder, refusing to leave one unresolved.

    ``safe_substitute`` would ship a literal ``$parent_skill`` inside the
    prompt, which reads as a plausible instruction and fails silently. The
    check runs on the template's own identifiers, before substitution, so a
    ``$`` appearing inside the substituted parent source is not mistaken for
    an unresolved placeholder.
    """
    missing = sorted(_template_identifiers(template) - set(context))
    if missing:
        raise KeyError(
            "prompt template references placeholders the context does not "
            f"supply: {', '.join(missing)}"
        )
    return Template(template).safe_substitute(context)


def _template_context(parent: ProblemProgram) -> dict[str, str]:
    """Legacy template helper, kept descriptor-free for audit callers."""
    return {
        # The parent's cell is withheld: from the prose, and from the tail of
        # its own source. With the real labels shown, 97% of 118 distinct
        # children declared the cell their parent already occupied, across only
        # 12 distinct cells. The completed child is classified only after both
        # mutation stages have finished.
        "parent_source": strip_label_declarations(
            strip_module_docstring(parent.source_code)
        ),
        "parent_problem": _parent_problem_text(parent),
    }


def _parent_problem_text(parent: ProblemProgram) -> str:
    """The parent's seed-0 statement, or a note saying it could not be run.

    The prompt used to show the source alone, which framed the task as editing
    a program. It is not: the object being mutated is a MATHEMATICAL PROBLEM,
    and the program is only the machine that emits it. A model shown code
    rewrites code -- measured, the largest single failure of distinct children
    was the child's own cross-check firing, i.e. mathematics committed to before
    it was worked out. Showing the statement lets the child be designed as a
    problem first.

    Never raises: the parent is an archived champion that ran at insertion time,
    but a resume can load a snapshot whose source no longer executes here, and a
    prompt is not the place to discover that.
    """
    try:
        instance = parent.execute(seed=0)
    except Exception:  # pragma: no cover - execute already swallows most of these
        instance = None
    if instance is None:
        return "(the parent program did not run here; read its code below instead)"
    return instance.problem.strip()


def build_solver_messages(problem: str) -> list[dict]:
    """The solver conversation: rules in the system turn, problem in the user turn.

    Every other measurement path in the codebase -- ``dataset.py``,
    ``math_eval.py``, the expansion experiments, the standalone vLLM eval --
    sends ``SOLVER_SYSTEM_PROMPT`` as a system turn. The rollout path used to
    concatenate it into the user turn instead, so the policy was trained on one
    prompt shape and scored on another.

    ``shot_internal.build_solver_messages`` is the same conversation with an
    optional worked example spliced in, and its no-shot branch returns exactly
    what this returns. The two stay in step because both read the one
    ``SOLVER_SYSTEM_PROMPT`` above; this is the production rollout/eval builder,
    that one belongs to the shot diagnostics.
    """
    return [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


_INFERRED_LABEL = r"^[ \t>*_#-]*INFERRED_{key}[\s*_]*:[ \t]*(.*)$"


def parse_inferred_labels(reply: str) -> tuple[str | None, str | None]:
    """Parse labels from historical probe replies for offline analysis only.

    The live stage-2 generator no longer emits these historical fields.

    Returns ``(group, skill)`` where each is a vocabulary member or None.
    """
    text = reply or ""
    out: list[str | None] = []
    for key, vocabulary in (("GROUP", GROUPS), ("SKILL", SKILLS)):
        found = None
        for match in re.finditer(
            _INFERRED_LABEL.format(key=key), text, re.IGNORECASE | re.MULTILINE
        ):
            raw_val = match.group(1).strip()
            # Clean decoration: backticks, quotes, asterisks, whitespace
            token = raw_val.strip(" \t`'\"*_.:#").lower()
            token = token.split()[0] if token.split() else ""
            token = token.strip(" \t`'\"*_.:#,")
            if token in vocabulary:
                found = token  # last wins, same convention as parse_declared
        out.append(found)
    return out[0], out[1]


# --- two-stage mutation -----------------------------------------------------
#
# Stage 1 writes the child PROBLEM in prose, with no program in front of it;
# stage 2 writes that problem's generator, with the parent pair shown only as a
# worked example of the statement-to-program mapping. Measured against asking
# one call to mutate the parent program: child/parent source similarity fell
# from 0.99 to 0.14, and the share of children declaring their parent's own
# cell fell from 96% to 0%. Both follow from the same thing -- a base policy
# shown a complete program to "mutate" reproduces it, and a base policy shown a
# program it must not reproduce (because the target statement is already fixed
# and different) does not.

FAMILY_SYSTEM_PROMPT_FILE = "diff_problem_system_prompt.txt"
FAMILY_USER_PROMPT_FILE = "diff_problem_user_prompt.txt"
GENERATOR_SYSTEM_PROMPT_FILE = "gen_program_system_prompt.txt"
GENERATOR_USER_PROMPT_FILE = "gen_program_user_prompt.txt"
# Four authored WORKED EXAMPLEs for the skills the shipped stage-2 prompt never
# demonstrates. Its own four answer INFERRED_SKILL with transformation /
# extremal_principle / counting / casework, and measured over 376 gated children
# those same four absorb 86.7% of everything stage 2 infers -- `invariant` came
# back 4 times in 376 and `construction` 4. The gate is therefore reading its own
# demo set more than the code. With the pool at eight, invariant's pass rate goes
# 2% -> 17% and construction's 46% -> 43-71%.
GENERATOR_EXTRA_EXAMPLES_FILE = "gen_program_extra_examples.txt"
STRUCTURAL_INSPIRATION_SYSTEM_NOTE_FILE = "structural_inspiration_system_note.txt"
STRUCTURAL_INSPIRATION_USER_BLOCK_FILE = "structural_inspiration_user_block.txt"

# --- few-shot rotation ------------------------------------------------------
#
# The stage-1 prompt's eight EXAMPLE blocks are fixed, and the policy copies them:
# 20.4% of children on the base model, 25.6% after 224 RL steps, with EXAMPLE 8
# (Mantel) alone accounting for 12-16% and holding a MAP cell outright. Showing a
# random three of the eight per call breaks that attractor -- copying falls to
# 10%, and no single example exceeds 2.5%. Stage-2's examples are NOT copied
# (0.3%): it transcribes a fixed specification, so there is nothing to invent and
# nothing to plagiarise. Its rotation buys label coverage instead.
#
# Neither rotation looks at the target cell. A stage-2 pool that always contained
# the target skill's example would leak the answer into INFERRED_SKILL, and the
# whole value of that gate is that it is a BLIND re-derivation. Measured both
# ways: target-blind selection is not worse, it is slightly better (invariant
# 17% vs 14%, 20 cells reached vs 17).
_EXAMPLE_HEAD = re.compile(r"^EXAMPLE \d+ — (.+)$", re.M)
_WORKED_HEAD = re.compile(r"^WORKED EXAMPLE \d+$", re.M)
_SKETCH_HEAD = re.compile(r"(?!)")
FAMILY_SHOTS_SHOWN = 3
GENERATOR_SHOTS_SHOWN = 4


def _split_family_system(text: str):
    """(head, [example blocks], tail) for diff_problem_system_prompt.txt."""
    hits = list(_EXAMPLE_HEAD.finditer(text))
    if not hits:
        return text, [], ""
    head = text[: hits[0].start()]
    blocks = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else None
        blocks.append(text[m.start() : end] if end else text[m.start() :])
    cut = blocks[-1].find("Now create a child family")
    tail = blocks[-1][cut:] if cut >= 0 else ""
    if cut >= 0:
        blocks[-1] = blocks[-1][:cut]
    return head, blocks, tail


def _split_generator_system(text: str):
    """(head, skill sketches, mid, worked examples, tail) for the stage-2 prompt."""
    sm = list(_SKETCH_HEAD.finditer(text))
    wm = list(_WORKED_HEAD.finditer(text))
    if not sm or not wm:
        return text, [], "", [], ""
    head = text[: sm[0].start()]
    sketches = []
    for i, m in enumerate(sm):
        end = sm[i + 1].start() if i + 1 < len(sm) else wm[0].start()
        sketches.append(text[m.start() : end])
    mid = text[sm[-1].start() : wm[0].start()][len(sketches[-1]) :]
    worked = []
    for i, m in enumerate(wm):
        end = wm[i + 1].start() if i + 1 < len(wm) else None
        worked.append(text[m.start() : end] if end else text[m.start() :])
    cut = worked[-1].find("After the closing ```")
    tail = worked[-1][cut:] if cut >= 0 else ""
    if cut >= 0:
        worked[-1] = worked[-1][:cut]
    return head, sketches, mid, worked, tail


def _extra_worked_blocks() -> list[str]:
    try:
        text = _load_template(GENERATOR_EXTRA_EXAMPLES_FILE)
    except FileNotFoundError:
        return []
    hits = list(_WORKED_HEAD.finditer(text))
    out = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else None
        out.append(text[m.start() : end] if end else text[m.start() :])
    return out


def _renumber(blocks, pattern, label):
    """Rewrite each block's ordinal so a rotated set still reads 1..N.

    ``_EXAMPLE_HEAD`` captures the block's title ("EXAMPLE 3 - CONTRADICTION");
    that has to survive, because the title is the only place stage 1 is told
    which SKILL the block demonstrates.
    """
    out = []
    for i, block in enumerate(blocks, 1):

        def repl(m, i=i):
            title = m.group(1) if m.re.groups else None
            return f"{label} {i} \u2014 {title}" if title else f"{label} {i}"

        out.append(pattern.sub(repl, block, count=1))
    return out


# What a stage-1 reply must carry for the child to be usable.
FAMILY_KEYS = ("STRUCTURAL MUTATION", "CHILD FAMILY")
# Asked for, parsed when present, but NOT required. WHY FINITE makes the model
# name the clause that bounds its own answer before it labels the problem --
# 31% of archived champions were judged ill-posed, nearly all of them "find the
# maximum X" with the clause that made X finite dropped somewhere in the
# mutation. Requiring it would be the wrong trade: stage-1 parse failures are
# already the largest single loss (230 mutation_failed of 2144 candidates), and
# a fifth mandatory field buys more of them. The value is in writing it, not in
# gating on it.
OPTIONAL_FAMILY_KEYS = ("WHY FINITE",)
# EVERY header the reply may contain, required or not. The lookahead that ends
# one field has to know all of them: a header missing from this list is not a
# boundary, so its whole line is swallowed into the previous field's value --
# which would have quietly appended the finiteness prose to CHILD FAMILY.
_FAMILY_KEY_ALT = "|".join(FAMILY_KEYS + OPTIONAL_FAMILY_KEYS)


def parse_family_plan(reply: str) -> dict[str, str] | None:
    """The stage-1 fields, or None if a REQUIRED one is missing.

    Takes the LAST value for each key that is not a leftover ``<...>``. A base
    policy re-emits the template before answering it -- 13 of 24 replies opened
    with the placeholder lines and only then wrote the real four -- so reading
    the first match reports a complete reply as incomplete.
    """
    text = reply or ""
    out: dict[str, str] = {}
    for key in FAMILY_KEYS:
        best = ""
        for match in re.finditer(
            rf"^[ \t]*{key}[ \t]*:[ \t]*(.+?)(?=^[ \t]*(?:{_FAMILY_KEY_ALT})[ \t]*:|\Z)",
            text,
            re.M | re.S,
        ):
            value = match.group(1).strip()
            if value and not value.startswith("<"):
                best = value
        if not best:
            return None
        out[key] = best
    for key in OPTIONAL_FAMILY_KEYS:
        for match in re.finditer(
            rf"^[ \t]*{key}[ \t]*:[ \t]*(.+?)(?=^[ \t]*(?:{_FAMILY_KEY_ALT})[ \t]*:|\Z)",
            text,
            re.M | re.S,
        ):
            value = match.group(1).strip()
            if value and not value.startswith("<"):
                out[key] = value
    return out


def _first_token(value: str, vocabulary) -> str | None:
    token = value.strip().strip("\"'").split()
    token = token[0].strip(".,") if token else ""
    return token if token in vocabulary else None


def build_family_task(
    parent: ProblemProgram,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    max_output_tokens: int | None = None,
    target_cell: tuple | None = None,
    rotate_shots: bool = False,
    rng: random.Random | None = None,
    inspiration_template: str | None = None,
    inspiration_donor: ProblemProgram | None = None,
    provenance: dict | None = None,
) -> MutationTask:
    """Stage 1: mutate the parent's problem FAMILY, in prose, with no program.

    The parent goes in as its ``problem = ...`` f-string rather than as one
    rendered instance. Shown concrete numbers a model changes the numbers --
    85-89% of children were near-copies of the statement they were given, one
    reporting its own mutation as "a different prime p and a different range".
    Against the braces, substituting values is visibly not a change.
    """
    instance = _parent_problem_text(parent)
    template = extract_problem_template(parent.source_code) or instance
    system_prompt = _load_template(FAMILY_SYSTEM_PROMPT_FILE)
    task_provenance = dict(provenance or {})
    rng = rng or random
    if rotate_shots:
        head, blocks, tail = _split_family_system(system_prompt)
        if len(blocks) > FAMILY_SHOTS_SHOWN:
            blocks = rng.sample(blocks, FAMILY_SHOTS_SHOWN)
        else:
            blocks = list(blocks)
            rng.shuffle(blocks)
        system_prompt = (
            head + "".join(_renumber(blocks, _EXAMPLE_HEAD, "EXAMPLE")) + tail
        )
    if inspiration_template:
        # The behavioural rules belong in the system turn; the donor's one
        # permitted artefact (its parameterized statement skeleton) belongs in
        # the user turn below. Keeping them separate makes it testable that no
        # donor source/answer/labels leaked into either stage.
        inspiration_system_note = _load_template(
            STRUCTURAL_INSPIRATION_SYSTEM_NOTE_FILE
        ).strip()
        system_prompt = system_prompt.rstrip() + "\n\n" + inspiration_system_note + "\n"
    user_prompt = _render_template(
        _load_template(FAMILY_USER_PROMPT_FILE),
        {
            "parent_template": template,
            "parent_problem": instance,
        },
    )
    if inspiration_template:
        inspiration_user_block = _load_template(STRUCTURAL_INSPIRATION_USER_BLOCK_FILE)
        quoted_inspiration = "\n".join(
            f"> {line}" for line in inspiration_template.strip().splitlines()
        )
        user_prompt = (
            user_prompt.rstrip()
            + "\n\n"
            + _render_template(
                inspiration_user_block,
                {"inspiration_template": quoted_inspiration},
            ).strip()
        )
        inspiration_audit = dict(task_provenance.get("structural_inspiration") or {})
        inspiration_audit.update(
            {
                "prompt_version": "structural_inspiration_v1",
                "prompt_contract_sha256": hashlib.sha256(
                    (inspiration_system_note + "\0" + inspiration_user_block).encode(
                        "utf-8"
                    )
                ).hexdigest(),
            }
        )
        task_provenance["structural_inspiration"] = inspiration_audit
    if target_cell is not None:
        raise ValueError(
            "target_cell is retired: mutation must not receive a desired "
            "DOMAIN or PROBLEM_TYPE"
        )
    if inspiration_template:
        inspiration_audit = dict(task_provenance["structural_inspiration"])
        inspiration_audit["stage1_prompt_sha256"] = hashlib.sha256(
            (system_prompt + "\0" + user_prompt).encode("utf-8")
        ).hexdigest()
        task_provenance["structural_inspiration"] = inspiration_audit
    return MutationTask(
        op=MUTATION_OP,
        prompt=f"{system_prompt}\n\n{user_prompt}",
        parent=parent,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stage="family",
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        provenance=task_provenance,
        inspiration_donor=inspiration_donor,
    )


def build_generator_task(
    parent: ProblemProgram,
    plan: dict[str, str],
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    max_output_tokens: int | None = None,
    rotate_shots: bool = False,
    rng: random.Random | None = None,
    provenance: dict | None = None,
    inspiration_donor: ProblemProgram | None = None,
) -> MutationTask:
    """Stage 2: implement and self-label an already fixed child family.

    Stage 1 is the creative mutation and sees no descriptor vocabulary.  This
    transcription stage may only attach one top-level DOMAIN to the immutable
    child family; it receives no target cell or parent label.  PROBLEM_TYPE is
    never model-declared and is derived by deterministic verification code.
    """
    rng = rng or random
    system_prompt = _render_template(
        _load_template(GENERATOR_SYSTEM_PROMPT_FILE),
        {"domain_values": ", ".join(DOMAINS)},
    )
    if rotate_shots:
        head, sketches, mid, worked, tail = _split_generator_system(system_prompt)
        if worked:
            pool = list(worked) + _extra_worked_blocks()
            shown = (
                rng.sample(pool, GENERATOR_SHOTS_SHOWN)
                if len(pool) > GENERATOR_SHOTS_SHOWN
                else list(pool)
            )
            rng.shuffle(shown)
            rng.shuffle(sketches)
            system_prompt = (
                head
                + "".join(sketches)
                + mid
                + "".join(_renumber(shown, _WORKED_HEAD, "WORKED EXAMPLE"))
                + tail
            )
    user_prompt = _render_template(
        _load_template(GENERATOR_USER_PROMPT_FILE),
        {
            "parent_template": extract_problem_template(parent.source_code)
            or _parent_problem_text(parent),
            "parent_source": strip_label_declarations(
                strip_module_docstring(parent.source_code)
            ),
            "new_problem": plan["CHILD FAMILY"],
        },
    )
    return MutationTask(
        op=MUTATION_OP,
        prompt=f"{system_prompt}\n\n{user_prompt}",
        parent=parent,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stage="generator",
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        # Carried for reporting only. No provenance field is substituted into
        # either generator prompt, so stage 2 cannot see the donor.
        provenance=dict(provenance or {}),
        inspiration_donor=inspiration_donor,
    )
