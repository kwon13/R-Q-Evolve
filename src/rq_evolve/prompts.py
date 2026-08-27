import ast
import hashlib
import os
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from string import Template

from .code_utils import (
    compile_stage2_reply,
    extract_problem_statement_template,
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
# Stage 1 receives no descriptor or destination. Stage 2 only implements the
# fixed child. A later seven-arm YES/NO labeler assigns DOMAIN, and deterministic
# code derives PROBLEM_TYPE from the visible request and verifier contract.
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
    # DOMAIN-labeling probe. With logprobs on and the answer restricted to YES
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
    # Executable references corresponding to one-shot examples shown by Stage 2.
    # The reference objects themselves are model-invisible comparison data: a
    # child copying one is rejected before solver rollouts.
    copy_exclusion_examples: tuple[ProblemProgram, ...] = ()


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
        "parent_source": _stage2_parent_source(parent.source_code),
        "parent_problem": _parent_problem_text(parent),
    }


def _neutralize_prompt_control_text(text: str) -> str:
    """Keep model-generated text inside its data boundary in a later prompt.

    Parents and child families are themselves model outputs. A Python string can
    legally contain chat-template tokens or a Markdown fence; copying it
    verbatim would let one generation create instructions for the next.
    Zero-width separators preserve human/model readability while making those
    control sequences lexically inert.
    """

    value = str(text or "").replace("\x00", "")
    return (
        value.replace("```", "``\u200b`")
        .replace("<|", "<\u200b|")
        .replace("|>", "|\u200b>")
        .replace("[INST]", "[\u200bINST]")
        .replace("[/INST]", "[\u200b/INST]")
    )


def _stage2_parent_source(source_code: str) -> str:
    """Descriptor-free, comment-free parent program used as read-only data."""

    cleaned = strip_label_declarations(strip_module_docstring(source_code)).strip()
    try:
        # Canonical rendering removes comments (the easiest injection channel)
        # while retaining the complete executable program and literal family.
        cleaned = ast.unparse(ast.parse(cleaned)).strip()
    except (SyntaxError, ValueError):
        # Archived parents are validated before insertion, but keep prompt
        # construction diagnostic-safe for hand-built/offline objects.
        pass
    return _neutralize_prompt_control_text(cleaned)


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
# Stage 1 writes the child PROBLEM in prose. Stage 2 receives the fixed family
# plus descriptor-free parent source/family as transformation context, then
# implements only that already-fixed child. It never emits a descriptor.

FAMILY_SYSTEM_PROMPT_FILE = "diff_problem_system_prompt.txt"
FAMILY_USER_PROMPT_FILE = "diff_problem_user_prompt.txt"
GENERATOR_SYSTEM_PROMPT_FILE = "gen_program_system_prompt.txt"
GENERATOR_USER_PROMPT_FILE = "gen_program_user_prompt.txt"
STRUCTURAL_INSPIRATION_SYSTEM_NOTE_FILE = "structural_inspiration_system_note.txt"
STRUCTURAL_INSPIRATION_USER_BLOCK_FILE = "structural_inspiration_user_block.txt"
DOMAIN_LABELING_SYSTEM_PROMPT_FILE = "domain_labeling_system_prompt.txt"
DOMAIN_LABELING_USER_PROMPT_FILE = "domain_labeling_user_prompt.txt"
DOMAIN_DEFINITIONS_FILE = "domain_definitions.txt"
DOMAIN_LABELING_RULESET = "omni-top-level-local-policy-binary-labeler-v2"
DOMAIN_LABELING_METHOD = "local_policy_binary_label_v1"

# --- few-shot rotation ------------------------------------------------------
#
# Stage 1 rotates semantically tagged examples. Stage 2 deliberately has no
# model-visible XML tags: the 4B policy copied those tags and continued into the
# surrounding prompt instead of returning the raw MODE/CORE protocol. Its
# executable copy-exclusion reference is kept below as model-invisible data.
_FAMILY_EXAMPLE_RE = re.compile(
    r"<FAMILY_EXAMPLE>\s*.*?</FAMILY_EXAMPLE>", re.DOTALL
)
FAMILY_SHOTS_SHOWN = 3

_STAGE2_COPY_EXCLUSION_FAMILY = (
    "Find the area of a rectangle with integer side lengths [[length]] and "
    "[[width]]."
)
_STAGE2_COPY_EXCLUSION_REPLY = """MODE: expression
CORE:
```python
def build_instance(rng):
    length = rng.randint(2, 20)
    width = rng.randint(2, 20)
    answer = length * width
    check = sum(1 for _row in range(length) for _column in range(width))
    parameters = {"length": length, "width": width}
    return parameters, answer, check
```"""


@lru_cache(maxsize=2)
def _stage2_copy_exclusion_examples(
    require_domain: bool = False,
) -> tuple[ProblemProgram, ...]:
    """Compile the model-invisible Stage-2 one-shot copy reference."""

    reply = _STAGE2_COPY_EXCLUSION_REPLY
    if require_domain:
        reply = "DOMAIN: geometry\n" + reply
    source, reason = compile_stage2_reply(
        reply,
        _STAGE2_COPY_EXCLUSION_FAMILY,
        require_domain=require_domain,
    )
    if source is None:
        raise ValueError(f"Stage-2 copy-excluded example does not compile: {reason}")
    example = ProblemProgram(
        source_code=source,
        metadata={
            "prompt_copy_exclusion_index": 1,
            "family_sha256": hashlib.sha256(
                _STAGE2_COPY_EXCLUSION_FAMILY.encode("utf-8")
            ).hexdigest(),
        },
    )
    for seed in range(5):
        if example.execute(seed) is None:
            raise ValueError(
                f"Stage-2 copy-excluded example fails at seed={seed}: "
                f"{example.last_execution_error or 'unknown execution error'}"
            )
    return (example,)


def _domain_definitions() -> dict[str, str]:
    definitions: dict[str, str] = {}
    for line in _load_template(DOMAIN_DEFINITIONS_FILE).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(
                f"{DOMAIN_DEFINITIONS_FILE} contains a line without ':'"
            )
        key, definition = line.split(":", 1)
        key, definition = key.strip(), definition.strip()
        if key in definitions or not definition:
            raise ValueError(
                f"invalid or duplicate domain definition for {key!r}"
            )
        definitions[key] = definition
    if set(definitions) != set(DOMAINS):
        raise ValueError(
            f"{DOMAIN_DEFINITIONS_FILE} must define exactly {list(DOMAINS)!r}; "
            f"got {sorted(definitions)!r}"
        )
    return definitions


def domain_labeling_ruleset_sha256() -> str:
    """Hash the exact binary readback rubric and seven domain definitions."""

    payload = "\0".join(
        (
            DOMAIN_LABELING_RULESET,
            _load_template(DOMAIN_LABELING_SYSTEM_PROMPT_FILE),
            _load_template(DOMAIN_LABELING_USER_PROMPT_FILE),
            _load_template(DOMAIN_DEFINITIONS_FILE),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_domain_labeling_task(
    *,
    parent: ProblemProgram,
    child_family: str,
    domain: str,
    allowed_token_ids: list[int] | None = None,
) -> MutationTask:
    """One binary arm of the independent seven-way DOMAIN labeler."""

    definitions = _domain_definitions()
    if domain not in definitions:
        raise ValueError(f"unknown domain labeling candidate: {domain!r}")
    system = _load_template(DOMAIN_LABELING_SYSTEM_PROMPT_FILE).strip()
    user = _render_template(
        _load_template(DOMAIN_LABELING_USER_PROMPT_FILE),
        {
            "child_family": _neutralize_prompt_control_text(
                str(child_family).strip()
            ),
            "domain": domain,
            "domain_definition": definitions[domain],
        },
    ).strip()
    return MutationTask(
        op="domain_labeling",
        prompt=user,
        parent=parent,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stage="domain_labeling",
        max_output_tokens=1,
        temperature=0.0,
        top_p=1.0,
        logprobs=1,
        allowed_token_ids=allowed_token_ids,
    )


def _split_family_system(text: str):
    """Return prompt head, semantic example blocks, and untouched task tail."""

    hits = list(_FAMILY_EXAMPLE_RE.finditer(text))
    if not hits:
        return text, [], ""
    if text.count("<FAMILY_EXAMPLE>") != len(hits) or text.count(
        "</FAMILY_EXAMPLE>"
    ) != len(hits):
        raise ValueError("every Stage-1 example must be one complete <FAMILY_EXAMPLE>")
    for left, right in zip(hits, hits[1:]):
        if text[left.end() : right.start()].strip():
            raise ValueError(
                "text between <FAMILY_EXAMPLE> blocks must be whitespace only"
            )
    head = text[: hits[0].start()]
    blocks = [match.group(0).strip() + "\n\n" for match in hits]
    tail = text[hits[-1].end() :].lstrip()
    return head, blocks, tail


# All three fields are part of the live Stage-1 contract.  In particular,
# accepting a family without WHY FINITE only to ask Stage 2 to return INVALID
# wastes a model call and records the failure at the wrong boundary.
FAMILY_KEYS = ("STRUCTURAL MUTATION", "CHILD FAMILY", "WHY FINITE")
OPTIONAL_FAMILY_KEYS: tuple[str, ...] = ()
# EVERY header the reply may contain, required or not. The lookahead that ends
# one field has to know all of them: a header missing from this list is not a
# boundary, so its whole line is swallowed into the previous field's value --
# which would have quietly appended the finiteness prose to CHILD FAMILY.
_FAMILY_KEY_ALT = "|".join(FAMILY_KEYS + OPTIONAL_FAMILY_KEYS)
_FAMILY_PLACEHOLDER_RE = re.compile(r"\[\[([a-z][a-z0-9_]*)\]\]")
# A bare one-letter braced symbol such as ``{x}`` is common mathematical set
# notation.  Multi-character names are the legacy Python-format spelling the
# live family contract retired; an all-legacy family is rejected independently
# because it contains no valid ``[[...]]`` token at all.
_LEGACY_FAMILY_PLACEHOLDER_RE = re.compile(
    r"(?<!\\)\{(?:[a-z][a-z0-9_]*_[a-z0-9_]*|[a-z][a-z0-9_]+)\}"
)


def _family_placeholder_names(family: str) -> list[str] | None:
    """Unique placeholder names, or None when the family syntax is invalid."""

    names = list(dict.fromkeys(_FAMILY_PLACEHOLDER_RE.findall(family or "")))
    if not names:
        return None
    without_placeholders = _FAMILY_PLACEHOLDER_RE.sub("", family)
    if "[[" in without_placeholders or "]]" in without_placeholders:
        return None
    if _LEGACY_FAMILY_PLACEHOLDER_RE.search(family):
        return None
    return names


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

    if _family_placeholder_names(out["CHILD FAMILY"]) is None:
        return None
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

    The parent goes in as its available family representation (a trusted
    ``[[placeholder]]`` template, a ``problem = ...`` assignment, or a rendered
    fallback) together with one rendered instance. This keeps the mathematical
    structure visible without claiming that every archived representation is a
    Python f-string.
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
        system_prompt = head + "".join(blocks) + tail
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
    emit_legacy_domain: bool = False,
) -> MutationTask:
    """Stage 2: implement an already fixed child family from parent context.

    Stage 1 is the creative mutation and sees no descriptor vocabulary. This
    transcription stage sees the descriptor-free parent generator/family as a
    worked transformation reference, but the child family remains immutable.
    It emits no DOMAIN or PROBLEM_TYPE; those are assigned downstream.
    """
    system_prompt = _load_template(GENERATOR_SYSTEM_PROMPT_FILE)
    if emit_legacy_domain:
        system_prompt = system_prompt.replace(
            "MODE: expression\nCORE:",
            "DOMAIN: geometry\nMODE: expression\nCORE:",
            1,
        )
        system_prompt += (
            "\n\nLEGACY SOURCE-DOMAIN COMPATIBILITY MODE:\n"
            "Prepend exactly `DOMAIN: token` before MODE, choosing one token "
            "from: "
            + ", ".join(DOMAINS)
            + ". This compatibility header is required only because the local "
            "DOMAIN labeler is disabled for this run.\n"
        )
    copy_exclusion_examples = _stage2_copy_exclusion_examples(emit_legacy_domain)
    placeholder_names = _family_placeholder_names(plan["CHILD FAMILY"])
    if placeholder_names is None:
        raise ValueError(
            "Stage 2 requires valid [[lower_snake_case]] placeholder syntax"
        )
    why_finite = plan.get("WHY FINITE", "").strip()
    if not why_finite:
        raise ValueError("Stage 2 requires a nonempty WHY FINITE argument")
    rendered_placeholder_names = "\n".join(
        f"- {name}" for name in placeholder_names
    )
    parent_source = _stage2_parent_source(parent.source_code)
    parent_family = (
        extract_problem_statement_template(parent.source_code)
        or "Parent family unavailable from source; use the fixed child family."
    )
    user_prompt = _render_template(
        _load_template(GENERATOR_USER_PROMPT_FILE),
        {
            "new_problem": _neutralize_prompt_control_text(plan["CHILD FAMILY"]),
            "why_finite": _neutralize_prompt_control_text(why_finite),
            "placeholder_names": rendered_placeholder_names,
            "parent_source": parent_source,
            "parent_family": _neutralize_prompt_control_text(parent_family),
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
        # Donor provenance is reporting-only. The primary parent is now shown
        # explicitly, but structural-inspiration donor content remains absent.
        provenance=dict(provenance or {}),
        inspiration_donor=inspiration_donor,
        copy_exclusion_examples=copy_exclusion_examples,
    )
