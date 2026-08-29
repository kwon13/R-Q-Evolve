import ast
import copy
from .safe_parse import ParserBomb, safe_ast_parse
import textwrap
import re

import io
import tokenize

from .concepts import DOMAINS
from .ast_contract import check_generator_contract
from .problem_type import annotate_problem_type
from .program import ALLOWED_IMPORT_ROOTS, ProblemInstance


TRUSTED_ASSEMBLER_VERSION = "trusted-instance-v2"

FORBIDDEN_SOURCE_PATTERNS = (
    "open(",
    "input(",
    "eval(",
    "exec(",
    "__import__",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "os.",
    "sys.",
)

_LATEX_ESCAPE_PATTERNS = re.compile(
    r"\\(?:"
    r"\(|\)|\[|\]|frac|boxed|text|sqrt|alpha|beta|theta|gamma|delta|epsilon|"
    r"pi|sigma|lambda|mu|leq|geq|neq|cdot|times|pm|left|right|sum|prod|int|"
    r"infty|binom|pmod|gcd|lcm|deg|log|ln|sin|cos|tan|cot|sec|csc|arcsin|"
    r"arccos|arctan|lim|to|quad|qquad|over|substack|prime|circ|triangle|angle"
    r")"
)


def sanitize_latex_raw_strings(code: str) -> str:
    """Prepend 'r' to string literals containing LaTeX backslashes without raw prefix.

    Prevents Python 3.12 SyntaxWarnings (e.g. invalid escape sequence '\\(') and avoids
    destructive ASCII control character substitutions (e.g. \\b -> backspace, \\t -> tab).
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError):
        return code

    modified = False
    new_tokens = []
    for tok in tokens:
        if tok.type == tokenize.STRING:
            val = tok.string
            prefix_match = re.match(r"^([a-zA-Z]*)(['\"].*)$", val, re.DOTALL)
            if prefix_match:
                prefix, quote_part = prefix_match.groups()
                # If not already raw (r/R), formatted (f/F), or binary (b/B)
                if (
                    "r" not in prefix.lower()
                    and "f" not in prefix.lower()
                    and "b" not in prefix.lower()
                ):
                    if (
                        _LATEX_ESCAPE_PATTERNS.search(quote_part)
                        or r"\(" in quote_part
                        or r"\)" in quote_part
                        or r"\[" in quote_part
                        or r"\]" in quote_part
                    ):
                        new_val = f"r{val}"
                        tok = tokenize.TokenInfo(
                            tok.type, new_val, tok.start, tok.end, tok.line
                        )
                        modified = True
        new_tokens.append(tok)

    if modified:
        try:
            return tokenize.untokenize(new_tokens)
        except Exception:
            return code
    return code


def extract_generator_code(text: str) -> str | None:
    """Extract the best parseable Python block containing ``generate``.

    Prefers the LAST parseable fenced block: when the model thinks first and a
    draft ``generate`` appears in a ```python``` fence inside <think>, the final
    program (the last fence) wins instead of the draft. Falls back to an
    import/def scan and the whole text when no fenced block parses.
    """
    # ast.parse raises ValueError ("source code string cannot contain null
    # bytes"), NOT SyntaxError, on NUL bytes in a generation -- which the
    # except-SyntaxError guards below would not catch, crashing the whole run.
    # Strip NULs up front so a poisoned generation degrades to a normal
    # parse-failure (rejected candidate) instead of killing training.
    text = text.replace("\x00", "")

    # Fenced blocks in REVERSE document order -> last fence tried first.
    candidates: list[str] = [
        match.group(1).strip()
        for match in re.finditer(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    ][::-1]

    lines = text.splitlines()
    # Cap the import/def-suffix candidates: a degenerate output with hundreds of
    # "def generate"/"import" lines would otherwise spawn hundreds of candidates,
    # each running the (bounded but non-trivial) prefix scan below -> the evolution
    # phase wedges the main thread for many minutes. A real program has 1-2 entry
    # points, so the first few suffixes are enough.
    _MAX_SUFFIX_CANDIDATES = 16
    suffix_added = 0
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("import ", "from ", "def generate")):
            candidates.append("\n".join(lines[i:]).strip())
            suffix_added += 1
            if suffix_added >= _MAX_SUFFIX_CANDIDATES:
                break

    candidates.append(text.strip())

    for candidate in candidates:
        if "def generate" not in candidate:
            continue
        # Sanitize candidate before checking parseable prefix
        candidate = sanitize_latex_raw_strings(candidate)
        trimmed = _trim_to_parseable_prefix(candidate)
        if trimmed is not None:
            return sanitize_latex_raw_strings(trimmed)
    return None


# Max ast.parse attempts per candidate. The prefix scan below is worst-case
# O(n^2) over lines; on a multi-thousand-line degenerate generation that wedges
# the (single-threaded) evolution phase for minutes. A genuine generate() program
# is small, so capping the attempts bounds the work without losing real code.
_TRIM_MAX_ATTEMPTS = 400


def _trim_to_parseable_prefix(code: str) -> str | None:
    """Longest line-prefix of ``code`` that parses AND defines ``generate``.

    Bounded: instead of trying every prefix length (O(n) parses), a SyntaxError
    jumps straight to just before the offending line, and the total attempts are
    capped at ``_TRIM_MAX_ATTEMPTS`` so a huge unparseable blob can never hang.
    """
    lines = code.splitlines()
    end = len(lines)
    attempts = 0
    while end > 0:
        attempts += 1
        if attempts > _TRIM_MAX_ATTEMPTS:
            return None
        snippet = "\n".join(lines[:end]).strip()
        try:
            tree = safe_ast_parse(snippet)
        except ParserBomb:
            # Shaving trailing lines cannot defuse a nesting bomb (the nesting
            # is within a line), and stepping through would retry the explosive
            # parse up to _TRIM_MAX_ATTEMPTS more times -- bail out of the
            # whole candidate. Must precede the SyntaxError arm: ParserBomb is
            # a SyntaxError so existing handlers elsewhere absorb it, but here
            # the jump-to-lineno recovery would defeat the point.
            return None
        except (SyntaxError, ValueError) as exc:
            # Jump to just before the failing line (O(#error-points), not O(n));
            # ValueError (NUL bytes) has no lineno -> fall back to a single step.
            lineno = getattr(exc, "lineno", None)
            if isinstance(lineno, int) and 0 < lineno - 1 < end:
                end = lineno - 1
            else:
                end -= 1
            continue
        if any(
            isinstance(node, ast.FunctionDef) and node.name == "generate"
            for node in tree.body
        ):
            return snippet
        end -= 1
    return None


def strip_module_docstring(source_code: str) -> str:
    """Drop top-level string-literal statements from a parent generator.

    Used to clean a parent's source before injecting it into a mutation prompt:
    the module docstring (and any stray top-level prose narrative) is the
    anchor the LLM tends to imitate as an output template. Everything else --
    imports, ``generate``, the CONCEPT_* constants, comments, formatting -- is
    preserved verbatim (line-based removal, not ``ast.unparse``). Returns the
    source unchanged if it does not parse. The child is still asked to write its
    own docstring; only the *parent shown in the prompt* is stripped.
    """
    try:
        tree = safe_ast_parse(source_code)
    except (SyntaxError, ValueError):
        return source_code

    drop_lines: set[int] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            end = node.end_lineno or node.lineno
            drop_lines.update(range(node.lineno, end + 1))
    if not drop_lines:
        return source_code

    kept = [
        line
        for i, line in enumerate(source_code.splitlines(), start=1)
        if i not in drop_lines
    ]
    return "\n".join(kept).strip()


def strip_label_declarations(source_code: str) -> str:
    """Delete top-level descriptor assignments from a parent generator.

    The parent shown in a mutation prompt must not carry its own cell. With the
    real labels visible, 97% of 118 distinct children declared the cell their
    parent already occupied, across only 12 distinct cells; hiding them cut that
    to 25%.

    Whatever sits in that tail is what the child copies, so there is no
    redaction that is safe on its own: `GROUP = "..."` produced children whose
    declared cell was literally `...`, and the skeleton placeholder produced
    `<one of the allowed GROUPS>`. Deleting the lines is the only form that
    cannot be copied -- but it removes the ending the contract requires, and on
    its own it drove "missing GROUP; missing SKILL" from 0 to 11 of 24. It is
    safe only because the system prompt now makes PART 1 commit to GROUP and
    SKILL before any code is written, so the two lines in the block are a
    transcription rather than a step the reply can reach the end without taking.

    Line-based, like :func:`strip_module_docstring`. Returns the source
    unchanged if it does not parse.
    """
    try:
        tree = safe_ast_parse(source_code)
    except (SyntaxError, ValueError):
        return source_code

    drop: set[int] = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name)
            and target.id in ("DOMAIN", "PROBLEM_TYPE", "GROUP", "SKILL")
            for target in targets
        ):
            drop.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    if not drop:
        return source_code

    kept = [
        line
        for i, line in enumerate(source_code.splitlines(), start=1)
        if i not in drop
    ]
    return "\n".join(kept).rstrip()


def lint_generator_source(source_code: str) -> list[str]:
    """Cheap static checks before executing a generated program."""
    reasons: list[str] = []
    lowered = source_code.lower()
    for pattern in FORBIDDEN_SOURCE_PATTERNS:
        if pattern in lowered:
            reasons.append(f"forbidden source pattern: {pattern}")

    try:
        tree = safe_ast_parse(source_code)
    except (SyntaxError, ValueError) as exc:
        return [f"syntax error: {exc}"]

    generate_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate"
    ]
    if len(generate_functions) != 1:
        reasons.append(
            "source must contain exactly one top-level generate function "
            f"(found {len(generate_functions)})"
        )

    if not any(
        isinstance(node, ast.FunctionDef) and node.name == "generate"
        for node in tree.body
    ):
        # Keep the historical phrase for callers that group this diagnostic.
        reasons.append("missing top-level generate function")

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    reasons.append(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                reasons.append(f"disallowed import: {node.module}")
        elif isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign)):
            continue
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        else:
            reasons.append(f"top-level executable statement: {type(node).__name__}")

    return reasons


def validated_domain_declaration(source_code: str) -> tuple[str | None, list[str]]:
    """Read one exact top-level literal ``DOMAIN`` declaration.

    This is the compatibility/manual-seed path. New Stage-2 children contain no
    DOMAIN declaration: the downstream local-policy YES/NO labeler writes their
    label into pinned metadata. For source-labelled programs, every occurrence
    remains constrained here to one top-level literal closed-vocabulary store
    with no reads, deletes, branch-local writes, or reassignment.
    """

    try:
        tree = safe_ast_parse(source_code)
    except (SyntaxError, ValueError) as exc:
        return None, [f"syntax error: {exc}"]

    stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "DOMAIN"
        and isinstance(node.ctx, ast.Store)
    ]
    reads_or_deletes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "DOMAIN"
        and not isinstance(node.ctx, ast.Store)
    ]
    declarations: list[ast.Assign | ast.AnnAssign] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "DOMAIN":
                    declarations.append(node)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "DOMAIN":
                declarations.append(node)

    errors: list[str] = []
    if len(stores) != 1 or len(declarations) != 1:
        errors.append(
            "generated program must contain exactly one top-level literal "
            "DOMAIN declaration"
        )
        return None, errors
    if reads_or_deletes:
        errors.append("DOMAIN is metadata only and may not be read or deleted")

    value_node = declarations[0].value
    value = (
        value_node.value
        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str)
        else None
    )
    if value not in DOMAINS:
        errors.append(
            "DOMAIN must be one of " + ", ".join(DOMAINS) + f"; got {value!r}"
        )
        return None, errors
    return value, errors


# ---------------------------------------------------------------------------
# Stage-2 trusted instance compiler
# ---------------------------------------------------------------------------

_STAGE2_PROTOCOL_RE = re.compile(
    r"\A(?:DOMAIN: ([a-z_]+)\n)?MODE: (expression|boolean|set)\n"
    r"CORE:\n```python\n(.*?)\n```",
    re.DOTALL,
)
_STAGE2_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FAMILY_PLACEHOLDER_RE = re.compile(r"\[\[([A-Za-z_][A-Za-z0-9_]*)\]\]")
_UNBOUNDED_LINE_DISTANCE_RE = re.compile(
    r"\b(?:maximum|greatest|largest)\s+(?:possible\s+)?distance\b"
    r"\s+from\s+(?:the\s+)?(?:origin|(?:given|fixed|specific)\s+point|"
    r"point\s+(?:[A-Z]|\([^)]{1,80}\)))\s+to\s+"
    r"(?:any|arbitrary)\s+point\s+on\s+(?:an?\s+|the\s+)?"
    r"(?:infinite\s+)?(?:straight\s+)?line\b(?!\s+segment)",
    re.IGNORECASE | re.DOTALL,
)
_LOCAL_LINE_BOUND_RE = re.compile(
    r"\b(?:within|inside|restricted\s+to|up\s+to|between)\b.{0,100}$|"
    r"\bline\b.{0,100}\b(?:within|inside|intersection\s+with|bounded\s+by)\b",
    re.IGNORECASE,
)
_UNSPECIFIED_DISTINCT_SET_RE = re.compile(
    r"\b(?:given|let|consider)\s+(?:an?\s+)?set\s+of\s+"
    r"\[\[[A-Za-z_][A-Za-z0-9_]*\]\]\s+distinct\s+"
    r"(?:integers|numbers)\b",
    re.IGNORECASE,
)
_CONTENT_SENSITIVE_SET_PREDICATE_RE = re.compile(
    r"\b(?:sum|product|divisib\w*|remainder|congruen\w*|gcd|lcm|"
    r"greatest\s+common\s+divisor|least\s+common\s+multiple|average|"
    r"mean|median)\b",
    re.IGNORECASE,
)
_SUBSET_REQUEST_RE = re.compile(
    r"\b(?:how\s+many|(?:find|determine|compute)\s+(?:the\s+)?number\s+of|"
    r"count)\b.{0,700}\bsubsets?\b.*",
    re.IGNORECASE | re.DOTALL,
)
_SPECIFIED_SET_ELEMENTS_RE = re.compile(
    r"\b(?:namely|whose\s+elements\s+are|consisting\s+exactly\s+of)\s+"
    r"(?:\[\[[A-Za-z_][A-Za-z0-9_]*\]\]|\{[^{}]+\})",
    re.IGNORECASE,
)
_BOOLEAN_TRICHOTOMY_RE = re.compile(
    r"\b(?:inside\s*,?\s*on\s*,?\s*or\s*outside|"
    r"positive\s*,?\s*zero\s*,?\s*or\s*negative)\b",
    re.IGNORECASE,
)
_STAGE2_DESCRIPTOR_NAMES = frozenset(
    {
        "DOMAIN",
        "MODE",
        "CORE",
        "PROBLEM_TYPE",
        "GROUP",
        "SKILL",
        "problem",
        "verifier",
        "FAMILY_TEMPLATE",
        "TRUSTED_ASSEMBLER_VERSION",
    }
)
_STAGE2_FORBIDDEN_CALLS = frozenset(
    {
        "open",
        "input",
        "eval",
        "exec",
        "compile",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "__import__",
        "breakpoint",
        "help",
    }
)
_STAGE2_GLOBAL_RANDOM_NAMES = frozenset(
    {
        "random",
        "Random",
        "randint",
        "randrange",
        "choice",
        "choices",
        "sample",
        "shuffle",
        "uniform",
        "triangular",
        "getrandbits",
        "randbytes",
        "seed",
        "setstate",
        "getstate",
    }
)
_STAGE2_MAX_CORE_CHARS = 40_000
_STAGE2_MAX_AST_NODES = 8_000


def _stage2_identifier_error(name: str) -> str | None:
    if name in _STAGE2_DESCRIPTOR_NAMES:
        return f"stage2 CORE may not use descriptor/trusted name {name!r}"
    if name == "generate":
        return "stage2 CORE may not define or bind generate"
    if name.startswith("__rq_"):
        return f"stage2 CORE may not use reserved name {name!r}"
    if name in _STAGE2_GLOBAL_RANDOM_NAMES:
        return f"stage2 CORE may not use global random name {name!r}"
    return None


def _stage2_bound_import_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    names: list[str] = []
    for alias in node.names:
        if alias.name == "*":
            names.append("*")
        elif alias.asname:
            names.append(alias.asname)
        elif isinstance(node, ast.Import):
            names.append(alias.name.split(".", 1)[0])
        else:
            names.append(alias.name)
    return names


def _stage2_own_nodes(function: ast.FunctionDef) -> list[ast.AST]:
    """Nodes executed by ``function``, excluding nested function scopes."""

    found: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
            ):
                continue
            found.append(child)
            visit(child)

    visit(function)
    return found


def _stage2_root_name(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _stage2_family_parts(
    family_template: str,
) -> tuple[list[tuple[str, str]], tuple[str, ...], str | None]:
    """Parse literal/parameter parts without using ``str.format`` semantics."""

    if not isinstance(family_template, str):
        return [], (), "family_template must be a string"
    if not family_template or len(family_template) > 4_000:
        return [], (), "family_template must contain 1..4000 characters"

    parts: list[tuple[str, str]] = []
    keys: list[str] = []
    cursor = 0
    for match in _FAMILY_PLACEHOLDER_RE.finditer(family_template):
        parts.append(("literal", family_template[cursor : match.start()]))
        key = match.group(1)
        parts.append(("parameter", key))
        if key not in keys:
            keys.append(key)
        cursor = match.end()
    parts.append(("literal", family_template[cursor:]))

    residue = _FAMILY_PLACEHOLDER_RE.sub("", family_template)
    if "[[" in residue or "]]" in residue:
        return [], (), "family_template contains a malformed [[identifier]] placeholder"
    if not keys:
        return (
            [],
            (),
            "family_template requires at least one [[identifier]] placeholder",
        )
    return parts, tuple(sorted(keys)), None


def _stage2_family_semantic_error(
    family_template: str, *, mode: str
) -> str | None:
    """Reject narrow, provably ill-posed fixed-family statement shapes.

    This is deliberately not a general natural-language oracle.  Each rule is
    restricted to a shape whose missing contract is mathematically decisive,
    so ordinary neighbouring questions remain admissible.
    """

    text = " ".join(str(family_template or "").split())
    unbounded = _UNBOUNDED_LINE_DISTANCE_RE.search(text)
    local_clause = text[unbounded.start() : unbounded.end() + 140] if unbounded else ""
    if unbounded and not _LOCAL_LINE_BOUND_RE.search(local_clause):
        return (
            "fixed family asks for a maximum distance over an unbounded line; "
            "restrict the feasible points to a bounded set"
        )
    unspecified_set = _UNSPECIFIED_DISTINCT_SET_RE.search(text)
    subset_request = (
        _SUBSET_REQUEST_RE.search(text, unspecified_set.end())
        if unspecified_set
        else None
    )
    if (
        unspecified_set
        and subset_request
        and _CONTENT_SENSITIVE_SET_PREDICATE_RE.search(subset_request.group(0))
        and not _SPECIFIED_SET_ELEMENTS_RE.search(text)
    ):
        return (
            "fixed family gives only the cardinality of a set of distinct "
            "integers, but asks a value-dependent subset property without "
            "specifying the elements or their construction"
        )
    if mode == "boolean" and _BOOLEAN_TRICHOTOMY_RE.search(text):
        return (
            "fixed family requests one of three outcomes but declares a "
            "Yes/No output contract"
        )
    return None


def _stage2_function_shape_errors(function: ast.FunctionDef) -> list[str]:
    errors: list[str] = []
    args = function.args
    if function.decorator_list:
        errors.append(f"function {function.name!r} may not have decorators")
    if function.returns is not None:
        errors.append(f"function {function.name!r} may not have a return annotation")
    all_args = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if any(arg.annotation is not None for arg in all_args):
        errors.append(f"function {function.name!r} may not have annotations")
    if args.vararg is not None or args.kwarg is not None:
        errors.append(f"function {function.name!r} may not use *args or **kwargs")
    if args.defaults or any(default is not None for default in args.kw_defaults):
        errors.append(f"function {function.name!r} may not have default arguments")
    return errors


def _normalize_stage2_core_ast(tree: ast.AST) -> ast.AST:
    """Normalize common benign return-statement variations and missing imports in build_instance AST."""
    # 0. Auto-inject missing whitelist imports if referenced in code without import
    existing_imports: set[str] = set()
    locally_bound_names: set[str] = set()
    loaded_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                existing_imports.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                existing_imports.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    locally_bound_names.add(target.id)
        elif isinstance(node, ast.FunctionDef):
            locally_bound_names.add(node.name)
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                locally_bound_names.add(arg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded_names.add(node.id)

    allowed_to_inject = (
        (set(ALLOWED_IMPORT_ROOTS) - {"random"})
        - existing_imports
        - locally_bound_names
    )
    needed_imports = allowed_to_inject & loaded_names
    if needed_imports:
        for pkg in sorted(needed_imports):
            tree.body.insert(0, ast.Import(names=[ast.alias(name=pkg, asname=None)]))

    builder = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_instance"
        ),
        None,
    )
    if builder is None or not builder.body:
        ast.fix_missing_locations(tree)
        return tree

    final_return = builder.body[-1] if isinstance(builder.body[-1], ast.Return) else None
    if final_return is None or not isinstance(final_return.value, ast.Tuple):
        return tree

    elts = list(final_return.value.elts)
    if len(elts) != 3:
        return tree

    assigns_before_params = []
    param_assign = None

    # 1. First element: parameters
    first_elt = elts[0]
    if isinstance(first_elt, ast.Dict):
        builder.body = [
            node
            for node in builder.body
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "parameters"
            )
        ]
        param_assign = ast.Assign(
            targets=[ast.Name(id="parameters", ctx=ast.Store())],
            value=first_elt,
        )
        elts[0] = ast.Name(id="parameters", ctx=ast.Load())
    elif isinstance(first_elt, ast.Name) and first_elt.id == "parameters":
        param_assign = next(
            (
                node
                for node in builder.body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "parameters"
                and isinstance(node.value, ast.Dict)
            ),
            None,
        )
        if param_assign is not None and param_assign in builder.body:
            builder.body.remove(param_assign)

    # 2. Second and Third elements: unwrap str / int / float / list / set
    for i in (1, 2):
        elt = elts[i]
        if (
            isinstance(elt, ast.Call)
            and isinstance(elt.func, ast.Name)
            and elt.func.id in ("str", "int", "float", "list", "set")
        ):
            if elt.args and len(elt.args) == 1:
                elts[i] = elt.args[0]

    # 3. Variable names
    ans_elt = elts[1]
    chk_elt = elts[2]

    if isinstance(ans_elt, ast.Name) and ans_elt.id != "answer":
        assigns_before_params.append(
            ast.Assign(
                targets=[ast.Name(id="answer", ctx=ast.Store())],
                value=ans_elt,
            )
        )
        elts[1] = ast.Name(id="answer", ctx=ast.Load())
    elif not isinstance(ans_elt, ast.Name):
        assigns_before_params.append(
            ast.Assign(
                targets=[ast.Name(id="answer", ctx=ast.Store())],
                value=ans_elt,
            )
        )
        elts[1] = ast.Name(id="answer", ctx=ast.Load())

    if isinstance(chk_elt, ast.Name) and chk_elt.id != "check":
        assigns_before_params.append(
            ast.Assign(
                targets=[ast.Name(id="check", ctx=ast.Store())],
                value=chk_elt,
            )
        )
        elts[2] = ast.Name(id="check", ctx=ast.Load())
    elif not isinstance(chk_elt, ast.Name):
        assigns_before_params.append(
            ast.Assign(
                targets=[ast.Name(id="check", ctx=ast.Store())],
                value=chk_elt,
            )
        )
        elts[2] = ast.Name(id="check", ctx=ast.Load())

    final_return.value.elts = elts

    if builder.body and builder.body[-1] is final_return:
        builder.body.pop()

    for stmt in assigns_before_params:
        builder.body.append(stmt)
    if param_assign is not None:
        builder.body.append(param_assign)
    builder.body.append(final_return)

    ast.fix_missing_locations(tree)
    return tree


def _validate_stage2_core(
    core: str,
    placeholder_keys: tuple[str, ...],
) -> tuple[ast.Module | None, ast.FunctionDef | None, dict[str, str], str | None]:
    """Validate the untrusted CORE and return its placeholder/local mapping."""

    if not core or len(core) > _STAGE2_MAX_CORE_CHARS:
        return (
            None,
            None,
            {},
            (f"stage2 CORE must contain 1..{_STAGE2_MAX_CORE_CHARS} characters"),
        )
    if "\x00" in core:
        return None, None, {}, "stage2 CORE contains a NUL byte"
    try:
        tree = safe_ast_parse(core)
    except (SyntaxError, ValueError) as exc:
        return None, None, {}, f"stage2 CORE syntax error: {exc}"
    try:
        tree = _normalize_stage2_core_ast(tree)
    except Exception:
        pass
    if sum(1 for _ in ast.walk(tree)) > _STAGE2_MAX_AST_NODES:
        return None, None, {}, "stage2 CORE AST is too large"

    errors: list[str] = []
    allowed_imports = set(ALLOWED_IMPORT_ROOTS) - {"random"}
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            errors.append(
                "stage2 CORE top level permits only imports and function definitions; "
                f"found {type(node).__name__}"
            )

    # Imports are executable wherever they occur.  Validate the whole tree so
    # moving a forbidden/random import into a helper cannot evade the top-level
    # shape check.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in allowed_imports:
                    errors.append(f"stage2 CORE import is not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if node.level or root not in allowed_imports:
                errors.append(
                    f"stage2 CORE import is not allowed: {node.module or '<relative>'}"
                )
            if any(alias.name == "*" for alias in node.names):
                errors.append("stage2 CORE may not use wildcard imports")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for bound_name in _stage2_bound_import_names(node):
                identifier_error = _stage2_identifier_error(bound_name)
                if identifier_error:
                    errors.append(identifier_error)
                if bound_name in {"rng", "build_instance"}:
                    errors.append(
                        f"stage2 CORE import may not bind reserved name {bound_name!r}"
                    )

    top_level_builders = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_instance"
    ]
    all_builders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "build_instance"
    ]
    if len(top_level_builders) != 1 or len(all_builders) != 1:
        errors.append(
            "stage2 CORE must define exactly one top-level build_instance(rng) "
            f"(found {len(all_builders)} total, "
            f"{len(top_level_builders)} top-level)"
        )
    builder = (
        top_level_builders[0]
        if len(top_level_builders) == 1 and len(all_builders) == 1
        else None
    )

    forbidden_nodes = (
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Lambda,
        ast.Global,
        ast.Nonlocal,
        ast.Await,
        ast.AsyncFor,
        ast.AsyncWith,
        ast.Yield,
        ast.YieldFrom,
    )
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            errors.append(f"stage2 CORE forbids {type(node).__name__}")
        if isinstance(node, ast.FunctionDef):
            errors.extend(_stage2_function_shape_errors(node))
            identifier_error = _stage2_identifier_error(node.name)
            if identifier_error and node.name != "build_instance":
                errors.append(identifier_error)
            for arg in [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ]:
                identifier_error = _stage2_identifier_error(arg.arg)
                if identifier_error and not (node is builder and arg.arg == "rng"):
                    errors.append(identifier_error)
        elif isinstance(node, ast.Name):
            identifier_error = _stage2_identifier_error(node.id)
            if identifier_error:
                # The one trusted RNG parameter may be loaded but never stored.
                if not (node.id == "rng" and isinstance(node.ctx, ast.Load)):
                    errors.append(identifier_error)
            if node.id == "rng" and isinstance(node.ctx, (ast.Store, ast.Del)):
                errors.append("stage2 CORE may not reassign or delete rng")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                errors.append("stage2 CORE may not access dunder attributes")
            if node.attr in {"seed", "setstate"}:
                errors.append(f"stage2 CORE may not call or access rng.{node.attr}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _STAGE2_FORBIDDEN_CALLS:
                errors.append(f"stage2 CORE forbids call to {node.func.id}")

    if builder is not None:
        args = builder.args
        exact_builder_signature = (
            not args.posonlyargs
            and len(args.args) == 1
            and args.args[0].arg == "rng"
            and not args.kwonlyargs
            and args.vararg is None
            and args.kwarg is None
            and not args.defaults
            and not args.kw_defaults
            and args.args[0].annotation is None
            and builder.returns is None
            and not builder.decorator_list
        )
        if not exact_builder_signature:
            errors.append(
                "build_instance must be undecorated and have exact signature "
                "build_instance(rng), with no defaults or annotations"
            )

    if errors or builder is None:
        return None, None, {}, "; ".join(list(dict.fromkeys(errors))[:8])

    own_nodes = _stage2_own_nodes(builder)
    returns = [node for node in own_nodes if isinstance(node, ast.Return)]
    final_return = builder.body[-1] if builder.body else None
    if (
        len(returns) != 1
        or not isinstance(final_return, ast.Return)
        or returns[0] is not final_return
        or not isinstance(final_return.value, ast.Tuple)
        or len(final_return.value.elts) != 3
        or [getattr(value, "id", None) for value in final_return.value.elts]
        != ["parameters", "answer", "check"]
        or not all(
            isinstance(value, ast.Name) and isinstance(value.ctx, ast.Load)
            for value in final_return.value.elts
        )
    ):
        return (
            None,
            None,
            {},
            (
                "build_instance must have one final return exactly "
                "`return parameters, answer, check`"
            ),
        )

    parameter_assignments = [
        node
        for node in own_nodes
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "parameters"
    ]
    parameter_stores = [
        node
        for node in own_nodes
        if isinstance(node, ast.Name)
        and node.id == "parameters"
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ]
    if len(parameter_assignments) != 1 or len(parameter_stores) != 1:
        return (
            None,
            None,
            {},
            ("build_instance must assign `parameters` exactly once as a literal dict"),
        )
    parameter_assignment = parameter_assignments[0]
    if parameter_assignment not in builder.body or not isinstance(
        parameter_assignment.value, ast.Dict
    ):
        return (
            None,
            None,
            {},
            ("`parameters` must be one top-level literal dict assignment"),
        )
    if len(builder.body) < 2 or builder.body[-2] is not parameter_assignment:
        return (
            None,
            None,
            {},
            (
                "the literal `parameters` assignment must immediately precede "
                "the final return"
            ),
        )
    parameter_dict = parameter_assignment.value
    if any(key is None for key in parameter_dict.keys):
        return None, None, {}, "parameters may not use dict unpacking"
    if not all(
        isinstance(key, ast.Constant) and isinstance(key.value, str)
        for key in parameter_dict.keys
    ):
        return None, None, {}, "parameters keys must be literal strings"
    keys = [key.value for key in parameter_dict.keys]
    if len(keys) != len(set(keys)):
        return None, None, {}, "parameters may not contain duplicate keys"
    if set(keys) != set(placeholder_keys):
        return (
            None,
            None,
            {},
            (
                "parameters keys must exactly match family placeholders: expected "
                f"{list(placeholder_keys)!r}, got {sorted(keys)!r}"
            ),
        )
    if not all(isinstance(value, ast.Name) for value in parameter_dict.values):
        return None, None, {}, "every parameters value must be a local Name"

    assignment_index = builder.body.index(parameter_assignment)
    preceding_nodes = [
        child
        for statement in builder.body[:assignment_index]
        for child in ast.walk(statement)
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    locals_before = {
        node.id
        for node in preceding_nodes
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    parameter_locals = {
        key: value.id
        for key, value in zip(keys, parameter_dict.values)
        if isinstance(value, ast.Name)
    }
    invalid_values = sorted(
        {
            local
            for local in parameter_locals.values()
            if local not in locals_before
            or local in {"rng", "parameters", "answer", "check"}
        }
    )
    if invalid_values:
        return (
            None,
            None,
            {},
            (
                "parameters values must be previously assigned local Names: "
                + ", ".join(invalid_values)
            ),
        )

    # `parameters` is an output snapshot, not a mutable work object. Allow its
    # one literal store and its one final load only; this also blocks aliasing it
    # and mutating through the alias before the trusted assembler sees it.
    parent_map = {
        child: parent
        for parent in ast.walk(builder)
        for child in ast.iter_child_nodes(parent)
    }
    for node in own_nodes:
        if not (
            isinstance(node, ast.Name)
            and node.id == "parameters"
            and isinstance(node.ctx, ast.Load)
        ):
            continue
        parent = parent_map.get(node)
        if parent is not final_return.value:
            return None, None, {}, ("parameters may only be loaded by the final return")
    for node in own_nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                _stage2_root_name(target) == "parameters"
                and not (
                    target is parameter_assignment.targets[0]
                    if isinstance(parameter_assignment, ast.Assign)
                    else False
                )
                for target in targets
            ):
                return None, None, {}, "parameters may not be mutated"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _stage2_root_name(node.func.value) == "parameters"
        ):
            return None, None, {}, "parameters may not be mutated or inspected"

    assigned_names = {
        node.id
        for node in own_nodes
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    if not {"answer", "check"} <= assigned_names:
        return None, None, {}, "build_instance must assign answer and check"

    # The synthesized wrapper asserts the *final* answer/check values, but an
    # internal assertion can otherwise create a convincing dead proof and then
    # overwrite both outputs.  One archived applied-math champion asserted two
    # equal integers and subsequently returned ``False, False`` by comparing
    # those integers with the string ``'Yes'``.  Inspect the reaching output
    # definitions, not merely the existence of an earlier assertion.
    route_assignments: dict[str, list[ast.AST]] = {"answer": [], "check": []}
    for node in own_nodes:
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            for name in {
                child.id
                for child in ast.walk(target)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
            }:
                if name in route_assignments:
                    route_assignments[name].append(node)

    final_route_assignment = {
        name: max(nodes, key=lambda node: (node.lineno, node.col_offset))
        for name, nodes in route_assignments.items()
        if nodes
    }
    final_answer = final_route_assignment.get("answer")
    final_check = final_route_assignment.get("check")
    def _loads(node: ast.AST) -> set[str]:
        return {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }

    if final_answer is not None and "check" in _loads(final_answer):
        return (
            None,
            None,
            {},
            "stage2 CORE failed independent-check contract: A4d: final answer "
            "may not be derived from check",
        )
    if final_check is not None and "answer" in _loads(final_check):
        return (
            None,
            None,
            {},
            "stage2 CORE failed independent-check contract: A4d: final check "
            "may not be derived from answer",
        )

    for node in own_nodes:
        if not isinstance(node, ast.Assert):
            continue
        if not (_loads(node.test) & {"answer", "check"}):
            continue
        later = [
            name
            for name, assignment in final_route_assignment.items()
            if (assignment.lineno, assignment.col_offset)
            > (node.lineno, node.col_offset)
        ]
        if later:
            return (
                None,
                None,
                {},
                "stage2 CORE failed independent-check contract: A4d: "
                "answer/check may not be overwritten after their cross-check: "
                + ", ".join(sorted(later)),
            )

    for name, assignment in final_route_assignment.items():
        value = getattr(assignment, "value", None)
        if value is None:
            continue
        for comparison in (
            node for node in ast.walk(value) if isinstance(node, ast.Compare)
        ):
            operands = [comparison.left, *comparison.comparators]
            if any(
                isinstance(operand, ast.Constant)
                and operand.value in {"Yes", "No"}
                for operand in operands
            ):
                return (
                    None,
                    None,
                    {},
                    f"final {name} may not be formed by comparing a computed "
                    "value with the string 'Yes' or 'No'",
                )
    return tree, builder, parameter_locals, None


def _synthesized_stage2_generate(
    tree: ast.Module,
    builder: ast.FunctionDef,
    family_parts: list[tuple[str, str]],
    parameter_locals: dict[str, str],
) -> str:
    """Inline candidate math so legacy AST checks inspect it, not the wrapper."""

    generated = copy.deepcopy(builder)
    generated.name = "generate"
    generated.args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg="seed")],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )
    generated.decorator_list = []
    generated.returns = None
    generated.body = [
        ast.Assign(
            targets=[ast.Name(id="rng", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="random", ctx=ast.Load()),
                    attr="Random",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id="seed", ctx=ast.Load())],
                keywords=[],
            ),
        ),
        # The penultimate statement is the validated ``parameters = {...}``
        # output snapshot.  Keeping it in this synthetic analysis function
        # connects every placeholder through one dict node, making P1 treat an
        # answer-irrelevant statement parameter as mathematically coupled to
        # every other one.  Rendering below already reads the parameter locals
        # directly, so the snapshot belongs only to the trusted runtime shell.
        *copy.deepcopy(builder.body[:-2]),
        ast.Assert(
            test=ast.Compare(
                left=ast.Name(id="answer", ctx=ast.Load()),
                ops=[ast.Eq()],
                comparators=[ast.Name(id="check", ctx=ast.Load())],
            ),
            msg=ast.Constant(value="answer/check mismatch"),
        ),
        ast.Assign(
            targets=[ast.Name(id="problem", ctx=ast.Store())],
            value=ast.JoinedStr(
                values=[
                    (
                        ast.Constant(value=value)
                        if kind == "literal"
                        else ast.FormattedValue(
                            value=ast.Name(id=parameter_locals[value], ctx=ast.Load()),
                            conversion=-1,
                        )
                    )
                    for kind, value in family_parts
                ]
            ),
        ),
        ast.Return(
            value=ast.Tuple(
                elts=[
                    ast.Name(id="problem", ctx=ast.Load()),
                    ast.Call(
                        func=ast.Name(id="str", ctx=ast.Load()),
                        args=[ast.Name(id="answer", ctx=ast.Load())],
                        keywords=[],
                    ),
                    ast.Dict(
                        keys=[ast.Constant(value="mode")],
                        values=[ast.Constant(value="expression")],
                    ),
                ],
                ctx=ast.Load(),
            )
        ),
    ]
    imports = [
        copy.deepcopy(node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    module = ast.Module(
        body=[ast.Import(names=[ast.alias(name="random")]), *imports, generated],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    return ast.unparse(module)


def _trusted_stage2_source(
    *,
    core: str,
    domain: str | None,
    mode: str,
    family_template: str,
    family_parts: list[tuple[str, str]],
    placeholder_keys: tuple[str, ...],
    normalize_counting_integers: bool,
) -> str:
    """Attach the fixed data-only assembler to a validated candidate CORE."""

    domain_line = f"DOMAIN = {domain!r}\n" if domain is not None else ""
    return f"""import math
import random

__rq_type = type
__rq_len = len
__rq_str = str
__rq_repr = repr
__rq_id = id
__rq_sorted = sorted
__rq_tuple = tuple
__rq_list = list
__rq_set = set
__rq_isfinite = math.isfinite


def __rq_validate_plain(value, *, depth=0, seen=None, budget=None):
    if seen is None:
        seen = __rq_set()
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > 256:
        raise ValueError("stage2 payload contains too many values")
    if depth > 6:
        raise ValueError("stage2 payload is nested too deeply")
    value_type = __rq_type(value)
    if value_type is str:
        if __rq_len(value) > 2048:
            raise ValueError("stage2 payload string is too long")
        return
    if value_type is int:
        if value.bit_length() > 4096:
            raise ValueError("stage2 payload integer is too large")
        return
    if value_type is float:
        if not __rq_isfinite(value):
            raise ValueError("stage2 payload float must be finite")
        return
    if value_type is bool:
        return
    if hasattr(value, "is_FiniteSet") or (hasattr(value, "args") and __rq_type(value).__name__ == "FiniteSet"):
        pass
    elif hasattr(value, "is_integer") and value.is_integer is True:
        return
    elif hasattr(value, "is_number") and value.is_number is True:
        return
    elif value_type in (set, frozenset):
        pass
    elif value_type not in (dict, list, tuple):
        raise ValueError("stage2 payload must contain exact built-in data only")
    marker = __rq_id(value)
    if marker in seen:
        raise ValueError("stage2 payload may not contain cycles")
    if __rq_len(value) > 64:
        raise ValueError("stage2 payload container is too large")
    seen.add(marker)
    try:
        if value_type is dict:
            for key, item in value.items():
                if __rq_type(key) is not str or not key or __rq_len(key) > 128:
                    raise ValueError("stage2 payload dict keys must be short strings")
                __rq_validate_plain(item, depth=depth + 1, seen=seen, budget=budget)
        else:
            for item in value:
                __rq_validate_plain(item, depth=depth + 1, seen=seen, budget=budget)
    finally:
        seen.remove(marker)


def __rq_scalar_text(value):
    value_type = __rq_type(value)
    if value_type is str:
        return value
    if value_type is bool:
        return "True" if value else "False"
    if value_type in (int, float):
        return __rq_str(value)
    if hasattr(value, "is_integer") and value.is_integer is True:
        return __rq_str(int(value))
    if hasattr(value, "is_number") and value.is_number is True:
        return __rq_str(value)
    raise ValueError("a scalar answer/parameter was required")


def __rq_render_parameter(value):
    value_type = __rq_type(value)
    if value_type in (str, int, float, bool):
        return __rq_scalar_text(value)
    if value_type is list:
        return "[" + ", ".join(__rq_render_parameter(v) for v in value) + "]"
    if value_type is tuple:
        return "(" + ", ".join(__rq_render_parameter(v) for v in value) + ")"
    if value_type in (set, frozenset):
        return "{" + ", ".join(__rq_render_parameter(v) for v in __rq_sorted(value, key=__rq_str)) + "}"
    if value_type is dict:
        return "{{" + ", ".join(
            __rq_repr(key) + ": " + __rq_render_parameter(value[key])
            for key in __rq_sorted(value)
        ) + "}}"
    raise ValueError("unsupported parameter value")


{core.rstrip()}


{domain_line.rstrip()}
FAMILY_TEMPLATE = {family_template!r}
TRUSTED_ASSEMBLER_VERSION = {TRUSTED_ASSEMBLER_VERSION!r}
__rq_mode = {mode!r}
__rq_normalize_counting_integers = {normalize_counting_integers!r}
__rq_template_parts = {tuple(family_parts)!r}
__rq_parameter_keys = {placeholder_keys!r}


def generate(seed):
    rng = random.Random(seed)
    payload = build_instance(rng)
    if __rq_type(payload) is not tuple or __rq_len(payload) != 3:
        raise ValueError("build_instance must return one exact 3-tuple")
    parameters = payload[0]
    answer = payload[1]
    check = payload[2]
    __rq_validate_plain(parameters)
    __rq_validate_plain(answer)
    __rq_validate_plain(check)
    if __rq_type(parameters) is not dict:
        raise ValueError("parameters must be an exact built-in dict")
    if __rq_tuple(__rq_sorted(parameters)) != __rq_parameter_keys:
        raise ValueError("runtime parameters keys do not match FAMILY_TEMPLATE")

    problem_parts = []
    for part_kind, part_value in __rq_template_parts:
        if part_kind == "literal":
            problem_parts.append(part_value)
        else:
            problem_parts.append(__rq_render_parameter(parameters[part_value]))
    problem = "".join(problem_parts)
    if not (10 <= __rq_len(problem) <= 4000):
        raise ValueError("assembled problem length must be between 10 and 4000")

    if __rq_mode == "expression":
        if __rq_normalize_counting_integers:
            if __rq_type(answer) is float and answer.is_integer() and -(2**53) <= answer <= 2**53:
                answer = int(answer)
            if __rq_type(check) is float and check.is_integer() and -(2**53) <= check <= 2**53:
                check = int(check)
        if __rq_type(answer) is not __rq_type(check):
            raise ValueError("answer and check must have the same exact built-in type")
        assert answer == check, "answer/check mismatch"
        if __rq_type(answer) not in (str, int, float) or __rq_type(answer) is bool:
            raise ValueError("expression answer must be an exact scalar")
        answer_text = __rq_scalar_text(answer).strip()
        check_text = __rq_scalar_text(check).strip()
        verifier = {{"mode": "expression"}}
    elif __rq_mode == "boolean":
        if __rq_type(answer) is not __rq_type(check):
            raise ValueError("answer and check must have the same exact built-in type")
        assert answer == check, "answer/check mismatch"
        if __rq_type(answer) is bool:
            answer_text = "Yes" if answer else "No"
            check_text = "Yes" if check else "No"
        elif __rq_type(answer) is str and answer in ("Yes", "No"):
            answer_text = answer
            check_text = check
        else:
            raise ValueError("boolean answer must be bool, Yes, or No")
        verifier = {{"mode": "boolean"}}
    else:
        if hasattr(answer, "is_FiniteSet") or (hasattr(answer, "args") and __rq_type(answer).__name__ == "FiniteSet"):
            try:
                answer = __rq_list(answer)
            except Exception:
                pass
        if hasattr(check, "is_FiniteSet") or (hasattr(check, "args") and __rq_type(check).__name__ == "FiniteSet"):
            try:
                check = __rq_list(check)
            except Exception:
                pass
        if __rq_type(answer) in (set, frozenset):
            answer = __rq_list(answer)
        if __rq_type(check) in (set, frozenset):
            check = __rq_list(check)
        if __rq_type(answer) not in (list, tuple) or __rq_type(check) not in (list, tuple):
            raise ValueError("set answer/check must be exact lists, tuples, or sets")
        elements = []
        for element in answer:
            if __rq_type(element) is bool:
                raise ValueError("set elements must be exact non-boolean scalars")
            try:
                elements.append(__rq_scalar_text(element).strip())
            except ValueError:
                raise ValueError("set elements must be exact non-boolean scalars")
        check_elements = []
        for element in check:
            if __rq_type(element) is bool:
                raise ValueError("set elements must be exact non-boolean scalars")
            try:
                check_elements.append(__rq_scalar_text(element).strip())
            except ValueError:
                raise ValueError("set elements must be exact non-boolean scalars")
        if any(
            not element or __rq_len(element) > 512
            for element in elements + check_elements
        ):
            raise ValueError("set elements must contain 1..512 characters")
        canonical_elements = __rq_sorted(__rq_set(elements))
        canonical_check_elements = __rq_sorted(__rq_set(check_elements))
        if __rq_len(canonical_elements) > 32 or __rq_len(canonical_check_elements) > 32:
            raise ValueError("set answer/check has too many unique elements")
        assert canonical_elements == canonical_check_elements, "answer/check mismatch"
        answer_text = r"\\{{" + ",".join(canonical_elements) + r"\\}}"
        check_text = r"\\{{" + ",".join(canonical_check_elements) + r"\\}}"
        verifier = {{"mode": "set", "elements": canonical_elements}}
    if not answer_text or __rq_len(answer_text) > 2048:
        raise ValueError("assembled answer must contain 1..2048 characters")
    assert answer_text == check_text, "serialized answer/check mismatch"
    return problem, answer_text, verifier
"""


def _strip_redundant_random_import(core: str) -> str:
    """Remove boilerplate `import random` / `from random import ...` from CORE.

    Models frequently emit `import random` out of reflex while correctly using
    the runtime-supplied `rng` parameter in `build_instance(rng)`. If the code
    actually uses global `random.<method>`, subsequent AST checking or name
    resolution will safely reject it.
    """
    try:
        tree = safe_ast_parse(core)
    except (SyntaxError, ValueError):
        return core

    class _RandomImportRemover(ast.NodeTransformer):
        def visit_Import(self, node: ast.Import):
            new_names = [
                alias
                for alias in node.names
                if alias.name.split(".", 1)[0] != "random"
            ]
            if not new_names:
                return None
            node.names = new_names
            return node

        def visit_ImportFrom(self, node: ast.ImportFrom):
            if (node.module or "").split(".", 1)[0] == "random":
                return None
            return node

    new_tree = _RandomImportRemover().visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree)
    except Exception:
        lines = []
        for line in core.splitlines():
            s = line.strip()
            if re.match(r"^import\s+random(\s+as\s+\w+)?$", s) or re.match(r"^from\s+random\s+import\s+.*$", s):
                continue
            lines.append(line)
        return "\n".join(lines)


def _extract_stage2_protocol(
    text: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Robustly extract (domain, mode, core, error_or_invalid_reason) from stage2 reply."""
    # 1. Model explicitly declared INVALID
    inv_match = re.search(r"^\s*[\.\*\#\-\>]*\s*INVALID:\s*([^\n]+)", text, re.M)
    if inv_match and not ("def build_instance" in text):
        reason = inv_match.group(1).strip()
        if not reason:
            return None, None, None, "stage2 INVALID requires a nonempty single-line reason"
        return None, None, None, f"stage2 reply declared INVALID: {reason}"

    # 2. Try standard strict match first
    strict_match = _STAGE2_PROTOCOL_RE.match(text)
    if strict_match:
        domain, mode, core = strict_match.groups()
        return domain, mode, core, None

    # 3. Clean leading bullet/markdown markers on header lines
    cleaned_lines = []
    for line in text.splitlines():
        cleaned_line = re.sub(
            r"^\s*[\.\*\#\-\>]+\s*(DOMAIN|MODE|CORE)\s*:",
            r"\1:",
            line,
            flags=re.I,
        )
        cleaned_lines.append(cleaned_line)
    cleaned_text = "\n".join(cleaned_lines)

    strict_match2 = _STAGE2_PROTOCOL_RE.search(cleaned_text)
    if strict_match2:
        domain, mode, core = strict_match2.groups()
        return domain, mode, core, None

    # 4. Search for DOMAIN and MODE flexibly
    domain_match = re.search(
        r"(?:^|\n)\s*DOMAIN\s*:\s*([a-z_]+)", cleaned_text, re.I
    )
    domain = domain_match.group(1).lower() if domain_match else None

    mode_match = re.search(
        r"(?:^|\n)\s*MODE\s*:\s*(expression|boolean|set)", cleaned_text, re.I
    )
    mode = mode_match.group(1).lower() if mode_match else None

    # 5. Extract CORE python code
    core = None
    fence_matches = list(
        re.finditer(r"```(?:python)?\s*\n(.*?)\n```", cleaned_text, re.DOTALL)
    )
    for fm in fence_matches:
        cand = fm.group(1).strip()
        if "def build_instance" in cand:
            core = cand
            break

    if not core and "def build_instance" in cleaned_text:
        lines = cleaned_text.splitlines()
        start_idx = 0
        for i, l in enumerate(lines):
            s = l.strip()
            if (
                s.startswith("import ")
                or s.startswith("from ")
                or s.startswith("def build_instance")
            ):
                start_idx = i
                break
        cand_block = "\n".join(lines[start_idx:]).strip()
        cand_block = re.sub(r"\n```.*$", "", cand_block, flags=re.DOTALL).strip()
        if "def build_instance" in cand_block:
            core = cand_block

    if not core:
        return None, None, None, (
            "stage2 reply must start with exact MODE/CORE and one python fence, "
            "or one line `INVALID: <specific reason>`"
        )

    # Clean core: remove any accidental DOMAIN/MODE/CORE lines inside the code block
    core_lines = []
    for l in core.splitlines():
        s = l.strip()
        if re.match(r"^(?:DOMAIN|MODE|CORE)\s*:", s, re.I):
            continue
        core_lines.append(l)
    cleaned_core = "\n".join(core_lines).strip()

    if not mode:
        return None, None, None, (
            "stage2 reply must declare MODE: expression, MODE: boolean, or MODE: set"
        )

    return domain, mode, cleaned_core, None


def compile_stage2_reply(
    reply: str,
    family_template: str,
    *,
    require_domain: bool = False,
) -> tuple[str | None, str | None]:
    """Compile one strict stage-2 reply into a self-contained generator.

    The model supplies only imports/helpers and ``build_instance(rng)``.  The
    fixed assembler owns seeding, statement rendering, payload validation,
    answer/check enforcement, and the declarative verifier.  It is deliberately
    data-only: candidate callables and custom objects are rejected before any
    string coercion or equality check can invoke their code.
    """

    if not isinstance(reply, str):
        return None, "stage2 reply must be a string"
    normalized = reply.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _STAGE2_THINK_RE.sub("", normalized).strip()
    if "<think>" in normalized or "</think>" in normalized:
        return None, "stage2 reply contains an unclosed <think> block"

    domain, mode, core, extract_error = _extract_stage2_protocol(normalized)
    if extract_error is not None:
        return None, extract_error

    if require_domain and domain is None:
        return None, "stage2 legacy path requires one DOMAIN header"
    if not require_domain and domain is not None:
        return None, "stage2 DOMAIN is assigned downstream and must be omitted"
    if domain is not None and domain not in DOMAINS:
        return None, "stage2 DOMAIN must be one of " + ", ".join(DOMAINS)

    core = _strip_redundant_random_import(core)

    family_parts, placeholder_keys, family_error = _stage2_family_parts(family_template)
    if family_error:
        return None, family_error
    semantic_error = _stage2_family_semantic_error(family_template, mode=mode)
    if semantic_error:
        return None, "stage2 fixed family failed semantic contract: " + semantic_error
    tree, builder, parameter_locals, core_error = _validate_stage2_core(
        core, placeholder_keys
    )
    if core_error or tree is None or builder is None:
        return None, core_error or "stage2 CORE validation failed"

    normalized_core = ast.unparse(tree)

    synthesized = _synthesized_stage2_generate(
        tree, builder, family_parts, parameter_locals
    )
    bare_draw = answer_is_bare_draw(synthesized)
    if bare_draw:
        return None, "stage2 CORE failed bare-answer contract: " + bare_draw
    findings = check_generator_contract(synthesized)
    if findings:
        return None, "stage2 CORE failed independent-check contract: " + "; ".join(
            str(finding) for finding in findings[:3]
        )

    source = _trusted_stage2_source(
        core=normalized_core,
        domain=domain,
        mode=mode,
        family_template=family_template,
        family_parts=family_parts,
        placeholder_keys=placeholder_keys,
        normalize_counting_integers=(
            annotate_problem_type(family_template).problem_type == "counting"
        ),
    )
    source_errors = lint_generator_source(source)
    if source_errors:
        return None, "compiled stage2 source failed lint: " + "; ".join(
            source_errors[:3]
        )
    return source, None


def lint_compiled_stage2_semantics(source_code: str) -> list[str]:
    """Re-audit the model-owned CORE embedded in a trusted Stage-2 source.

    Snapshot champions store the assembled generator, not the original
    ``MODE/CORE`` reply.  Re-running only ``check_generator_contract`` on that
    wrapper sees ``answer, check = build_instance(...)`` as an opaque call and
    cannot inspect the two routes inside.  Reconstruct the candidate-only CORE
    and the same synthetic inline generator used at initial compilation so a
    resumed archive is judged by today's stricter rules.
    """

    if "TRUSTED_ASSEMBLER_VERSION" not in str(source_code or ""):
        return []
    try:
        tree = safe_ast_parse(source_code)
    except (SyntaxError, ValueError):
        return []
    family = extract_problem_statement_template(source_code)
    if not family:
        return ["compiled Stage-2 source has no fixed FAMILY_TEMPLATE"]
    family_parts, placeholder_keys, family_error = _stage2_family_parts(family)
    if family_error:
        return [family_error]

    candidate_body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # The trusted shell owns ``random``; the candidate CORE is forbidden
            # from importing it and does not need it for this static rebuild.
            if isinstance(node, ast.Import) and any(
                alias.name.split(".", 1)[0] == "random" for alias in node.names
            ):
                continue
            candidate_body.append(copy.deepcopy(node))
        elif isinstance(node, ast.FunctionDef) and (
            node.name == "build_instance" or not node.name.startswith("__rq_")
        ) and node.name != "generate":
            candidate_body.append(copy.deepcopy(node))
    candidate_module = ast.Module(body=candidate_body, type_ignores=[])
    ast.fix_missing_locations(candidate_module)
    core = ast.unparse(candidate_module)
    checked_tree, builder, parameter_locals, core_error = _validate_stage2_core(
        core, placeholder_keys
    )
    if core_error or checked_tree is None or builder is None:
        return [core_error or "embedded Stage-2 CORE validation failed"]
    synthesized = _synthesized_stage2_generate(
        checked_tree, builder, family_parts, parameter_locals
    )
    bare_draw = answer_is_bare_draw(synthesized)
    if bare_draw:
        return ["stage2 CORE failed bare-answer contract: " + bare_draw]
    findings = check_generator_contract(synthesized)
    if findings:
        return [
            "stage2 CORE failed independent-check contract: "
            + "; ".join(str(finding) for finding in findings[:3])
        ]
    return []


def lint_mutation_generator_source(
    source_code: str,
    *,
    reject_descriptor_markers: bool = True,
    require_assert: bool = False,
    reject_trivial_assert: bool = True,
    reject_unbounded_sampling: bool = True,
    require_answer_routes: bool = False,
    require_canonical_instance_data: bool = False,
    require_mechanical_shape: bool = True,
) -> list[str]:
    """Extra static contract for model-generated mutation children.

    The standard contract uses one executable ``answer`` route. The historical
    independent ``answer_insight``/``answer_brute`` check remains available only
    when a caller explicitly enables both route-related flags.
    """
    try:
        tree = safe_ast_parse(source_code)
    except (SyntaxError, ValueError) as exc:
        return [f"syntax error: {exc}"]

    generate = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "generate"
        ),
        None,
    )
    if generate is None:
        return ["missing top-level generate function"]

    reasons: list[str] = []
    if reject_descriptor_markers:
        domain, domain_errors = validated_domain_declaration(source_code)
        reasons.extend(domain_errors)
        forbidden_descriptor_names = {"PROBLEM_TYPE", "GROUP", "SKILL"}

        def assigned_names(target: ast.AST) -> set[str]:
            if isinstance(target, ast.Name):
                return {target.id}
            if isinstance(target, (ast.Tuple, ast.List)):
                return {
                    name for element in target.elts for name in assigned_names(element)
                }
            return set()

        ast_markers: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                targets = [node.target]
            else:
                continue
            ast_markers.update(
                name
                for target in targets
                for name in assigned_names(target)
                if name in forbidden_descriptor_names
            )
        text_markers = set(
            re.findall(
                r"\b(PROBLEM_TYPE|GROUP|SKILL)\s*[:=]|\b(DOMAIN)\s*:",
                source_code,
            )
        )
        flattened_text_markers = {
            token for match in text_markers for token in match if token
        }
        markers = sorted(ast_markers | flattened_text_markers)
        if markers:
            reasons.append(
                "generated mutation may not contain PROBLEM_TYPE/GROUP/SKILL "
                "declarations or descriptor field markers: " + ", ".join(markers)
            )

    max_attempts = next(
        (
            node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "MAX_ATTEMPTS"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ),
        None,
    )
    if require_mechanical_shape and max_attempts != 200:
        reasons.append("generated mutation requires top-level MAX_ATTEMPTS = 200")

    assertions = [node for node in ast.walk(generate) if isinstance(node, ast.Assert)]
    if require_assert and not assertions:
        reasons.append("generated mutation requires an assert inside generate()")

    if reject_trivial_assert:
        for assertion in assertions:
            test = assertion.test
            if isinstance(test, ast.Compare) and len(test.comparators) == 1:
                left = ast.dump(test.left, include_attributes=False)
                right = ast.dump(test.comparators[0], include_attributes=False)
                if left == right:
                    reasons.append("trivial self-comparison assert")
                    break

    if require_canonical_instance_data:
        assignments: dict[str, list[ast.AST]] = {}
        assignment_nodes: dict[
            str,
            list[ast.Assign | ast.AnnAssign],
        ] = {}
        for node in ast.walk(generate):
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            else:
                continue
            if value is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(value)
                    assignment_nodes.setdefault(target.id, []).append(node)

        def expression_depends_on(
            expression: ast.AST,
            dependency: str,
            *,
            seen: frozenset[str] = frozenset(),
        ) -> bool:
            names = {
                child.id
                for child in ast.walk(expression)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
            }
            if dependency in names:
                return True
            for name in names - set(seen):
                if name == dependency:
                    return True
                for value in assignments.get(name, ()):
                    if expression_depends_on(
                        value,
                        dependency,
                        seen=seen | {name},
                    ):
                        return True
            return False

        canonical_nodes = assignment_nodes.get("instance_data", [])
        if len(canonical_nodes) != 1:
            reasons.append(
                "generated mutation must assign canonical `instance_data` "
                "exactly once after all sampled-object transformations"
            )
        else:
            canonical_line = canonical_nodes[0].lineno
            for name in ("answer", "problem"):
                if any(
                    node.lineno <= canonical_line
                    for node in assignment_nodes.get(name, ())
                ):
                    reasons.append(
                        f"generated mutation must assign `{name}` only after "
                        "canonical `instance_data`"
                    )

            if not any(
                expression_depends_on(value, "instance_data")
                for value in assignments.get("answer", ())
            ):
                reasons.append(
                    "generated mutation must compute `answer` from " "`instance_data`"
                )
            if not any(
                expression_depends_on(value, "instance_data")
                for value in assignments.get("problem", ())
            ):
                reasons.append(
                    "generated mutation must render `problem` from " "`instance_data`"
                )

            helper_names = {
                node.name
                for node in ast.walk(generate)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            helper_names.update(
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            harmless_names = {
                "abs",
                "enumerate",
                "format",
                "int",
                "len",
                "list",
                "map",
                "max",
                "min",
                "range",
                "repr",
                "sorted",
                "str",
                "sum",
                "tuple",
                "zip",
                "math",
                "sympy",
            }
            stale_problem_names: set[str] = set()
            for value in assignments.get("problem", ()):
                direct_names = {
                    child.id
                    for child in ast.walk(value)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
                }
                for name in direct_names:
                    if (
                        name == "instance_data"
                        or name in helper_names
                        or name in harmless_names
                        or any(
                            expression_depends_on(
                                assigned_value,
                                "instance_data",
                            )
                            for assigned_value in assignments.get(name, ())
                        )
                    ):
                        continue
                    stale_problem_names.add(name)
            if stale_problem_names:
                reasons.append(
                    "generated mutation renders `problem` from names outside "
                    "canonical `instance_data`: "
                    + ", ".join(sorted(stale_problem_names))
                )

            has_semantic_consistency_assert = False
            answer_value_dumps = {
                ast.dump(value, include_attributes=False)
                for value in assignments.get("answer", ())
            }
            for assertion in assertions:
                if assertion.lineno <= canonical_line:
                    continue
                if not isinstance(assertion.test, ast.Compare):
                    continue
                direct_names = {
                    child.id
                    for child in ast.walk(assertion.test)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
                }
                if "answer" not in direct_names:
                    continue
                other_names = direct_names - {"answer"}
                links_canonical_data = "instance_data" in other_names or any(
                    any(
                        expression_depends_on(value, "instance_data")
                        for value in assignments.get(name, ())
                    )
                    for name in other_names
                )
                comparison_parts = [
                    assertion.test.left,
                    *assertion.test.comparators,
                ]
                non_answer_parts = [
                    part
                    for part in comparison_parts
                    if not (isinstance(part, ast.Name) and part.id == "answer")
                ]
                merely_repeats_answer_rhs = bool(non_answer_parts) and all(
                    ast.dump(part, include_attributes=False) in answer_value_dumps
                    for part in non_answer_parts
                )
                if links_canonical_data and not merely_repeats_answer_rhs:
                    has_semantic_consistency_assert = True
                    break
            if not has_semantic_consistency_assert:
                reasons.append(
                    "generated mutation must assert a non-trivial semantic "
                    "consistency comparison linking `instance_data` and "
                    "`answer` without repeating the answer assignment"
                )

            def root_name(node: ast.AST) -> str | None:
                current = node
                while isinstance(current, (ast.Attribute, ast.Subscript)):
                    current = current.value
                return current.id if isinstance(current, ast.Name) else None

            mutating_methods = {
                "add",
                "append",
                "clear",
                "discard",
                "extend",
                "insert",
                "pop",
                "remove",
                "reverse",
                "setdefault",
                "sort",
                "update",
            }
            canonical_source_names = {
                child.id
                for value in assignments.get("instance_data", ())
                for child in ast.walk(value)
                if isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
                and child.id in assignments
            }
            protected_roots = {"instance_data", *canonical_source_names}
            mutated_after_canonical = any(
                (
                    isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                    and any(
                        root_name(target) in protected_roots
                        and (
                            not isinstance(target, ast.Name)
                            or target.id != "instance_data"
                        )
                        for target in (
                            node.targets
                            if isinstance(node, ast.Assign)
                            else [node.target]
                        )
                    )
                )
                or (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and root_name(node.func.value) in protected_roots
                    and node.func.attr in mutating_methods
                )
                for node in ast.walk(generate)
                if getattr(node, "lineno", 0) > canonical_line
            )
            if mutated_after_canonical:
                reasons.append(
                    "generated mutation may not mutate `instance_data` after "
                    "canonicalization"
                )

    if reject_unbounded_sampling:
        for node in ast.walk(generate):
            if (
                isinstance(node, ast.While)
                and isinstance(node.test, ast.Constant)
                and node.test.value is True
            ):
                reasons.append("generated mutation may not use while True")
                break
        has_bounded_sampler = any(
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and any(
                isinstance(arg, ast.Name) and arg.id == "MAX_ATTEMPTS"
                for arg in node.iter.args
            )
            and any(
                isinstance(child, ast.Raise)
                and isinstance(child.exc, ast.Call)
                and isinstance(child.exc.func, ast.Name)
                and child.exc.func.id == "RuntimeError"
                for branch in node.orelse
                for child in ast.walk(branch)
            )
            for node in ast.walk(generate)
        )
        if not has_bounded_sampler:
            reasons.append(
                "generated mutation requires `for ... in range(MAX_ATTEMPTS)` "
                "with an exhaustion else clause that raises RuntimeError"
            )

    if require_answer_routes:
        assigned_names = {
            target.id
            for node in ast.walk(generate)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                list(node.targets) if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        for route_name in ("answer_insight", "answer_brute"):
            if route_name not in assigned_names:
                reasons.append(f"generated mutation must assign `{route_name}`")

        route_assignments: dict[str, list[ast.AST]] = {
            "answer_insight": [],
            "answer_brute": [],
        }
        for node in ast.walk(generate):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                list(node.targets) if isinstance(node, ast.Assign) else [node.target]
            )
            for target in targets:
                if isinstance(target, ast.Name) and target.id in route_assignments:
                    route_assignments[target.id].append(node.value)

        # Catch the common false cross-check where both route variables are
        # aliases for one precomputed value. Restrict identical-RHS rejection
        # to the single-assignment case so independent accumulators that both
        # initialize at zero remain valid.
        insight_values = route_assignments["answer_insight"]
        brute_values = route_assignments["answer_brute"]
        if len(insight_values) == 1 and len(brute_values) == 1:
            if ast.dump(
                insight_values[0],
                include_attributes=False,
            ) == ast.dump(
                brute_values[0],
                include_attributes=False,
            ):
                reasons.append(
                    "answer_insight and answer_brute have identical assignments"
                )
        for route_name, other_name in (
            ("answer_insight", "answer_brute"),
            ("answer_brute", "answer_insight"),
        ):
            if any(
                isinstance(child, ast.Name) and child.id == other_name
                for value in route_assignments[route_name]
                for child in ast.walk(value)
            ):
                reasons.append(
                    f"`{route_name}` may not be computed from `{other_name}`"
                )
                break

        has_route_equivalence_assert = any(
            isinstance(assertion.test, ast.Compare)
            and any(isinstance(op, ast.Eq) for op in assertion.test.ops)
            and {
                part.id
                for part in [
                    assertion.test.left,
                    *assertion.test.comparators,
                ]
                if isinstance(part, ast.Name)
            }
            >= {"answer_insight", "answer_brute"}
            for assertion in assertions
        )
        if not has_route_equivalence_assert:
            reasons.append(
                "generated mutation must assert " "`answer_insight == answer_brute`"
            )

    # Randomness must flow through the seed-local rng. Creating Random(seed) is
    # required, and direct module-level random calls inside generate are rejected.
    has_local_rng = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "rng" for t in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "random"
        and node.value.func.attr == "Random"
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "seed"
        for node in ast.walk(generate)
    )
    if not has_local_rng:
        reasons.append("generated mutation requires `rng = random.Random(seed)`")
    for node in ast.walk(generate):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "random"
            and node.func.attr != "Random"
        ):
            reasons.append(f"direct random.{node.func.attr} call; use rng instead")
            break

    for node in ast.walk(generate):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Attribute)
            and node.iter.func.attr in {"items", "keys", "values"}
        ):
            reasons.append(
                "dict/set-derived iteration must be wrapped in sorted() "
                "for cross-process determinism"
            )
            break

    returns = [node for node in ast.walk(generate) if isinstance(node, ast.Return)]
    has_integer_serialization = any(
        isinstance(node.value, ast.Tuple)
        and len(node.value.elts) == 2
        and isinstance(node.value.elts[1], ast.Call)
        and isinstance(node.value.elts[1].func, ast.Name)
        and node.value.elts[1].func.id == "str"
        and len(node.value.elts[1].args) == 1
        and isinstance(node.value.elts[1].args[0], ast.Call)
        and isinstance(node.value.elts[1].args[0].func, ast.Attribute)
        and isinstance(node.value.elts[1].args[0].func.value, ast.Name)
        and node.value.elts[1].args[0].func.value.id == "sympy"
        and node.value.elts[1].args[0].func.attr == "Integer"
        for node in returns
    )
    if require_mechanical_shape and not has_integer_serialization:
        reasons.append(
            "generated mutation must return " "`problem, str(sympy.Integer(answer))`"
        )

    return reasons


# A competition-math statement that runs past this is not a hard problem, it is
# a generator dumping its sampled object into the prose. The bound matters
# operationally, not just aesthetically: problem_text is concatenated into the
# evaluator prompt, the solver prompt, and every training row, so one runaway
# statement can exceed the rollout context and take the whole run down. Real
# instances (14 seeds + 8 verified fixtures, 110 samples) top out at 397 chars,
# so this leaves an order of magnitude of headroom.
MAX_PROBLEM_TEXT_CHARS = 4000


def lint_problem_instance(instance: ProblemInstance) -> list[str]:
    """Reject obviously poor training examples."""
    reasons: list[str] = []
    problem = instance.problem.strip()
    answer = instance.answer.strip()

    if len(problem) < 10:
        reasons.append("problem text too short")
    if len(problem) > MAX_PROBLEM_TEXT_CHARS:
        reasons.append(
            f"problem text too long: {len(problem)} chars "
            f"(limit {MAX_PROBLEM_TEXT_CHARS})"
        )
    if not answer:
        reasons.append("empty answer")
    if any(token in answer.lower() for token in ("nan", "inf", "undefined")):
        reasons.append("non-finite answer")
    if instance.verifier.get("mode", "expression") == "expression" and (
        "," in answer or ";" in answer
    ):
        reasons.append("multi-part answer")

    lowered = problem.lower()

    # 1) repr / object leakage
    if re.search(r"<(?:function|built-in|class|bound method|module)\b", problem):
        reasons.append("object repr leaked into problem text")
    if re.search(r"0x[0-9a-fA-F]{6,}", problem):
        reasons.append("memory address leaked into problem text")

    # 2) literal answer appears in the problem text
    boolean_answer = instance.verifier.get("mode") == "boolean"
    if not boolean_answer and _answer_leaks_into_problem(answer, problem):
        reasons.append("answer leaked into problem text")

    # 3) answer disguised as a variable assignment
    if not boolean_answer and _answer_leaks_as_assignment(answer, problem):
        reasons.append("answer leaked via variable assignment")

    # 4) soft concatenation cue: marker + multiple imperatives
    concat_markers = (
        "additionally",
        "now consider",
        "now, consider",
        "find the value of x in the following",
        "compute the sum of the first",
        "also compute",
        "and then calculate",
    )
    hits = [m for m in concat_markers if m in lowered]
    if len(hits) >= 1 and _looks_multi_answer(problem):
        reasons.append(f"possible concatenation: {hits}")

    # 4b) strong concatenation markers
    if re.search(
        r"\b(also compute|and then (?:compute|calculate)|"
        r"separately(?: compute)?|total sum of all parts)\b",
        problem,
        re.IGNORECASE,
    ):
        reasons.append("explicit concatenation marker")

    # 5) self-contradictory numeric range
    for lo, var, hi in re.findall(r"(\d+)\s*<\s*([A-Za-z]\w*)\s*<\s*(\d+)", problem):
        if int(lo) >= int(hi):
            reasons.append(f"contradictory range: {lo} < {var} < {hi}")

    # 6) intermediate computed-value leak
    if re.search(
        r"\bwhich (?:is|equals|gives)\s+-?\d{2,}", problem, re.IGNORECASE
    ) or re.search(r"\bsum\b[^.]{0,40}\bis\s+-?\d{3,}", problem, re.IGNORECASE):
        reasons.append("intermediate result leaked into problem text")

    # 7) pre-computed data dump
    if len(re.findall(r"[=:]\s*-?\d{4,}\b", problem)) >= 2:
        reasons.append("pre-computed data dump in problem text")

    # 8) malformed / nested LaTeX delimiters
    if re.search(r"\\\([^)]*\\\(", problem):
        reasons.append("malformed/nested LaTeX delimiters")

    # 9) structural multi-problem: two or more independent questions, even with
    #    NO concatenation marker (catches the marker-free stapling that 4/4b miss,
    #    e.g. a number-theory paragraph followed by a committee paragraph).
    if _counts_independent_questions(problem) >= 2:
        reasons.append("multiple independent questions in problem text")

    return reasons


def _counts_independent_questions(problem: str) -> int:
    """Count distinct answer-demanding questions.

    A single multi-step problem normally issues ONE final demand ("find X").
    Two or more separate demands — each phrased as its own question — is the
    signature of a stapled multi-problem, regardless of any linking word.
    """
    # explicit question marks
    q_marks = problem.count("?")
    # imperative answer-demands ("how many", "find", "compute", "solve for",
    # "determine", "calculate"), counted as occurrences
    demands = re.findall(
        r"\b(how many|find|compute|calculate|determine|solve for|" r"evaluate)\b",
        problem,
        re.IGNORECASE,
    )
    # Question marks are a clean signal (a single problem normally has one "?").
    # Demand verbs are NOISY: a legitimate multi-step problem routinely chains
    # two of them on one object -- "compute ... find ... mod 1000", "find all x
    # ... and sum them" -- which are exactly the high-RQ AIME-style problems we
    # must not reject. So count demands as multiple questions only at 3+ (i.e.
    # demands - 1), while 2+ question marks alone still flags a stapled pair.
    return max(q_marks, len(demands) - 1)


def _answer_leaks_into_problem(answer: str, problem: str) -> bool:
    """True if the exact answer value shows up in the problem body."""
    a = answer.strip()
    # skip very short answers (0-9, single char) — too many false positives
    if len(a) <= 2:
        return False
    # match the answer as a standalone token (not a substring of a longer number)
    pattern = r"(?<![\d.])" + re.escape(a) + r"(?![\d.])"
    return re.search(pattern, problem) is not None


def _looks_multi_answer(problem: str) -> bool:
    """Heuristic: more than one imperative 'compute/find' verb suggests
    independent subproblems."""
    verbs = re.findall(
        r"\b(compute|find|calculate|determine|evaluate|how many)\b",
        problem,
        re.IGNORECASE,
    )
    return len(verbs) >= 2


def _answer_leaks_as_assignment(answer: str, problem: str) -> bool:
    """True if the answer appears as a bare 'var = <number>' assignment.

    Catches the pattern where the model writes a chain like
        x = 8^10
        y = x + 8^10
        z = y + 8^10
        z = 1073741824
    disguising the final answer as another equation line.
    """
    a = answer.strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", a):  # only for numeric answers
        return False
    # any line of the form  <identifier> = <pure number literal>
    # where that number equals the answer
    for m in re.finditer(r"(?m)^\s*[A-Za-z_]\w*\s*=\s*(-?\d+(?:\.\d+)?)\s*$", problem):
        if m.group(1) == a:
            return True
    return False


def answer_is_bare_draw(source_code: str) -> str | None:
    """Error when ``answer`` is just a name the generator sampled, unchanged.

    The pattern is ``n = rng.randint(3, 10)`` followed by ``answer = n``. Nothing
    between the statement and the returned value does any work, so the program's
    mathematics is whatever the prose happens to claim and the assert is
    guaranteed to pass -- an archived champion paired "Construct a sequence of n
    distinct positive integers whose pairwise sums avoid n; find the smallest
    possible a_n" with ``answer = point_count`` and a `check` that read the last
    element of ``range(1, n + 1)``, i.e. the same n wearing a hat.

    Measured over the 48 champions of the 4B run: 10 (21%) are built this way,
    and their s_hat averages 0.90 against the archive's 0.81 -- solvers "succeed"
    because the answer is the number printed in the question.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None
    drawn: set[str] = set()
    answer_value: ast.expr | None = None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr in ("randint", "choice", "randrange", "sample")
        ):
            drawn.add(target.id)
        if target.id == "answer":
            answer_value = value
    if answer_value is None:
        return None
    if isinstance(answer_value, ast.Name) and answer_value.id in drawn:
        return (
            f"answer is the sampled parameter {answer_value.id!r} unchanged: the "
            "program computes nothing the statement asks for"
        )
    return None


def answer_leaks_in_every_instance(instances) -> str | None:
    """Error when the answer appears verbatim in the problem text of EVERY seed.

    ``_answer_leaks_into_problem`` already looks for this on ONE instance, but it
    returns early for answers of two characters or fewer -- and that guard is why
    it has never fired: measured over the 48 champions of the 4B run it flags 0
    of them, while 12 (25%) do leak, all with short answers like ``n`` drawn from
    ``randint(3, 10)`` and printed in the statement as "Let n = 7".

    Requiring the leak on EVERY verification seed is what makes short answers
    safe to check. A single coincidence is common; the same coincidence five
    independent draws running is not. Measured on the same 48: demanding one seed
    flags 18 (38%, mean s_hat 0.82 -- it is catching real problems too), while
    demanding all five flags 12 (25%, mean s_hat 0.90) and those are exactly the
    trivial ones.
    """
    # Yes/no prompts necessarily print the two legal answer words (for example
    # "Answer Yes or No"). That is the output format, not a leaked truth value.
    # Per-instance lint already makes the same distinction; keep the stronger
    # across-seed check aligned with it.
    seen = [
        i for i in instances if i is not None and i.verifier.get("mode") != "boolean"
    ]
    if len(seen) < 2:
        return None
    for inst in seen:
        answer = str(inst.answer).strip()
        if not answer:
            return None
        # A sentence-ending period is a boundary; a period followed by a digit
        # is part of a larger decimal. The older ``(?![\d.])`` missed the most
        # natural leak spelling, ``The answer is 17.``.
        pattern = r"(?<![\d.])" + re.escape(answer) + r"(?!\d|\.\d)"
        if re.search(pattern, inst.problem) is None:
            return None
    return (
        "the answer is printed in the problem text on every seed: a solver that "
        "copies the number out of the question scores without solving anything"
    )


def set_label_declarations(source_code: str, group: str, skill: str) -> str:
    """Return ``source_code`` with exactly one GROUP / SKILL pair, at the end.

    The two-stage mutation decides the labels in stage 1, from the problem and
    its solution, before any code exists. Stage 2 is shown a parent with its
    label lines removed and is told not to write any, so appending them here is
    what puts them back -- which means they cannot be omitted (the failure that
    cost 15 of 24 children when the model was asked for them) and cannot
    disagree with stage 1.

    Any label lines the patch added anyway are dropped first, so this is
    idempotent and there is never a second pair.
    """
    body = strip_label_declarations(source_code).rstrip()
    return f'{body}\n\n\nGROUP = "{group}"\nSKILL = "{skill}"\n'


def _top_level_family_template(tree: ast.Module) -> str | None:
    """Trusted assembler family text, when present as one literal constant."""

    values: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "FAMILY_TEMPLATE"
            for target in targets
        ):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            values.append(value.value)
        else:
            return None
    return values[0] if len(values) == 1 else None


def extract_problem_template(source_code: str) -> str | None:
    """The parent's ``problem = ...`` assignment, verbatim, as it is written.

    Stage 1 of the two-stage mutation used to be shown one rendered instance --
    seed 0, with every parameter already a number. Shown that, a model changes
    the numbers: measured, 85-89% of children were near-copies of the statement
    they were given, and one child's own account of its mutation was "a
    different prime p, a different g, and a different range".

    The template makes that move visibly empty. Where the instance says
    ``p = 13``, the template says ``p = {p}``, so substituting a different value
    is not a change to anything the model can see. What is left to alter is the
    structure around the holes.

    Returns None when the source does not parse or has no such assignment; the
    caller falls back to the rendered instance rather than dropping the parent.
    """
    try:
        tree = safe_ast_parse(source_code)
    except (SyntaxError, ValueError):
        return None

    family_template = _top_level_family_template(tree)
    if family_template is not None:
        return family_template

    lines = source_code.splitlines()
    best: str | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == "problem" for t in targets):
            continue
        end = node.end_lineno or node.lineno
        # Last assignment wins: a generator that builds the text in pieces ends
        # with the one that is actually returned.
        best = "\n".join(lines[node.lineno - 1 : end])
    return textwrap.dedent(best).strip() if best else None


def extract_problem_statement_template(source_code: str) -> str | None:
    """Return only the text skeleton of the final ``problem`` assignment.

    This is deliberately narrower than :func:`extract_problem_template`. It is
    used both for the Stage-2 parent-family summary and for an untrusted
    structural-inspiration donor, so it contributes only the natural-language
    statement: no assignment syntax, helper code, label declarations,
    answer/check route, comments, or metadata.

    Literal strings and f-strings joined with ``+`` are supported.  A template
    that interpolates the program's ``answer`` or ``check`` variable is rejected
    instead of risking an answer leak.  More dynamic constructions are also
    rejected; the caller can simply sample another archive champion.
    """
    try:
        tree = safe_ast_parse(source_code)
    except (SyntaxError, ValueError):
        return None

    family_template = _top_level_family_template(tree)
    if family_template is not None:
        return family_template

    value: ast.expr | None = None
    value_lineno = -1
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if (
            any(isinstance(t, ast.Name) and t.id == "problem" for t in targets)
            and node.lineno >= value_lineno
        ):
            # Match extract_problem_template's last-assignment-wins contract.
            value = node.value
            value_lineno = node.lineno
    if value is None:
        return None

    rendered = _render_problem_text_expression(value)
    if rendered is None:
        return None
    rendered = textwrap.dedent(rendered).strip()
    return rendered or None


def _render_problem_text_expression(node: ast.AST) -> str | None:
    """Render a safe string/f-string AST as a readable parameterized statement."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        pieces: list[str] = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                pieces.append(part.value)
                continue
            if not isinstance(part, ast.FormattedValue):
                return None
            referenced = {
                name.id.lower()
                for name in ast.walk(part.value)
                if isinstance(name, ast.Name)
            }
            forbidden = {
                "answer",
                "check",
                "group",
                "skill",
                "domain",
                "problem_type",
                "concept_group",
                "concept_type",
            }
            if any(
                name in forbidden
                or name.endswith("_answer")
                or name.startswith("answer_")
                or name.endswith("_check")
                for name in referenced
            ):
                return None
            expression = ast.unparse(part.value).strip()
            if not expression:
                return None
            conversion = f"!{chr(part.conversion)}" if part.conversion != -1 else ""
            format_spec = ""
            if part.format_spec is not None:
                rendered_spec = _render_problem_text_expression(part.format_spec)
                if rendered_spec is None:
                    return None
                format_spec = f":{rendered_spec}"
            pieces.append(f"{{{expression}{conversion}{format_spec}}}")
        return "".join(pieces)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _render_problem_text_expression(node.left)
        right = _render_problem_text_expression(node.right)
        if left is None or right is None:
            return None
        return left + right
    return None


_INSPIRATION_ROLE_MARKER_RE = re.compile(
    r"<\|[^|\r\n]{1,64}\|>|\[/?INST\]|<<\s*/?SYS\s*>>",
    re.IGNORECASE,
)
_INSPIRATION_FIELD_MARKER_RE = re.compile(
    r"^\s*(?:system|developer|assistant|user|target\s+group|target\s+skill|"
    r"target\s+domain|target\s+problem[_ ]type|group|skill|domain|"
    r"problem[_ ]type|problem[_ ]text|reference[_ ]answer|"
    r"declarative[_ ]verifier|valid|domain[_ ]confidence|domain[_ ]evidence|"
    r"type[_ ]confidence|type[_ ]evidence|failure[_ ]reason|answer|"
    r"structural\s+mutation|child\s+family|why\s+finite)"
    r"\s*[:=]",
    re.IGNORECASE | re.MULTILINE,
)
_INSPIRATION_INLINE_LABEL_RE = re.compile(
    r"\b(?:declared\s+|target\s+)?(?:GROUP|SKILL|DOMAIN|PROBLEM_TYPE)\s*[:=]",
    re.IGNORECASE,
)
_INSPIRATION_OVERRIDE_RE = re.compile(
    r"\b(?:ignore|disregard|override)\b.{0,48}\b(?:instruction|prompt|rule)s?\b|"
    r"\bfollow\s+(?:these|the following|my)\s+instructions?\b|"
    r"\breply\s+with\s+exactly\b",
    re.IGNORECASE | re.DOTALL,
)


def structural_inspiration_safety_reason(template: str) -> str | None:
    """Classify prompt-control text that makes a donor unsafe to embed.

    Markdown quoting is presentation, not a security boundary.  In particular,
    Qwen tokenizes strings such as ``<|im_start|>`` as real chat-control tokens,
    so an archive-generated problem statement containing one must be rejected,
    not merely prefixed with ``>``.  The filters are intentionally narrow enough
    to keep ordinary mathematical uses of words such as "group" and "target".
    """
    text = str(template or "")
    if _INSPIRATION_ROLE_MARKER_RE.search(text):
        return "chat_control_marker"
    if "```" in text:
        return "code_fence"
    if _INSPIRATION_FIELD_MARKER_RE.search(text):
        return "role_label_or_output_marker"
    if _INSPIRATION_INLINE_LABEL_RE.search(text):
        return "explicit_label_marker"
    if _INSPIRATION_OVERRIDE_RE.search(text):
        return "instruction_override"
    return None
