"""One ``ast.parse`` for the whole package, hardened against parser bombs.

Model-written source reaches ``ast.parse`` from eight call sites (extraction,
linting, the AST contract, label reading, constancy, template extraction). A
degenerate generation -- a base model looping ``[1,`` for thousands of tokens
-- exhausts CPython's PEG-parser arena and raises ``MemoryError`` with host
RAM to spare; pure paren runs get the cheap "too many nested parentheses"
``SyntaxError``, but mixed nesting does not. One such reply killed a training
run mid-iteration (2026-08-23 18:29), and had it first reached
``declared_group`` on an archive-restored champion, every restart would have
replayed the crash.

``ParserBomb`` subclasses ``SyntaxError`` on purpose: every call site already
treats ``SyntaxError`` as "this source cannot be parsed", which is exactly
what a bomb is. Routing every parse through here makes the defense a property
of the package rather than of the call order.
"""

import ast
from collections import deque

__all__ = ["MAX_AST_DEPTH", "ParserBomb", "check_ast_depth", "safe_ast_parse"]

MAX_AST_DEPTH = 100


class ParserBomb(SyntaxError):
    """The parser exhausted itself (``MemoryError``/``RecursionError``) on
    hostile nesting. A ``SyntaxError`` subclass so existing handlers treat it
    as the unparseable source it is."""


def check_ast_depth(tree: ast.AST, max_depth: int = MAX_AST_DEPTH) -> None:
    """Iteratively check AST tree depth using BFS to prevent RecursionError."""
    queue = deque([(tree, 1)])
    while queue:
        node, depth = queue.popleft()
        if depth > max_depth:
            raise ParserBomb(
                f"AST depth ({depth}) exceeds safety limit ({max_depth})"
            )
        for child in ast.iter_child_nodes(node):
            queue.append((child, depth + 1))


def safe_ast_parse(
    source: str, filename: str = "<generated>", max_depth: int = MAX_AST_DEPTH
) -> ast.Module:
    try:
        tree = ast.parse(source, filename)
    except (MemoryError, RecursionError) as exc:
        raise ParserBomb(
            f"parser exhausted on a {len(source)}-char source ({type(exc).__name__})"
        ) from exc
    check_ast_depth(tree, max_depth=max_depth)
    return tree
