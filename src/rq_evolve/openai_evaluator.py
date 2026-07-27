"""OpenAI-backed evaluator gate for generated math problems."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re


class EvaluatorConfigurationError(RuntimeError):
    """Evaluator cannot start because its local configuration is invalid."""


class EvaluatorRuntimeError(RuntimeError):
    """Evaluator failed during a run; continuing would invalidate the run."""


@dataclass(slots=True)
class OpenAIEvaluatorConfig:
    model: str
    reasoning_effort: str
    timeout_s: float
    max_output_tokens: int


def load_project_dotenv(project_root: str | Path) -> None:
    """Load project ``.env`` values without overwriting the shell environment.

    The training entrypoint historically did not load ``R-Q-Evolve/.env`` even
    though the evaluation scripts did. Keeping this tiny loader dependency-free
    makes the evaluator behave consistently in the training process and Ray
    workers. Secrets are never printed.
    """
    path = Path(project_root) / ".env"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise EvaluatorConfigurationError(
            f"Unable to read evaluator environment file: {path}: {exc}"
        ) from exc
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^export\s+", "", line)
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not match:
            continue
        key, value = match.groups()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def validate_openai_evaluator_environment() -> None:
    """Fail before the first candidate if the OpenAI credential is absent."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise EvaluatorConfigurationError(
            "OpenAI evaluator is enabled but OPENAI_API_KEY is not set. "
            "Set it in the shell or in R-Q-Evolve/.env before starting training."
        )


def evaluate_messages_with_openai(
    messages: list[dict],
    config: OpenAIEvaluatorConfig,
) -> str:
    """Return the evaluator's text verdict from the OpenAI Responses API."""
    validate_openai_evaluator_environment()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI evaluator requires the `openai` Python package. "
            "Install project dependencies or run `pip install openai`."
        ) from exc

    client = OpenAI(timeout=float(config.timeout_s))
    response = client.responses.create(
        model=config.model,
        input=messages,
        reasoning={"effort": config.reasoning_effort},
        max_output_tokens=int(config.max_output_tokens),
        store=False,
    )
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    return _extract_output_text(response)


def _extract_output_text(response) -> str:
    """Best-effort text extraction for SDK response objects and dicts."""
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif isinstance(response, dict):
        payload = response
    else:
        return str(response)

    parts: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)
