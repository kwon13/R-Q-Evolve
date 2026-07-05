"""OpenAI-backed evaluator gate for generated math problems."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OpenAIEvaluatorConfig:
    model: str
    reasoning_effort: str
    timeout_s: float
    max_output_tokens: int


def evaluate_messages_with_openai(
    messages: list[dict],
    config: OpenAIEvaluatorConfig,
) -> str:
    """Return the evaluator's text verdict from the OpenAI Responses API."""
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
