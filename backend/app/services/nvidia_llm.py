"""Wraps NVIDIA Nemotron 3.5 Lightning via its OpenAI-compatible endpoint.
Same function signature as services/gemini.py's generate_guidance(), so
routes.py can fall back to Gemini transparently if this raises."""
import json

from openai import OpenAI

from app import config
from app.schemas.first_aid import FirstAidResponse

_client = OpenAI(api_key=config.NVIDIA_API_KEY, base_url=config.NVIDIA_BASE_URL, timeout=20.0)

_PROMPT_TEMPLATE = config.PROMPT_PATH.read_text(encoding="utf-8")


def _parse_json_response(text: str) -> dict:
    """Extract the JSON object even if the model adds reasoning, <think> tags,
    or ```json fences around it."""
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in Nemotron response")
    return json.loads(text[start:end + 1])


def generate_guidance(condition: str, confidence: float, retrieved_context: str) -> FirstAidResponse:
    prompt = _PROMPT_TEMPLATE.format(
        condition=condition, confidence=confidence, retrieved_context=retrieved_context
    )
    completion = _client.chat.completions.create(
        model=config.NEMOTRON_LLM_MODEL,
        temperature=0.2,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = completion.choices[0].message.content or ""
    parsed = _parse_json_response(raw)
    parsed.pop("source", None)  # we set source ourselves
    parsed.setdefault("condition", condition)
    parsed.setdefault("confidence", confidence)
    return FirstAidResponse(**parsed, source="nemotron")
