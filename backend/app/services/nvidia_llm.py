"""Wraps NVIDIA Nemotron 3.5 Lightning via its OpenAI-compatible endpoint.
Same function signature as services/gemini.py's generate_guidance(), so
routes.py can fall back to Gemini transparently if this raises."""
import json

from openai import OpenAI

from app import config
from app.schemas.first_aid import FirstAidResponse

_client = OpenAI(api_key=config.NVIDIA_API_KEY, base_url=config.NVIDIA_BASE_URL)

_PROMPT_TEMPLATE = config.PROMPT_PATH.read_text(encoding="utf-8")


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def generate_guidance(condition: str, confidence: float, retrieved_context: str) -> FirstAidResponse:
    prompt = _PROMPT_TEMPLATE.format(
        condition=condition, confidence=confidence, retrieved_context=retrieved_context
    )
    completion = _client.chat.completions.create(
        model=config.NEMOTRON_LLM_MODEL,
        temperature=0.2,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = completion.choices[0].message.content or ""
    parsed = _parse_json_response(raw)
    return FirstAidResponse(**parsed, source="nemotron")
