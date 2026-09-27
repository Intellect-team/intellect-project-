"""Fallback reasoning path — used automatically if Nemotron fails or is unavailable.
Mirrors nvidia_llm.generate_guidance()'s signature exactly."""
import json

from openai import OpenAI

from app import config
from app.schemas.first_aid import FirstAidResponse

_client = OpenAI(api_key=config.GEMINI_API_KEY, base_url=config.GEMINI_BASE_URL)

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
        model=config.GEMINI_LLM_MODEL,
        temperature=0.2,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = completion.choices[0].message.content or ""
    parsed = _parse_json_response(raw)
    return FirstAidResponse(**parsed, source="gemini")


def classify_image(image_bytes: bytes, media_type: str) -> dict:
    """Placeholder classification step, used until the custom-trained model
    (see the custom-model guide) is wired into services/classifier.py."""
    import base64

    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{media_type};base64,{b64}"
    system_prompt = (
        "Classify the visible first-aid situation in the photo. Respond with ONLY a JSON "
        f"object: {{\"label\": <one of {config.ALLOWED_CONDITIONS}>, \"confidence\": <0-100 integer>}}. "
        "Use 'normal' if unclear or no injury is visible — never guess confidently on an ambiguous photo."
    )
    completion = _client.chat.completions.create(
        model=config.GEMINI_VISION_MODEL,
        max_tokens=200,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify this photo."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    return _parse_json_response(completion.choices[0].message.content or "")
