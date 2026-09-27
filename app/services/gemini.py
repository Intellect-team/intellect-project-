"""Fallback reasoning path - used automatically if Nemotron fails or is unavailable.
Mirrors nvidia_llm.generate_guidance()'s signature exactly.
Also provides the fallback image classifier used by services/classifier.py
when our trained model is unsure."""
import base64
import json

from openai import OpenAI

from app import config
from app.schemas.first_aid import FirstAidResponse

_client = OpenAI(api_key=config.GEMINI_API_KEY, base_url=config.GEMINI_BASE_URL)

_PROMPT_TEMPLATE = config.PROMPT_PATH.read_text(encoding="utf-8")


def _parse_json_response(text: str) -> dict:
    """Extract the JSON object even if the model adds text or ```json fences."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in Gemini response")
    return json.loads(text[start:end + 1])


def generate_guidance(condition: str, confidence: float, retrieved_context: str) -> FirstAidResponse:
    prompt = _PROMPT_TEMPLATE.format(
        condition=condition, confidence=confidence, retrieved_context=retrieved_context
    )
    completion = _client.chat.completions.create(
        model=config.GEMINI_LLM_MODEL,
        temperature=0.2,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = completion.choices[0].message.content or ""
    parsed = _parse_json_response(raw)
    return FirstAidResponse(**parsed, source="gemini")


def classify_image(image_bytes: bytes, media_type: str) -> dict:
    """Fallback classifier - used by services/classifier.py when our trained
    YOLO11 model is unsure (confidence below the threshold) or unavailable."""
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{media_type};base64,{b64}"
    system_prompt = (
        "Classify the visible first-aid situation in the photo. Respond with ONLY a JSON "
        f"object: {{\"label\": <one of {config.ALLOWED_CONDITIONS}>, \"confidence\": <0-100 integer>}}. "
        "Use 'normal' only if the skin is clearly healthy with no injury. "
        "Use 'unknown' if the photo is unclear, ambiguous, or you are not sure - "
        "never guess confidently on an ambiguous photo."
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