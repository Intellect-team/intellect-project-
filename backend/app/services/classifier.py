"""
Image -> (condition, confidence).

TODO: once the custom-trained classifier (MobileNetV3/EfficientNet-Lite, per the
training guide) is exported, replace the body of classify() with local inference,
e.g. using ultralytics YOLO or a torchvision model loaded from models/classifier/.
Nothing outside this file needs to change — routes.py only calls classify().
"""
from app import config
from app.services import gemini


def classify(image_bytes: bytes, media_type: str) -> tuple[str, float]:
    result = gemini.classify_image(image_bytes, media_type)
    label = result.get("label", "normal")
    if label not in config.ALLOWED_CONDITIONS:
        label = "normal"
    confidence = float(result.get("confidence", 0)) / 100.0
    return label, confidence
