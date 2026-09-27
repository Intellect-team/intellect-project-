"""
Image -> (condition, confidence).

1. First uses our trained wound classifier (YOLO11s-cls, fine-tuned on NVIDIA Brev),
   loaded from models/classifier/classifier.pt.
2. If the model is unsure (confidence < config.CLASSIFIER_CONF_THRESHOLD), missing,
   or fails, falls back to Gemini vision.
3. If everything fails, returns "unknown" so the safety layer escalates.

Nothing outside this file needs to change — routes.py only calls classify().
"""
import io
import logging

from PIL import Image

from app import config
from app.services import gemini

log = logging.getLogger(__name__)

_model = None
_model_failed = False


def _load_model():
    """Load the trained model once and reuse it. Returns None if unavailable."""
    global _model, _model_failed
    if _model is None and not _model_failed:
        try:
            from ultralytics import YOLO
            _model = YOLO(str(config.CLASSIFIER_MODEL_PATH))
            log.info("Loaded trained classifier from %s", config.CLASSIFIER_MODEL_PATH)
        except Exception as e:
            _model_failed = True
            log.warning("Could not load trained classifier (%s) — using Gemini only.", e)
    return _model


def _classify_with_model(image_bytes: bytes):
    """Run our trained model. Returns (label, confidence) or None if unavailable."""
    model = _load_model()
    if model is None:
        return None
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = model.predict(image, device="cpu", verbose=False)[0]
    class_name = result.names[int(result.probs.top1)]
    confidence = float(result.probs.top1conf)
    label = config.MODEL_TO_CONDITION.get(class_name, "unknown")
    return label, confidence


def _classify_with_gemini(image_bytes: bytes,
