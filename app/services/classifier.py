"""
Image -> (condition, confidence).

Primary path: our trained YOLO11s-cls model at models/classifier.pt
(10 classes, 224x224 input, trained with Ultralytics).

Fallback path: Gemini vision, used only when
  - the local model could not be loaded, or
  - the local model's top-1 confidence is below CLASSIFIER_CONF_THRESHOLD
    (and CLASSIFIER_GEMINI_FALLBACK is on).
If Gemini also fails, we keep the local model's answer rather than erroring.

routes.py only calls classify(); nothing else needs to know which path ran.
"""
import io
import logging
import os
import threading

# Stop Ultralytics from pip-installing extra packages at request time on a server.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from PIL import Image, ImageOps

from app import config

log = logging.getLogger(__name__)

_model = None
_model_error: str | None = None
_lock = threading.Lock()

# Last classification's origin ("model", "model_low_conf", "gemini"); handy for debugging.
last_source: str | None = None


def load_model():
    """Load the YOLO model once. Safe to call repeatedly / from several threads.
    Returns the model, or None if it can't be loaded (error kept in _model_error)."""
    global _model, _model_error
    if _model is not None or _model_error is not None:
        return _model
    with _lock:
        if _model is not None or _model_error is not None:
            return _model
        try:
            from ultralytics import YOLO

            path = config.CLASSIFIER_MODEL_PATH
            if not path.exists():
                raise FileNotFoundError(f"classifier weights not found at {path}")
            model = YOLO(str(path), task="classify")
            unknown = set(model.names.values()) - set(config.MODEL_LABEL_MAP)
            if unknown:
                log.warning("Model classes with no MODEL_LABEL_MAP entry: %s", sorted(unknown))
            _model = model
            log.info("Loaded classifier %s with classes %s", path.name, model.names)
        except Exception as e:  # missing deps, bad file, etc.
            _model_error = f"{type(e).__name__}: {e}"
            log.error("Could not load local classifier, will use Gemini only: %s", _model_error)
    return _model


def model_status() -> dict:
    load_model()
    return {
        "loaded": _model is not None,
        "path": str(config.CLASSIFIER_MODEL_PATH),
        "classes": list(_model.names.values()) if _model is not None else [],
        "error": _model_error,
        "conf_threshold": config.CLASSIFIER_CONF_THRESHOLD,
        "gemini_fallback": config.CLASSIFIER_GEMINI_FALLBACK,
    }


class InvalidImageError(ValueError):
    """Uploaded bytes could not be decoded as an image."""


def _decode(image_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)  # phone photos: respect camera rotation
        return img.convert("RGB")
    except Exception as e:  # PIL errors, or ultralytics' HEIC patch failing
        raise InvalidImageError(str(e)) from e


def _classify_local(image_bytes: bytes) -> tuple[str, float]:
    img = _decode(image_bytes)
    result = _model.predict(img, imgsz=224, device="cpu", verbose=False)[0]
    idx = int(result.probs.top1)
    raw_label = _model.names[idx]
    label = config.MODEL_LABEL_MAP.get(raw_label, "normal")
    return label, float(result.probs.top1conf)


def _classify_gemini(image_bytes: bytes, media_type: str) -> tuple[str, float]:
    from app.services import gemini  # imported lazily so the model path works without a Gemini key

    result = gemini.classify_image(image_bytes, media_type)
    label = result.get("label", "normal")
    if label not in config.ALLOWED_CONDITIONS:
        label = "normal"
    confidence = float(result.get("confidence", 0)) / 100.0
    return label, confidence


def classify(image_bytes: bytes, media_type: str) -> tuple[str, float]:
    global last_source
    model = load_model()

    if model is None:
        last_source = "gemini"
        return _classify_gemini(image_bytes, media_type)

    label, confidence = _classify_local(image_bytes)
    last_source = "model"

    if confidence < config.CLASSIFIER_CONF_THRESHOLD and config.CLASSIFIER_GEMINI_FALLBACK:
        try:
            g_label, g_conf = _classify_gemini(image_bytes, media_type)
            if g_conf > confidence:
                last_source = "gemini"
                return g_label, g_conf
        except Exception as e:
            log.warning("Gemini fallback failed, keeping local prediction: %s", e)
        last_source = "model_low_conf"

    return label, confidence
