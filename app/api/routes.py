from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.schemas.first_aid import FirstAidResponse
from app.schemas.speech import TranscribeResponse, TTSRequest
from app.services import classifier, gemini, nvidia_llm, retriever, safety, speech_to_text, text_to_speech

router = APIRouter()

MEDIA_TYPES = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp",
    "heic": "image/heic", "heif": "image/heif",  # iPhone camera photos (decoded via pi-heif)
}
AUDIO_EXTENSIONS = {"wav", "flac", "ogg"}  # Riva's offline_recognize expects one of these containers


@router.get("/")
def root():
    return {"status": "ok"}


@router.get("/health")
def health():
    """Dedicated health-check path for hosting platforms (Render, Railway,
    Fly.io, etc.) that expect /health or /healthz rather than "/"."""
    return {"status": "ok"}


@router.get("/classifier/status")
def classifier_status():
    """Is the local classifier.pt loaded, and with which classes?"""
    return classifier.model_status()


def _read_image(image: UploadFile, raw: bytes) -> str:
    ext = (image.filename or "").rsplit(".", 1)[-1].lower()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 8MB for this demo).")
    return MEDIA_TYPES.get(ext, "image/jpeg")


def _classify_or_400(raw: bytes, media_type: str) -> tuple[str, float]:
    try:
        return classifier.classify(raw, media_type)
    except classifier.InvalidImageError:
        raise HTTPException(status_code=400, detail="Could not read that file as an image.")


@router.post("/classify")
async def classify_only(image: UploadFile = File(...)):
    """Classification only (no RAG / LLM) — for testing the model directly."""
    raw = await image.read()
    media_type = _read_image(image, raw)
    condition, confidence = _classify_or_400(raw, media_type)
    return {"condition": condition, "confidence": confidence, "classifier_source": classifier.last_source}


@router.post("/analyze", response_model=FirstAidResponse)
async def analyze(image: UploadFile = File(...)):
    raw = await image.read()
    media_type = _read_image(image, raw)

    # 1. Classify (local YOLO11 model, Gemini only as low-confidence fallback)
    condition, confidence = _classify_or_400(raw, media_type)

    if condition == "normal":
        return FirstAidResponse(
            condition="normal",
            confidence=confidence,
            steps=[],
            warnings=[],
            seek_emergency_help=False,
            uncertainty_message="No clear first-aid situation was recognized in this photo.",
            source="classifier",
        )

    # 2. Retrieve trusted protocol (RAG)
    try:
        context = retriever.retrieve(condition.replace("_", " "))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Generate guidance — Nemotron first, Gemini as fallback
    try:
        response = nvidia_llm.generate_guidance(condition, confidence, context)
    except Exception as nvidia_error:
        try:
            response = gemini.generate_guidance(condition, confidence, context)
        except Exception as gemini_error:
            raise HTTPException(
                status_code=502,
                detail=f"Both Nemotron and Gemini failed. Nemotron: {nvidia_error}. Gemini: {gemini_error}",
            )

    # 4. Optional safety validation
    response = safety.validate(response)

    return response


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...), language_code: str = Form("en-US")):
    ext = (audio.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type '.{ext}'. Use one of: {sorted(AUDIO_EXTENSIONS)}",
        )

    raw = await audio.read()

    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio too large (max 20MB for this demo).")

    try:
        transcript = speech_to_text.transcribe(raw, language_code=language_code)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ASR request failed: {e}")

    return TranscribeResponse(transcript=transcript, language_code=language_code)


@router.post("/speak")
async def speak(payload: TTSRequest):
    try:
        audio_bytes = text_to_speech.synthesize(payload.text, language_code=payload.language_code)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS request failed: {e}")

    return Response(content=audio_bytes, media_type="audio/wav")
