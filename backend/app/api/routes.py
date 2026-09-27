from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.first_aid import FirstAidResponse
from app.services import classifier, gemini, nvidia_llm, retriever, safety

router = APIRouter()

MEDIA_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


@router.get("/")
def health():
    return {"status": "ok"}


@router.post("/analyze", response_model=FirstAidResponse)
async def analyze(image: UploadFile = File(...)):
    ext = (image.filename or "").rsplit(".", 1)[-1].lower()
    media_type = MEDIA_TYPES.get(ext, "image/jpeg")
    raw = await image.read()

    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 8MB for this demo).")

    # 1. Classify
    condition, confidence = classifier.classify(raw, media_type)

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
        context = retriever.retrieve(condition)
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
