"""Optional final validation layer using NVIDIA's content-safety models.
Per the handout: add this only after the core pipeline works end to end.
Currently a no-op passthrough — flip ENABLE_SAFETY_CHECK on once you're ready
to wire in an actual NVIDIA safety model call here."""
from app.schemas.first_aid import FirstAidResponse

ENABLE_SAFETY_CHECK = False


def validate(response: FirstAidResponse) -> FirstAidResponse:
    if not ENABLE_SAFETY_CHECK:
        return response
    # TODO: call an NVIDIA content-safety model on response.steps / response.warnings here.
    # If it flags the content, either regenerate or fall back to a generic
    # "seek professional help" response rather than showing unsafe guidance.
    return response
