"""
Speech-to-text via NVIDIA's hosted Riva ASR endpoint (same NVIDIA_API_KEY as
nvidia_llm.py, but Riva speaks gRPC rather than the OpenAI-compatible REST API,
so it needs its own client here).

Requires: pip install nvidia-riva-client
"""
import io
import wave

import riva.client

from app import config

_auth = riva.client.Auth(
    uri=config.NVIDIA_ASR_SERVER,
    use_ssl=True,
    metadata_args=[
        ["function-id", config.NVIDIA_ASR_FUNCTION_ID],
        ["authorization", f"Bearer {config.NVIDIA_API_KEY}"],
    ],
)
_asr_service = riva.client.ASRService(_auth)


def _detect_sample_rate(audio_bytes: bytes, default: int = 16000) -> int:
    """Reads the sample rate out of a WAV header when possible; falls back to default
    for non-WAV formats (mp3/ogg/etc. — Riva will still attempt to decode them)."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            return wf.getframerate()
    except (wave.Error, EOFError):
        return default


def transcribe(audio_bytes: bytes, language_code: str = "en-US") -> str:
    """Sends raw audio bytes to NVIDIA's hosted ASR model and returns the transcript.
    Raises RuntimeError on empty/failed transcription so callers can decide how to
    surface it (mirrors retriever.py's error-handling style)."""
    if not audio_bytes:
        raise RuntimeError("No audio data received.")

    sample_rate = _detect_sample_rate(audio_bytes)

    recognize_config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        language_code=language_code,
        sample_rate_hertz=sample_rate,
        max_alternatives=1,
        enable_automatic_punctuation=True,
    )

    response = _asr_service.offline_recognize(audio_bytes, recognize_config)

    if not response.results:
        raise RuntimeError("ASR returned no results — check audio format (expects 16-bit mono WAV).")

    transcript = "".join(
        result.alternatives[0].transcript
        for result in response.results
        if result.alternatives
    ).strip()

    if not transcript:
        raise RuntimeError("ASR returned an empty transcript.")

    return transcript
