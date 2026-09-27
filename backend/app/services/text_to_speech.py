"""
Text-to-speech via NVIDIA's hosted Riva TTS endpoint (same NVIDIA_API_KEY as
speech_to_text.py — ASR and TTS are separate Riva models behind the same gRPC
gateway, each addressed by its own function-id).

Requires: pip install nvidia-riva-client
"""
import io
import wave

import riva.client
from riva.client.proto.riva_audio_pb2 import AudioEncoding

from app import config

_auth = riva.client.Auth(
    uri=config.NVIDIA_TTS_SERVER,
    use_ssl=True,
    metadata_args=[
        ["function-id", config.NVIDIA_TTS_FUNCTION_ID],
        ["authorization", f"Bearer {config.NVIDIA_API_KEY}"],
    ],
)
_tts_service = riva.client.SpeechSynthesisService(_auth)


def synthesize(text: str, language_code: str = "en-US") -> bytes:
    """Converts text to speech and returns a complete WAV file as bytes.
    Raises RuntimeError on empty input or an empty result (mirrors
    speech_to_text.py's error-handling style)."""
    text = (text or "").strip()
    if not text:
        raise RuntimeError("No text provided to synthesize.")

    voice_name = config.NVIDIA_TTS_VOICES.get(
        language_code, config.NVIDIA_TTS_VOICES["en-US"]
    )
    sample_rate_hz = config.NVIDIA_TTS_SAMPLE_RATE

    response = _tts_service.synthesize(
        text,
        voice_name,
        language_code,
        sample_rate_hz=sample_rate_hz,
        encoding=AudioEncoding.LINEAR_PCM,
    )

    if not response.audio:
        raise RuntimeError("TTS returned no audio.")

    # Riva hands back raw PCM samples with no container — wrap them in a real
    # WAV file so the browser's <audio>/Audio() can play the response directly.
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate_hz)
        wf.writeframes(response.audio)

    return buffer.getvalue()
