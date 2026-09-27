"""Central config — every other module reads settings from here, never os.environ directly."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

NEMOTRON_LLM_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
NEMOTRON_EMBED_MODEL = "nvidia/nemotron-3-embed-1b"

# Riva ASR (speech-to-text) — hosted on NVIDIA's gRPC endpoint, uses the same
# NVIDIA_API_KEY as Nemotron above. function-id is model-specific: grab the
# current one for your chosen ASR model from its "API Reference" tab at
# build.nvidia.com (e.g. search "parakeet" or "conformer-ctc-asr").
NVIDIA_ASR_SERVER = "grpc.nvcf.nvidia.com:443"
NVIDIA_ASR_FUNCTION_ID = os.environ.get(
    "NVIDIA_ASR_FUNCTION_ID", "1598d209-5e27-4d3c-8079-4751568b1081"
)

# Riva TTS (text-to-speech) — same NVIDIA_API_KEY, different function-id/model.
# Voice names are model-specific: check the "API Reference" tab for your chosen
# TTS model at build.nvidia.com (search "tts", e.g. "magpie-tts-multilingual" or
# "chatterbox-multilingual-tts") and use --list-voices to see valid voice names.
# NOTE: verify Arabic is actually supported by whichever TTS model you pick —
# not all hosted TTS models cover every language; the ar-US entry below falls
# back to an English voice until you confirm and swap in a real Arabic voice.
NVIDIA_TTS_SERVER = "grpc.nvcf.nvidia.com:443"
NVIDIA_TTS_FUNCTION_ID = os.environ.get(
    "NVIDIA_TTS_FUNCTION_ID", "877104f7-e885-42b9-8de8-f6e4c6303969"
)
NVIDIA_TTS_SAMPLE_RATE = 22050
NVIDIA_TTS_VOICES = {
    "en-US": os.environ.get("NVIDIA_TTS_VOICE_EN", "Magpie-Multilingual.EN-US.Sofia"),
    "fr-FR": os.environ.get("NVIDIA_TTS_VOICE_FR", "Magpie-Multilingual.FR-FR.Louise"),
    "ar-AR": os.environ.get("NVIDIA_TTS_VOICE_AR", "Magpie-Multilingual.EN-US.Sofia"),
}
GEMINI_LLM_MODEL = "gemini-2.5-flash"
GEMINI_VISION_MODEL = "gemini-2.5-flash"  # low-confidence / fallback classifier, see services/classifier.py

KNOWLEDGE_DIR = BASE_DIR / "knowledge"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
PROMPT_PATH = BASE_DIR / "app" / "prompts" / "first_aid_prompt.txt"

# --- Local image classifier (YOLO11s-cls, trained with Ultralytics) ---------
CLASSIFIER_MODEL_PATH = Path(
    os.environ.get("CLASSIFIER_MODEL_PATH", BASE_DIR / "models" / "classifier.pt")
)
# Below this top-1 confidence (0-1) we ask Gemini vision for a second opinion.
CLASSIFIER_CONF_THRESHOLD = float(os.environ.get("CLASSIFIER_CONF_THRESHOLD", "0.5"))
# Set to "0" to disable the Gemini second opinion entirely (fully offline classify).
CLASSIFIER_GEMINI_FALLBACK = os.environ.get("CLASSIFIER_GEMINI_FALLBACK", "1") != "0"

# The trained model's class folder names -> the condition ids used everywhere
# else in the app (knowledge base, prompts, frontends).
MODEL_LABEL_MAP = {
    "abrasions": "abrasion",
    "bruises": "bruise",
    "burns": "burn",
    "cuts": "cut",
    "laceration": "laceration",
    "diabetic_wounds": "diabetic_wound",
    "pressure_wounds": "pressure_wound",
    "ulcer_wounds": "ulcer_wound",
    "venous_wounds": "venous_wound",
    "normal": "normal",
}

ALLOWED_CONDITIONS = [
    "burn", "cut", "bleeding", "choking", "bruise", "abrasion",
    "laceration", "diabetic_wound", "pressure_wound", "ulcer_wound", "venous_wound",
    "normal",
]

if not NVIDIA_API_KEY:
    print("[warning] NVIDIA_API_KEY not set — Nemotron calls will fail until it is.")
if not GEMINI_API_KEY:
    print("[warning] GEMINI_API_KEY not set — the Gemini fallback and classifier placeholder will fail until it is.")
