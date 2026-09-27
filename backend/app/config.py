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
GEMINI_LLM_MODEL = "gemini-2.5-flash"
GEMINI_VISION_MODEL = "gemini-2.5-flash"  # fallback classifier when our trained model is unsure

KNOWLEDGE_DIR = BASE_DIR / "knowledge"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
PROMPT_PATH = BASE_DIR / "app" / "prompts" / "first_aid_prompt.txt"

ALLOWED_CONDITIONS = ["burn", "cut", "bleeding", "choking", "bruise", "abrasion",
                      "chronic_wound", "normal", "unknown"]

# Trained wound classifier (YOLO11s-cls, fine-tuned on NVIDIA Brev)
CLASSIFIER_MODEL_PATH = BASE_DIR / "models" / "classifier" / "classifier.pt"
CLASSIFIER_CONF_THRESHOLD = 0.6  # below this, fall back to Gemini vision

# Maps the trained model's class names to our app's condition labels
MODEL_TO_CONDITION = {
    "burns": "burn",
    "cuts": "cut",
    "laceration": "cut",
    "abrasions": "abrasion",
    "bruises": "bruise",
    "normal": "normal",
    "diabetic_wounds": "chronic_wound",
    "pressure_wounds": "chronic_wound",
    "ulcer_wounds": "chronic_wound",
    "venous_wounds": "chronic_wound",
}

if not NVIDIA_API_KEY:
    print("[warning] NVIDIA_API_KEY not set — Nemotron calls will fail until it is.")
if not GEMINI_API_KEY:
    print("[warning] GEMINI_API_KEY not set — Gemini fallback will fail until it is.")
if not CLASSIFIER_MODEL_PATH.exists():
    print(f"[warning] Classifier model not found at {CLASSIFIER_MODEL_PATH} — using Gemini only.")
