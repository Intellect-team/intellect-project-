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
GEMINI_VISION_MODEL = "gemini-2.5-flash"  # used for the classifier placeholder, see services/classifier.py

KNOWLEDGE_DIR = BASE_DIR / "knowledge"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
PROMPT_PATH = BASE_DIR / "app" / "prompts" / "first_aid_prompt.txt"

ALLOWED_CONDITIONS = ["burn", "cut", "bleeding", "choking", "bruise", "abrasion", "normal"]

if not NVIDIA_API_KEY:
    print("[warning] NVIDIA_API_KEY not set — Nemotron calls will fail until it is.")
if not GEMINI_API_KEY:
    print("[warning] GEMINI_API_KEY not set — the Gemini fallback and classifier placeholder will fail until it is.")
