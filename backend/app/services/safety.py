"""
Final safety layer - plain-code rules that always run, no AI involved,
so they are fast, predictable and cannot be "talked out of" by a model.

1. Force the emergency banner for dangerous or unsure cases.
2. Chronic wounds -> "see a doctor soon" (no emergency banner by itself).
3. Block medicine / dose advice in the steps (but allow "Do not ..." warnings).
4. Never return empty guidance.
5. Never turn the emergency banner OFF - only ON.

An NVIDIA content-safety model can be added later (see ENABLE_NVIDIA_SAFETY_MODEL).
"""
import logging
import re

from app import config
from app.schemas.first_aid import FirstAidResponse

log = logging.getLogger(__name__)

ENABLE_NVIDIA_SAFETY_MODEL = False  # optional extra check, future work

EMERGENCY_CONDITIONS = {"choking", "unknown", "bleeding"}
SEE_DOCTOR_CONDITIONS = {"chronic_wound"}
SEVERE_LEVELS = {"severe", "critical", "high", "life-threatening"}

MEDICINE_PATTERN = re.compile(
    r"\b(\d+\s?(mg|ml|mcg)\b|dose|dosage|tablet|pill|capsule|inject|ibuprofen|"
    r"paracetamol|acetaminophen|aspirin|antibiotic|painkiller|pain reliever|"
    r"pain-reliever|nonprescription|medication|prescription)",
    re.IGNORECASE,
)
NEGATIVE_PREFIXES = ("do not", "don't", "dont", "never", "avoid")

EMERGENCY_STEP = "Call emergency services now - in Algeria: 14 (Civil Protection) or 115 (SAMU)."
SEE_DOCTOR_STEP = "Have this wound checked by a doctor or nurse soon - it may need professional treatment."
SAFE_FALLBACK_STEPS = [
    "Stay calm and keep the person safe and still.",
    "If there is bleeding, press firmly on the wound with a clean cloth.",
    "Do not give any medicine unless a medical professional tells you to.",
    "Stay with the person until help arrives and watch their breathing.",
]
UNCERTAIN_MESSAGE = "We couldn't identify this clearly. If you are worried, call emergency services."
NO_MEDICINE_WARNING = "Do not take or give any medicine without medical advice."


def _gives_medicine_advice(step: str) -> bool:
    """True if a step recommends medicine. 'Do not ...' warnings are allowed."""
    if step.strip().lower().startswith(NEGATIVE_PREFIXES):
        return False
    return bool(MEDICINE_PATTERN.search(step))


def validate(response: FirstAidResponse) -> FirstAidResponse:
    r = response.model_copy(deep=True)
    escalate = False
    reasons = []

    # 1) Dangerous or unknown condition
    if r.condition in EMERGENCY_CONDITIONS:
        escalate = True
        reasons.append(f"condition={r.condition}")

    # 2) Severe case
    if (r.severity or "").lower() in SEVERE_LEVELS:
        escalate = True
        reasons.append(f"severity={r.severity}")

    # 3) Low confidence -> escalate + explain uncertainty
    if r.confidence < config.CLASSIFIER_CONF_THRESHOLD:
        escalate = True
        reasons.append(f"low_confidence={r.confidence:.2f}")
        if not r.uncertainty_message:
            r.uncertainty_message = UNCERTAIN_MESSAGE

    # 4) Medicine / dose advice in steps -> replace with safe generic steps
    if any(_gives_medicine_advice(step) for step in r.steps):
        r.steps = SAFE_FALLBACK_STEPS.copy()
        if NO_MEDICINE_WARNING not in r.warnings:
            r.warnings.append(NO_MEDICINE_WARNING)
        escalate = True
        reasons.append("medicine_advice_blocked")

    # 5) Never return empty guidance (except for healthy skin)
    if not r.steps and r.condition != "normal":
        r.steps = SAFE_FALLBACK_STEPS.copy()
        escalate = True
        reasons.append("empty_steps")

    # 6) Chronic wounds -> always recommend a doctor visit
    if r.condition in SEE_DOCTOR_CONDITIONS:
        if not any("doctor" in s.lower() or "nurse" in s.lower() for s in r.steps):
            r.steps.append(SEE_DOCTOR_STEP)

    # 7) Banner can only be turned ON, never OFF
    if escalate or r.seek_emergency_help:
        r.seek_emergency_help = True
        if not any("emergency" in step.lower() for step in r.steps):
            r.steps.insert(0, EMERGENCY_STEP)

    if reasons:
        log.info("safety escalation: %s", ", ".join(reasons))

    # Optional future work: NVIDIA content-safety model check here.
    return r
