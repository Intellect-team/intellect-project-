from typing import List, Optional

from pydantic import BaseModel


class FirstAidResponse(BaseModel):
    condition: str
    confidence: float
    severity: str = "unknown"
    steps: List[str] = []
    warnings: List[str] = []
    seek_emergency_help: bool = False
    uncertainty_message: Optional[str] = None
    source: str = "nemotron"  # "nemotron" or "gemini" — which model actually generated this response
