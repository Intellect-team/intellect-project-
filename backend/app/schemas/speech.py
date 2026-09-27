from pydantic import BaseModel


class TranscribeResponse(BaseModel):
    transcript: str
    language_code: str = "en-US"


class TTSRequest(BaseModel):
    text: str
    language_code: str = "en-US"
