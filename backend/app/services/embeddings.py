"""Wraps NVIDIA's Nemotron-3-Embed-1B via its OpenAI-compatible endpoint."""
from typing import List

from openai import OpenAI

from app import config

_client = OpenAI(api_key=config.NVIDIA_API_KEY, base_url=config.NVIDIA_BASE_URL)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Returns one embedding vector per input string, same order."""
    response = _client.embeddings.create(
        model=config.NEMOTRON_EMBED_MODEL,
        input=texts,
        extra_body={"input_type": "passage"},  # NVIDIA's NeMo Retriever embed models expect this
    )
    return [item.embedding for item in response.data]


def embed_query(text: str) -> List[float]:
    """Embeds a single search query. NVIDIA's retrieval embed models distinguish
    query vs. passage embeddings for better retrieval quality — use this for the
    condition/query side and embed_texts() for the knowledge-base side."""
    response = _client.embeddings.create(
        model=config.NEMOTRON_EMBED_MODEL,
        input=[text],
        extra_body={"input_type": "query"},
    )
    return response.data[0].embedding
