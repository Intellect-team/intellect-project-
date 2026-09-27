"""Loads the FAISS index built by scripts/build_embeddings.py and retrieves the
most relevant knowledge-base passage(s) for a given condition/query."""
import json
from typing import List

import faiss
import numpy as np

from app import config
from app.services.embeddings import embed_query

_index = None
_chunks: List[dict] = []


def _load():
    global _index, _chunks
    if _index is not None:
        return
    index_path = config.VECTOR_STORE_DIR / "index.faiss"
    chunks_path = config.VECTOR_STORE_DIR / "chunks.json"
    if not index_path.exists() or not chunks_path.exists():
        raise RuntimeError(
            "Vector store not built yet. Run: python scripts/build_embeddings.py"
        )
    _index = faiss.read_index(str(index_path))
    _chunks = json.loads(chunks_path.read_text(encoding="utf-8"))


def retrieve(query: str, top_k: int = 2) -> str:
    """Returns the top matching passage(s) from the knowledge base, concatenated,
    ready to drop into the Nemotron prompt as 'the verified protocol'."""
    _load()
    vector = np.array([embed_query(query)], dtype="float32")
    faiss.normalize_L2(vector)
    scores, indices = _index.search(vector, top_k)
    passages = [_chunks[i]["text"] for i in indices[0] if i != -1]
    return "\n\n---\n\n".join(passages) if passages else "No matching protocol found."
