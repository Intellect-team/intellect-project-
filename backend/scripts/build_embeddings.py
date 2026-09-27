"""Run this once (and again any time you edit knowledge/*.md):
    python scripts/build_embeddings.py

Chunks each knowledge file by its ## sections, embeds each chunk with
Nemotron-3-Embed-1B, and writes a FAISS index + chunk lookup to vector_store/.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np

from app import config
from app.services.embeddings import embed_texts


def chunk_markdown(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    title = text.splitlines()[0].lstrip("# ").strip()
    sections = re.split(r"\n(?=## )", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section or section.startswith("# "):
            continue
        chunks.append(f"[{title}]\n{section}")
    return chunks


def main():
    config.VECTOR_STORE_DIR.mkdir(exist_ok=True)
    all_chunks = []
    for md_file in sorted(config.KNOWLEDGE_DIR.glob("*.md")):
        all_chunks.extend(chunk_markdown(md_file))

    if not all_chunks:
        raise RuntimeError(f"No markdown files found in {config.KNOWLEDGE_DIR}")

    print(f"Embedding {len(all_chunks)} chunks from {config.KNOWLEDGE_DIR}...")
    vectors = np.array(embed_texts(all_chunks), dtype="float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(config.VECTOR_STORE_DIR / "index.faiss"))
    (config.VECTOR_STORE_DIR / "chunks.json").write_text(
        json.dumps([{"text": c} for c in all_chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Wrote index + {len(all_chunks)} chunks to {config.VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()
