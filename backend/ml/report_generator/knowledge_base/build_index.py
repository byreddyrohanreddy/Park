"""
build_index.py — Embeds and indexes seed_documents.md into a local vector store.

Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings (no API key required).
Chunks are split by the '---' delimiter in seed_documents.md.
Index is saved as a numpy .npz file for fast loading.
"""

import os
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer

SEED_DOC_PATH = os.path.join(os.path.dirname(__file__), "seed_documents.md")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "vector_index.npz")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "chunks.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_chunks(filepath: str) -> list[dict]:
    """Parse seed_documents.md into labeled chunks.
    
    Each chunk is delimited by '---' and must contain a '## CHUNK:' header.
    Returns list of {title, source, text} dicts.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split on horizontal rules
    raw_sections = re.split(r"\n---\n", content)
    
    chunks = []
    for section in raw_sections:
        section = section.strip()
        if not section or "## CHUNK:" not in section:
            continue
        
        # Extract title
        title_match = re.search(r"## CHUNK:\s*(.+)", section)
        title = title_match.group(1).strip() if title_match else "Untitled"
        
        # Extract source
        source_match = re.search(r"\*\*Source:\*\*\s*(.+)", section)
        source = source_match.group(1).strip() if source_match else "Unknown"
        
        # Extract body text (everything after source line)
        lines = section.split("\n")
        body_lines = []
        past_source = False
        for line in lines:
            if "**Source:**" in line:
                past_source = True
                continue
            if past_source and line.strip():
                body_lines.append(line.strip())
        
        text = " ".join(body_lines)
        
        chunks.append({
            "title": title,
            "source": source,
            "text": text
        })
    
    return chunks


def build_index():
    """Build the vector index from seed documents."""
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Parsing chunks from: {SEED_DOC_PATH}")
    chunks = parse_chunks(SEED_DOC_PATH)
    print(f"Found {len(chunks)} chunks.")
    
    # Embed all chunk texts
    texts = [c["text"] for c in chunks]
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Save embeddings
    np.savez(INDEX_PATH, embeddings=embeddings)
    print(f"Saved vector index to: {INDEX_PATH}")
    
    # Save chunk metadata
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved chunk metadata to: {CHUNKS_PATH}")
    
    return chunks, embeddings


if __name__ == "__main__":
    build_index()
