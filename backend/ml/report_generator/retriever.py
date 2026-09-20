"""
retriever.py — Step 2: Local vector retrieval over the seed knowledge base.

Uses sentence-transformers/all-MiniLM-L6-v2 for query embedding and
cosine similarity against the pre-built vector index. No external API
required — works entirely offline.
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

_KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
_INDEX_PATH = os.path.join(_KNOWLEDGE_DIR, "vector_index.npz")
_CHUNKS_PATH = os.path.join(_KNOWLEDGE_DIR, "chunks.json")
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level cache to avoid reloading on every call
_model = None
_embeddings = None
_chunks = None


def _load():
    """Lazy-load the embedding model, index, and chunks."""
    global _model, _embeddings, _chunks
    
    if _model is not None:
        return
    
    if not os.path.exists(_INDEX_PATH) or not os.path.exists(_CHUNKS_PATH):
        # Auto-build index if it doesn't exist yet
        from knowledge_base.build_index import build_index
        build_index()
    
    _model = SentenceTransformer(_MODEL_NAME)
    
    data = np.load(_INDEX_PATH)
    _embeddings = data["embeddings"]
    
    with open(_CHUNKS_PATH, "r", encoding="utf-8") as f:
        _chunks = json.load(f)


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Retrieve the top-k most relevant chunks for a query.
    
    Args:
        query: Natural language query string.
        k: Number of chunks to return (default: 3).
    
    Returns:
        List of dicts, each with keys: text, source, title, score.
    """
    _load()
    
    # Embed the query
    query_vec = _model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)
    
    # Cosine similarity (embeddings are already normalized)
    scores = np.dot(_embeddings, query_vec.T).flatten()
    
    # Get top-k indices
    top_k_idx = np.argsort(scores)[::-1][:k]
    
    results = []
    for idx in top_k_idx:
        results.append({
            "text": _chunks[idx]["text"],
            "source": _chunks[idx]["source"],
            "title": _chunks[idx]["title"],
            "score": float(scores[idx])
        })
    
    return results
