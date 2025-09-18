# tools/vertex_embed.py
from __future__ import annotations
from typing import List
import math
from vertexai.language_models import TextEmbeddingModel
from config import (
    VERTEX_EMBED_MODEL,  # e.g., "textembedding-gecko@003"
    VERTEX_LOCATION,
    GCP_PROJECT,
)

# Conservative per-chunk char cap to avoid hitting 20k token per request limits.
# 4000 chars ~ 1000-1500 tokens roughly; well under model limits.
_HARD_CHUNK_CHARS = 4000
_MAX_BATCH = 16  # small batches = fewer gRPC payloads

def _chunk_text(t: str, max_chars: int = _HARD_CHUNK_CHARS) -> List[str]:
    if not t:
        return [""]
    if len(t) <= max_chars:
        return [t]
    out = []
    i = 0
    L = len(t)
    while i < L:
        j = min(i + max_chars, L)
        out.append(t[i:j])
        i = j
    return out

def _mean_pool(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        # guard against empty vecs
        if not v:
            continue
        for i in range(dim):
            acc[i] += v[i]
    n = max(1, len(vectors))
    return [x / n for x in acc]

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Returns one embedding per input text. Internally:
      - split oversized texts into safe chunks,
      - embed chunks in small batches,
      - average chunk vectors back to a single vector per original text.
    """
    model = TextEmbeddingModel.from_pretrained(VERTEX_EMBED_MODEL, 
                                               location=VERTEX_LOCATION, 
                                               project=GCP_PROJECT)

    # Pre-chunk every text aggressively to prevent 20k token errors.
    # Keep a mapping so we can pool chunk vectors per original text.
    all_chunks: List[str] = []
    idx_map: List[tuple[int, int]] = []  # (orig_idx, chunk_count_so_far) unused but kept for clarity
    chunk_slices: List[slice] = []       # per original text: slice range in all_chunks

    start = 0
    for i, t in enumerate(texts):
        chunks = _chunk_text(t or "")
        all_chunks.extend(chunks)
        end = start + len(chunks)
        chunk_slices.append(slice(start, end))
        start = end

    # Embed chunks in small batches
    chunk_embeddings: List[List[float]] = []
    for b in range(0, len(all_chunks), _MAX_BATCH):
        batch = all_chunks[b:b+_MAX_BATCH]
        # Vertex SDK expects a list of strings
        pred = model.get_embeddings(batch)  # returns list of Embedding
        # Extract floats (Embedding.values) for each chunk
        for emb in pred:
            # Some SDK versions expose "values", others "embedding.values"
            vec = getattr(emb, "values", None)
            if vec is None and hasattr(emb, "embedding"):
                vec = getattr(emb.embedding, "values", None)
            chunk_embeddings.append(list(vec or []))

    # Pool back to one vector per original text
    out: List[List[float]] = []
    for sl in chunk_slices:
        vecs = chunk_embeddings[sl]
        out.append(_mean_pool(vecs))
    return out
