# tools/vertex_embed.py
from __future__ import annotations
from typing import List
from google.cloud import aiplatform
from vertexai import init
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from config import GCP_PROJECT, GCP_LOCATION, VERTEX_EMBED_MODEL

_init_done = False
def _ensure_init():
    global _init_done
    if _init_done: 
        return
    init(project=GCP_PROJECT, location=GCP_LOCATION)
    _init_done = True

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Returns one embedding vector per input text using Vertex AI text-embedding-004."""
    _ensure_init()
    model = TextEmbeddingModel.from_pretrained(VERTEX_EMBED_MODEL)
    # task_type "RETRIEVAL_DOCUMENT" is good for chunk embeddings
    items = [TextEmbeddingInput(text=t[:8000], task_type="RETRIEVAL_DOCUMENT") for t in texts]
    res = model.get_embeddings(items)
    return [e.values for e in res]
