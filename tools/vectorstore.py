# tools/vectorstore.py
from __future__ import annotations
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings  # <-- important

# --- Embedding adapters -------------------------------------------------------

class VertexEmbeddings(Embeddings):
    """LangChain-compatible wrapper around Vertex AI text-embedding-004."""
    def __init__(self):
        # lazy import so app still runs if Vertex isn't configured
        from tools.vertex_embed import embed_texts
        self._embed_texts = embed_texts

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_texts([text])[0]


class HFEmbeddings(Embeddings):
    """Fallback to local HuggingFace embeddings (no GCP auth needed)."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # langchain-huggingface replaces deprecated import path
        from langchain_huggingface import HuggingFaceEmbeddings
        self._emb = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._emb.embed_query(text)


def _get_embeddings() -> Embeddings:
    """
    Try Vertex first; if ADC/Vertex not set up, fall back to local HF.
    Returns an Embeddings subclass instance.
    """
    try:
        # quick sanity import to detect Vertex availability
        from tools.vertex_embed import embed_texts  # noqa: F401
        return VertexEmbeddings()
    except Exception:
        return HFEmbeddings()

# --- Index builder ------------------------------------------------------------

def build_index(docs, chunk_size: int = 1400, overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    splits = splitter.split_documents(docs)
    texts = [d.page_content for d in splits]
    metadatas = [d.metadata for d in splits]

    emb = _get_embeddings()  # proper Embeddings object
    return FAISS.from_texts(texts, embedding=emb, metadatas=metadatas)
