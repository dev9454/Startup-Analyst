# tools/vectorstore.py
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tools.vertex_embed import embed_texts  # <-- new

class _VertexEmbeddings:
    """LangChain-style wrapper so FAISS.from_texts works."""
    def embed_documents(self, texts):
        return embed_texts(texts)
    def embed_query(self, text):
        return embed_texts([text])[0]

EMB = _VertexEmbeddings()

def build_index(docs, chunk_size: int = 1400, overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    splits = splitter.split_documents(docs)
    texts = [d.page_content for d in splits]
    metadatas = [d.metadata for d in splits]
    return FAISS.from_texts(texts, embedding=EMB, metadatas=metadatas)
