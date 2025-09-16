# tools/vectorstore.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

EMB = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_index(docs, chunk_size: int = 1400, overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    splits = splitter.split_documents(docs)
    return FAISS.from_documents(splits, EMB)
