import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document
from utils import load_documents, chunk_documents

EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
INDEX_DIR = "data/embeddings/vectorstore"

def build_vectorstore(docs: List[Document]) -> FAISS:
    """Create a FAISS vectorstore"""

    if not docs:
        raise ValueError("No documents provided to build the vectorstore")
    
    chunks = chunk_documents(docs)
    emb =   OllamaEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_documents(chunks, emb)
    return vs

def save_vectorstore(vs: FAISS, path: str = INDEX_DIR) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vs.save_local(path)

## Testing
def main():
    print(" Loading documents from data/raw_docs/ ...")
    docs = load_documents("data/raw_docs/")
    if not docs:
        print(" No documents found in data/raw_docs/. Add PDFs/TXT/DOCX and rerun.")
        return

    print(f"Loaded {len(docs)} documents. Chunking + embedding with Ollama...")
    vs = build_vectorstore(docs)
    save_vectorstore(vs, INDEX_DIR)
    print(f"Vectorstore saved to: {INDEX_DIR}")

if __name__ == "__main__":
    main()