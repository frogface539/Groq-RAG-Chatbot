import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.docstore.document import Document

SUPPORTED_EXTS = {".pdf", ".txt", ".docx"}

def load_documents(raw_docs_path: str = "data/raw_docs/") -> List[Document]:

    """Load PDF/TXT/DOCX"""

    if not os.path.isdir(raw_docs_path):
        os.makedirs(raw_docs_path, exist_ok=True)

    docs: List[Document] = []
    for fname in os.listdir(raw_docs_path):
        fpath = os.path.join(raw_docs_path, fname)
        ext = os.path.splitext(fname)[1].lower()

        if not os.path.isfile(fpath) or ext not in SUPPORTED_EXTS:
            continue

        if ext == ".pdf":
            loader = PyPDFLoader(fpath)
        
        elif ext == ".txt":
            loader = TextLoader(fpath, encoding="utf-8")

        elif ext == ".docx":
            loader = Docx2txtLoader(fpath)
        else:
            continue

        loaded = loader.load()

        for d in loaded:
            d.metadata = d.metadata or {}
            d.metadata["filename"] = fname
        docs.extend(loaded)
    
    return docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    
    """Split documents into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    
    return chunks