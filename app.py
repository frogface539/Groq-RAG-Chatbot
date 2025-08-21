# app.py
import os
import tempfile
import streamlit as st
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from src.chatbot import load_vectorstore, ask
from src.utils import SUPPORTED_EXTS, chunk_documents
from langchain.docstore.document import Document

st.set_page_config(page_title="Groq RAG Chatbot", layout="wide")
st.title("💬 Groq RAG Chatbot ")

# Sidebar: settings
st.sidebar.header("⚙️ Settings")
groq_api_key = st.sidebar.text_input("GROQ API Key", type="password")
model_choice = st.sidebar.selectbox(
    "Groq Model",
    [
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma-7b-it",
    ],
    index=0,
)

# Vectorstore section
INDEX_DIR = "data/embeddings/vectorstore"

# Session state
if "history" not in st.session_state:
    st.session_state.history = []  # [(human, ai), ...]
if "vs" not in st.session_state:

    try:
        st.session_state.vs = load_vectorstore(INDEX_DIR)
        st.success("Loaded existing vectorstore.")
    except Exception as e:
        st.info("No vectorstore found yet. Ingest documents or upload files below.")
        st.session_state.vs = None

st.subheader("📂 Upload documents (PDF/TXT/DOCX)")
uploads = st.file_uploader(
    "Add files to the knowledge base",
    type=[e[1:] for e in SUPPORTED_EXTS],
    accept_multiple_files=True,
)

def _temp_docs_from_uploads(files) -> List[Document]:
    docs = []
    for f in files:
        suffix = os.path.splitext(f.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read())
            path = tmp.name

        if f.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif f.name.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif f.name.lower().endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue

        loaded = loader.load()
        for d in loaded:
            d.metadata = d.metadata or {}
            d.metadata["filename"] = f.name
        docs.extend(loaded)
    return docs

if uploads:
    st.write("⚙️ Indexing uploaded files with Ollama embeddings…")
    docs = _temp_docs_from_uploads(uploads)
    if docs:
        chunks = chunk_documents(docs)
        emb = OllamaEmbeddings(model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
        new_vs = FAISS.from_documents(chunks, emb)
        if st.session_state.vs is None:
            st.session_state.vs = new_vs
        else:
            st.session_state.vs.merge_from(new_vs)
        # Persist
        os.makedirs("data/embeddings", exist_ok=True)
        st.session_state.vs.save_local(INDEX_DIR)
        st.success(f"Indexed {len(chunks)} chunks. Vectorstore saved.")

st.subheader("🗣️ Ask a question")
q = st.text_input("Your question")

if st.button("Ask") or (q and st.session_state.get("_auto_submit") == q):
    if not groq_api_key:
        st.error("Please provide your GROQ API key in the sidebar.")
    elif st.session_state.vs is None:
        st.error("No knowledge base yet. Upload files or pre-ingest documents.")
    elif not q:
        st.warning("Type a question first.")
    else:
        answer, sources = ask(
            vs=st.session_state.vs,
            groq_api_key=groq_api_key,
            model_name=model_choice,
            question=q,
            history=st.session_state.history,
        )
        st.session_state.history.append((q, answer))
        st.write("### 🤖 Answer")
        st.write(answer)

        st.write("### 📚 Sources")
        for s in sources:
            line = f"- **{s.get('filename','Unknown')}**"
            if s.get("page") is not None:
                line += f" (p. {s['page']})"
            st.markdown(line)
            with st.expander("snippet"):
                st.write(s.get("snippet",""))

if st.session_state.history:
    st.markdown("---")
    st.write("### 🧠 Conversation History")
    for i, (u, a) in enumerate(st.session_state.history, 1):
        st.markdown(f"**You {i}:** {u}")
        st.markdown(f"**Bot {i}:** {a}")

col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear chat"):
        st.session_state.history = []
        st.rerun()
with col2:
    if st.button("🗑️ Reset vectorstore (delete index)"):
        import shutil
        try:
            shutil.rmtree(INDEX_DIR)
            st.session_state.vs = None
            st.success("Vectorstore deleted. Re-upload or re-ingest to rebuild.")
        except FileNotFoundError:
            st.info("No index to delete.")
