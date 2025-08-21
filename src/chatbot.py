# src/chatbot.py
import os
from typing import Tuple, List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

INDEX_DIR = "data/embeddings/vectorstore"

def load_vectorstore(index_dir: str = INDEX_DIR) -> FAISS:
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(
            f"Vectorstore not found at {index_dir}. Run: python src/ingest.py"
        )
    return FAISS.load_local(index_dir, embeddings=None, allow_dangerous_deserialization=True)

def make_llm(model_name: str, groq_api_key: str):
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing.")
    return ChatGroq(model_name=model_name, api_key=groq_api_key, temperature=0.2)

def make_chain(vs: FAISS, llm) -> ConversationalRetrievalChain:
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    # Grounding prompt (short & strict)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Use ONLY the provided context. If the answer is not in the context, say you don't know."),
            ("system", "Cite sources by filename and page when possible."),
            ("human", "{question}\n\nContext:\n{context}"),
        ]
    )
    # ConversationalRetrievalChain handles chat_history injection
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return chain

def ask(
    vs: FAISS,
    groq_api_key: str,
    model_name: str,
    question: str,
    history: List[Tuple[str, str]],
) -> Tuple[str, List[Dict[str, Any]]]:
    llm = make_llm(model_name, groq_api_key)
    chain = make_chain(vs, llm)
    result = chain.invoke({"question": question, "chat_history": history})
    answer: str = result["answer"]
    sources = []
    for d in result.get("source_documents", []):
        meta = d.metadata or {}
        sources.append(
            {
                "filename": meta.get("filename") or meta.get("source") or "Unknown",
                "page": meta.get("page"),
                "snippet": d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""),
            }
        )
    return answer, sources
