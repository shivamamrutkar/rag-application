# app.py
"""
Streamlit RAG demo using LangChain + Gemini + FAISS.
- Upload PDFs
- Ingest + chunk + embed -> FAISS (in-memory)
- Ask questions -> RetrievalQA with Gemini
"""

import os
import streamlit as st
from dotenv import load_dotenv
import asyncio

import nest_asyncio
nest_asyncio.apply()

if not asyncio.get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())


# LangChain / Gemini imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Setup ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Set GOOGLE_API_KEY in .env and restart. See README instructions.")
    st.stop()

st.set_page_config(page_title="RAG with Gemini", layout="wide")

# Sidebar controls
st.sidebar.header("RAG Controls")
model_choice = st.sidebar.selectbox("Gemini model", ["gemini-1.5-flash", "gemini-1.5-pro"])
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.1, 0.05)
max_output_tokens = st.sidebar.number_input("max_output_tokens", min_value=64, max_value=2048, value=512, step=64)
k = st.sidebar.slider("retriever k (top-k)", 1, 8, 3)
chunk_size = st.sidebar.number_input("chunk_size (chars)", min_value=200, max_value=2000, value=800, step=50)
chunk_overlap = st.sidebar.number_input("chunk_overlap (chars)", min_value=0, max_value=500, value=100, step=10)

st.title("RAG Chat — Gemini + LangChain + FAISS")
st.markdown("Upload PDF(s) then click `Ingest` to build the index. Ask questions in the chat box.")

# Session state to hold the vectorstore and conversation
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "docs_meta" not in st.session_state:
    st.session_state.docs_meta = []

# File uploader
uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("Ingest uploaded PDFs"):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Loading, chunking, embedding... (this may take a moment)"):
                # Instantiate embeddings and LLM (Gemini)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                llm = ChatGoogleGenerativeAI(model=model_choice,
                                             temperature=temperature,
                                             max_output_tokens=max_output_tokens)

                all_docs = []
                st.session_state.docs_meta = []

                for uploaded in uploaded_files:
                    # Save to temp path for PyPDFLoader which expects a path
                    tmp_path = os.path.join("temp_uploads", uploaded.name)
                    os.makedirs("temp_uploads", exist_ok=True)
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded.getbuffer())

                    # Load PDF and split into chunks
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    chunks = splitter.split_documents(docs)

                    # Keep metadata to show source pages later
                    for c in chunks:
                        # attach filename to metadata
                        c.metadata["source_file"] = uploaded.name
                    all_docs.extend(chunks)
                    st.session_state.docs_meta.append({"file": uploaded.name, "chunks": len(chunks)})

                # Build FAISS index
                if not all_docs:
                    st.error("No text extracted from PDFs.")
                else:
                    vectorstore = FAISS.from_documents(all_docs, embeddings)
                    st.session_state.vectorstore = vectorstore

                    # Create retriever + QA chain
                    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
                    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
                    st.session_state.qa_chain = qa_chain
                    st.success(f"Ingested {len(all_docs)} chunks from {len(uploaded_files)} files.")

with col2:
    if st.session_state.docs_meta:
        st.write("Ingest summary:")
        for m in st.session_state.docs_meta:
            st.write(f"- {m['file']}: {m['chunks']} chunks")
    else:
        st.info("No index in session. Upload and ingest PDFs to start.")

st.markdown("---")

# Chat UI
query = st.text_input("Ask a question about the uploaded documents:", value="", placeholder="e.g., What is the PTO policy?")
if st.button("Run Query") and query.strip():
    if not st.session_state.qa_chain:
        st.warning("No QA chain available. Ingest PDFs first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            # Update retriever's k in case user changed it after ingest
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k})
            st.session_state.qa_chain.retriever = retriever

            res = st.session_state.qa_chain(query)
            answer = res.get("result") or res.get("answer") or ""
            source_docs = res.get("source_documents", [])

        st.subheader("Answer")
        st.write(answer)

        if source_docs:
            st.subheader("Source snippets")
            for i, doc in enumerate(source_docs):
                meta = doc.metadata
