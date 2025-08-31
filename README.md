# 📘 Simple RAG App (Gemini + FAISS + LangChain)

A lightweight **Retrieval-Augmented Generation (RAG)** implementation using:

* **Google Gemini** for embeddings + text generation
* **FAISS** for efficient vector similarity search
* **LangChain** for pipeline management

This app lets you **upload a PDF and ask questions** about it, with Gemini grounding answers in the retrieved document chunks.

---

## 🔗 Link: [Live Demo](https://simple-rag.streamlit.app/)

## 🚀 Features

* 📑 **PDF ingestion** – extracts text from uploaded PDFs.
* ✂️ **Chunking** – splits text into chunks for retrieval.
* 🔍 **Vector store (FAISS)** – efficient semantic search over embeddings.
* 🤖 **Gemini LLM** – answers questions based on retrieved chunks.
* ⚡ **Simple workflow** – one script, no heavy UI, just pure RAG.

---

## 📂 Project Structure

```
simple-rag/
│
├── rag_app.py          # Main script
├── requirements.txt    # Dependencies
├── sample.pdf          # Example input PDF
└── README.md           # Documentation
```

---

## 🛠️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/simple-rag.git
cd simple-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
langchain
faiss-cpu
PyPDF2
google-generativeai
```

### 3. Set up API key

Export your Gemini API key as an environment variable:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

Or create a `.env` file with:

```
GOOGLE_API_KEY=your_api_key_here
```

---

## ▶️ Usage

Run the script:

```bash
python rag_app.py
```

You’ll be prompted to:

1. Upload a PDF
2. Ask a question
3. See Gemini’s answer (with retrieval grounding)

---

## 🧩 How It Works

1. **PDF → Text**: Extracts text from the PDF.
2. **Chunking**: Splits into chunks (`chunk_size`, `chunk_overlap`).
3. **Embeddings**: Converts chunks into embeddings via Gemini.
4. **Vector Store**: Stores embeddings in FAISS.
5. **Query → Retrieval**: User’s question is embedded and top-`k` chunks are retrieved.
6. **LLM Generation**: Gemini answers based on question + retrieved context.

---

## 📜 License

MIT License — free to use, modify, and learn from.
