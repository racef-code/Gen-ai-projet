# Intelligent Document Analysis System
### Local RAG-Powered Document Intelligence — 100% Private, No Cloud APIs

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [System Requirements](#3-system-requirements)
4. [Installing Ollama](#4-installing-ollama)
5. [Project Installation](#5-project-installation)
6. [Running the Application](#6-running-the-application)
7. [Folder Structure](#7-folder-structure)
8. [Configuration](#8-configuration)
9. [How to Use](#9-how-to-use)
10. [Dependencies Reference](#10-dependencies-reference)

---

## 1. Project Overview

This project is a **fully local, privacy-preserving intelligent document analysis system** built with Python and Streamlit. It allows users to upload documents (PDF, Word, TXT, Markdown) or scrape web pages, and then interact with that content through a conversational AI interface — without sending any data to external cloud services.

The system is built on three pillars:

- **RAG (Retrieval-Augmented Generation):** Instead of relying on a generic LLM, the system retrieves the most relevant passages from your own documents and uses them as context for the LLM's answer. This ensures responses are grounded in your actual content.
- **Local LLM inference via Ollama:** All AI computations (embeddings and text generation) run locally on your machine using Ollama. No OpenAI, no Anthropic, no data leaves your computer.
- **Incremental analytics:** After each document ingestion, the system automatically updates a topic model (BERTopic) and provides an analytics dashboard with five interactive visualizations.

**Tech stack at a glance:**

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM Orchestration | LangChain + ChatOllama |
| Embeddings | nomic-embed-text (via Ollama) |
| Vector Database | ChromaDB (persistent, on-disk) |
| Metadata Storage | SQLite (via SQLAlchemy Core) |
| Topic Modeling | BERTopic + UMAP + HDBSCAN |
| Visualizations | Plotly + WordCloud (PIL) |

---

## 2. Key Features

### Document Ingestion
- Upload **PDF**, **DOCX**, **TXT**, and **Markdown** files via drag-and-drop
- Scrape any **web URL** with automatic noise removal (navigation bars, footers, ads)
- File size validation (configurable limit, default 50 MB)
- Text preview before committing to ingestion
- Two chunking strategies: **paragraph-based** (semantic) and **sliding window** (fixed-size with overlap)
- **SHA-256 deduplication**: identical documents are detected and rejected automatically

### RAG Chat Interface
- Ask natural-language questions about your ingested documents
- **Transparency panel**: shows exactly which document excerpts were used to generate each answer, with cosine similarity scores
- Filter queries to specific documents
- Conversation history persisted in SQLite

### Analytics Dashboard (5 visualizations)
1. **Word Frequency Bar Chart** — top-N most frequent words with configurable stopword filtering (199 FR + EN stopwords built-in, no NLTK required)
2. **Word Cloud** — proportional word visualization using the wordcloud library
3. **Topic Clusters (UMAP 2D)** — BERTopic-generated semantic clusters projected to 2D using UMAP, with LLM-generated topic labels
4. **Topic Timeline** — stacked bar chart showing which topics appear in which documents
5. **Document Similarity Heatmap** — cosine similarity matrix between all ingested documents

### Persistence & Incremental Updates
- Full state restored on application restart (SQLite + ChromaDB + pickle)
- **Incremental topic model updates**: only new chunks are processed with `partial_fit`, embeddings are never recalculated
- Automatic LLM-powered labeling of newly detected topic clusters

---

## 3. System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Windows 10, macOS 12, Linux | Linux / macOS |
| Python | 3.10 | 3.11+ |
| RAM | 8 GB | 16 GB |
| Disk space | 10 GB free | 20 GB free |
| GPU | Not required | NVIDIA GPU (speeds up Ollama) |

> **Note on RAM:** Running `llama3` locally requires approximately 5–6 GB of RAM. `nomic-embed-text` requires an additional ~500 MB. Ensure you have enough free memory before starting.

---

## 4. Installing Ollama

Ollama is the local LLM runtime that powers both the embedding model and the text generation model. It must be installed and running before launching the application.

### Step 1 — Install Ollama

**macOS / Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download and run the installer from [https://ollama.com/download](https://ollama.com/download)

### Step 2 — Start the Ollama server

Open a terminal and run:
```bash
ollama serve
```

Keep this terminal open. The server runs on `http://localhost:11434` by default.

### Step 3 — Pull the required models

Open a **second terminal** and download both models:

```bash
# Embedding model (required for document indexing and search)
ollama pull nomic-embed-text

# Language model (required for chat responses and topic labeling)
ollama pull llama3
```

> **Alternative LLMs:** If `llama3` is too heavy for your machine, you can use `mistral` (smaller, faster) or `phi3` (very lightweight). Update `LLM_MODEL` in `app/config.py` accordingly.

### Step 4 — Verify the installation

```bash
ollama list
```

Expected output (both models must appear):
```
NAME                    ID              SIZE    MODIFIED
llama3:latest           ...             4.7 GB  ...
nomic-embed-text:latest ...             274 MB  ...
```

---

## 5. Project Installation

### Step 1 — Clone the repository

```bash
git clone <repository-url>
cd Gen-ai-projet
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Create the environment
python -m venv venv

# Activate it
# On macOS / Linux:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate.bat

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### Step 3 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Installation time:** Approximately 5–15 minutes depending on your internet speed. The largest packages are `torch` (pulled by BERTopic), `chromadb`, and `umap-learn`.

### Step 4 — Verify the installation

```bash
python -c "import streamlit, chromadb, bertopic, langchain_ollama; print('All dependencies OK')"
```

---

## 6. Running the Application

Make sure Ollama is running (`ollama serve` in a separate terminal) before proceeding.

```bash
# From the project root directory
streamlit run app/main.py
```

The application will automatically open in your browser at:
```
http://localhost:8501
```

**First-time startup:** The `data/` directory and all required subdirectories (`chroma_db/`, `topic_model/`) are created automatically on first launch.

---

## 7. Folder Structure

```
Gen-ai-projet/
│
├── app/                        Core application layer
│   ├── main.py                 Entry point: Streamlit config, page routing, sidebar
│   ├── config.py               All constants (paths, model names, chunking params)
│   └── state.py                Session state manager + persistence restoration
│
├── agents/                     AI logic layer
│   ├── rag_agent.py            RAG pipeline: embed → retrieve → prompt → LLM
│   └── query_handler.py        Query validation, error handling, chat history
│
├── ui/                         Streamlit page renderers (display only, no business logic)
│   ├── ingestion_ui.py         Upload/scrape page with pipeline progress bar
│   ├── chat_ui.py              Chat interface with transparency panel
│   └── analytics_ui.py         5-tab analytics dashboard with caching
│
├── ingestion/                  Document processing pipeline
│   ├── loaders.py              File extractors: PDF (pypdf), DOCX (python-docx), TXT, MD
│   ├── scraper.py              Web scraper: BeautifulSoup, noise removal, content selection
│   ├── chunker.py              Text splitting: paragraph and sliding window strategies
│   └── embedder.py             Chunk vectorization via Ollama (nomic-embed-text, 768-dim)
│
├── analytics/                  Analysis and visualization modules
│   ├── text_stats.py           Word frequency counter with built-in FR/EN stopwords
│   ├── topic_model.py          BERTopic incremental model, UMAP, LLM auto-labeling
│   ├── similarity.py           Cosine similarity matrix (NumPy, no external dependencies)
│   └── visualizations.py       5 Plotly figures + PIL word cloud image
│
├── persistence/                Data storage layer
│   ├── vector_store.py         ChromaDB wrapper (HNSW index, cosine metric, on-disk)
│   ├── metadata_store.py       SQLite via SQLAlchemy Core (documents + chat history tables)
│   └── incremental.py          Post-ingestion orchestrator: dedup, topic update, stats
│
├── data/                       Runtime data (auto-created, excluded from git)
│   ├── chroma_db/              ChromaDB persistent storage (vector index + metadata)
│   ├── topic_model/            BERTopic state: topic_state.pkl (model + UMAP + labels)
│   └── metadata.db             SQLite database (document registry + chat history)
│
├── requirements.txt            Python package dependencies with version pins
└── README.md                   Internal technical documentation (developer reference)
```

### Module Dependency Rules

The project enforces a strict layered architecture — upper layers depend on lower layers, never the reverse:

```
ui/          →  agents/, ingestion/, analytics/, persistence/
agents/      →  ingestion/ (embedder), persistence/ (vector_store)
analytics/   →  persistence/ (vector_store, metadata_store)
ingestion/   →  app/ (config) only
persistence/ →  app/ (config) only
app/         →  no internal dependencies (base layer)
```

---

## 8. Configuration

All application parameters are centralized in `app/config.py`. No `.env` file or external configuration is needed for local deployment.

```python
# app/config.py — key parameters

# Ollama endpoints
OLLAMA_BASE_URL = "http://localhost:11434"   # Ollama server address
LLM_MODEL       = "llama3"                  # Generation model (swap for "mistral", "phi3", etc.)
EMBED_MODEL     = "nomic-embed-text"         # Embedding model (768-dimensional vectors)
OLLAMA_TIMEOUT  = 120                        # Request timeout in seconds

# File ingestion
MAX_FILE_SIZE_MB = 50                        # Maximum upload size per file (MB)

# Chunking
CHUNK_STRATEGY  = "paragraph"               # "paragraph" or "sliding_window"
CHUNK_SIZE      = 512                       # Approx. tokens per chunk (sliding_window only)
CHUNK_OVERLAP   = 64                        # Overlap between adjacent chunks (sliding_window only)
MIN_CHUNK_CHARS = 80                        # Discard chunks shorter than this (characters)

# Retrieval
RETRIEVAL_TOP_K = 5                         # Number of chunks retrieved per query

# Topic Modeling
MIN_TOPIC_SIZE  = 3                         # Minimum cluster size for BERTopic
```

**To switch LLM models**, change `LLM_MODEL` in `app/config.py` and make sure the model is pulled in Ollama:
```bash
ollama pull mistral    # then set LLM_MODEL = "mistral"
```

**Storage paths** are automatically resolved relative to the project root — no absolute paths need to be configured.

---

## 9. How to Use

### Recommended Workflow

**Step 1 — Ingest Documents (Ingestion page)**

1. Navigate to the **Ingestion** tab in the sidebar.
2. Upload one or more documents using the file uploader (PDF, DOCX, TXT, MD), or paste a URL in the **URL Web** tab.
3. Preview the extracted text and adjust the chunking strategy if needed.
4. Click **"Intégrer dans la base de connaissances"**.
5. A progress bar tracks the pipeline: chunking → embedding → ChromaDB → SQLite → topic update.

**Step 2 — Ask Questions (Chat page)**

1. Navigate to the **Chat & RAG** tab.
2. Type a question in the input box and press Enter.
3. The answer appears in the left panel; the **transparency panel** on the right shows the exact source excerpts used.
4. Optionally filter to specific documents using the multiselect dropdown.

**Step 3 — Explore Analytics (Tableau de Bord page)**

1. Navigate to the **Tableau de Bord** tab.
2. Use the five sub-tabs to explore word frequencies, the word cloud, UMAP topic clusters, the topic timeline, and the document similarity heatmap.
3. Click **"Actualiser les analyses"** to refresh visualizations after adding new documents.

---

## 10. Dependencies Reference

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.35.0 | Web UI framework |
| `langchain` | ≥0.2.0 | LLM orchestration framework |
| `langchain-community` | ≥0.2.0 | LangChain community integrations |
| `langchain-ollama` | ≥0.1.0 | LangChain adapter for Ollama |
| `pypdf` | ≥4.0.0 | PDF text extraction |
| `python-docx` | ≥1.1.0 | DOCX text extraction |
| `beautifulsoup4` | ≥4.12.0 | HTML parsing for web scraping |
| `requests` | ≥2.32.0 | HTTP client for web scraping |
| `lxml` | ≥5.2.0 | Fast HTML/XML parser (BeautifulSoup backend) |
| `chromadb` | ≥0.5.0 | Local vector database with HNSW index |
| `bertopic` | ≥0.16.0 | Topic modeling with incremental updates |
| `scikit-learn` | ≥1.5.0 | Machine learning utilities (used by BERTopic) |
| `umap-learn` | ≥0.5.6 | Dimensionality reduction for topic visualization |
| `river` | ≥0.21.0 | Online/incremental clustering (required by BERTopic `partial_fit`) |
| `plotly` | ≥5.22.0 | Interactive charts and visualizations |
| `wordcloud` | ≥1.9.0 | Word cloud image generation |
| `sqlalchemy` | ≥2.0.0 | SQL toolkit for SQLite metadata storage |
| `numpy` | ≥1.26.0 | Numerical computing (cosine similarity, embedding matrices) |
| `pandas` | ≥2.2.0 | Data manipulation for analytics |
| `tqdm` | ≥4.66.0 | Progress bars |

Install all dependencies in one command:
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

**"Ollama inaccessible" error on the ingestion page**
- Make sure `ollama serve` is running in a separate terminal.
- Verify the server is reachable: `curl http://localhost:11434`
- Check that `nomic-embed-text` is pulled: `ollama list`

**Streamlit page is blank or crashes on startup**
- Confirm you are in the project root directory when running `streamlit run app/main.py`.
- Check that all dependencies installed without errors: `pip install -r requirements.txt`

**Topic clusters tab shows "not enough data"**
- BERTopic requires a minimum number of chunks to form meaningful clusters (default: at least 5 chunks minimum across documents). Ingest more documents or longer documents.

**Very slow response times in the Chat tab**
- LLM inference speed depends on hardware. A GPU will significantly accelerate response times. If no GPU is available, consider using a lighter model: `ollama pull phi3` and set `LLM_MODEL = "phi3"` in `app/config.py`.

---

*Built with Python 3.11 — All AI inference runs locally via Ollama — No data is sent to any external service.*
