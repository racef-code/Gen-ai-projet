# Intelligent Document Analysis System
### Local RAG-Powered Document Intelligence — 100% Private, No Cloud APIs

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [System Requirements](#3-system-requirements)
4. [Setting Up LM Studio](#4-setting-up-lm-studio)
5. [Project Installation](#5-project-installation)
6. [Running the Application](#6-running-the-application)
7. [Folder Structure](#7-folder-structure)
8. [Configuration](#8-configuration)
9. [How to Use](#9-how-to-use)
10. [Dependencies Reference](#10-dependencies-reference)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Project Overview

This project is a **fully local, privacy-preserving intelligent document analysis system** built with Python and Streamlit. It allows users to upload documents (PDF, Word, TXT, Markdown) or scrape web pages, and then interact with that content through a conversational AI interface — without sending any data to external cloud services.

The system is built on three pillars:

- **RAG (Retrieval-Augmented Generation):** Instead of relying on a generic LLM, the system retrieves the most relevant passages from your own documents and uses them as context for the LLM's answer. This ensures responses are grounded in your actual content.
- **Local LLM inference via LM Studio:** All AI computations (embeddings and text generation) run locally on your machine. No OpenAI, no Anthropic, no data leaves your computer.
- **Incremental analytics:** After each document ingestion, the system automatically updates a topic model (BERTopic) and provides an analytics dashboard with five interactive visualizations.

**Tech stack at a glance:**

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM Orchestration | LangChain + ChatOpenAI (LM Studio backend) |
| LLM | Mistral 7B Instruct v0.3 Q4_K_M (via LM Studio) |
| Embeddings | nomic-embed-text-v1.5 (via LM Studio) |
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

| Requirement | This project (tested on) |
|---|---|
| OS | Windows 10/11, macOS 12+, Linux |
| Python | 3.10 or higher |
| RAM | **16 GB** (Mistral 7B Q4 uses ~4.4 GB + OS overhead) |
| CPU | Any modern x86-64 (AMD Ryzen / Intel Core) |
| GPU | Not required — runs 100% on CPU |
| Disk space | ~8 GB free (models + dependencies + data) |

> **CPU-only note:** Without a GPU, expect response times of **3–8 seconds per token** in the chat. This is normal — Mistral 7B Q4 is optimized to be the best quality/speed tradeoff for CPU inference on 16 GB RAM.

---

## 4. Setting Up LM Studio

LM Studio is a desktop application that runs LLM models locally and exposes an **OpenAI-compatible API** on port 1234. It must be running before launching the application.

### Step 1 — Download and install LM Studio

Go to **[https://lmstudio.ai](https://lmstudio.ai)** and download the installer for your OS (Windows / macOS / Linux).

Install and open LM Studio.

---

### Step 2 — Download the LLM model (Mistral 7B Q4)

1. In LM Studio, click the **Search** tab (magnifying glass icon on the left sidebar).
2. Search for: `mistral-7b-instruct`
3. Find **Mistral 7B Instruct v0.3** by MistralAI.
4. Select the file: **`mistral-7b-instruct-v0.3.Q4_K_M.gguf`** (~4.4 GB)
5. Click **Download**.

> **Why Q4_K_M?** It is the best quantization level for 16 GB RAM — good quality with minimal size. Q2 would be too degraded, Q8 would not fit comfortably alongside the embedding model.

---

### Step 3 — Download the Embedding model (nomic-embed-text)

1. Still in the **Search** tab, search for: `nomic-embed-text`
2. Find **nomic-embed-text-v1.5** by nomic-ai.
3. Select the **GGUF** version (~274 MB).
4. Click **Download**.

> The embedding model converts text chunks into numeric vectors for semantic search. It must be loaded alongside the LLM.

---

### Step 4 — Start the Local Server

1. In LM Studio, click the **Local Server** tab (the `<->` icon on the left sidebar).
2. In the **"Select a model to load"** dropdown at the top, select **Mistral 7B Instruct v0.3 Q4_K_M**.
3. Click **"Start Server"**.

The server is now running at:
```
http://localhost:1234
```

You should see a green **"Running"** indicator.

---

### Step 5 — Load the Embedding model (Multi-Model)

The application needs **two models active simultaneously**: the LLM for chat and the embedding model for document indexing.

1. In the Local Server tab, look for **"Load Model"** or the **"+" button** to add a second model.
2. Load **nomic-embed-text-v1.5**.
3. Both models should now appear as loaded in the server panel.

> **If LM Studio only allows one model at a time** (older version): load the embedding model first when ingesting documents, then switch to Mistral for chat. The app will work either way — it just requires the right model to be active for each task.

---

### Step 6 — Verify the server is working

Open a terminal and run:

```bash
curl http://localhost:1234/v1/models
```

Expected output (a JSON list of loaded models):
```json
{"object":"list","data":[{"id":"mistral-7b-instruct-v0.3","object":"model",...}]}
```

If you see this, LM Studio is ready.

---

## 5. Project Installation

### Step 1 — Clone the repository

```bash
git clone <repository-url>
cd Gen-ai-projet
```

### Step 2 — Create a virtual environment (strongly recommended)

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

You should see `(venv)` appear at the beginning of your terminal prompt.

### Step 3 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Installation time:** Approximately 5–15 minutes. The largest packages are `torch` (pulled by BERTopic), `chromadb`, and `umap-learn`.

### Step 4 — Verify the installation

```bash
python -c "import streamlit, chromadb, bertopic, langchain_openai; print('All dependencies OK')"
```

Expected output:
```
All dependencies OK
```

---

## 6. Running the Application

### Prerequisites checklist — do this BEFORE launching

- [ ] LM Studio is **open**
- [ ] Mistral 7B Q4 is **loaded** in LM Studio
- [ ] nomic-embed-text-v1.5 is **loaded** in LM Studio
- [ ] The local server is **started** (green "Running" in LM Studio, port 1234)
- [ ] Your Python virtual environment is **activated** (`(venv)` in terminal)

### Launch command

Open a terminal in the project root directory and run:

```bash
streamlit run app/main.py
```

The application opens automatically in your browser at:
```
http://localhost:8501
```

**First-time startup:** The `data/` directory and all subdirectories (`chroma_db/`, `topic_model/`) are created automatically — nothing to configure manually.

### Verify LM Studio is connected

On the **Ingestion** page, look at the sidebar — there is a **"Statut LM Studio"** section with a **"Vérifier la connexion"** button. Click it to confirm the app can reach LM Studio before ingesting documents.

---

## 7. Folder Structure

```
Gen-ai-projet/
│
├── app/                        Core application layer
│   ├── main.py                 Entry point: Streamlit config, page routing, sidebar
│   ├── config.py               All constants (LM Studio URL, model names, chunking params)
│   └── state.py                Session state manager + persistence restoration on restart
│
├── agents/                     AI logic layer
│   ├── rag_agent.py            RAG pipeline: embed → retrieve → prompt → LLM
│   └── query_handler.py        Query validation, error handling, chat history persistence
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
│   └── embedder.py             Chunk vectorization via LM Studio (nomic-embed-text, 768-dim)
│
├── analytics/                  Analysis and visualization modules
│   ├── text_stats.py           Word frequency counter with built-in FR/EN stopwords
│   ├── topic_model.py          BERTopic incremental model, UMAP, LLM auto-labeling
│   ├── similarity.py           Cosine similarity matrix (NumPy only)
│   └── visualizations.py       5 Plotly figures + PIL word cloud image
│
├── persistence/                Data storage layer
│   ├── vector_store.py         ChromaDB wrapper (HNSW index, cosine metric, on-disk)
│   ├── metadata_store.py       SQLite via SQLAlchemy Core (documents + chat history tables)
│   └── incremental.py          Post-ingestion orchestrator: dedup, topic update, stats
│
├── data/                       Runtime data (auto-created, excluded from git)
│   ├── chroma_db/              ChromaDB persistent storage (vector index)
│   ├── topic_model/            BERTopic state: topic_state.pkl
│   └── metadata.db             SQLite database (document registry + chat history)
│
├── requirements.txt            Python package dependencies
├── README.md                   Internal technical documentation (developer reference)
└── README2.md                  This file — setup and usage guide
```

---

## 8. Configuration

All parameters are in `app/config.py`. The values below match the current setup for **LM Studio + Mistral 7B Q4 on 16 GB RAM**.

```python
# app/config.py — current configuration

# LM Studio server (OpenAI-compatible API)
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"   # LM Studio default port
LM_STUDIO_API_KEY  = "lm-studio"                  # Any string — LM Studio ignores it
LLM_MODEL          = "mistral-7b-instruct-v0.3"   # Must match the model name shown in LM Studio
EMBED_MODEL        = "nomic-embed-text-v1.5"       # Must match the embedding model in LM Studio
LLM_TIMEOUT        = 180                           # Seconds — generous for CPU-only inference

# File ingestion limits
MAX_FILE_SIZE_MB = 50                              # Files larger than this are rejected

# Chunking
CHUNK_STRATEGY  = "paragraph"                      # "paragraph" or "sliding_window"
CHUNK_SIZE      = 512                              # Tokens per chunk (sliding_window only)
CHUNK_OVERLAP   = 64                               # Overlap tokens (sliding_window only)
MIN_CHUNK_CHARS = 80                               # Chunks shorter than this are discarded

# Retrieval
RETRIEVAL_TOP_K = 5                                # Chunks retrieved per chat query

# Topic Modeling
MIN_TOPIC_SIZE  = 3                                # Min documents per topic cluster
```

### Changing the model name

The `LLM_MODEL` value **must exactly match** the model identifier shown in LM Studio's server panel. If LM Studio shows `mistral-7b-instruct-v0.3.Q4_K_M`, update the config accordingly:

```python
LLM_MODEL = "mistral-7b-instruct-v0.3.Q4_K_M"
```

---

## 9. How to Use

### Recommended Workflow

**Step 1 — Ingest Documents (Ingestion page)**

1. Navigate to the **Ingestion** tab in the sidebar.
2. Upload one or more files using the file uploader (PDF, DOCX, TXT, MD), or paste a URL in the **URL Web** tab.
3. A text preview appears — verify the content was correctly extracted.
4. Choose a chunking strategy (leave **paragraph** for most documents).
5. Click **"Intégrer dans la base de connaissances"**.
6. A progress bar tracks: chunking → embedding → ChromaDB → SQLite → topic model update.

**Step 2 — Ask Questions (Chat page)**

1. Navigate to the **Chat & RAG** tab.
2. Type a question in French or English and press Enter.
3. The answer appears in the main panel; the **transparency panel** on the right shows the exact source excerpts used with similarity scores.
4. Use the document filter (multiselect) to restrict answers to specific files.

**Step 3 — Explore Analytics (Tableau de Bord page)**

1. Navigate to the **Tableau de Bord** tab.
2. Explore the five sub-tabs: word frequencies, word cloud, UMAP topic clusters, topic timeline, and document similarity heatmap.
3. Click **"Actualiser les analyses"** to refresh after adding new documents.

---

## 10. Dependencies Reference

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.35.0 | Web UI framework |
| `langchain` | ≥0.2.0 | LLM orchestration framework |
| `langchain-community` | ≥0.2.0 | LangChain community integrations |
| `langchain-openai` | ≥0.1.0 | OpenAI-compatible client → used with LM Studio |
| `pypdf` | ≥4.0.0 | PDF text extraction |
| `python-docx` | ≥1.1.0 | DOCX text extraction |
| `beautifulsoup4` | ≥4.12.0 | HTML parsing for web scraping |
| `requests` | ≥2.32.0 | HTTP client for web scraping |
| `lxml` | ≥5.2.0 | Fast HTML/XML parser (BeautifulSoup backend) |
| `chromadb` | ≥0.5.0 | Local vector database with HNSW index |
| `bertopic` | ≥0.16.0 | Topic modeling with incremental updates |
| `scikit-learn` | ≥1.5.0 | Machine learning utilities (used by BERTopic) |
| `umap-learn` | ≥0.5.6 | Dimensionality reduction for topic visualization |
| `river` | ≥0.21.0 | Online/incremental clustering (BERTopic `partial_fit`) |
| `plotly` | ≥5.22.0 | Interactive charts and visualizations |
| `wordcloud` | ≥1.9.0 | Word cloud image generation |
| `sqlalchemy` | ≥2.0.0 | SQL toolkit for SQLite metadata storage |
| `numpy` | ≥1.26.0 | Numerical computing (cosine similarity, embedding matrices) |
| `pandas` | ≥2.2.0 | Data manipulation for analytics |
| `tqdm` | ≥4.66.0 | Progress bars |

---

## 11. Troubleshooting

### "LM Studio inaccessible" on the Ingestion page

**Cause:** The app cannot reach `http://localhost:1234`.

**Fix:**
1. Open LM Studio.
2. Go to the **Local Server** tab.
3. Make sure a model is loaded in the dropdown.
4. Click **Start Server** — the status should turn green.
5. Click **"Vérifier la connexion"** in the app sidebar to confirm.

---

### Embedding works but chat answers are wrong or empty

**Cause:** The wrong model may be active in LM Studio (embedding model loaded instead of LLM).

**Fix:**
- In LM Studio's server panel, confirm that **Mistral 7B Instruct** is the active model for `/v1/chat/completions`.
- If LM Studio only supports one model at a time, load Mistral for chat and nomic-embed-text only during ingestion.

---

### `LLM_MODEL` name mismatch error

**Cause:** The model name in `app/config.py` does not match what LM Studio reports.

**Fix:**
1. In LM Studio → Local Server, note the exact model identifier displayed (e.g., `mistral-7b-instruct-v0.3.Q4_K_M`).
2. Open `app/config.py` and update:
```python
LLM_MODEL = "mistral-7b-instruct-v0.3.Q4_K_M"   # exact match required
```

---

### Streamlit page is blank or crashes on startup

**Fix:**
- Confirm you are in the project root directory: `cd Gen-ai-projet`
- Confirm the virtual environment is active: `(venv)` must appear in the terminal prompt
- Reinstall dependencies: `pip install -r requirements.txt`

---

### Topic clusters tab shows "not enough data"

**Cause:** BERTopic needs a minimum number of text chunks to form clusters.

**Fix:** Ingest at least 2–3 documents of reasonable length (a few paragraphs each). Short single-sentence documents will not produce meaningful clusters.

---

### Very slow responses in the Chat tab

**Cause:** CPU-only inference on Mistral 7B is inherently slower than GPU-accelerated inference.

**Expected speed:** 3–8 tokens/second on a Ryzen 7 5800H — a full answer of ~200 tokens takes roughly 30–60 seconds.

**What you can do:**
- Be patient — this is normal for local CPU inference.
- Keep questions focused and concise to reduce answer length.
- If speed is critical, try a smaller model like **Phi-3 Mini** in LM Studio and update `LLM_MODEL` in `app/config.py`.

---

*Built with Python 3.11 — All AI inference runs locally via LM Studio — No data is sent to any external service.*
