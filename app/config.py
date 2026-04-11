"""
Configuration centrale de l'application.
Toutes les constantes et paramètres ajustables sont ici.
"""
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
TOPIC_MODEL_DIR = DATA_DIR / "topic_model"
DB_PATH = DATA_DIR / "metadata.db"

# Création automatique des dossiers au démarrage
for _dir in [DATA_DIR, CHROMA_DIR, TOPIC_MODEL_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3"           # ou "mistral", "phi3"
EMBED_MODEL = "nomic-embed-text"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "documents"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_STRATEGY = "paragraph"   # "paragraph" | "sliding_window"
CHUNK_SIZE = 512               # tokens (pour sliding_window)
CHUNK_OVERLAP = 64             # tokens (pour sliding_window)
MIN_CHUNK_CHARS = 80           # ignorer les chunks trop courts

# ── RAG ───────────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 5            # nombre de chunks retournés par le retriever

# ── Topic Modeling ────────────────────────────────────────────────────────────
MIN_TOPIC_SIZE = 3             # taille minimale d'un cluster BERTopic
NR_TOPICS = "auto"             # ou un entier fixe

# ── Timeout & Limites ────────────────────────────────────────────────────────
OLLAMA_TIMEOUT = 120           # secondes (timeout par requête Ollama)
MAX_FILE_SIZE_MB = 50          # taille max d'un fichier uploadé (Mo)

# ── Scraper ───────────────────────────────────────────────────────────────────
SCRAPER_TIMEOUT = 15           # secondes
SCRAPER_USER_AGENT = (
    "Mozilla/5.0 (compatible; DocAnalysisBot/1.0)"
)

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE = "Système d'Analyse Intelligente de Documents"
PREVIEW_MAX_CHARS = 2000       # caractères affichés dans l'aperçu
