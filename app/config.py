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

# ── LM Studio ────────────────────────────────────────────────────────────────
# LM Studio expose une API compatible OpenAI sur le port 1234.
# 1. Ouvrir LM Studio → onglet "Local Server" → charger le modèle → Start Server
# 2. Le modèle d'embedding doit aussi être chargé (onglet "Multi-Model" ou séparément)
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LM_STUDIO_API_KEY  = "lm-studio"          # Ignorée par LM Studio, requise par le client OpenAI
LLM_MODEL   = "mistral-7b-instruct-v0.3"  # Doit correspondre au nom exact dans LM Studio
EMBED_MODEL = "nomic-embed-text-v1.5"     # Modèle d'embedding chargé dans LM Studio

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
LLM_TIMEOUT = 180              # secondes (CPU-only sur Ryzen 7 = plus lent, marge large)
MAX_FILE_SIZE_MB = 50          # taille max d'un fichier uploadé (Mo)

# ── Scraper ───────────────────────────────────────────────────────────────────
SCRAPER_TIMEOUT = 15           # secondes
SCRAPER_USER_AGENT = (
    "Mozilla/5.0 (compatible; DocAnalysisBot/1.0)"
)

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE = "Système d'Analyse Intelligente de Documents"
PREVIEW_MAX_CHARS = 2000       # caractères affichés dans l'aperçu
