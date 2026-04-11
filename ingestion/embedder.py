"""
Vectorisation des chunks via nomic-embed-text (Ollama).

Architecture :
- Utilise langchain-ollama pour appeler le modèle d'embedding local.
- Traite les chunks par batch pour éviter de saturer Ollama.
- Retourne les vecteurs dans le même ordre que les chunks d'entrée.
- Gère le cas où Ollama n'est pas disponible avec un message d'erreur clair.
- Retry automatique (3 tentatives, backoff exponentiel 1s/2s/4s) sur chaque
  batch et sur les requêtes individuelles.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from app.config import EMBED_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT

if TYPE_CHECKING:
    from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Taille des batches envoyés à Ollama (évite les timeouts sur gros volumes)
_EMBED_BATCH_SIZE = 32
# Nombre maximal de tentatives en cas d'erreur transitoire Ollama
_MAX_RETRY = 3


def _with_retry(fn, max_attempts: int = _MAX_RETRY, base_delay: float = 1.0):
    """
    Exécute fn avec backoff exponentiel en cas d'échec.

    Tentatives : 1s → 2s → 4s (par défaut).
    Relance la dernière exception si toutes les tentatives échouent.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Ollama embedding : tentative %d/%d échouée (%s). "
                    "Nouvel essai dans %.1fs...",
                    attempt + 1, max_attempts, exc, delay,
                )
                time.sleep(delay)
    raise RuntimeError(
        f"Embedding échoué après {max_attempts} tentatives : {last_exc}"
    ) from last_exc


def embed_chunks(chunks: "list[Chunk]") -> list[list[float]]:
    """
    Calcule les embeddings de tous les chunks via Ollama.

    Args:
        chunks: Liste de Chunk (produits par ingestion/chunker.py).

    Returns:
        Liste de vecteurs (float) dans le même ordre que l'entrée.

    Raises:
        ConnectionError: Si Ollama n'est pas accessible.
        RuntimeError:    Si l'embedding échoue après toutes les tentatives.
    """
    if not chunks:
        return []

    embedder = _get_embedder()
    texts = [c.text for c in chunks]
    vectors: list[list[float]] = []

    total = len(texts)
    logger.info("Embedding de %d chunks via '%s'...", total, EMBED_MODEL)

    for batch_start in range(0, total, _EMBED_BATCH_SIZE):
        batch = texts[batch_start: batch_start + _EMBED_BATCH_SIZE]
        try:
            batch_vectors = _with_retry(lambda b=batch: embedder.embed_documents(b))
        except RuntimeError as exc:
            raise RuntimeError(
                f"Erreur d'embedding (batch {batch_start}–{batch_start + len(batch)}): {exc}"
            ) from exc

        vectors.extend(batch_vectors)
        logger.debug(
            "Batch %d/%d terminé (%d chunks)",
            batch_start // _EMBED_BATCH_SIZE + 1,
            (total + _EMBED_BATCH_SIZE - 1) // _EMBED_BATCH_SIZE,
            len(batch),
        )

    logger.info("Embedding terminé : %d vecteurs produits.", len(vectors))
    return vectors


def embed_query(query: str) -> list[float]:
    """
    Calcule l'embedding d'une requête utilisateur (texte unique).

    Args:
        query: Texte de la question.

    Returns:
        Vecteur d'embedding (float).
    """
    embedder = _get_embedder()
    try:
        return _with_retry(lambda: embedder.embed_query(query))
    except RuntimeError as exc:
        raise RuntimeError(f"Erreur d'embedding de la requête : {exc}") from exc


def check_ollama_connection() -> bool:
    """
    Vérifie qu'Ollama est accessible et que le modèle d'embedding est disponible.

    Returns:
        True si tout est OK, False sinon.
    """
    try:
        embedder = _get_embedder()
        # Test minimal : on embed une phrase courte
        result = embedder.embed_query("test connexion ollama")
        return len(result) > 0
    except Exception as exc:
        logger.warning("Ollama inaccessible : %s", exc)
        return False


# ── Interne ───────────────────────────────────────────────────────────────────

def _get_embedder():
    """
    Instancie OllamaEmbeddings (langchain-ollama).
    L'import est fait ici pour donner un message d'erreur explicite
    si la dépendance manque.
    """
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama est requis : pip install langchain-ollama"
        ) from exc

    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
        timeout=OLLAMA_TIMEOUT,
    )
