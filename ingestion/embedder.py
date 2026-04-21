"""
Vectorisation des chunks via nomic-embed-text (LM Studio).

Architecture :
- Appelle directement l'API HTTP /v1/embeddings de LM Studio via requests
  (compatible OpenAI) pour éviter les bugs de sérialisation de langchain-openai
  qui envoie parfois {"input": {"text": "..."}} au lieu de {"input": "..."}.
- Traite les chunks par batch pour éviter les timeouts sur gros volumes.
- Retourne les vecteurs dans le même ordre que les chunks d'entrée.
- Retry automatique (3 tentatives, backoff exponentiel 1s/2s/4s) sur chaque
  batch et sur les requêtes individuelles.

Prérequis LM Studio :
  1. Charger le modèle d'embedding (nomic-embed-text-v1.5) dans LM Studio
  2. Démarrer le serveur local (onglet "Local Server") → port 1234
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import requests

from app.config import EMBED_MODEL, LM_STUDIO_API_KEY, LM_STUDIO_BASE_URL, LLM_TIMEOUT

if TYPE_CHECKING:
    from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Taille des batches envoyés à LM Studio (évite les timeouts sur gros volumes)
_EMBED_BATCH_SIZE = 32
# Nombre maximal de tentatives en cas d'erreur transitoire
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
                    "LM Studio embedding : tentative %d/%d échouée (%s). "
                    "Nouvel essai dans %.1fs...",
                    attempt + 1, max_attempts, exc, delay,
                )
                time.sleep(delay)
    raise RuntimeError(
        f"Embedding échoué après {max_attempts} tentatives : {last_exc}"
    ) from last_exc


def embed_chunks(chunks: "list[Chunk]") -> list[list[float]]:
    """
    Calcule les embeddings de tous les chunks via LM Studio.

    Args:
        chunks: Liste de Chunk (produits par ingestion/chunker.py).

    Returns:
        Liste de vecteurs (float) dans le même ordre que l'entrée.

    Raises:
        ConnectionError: Si LM Studio n'est pas accessible.
        RuntimeError:    Si l'embedding échoue après toutes les tentatives.
    """
    if not chunks:
        return []

    texts = [c.text for c in chunks]
    vectors: list[list[float]] = []

    total = len(texts)
    logger.info("Embedding de %d chunks via '%s' (LM Studio)...", total, EMBED_MODEL)

    for batch_start in range(0, total, _EMBED_BATCH_SIZE):
        batch = texts[batch_start: batch_start + _EMBED_BATCH_SIZE]
        try:
            batch_vectors = _with_retry(lambda b=batch: _embed_via_http(b))
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
    try:
        results = _with_retry(lambda: _embed_via_http([query]))
        return results[0]
    except RuntimeError as exc:
        raise RuntimeError(f"Erreur d'embedding de la requête : {exc}") from exc


def check_lm_studio_connection() -> bool:
    """
    Vérifie que LM Studio est accessible et que le modèle d'embedding répond.

    Returns:
        True si tout est OK, False sinon.
    """
    try:
        result = _embed_via_http(["test connexion lm studio"])
        return len(result) > 0 and len(result[0]) > 0
    except Exception as exc:
        logger.warning("LM Studio inaccessible : %s", exc)
        return False


# Alias pour compatibilité avec les imports existants
check_ollama_connection = check_lm_studio_connection


# ── Interne ───────────────────────────────────────────────────────────────────

def _embed_via_http(texts: list[str]) -> list[list[float]]:
    """
    Appelle directement POST /v1/embeddings de LM Studio via requests.

    LM Studio attend : {"model": "...", "input": ["texte1", "texte2", ...]}
    Cette approche contourne les bugs de sérialisation de langchain-openai
    qui envoie parfois {"input": {"text": "..."}} au lieu de {"input": [...]}.

    Args:
        texts: Liste de textes à vectoriser (doit être non vide).

    Returns:
        Liste de vecteurs float dans le même ordre que l'entrée.

    Raises:
        ConnectionError: Si LM Studio n'est pas accessible.
        RuntimeError:    Si l'API retourne une erreur HTTP.
    """
    url = f"{LM_STUDIO_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {LM_STUDIO_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBED_MODEL,
        "input": texts,  # toujours une liste de strings — jamais un objet
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT)
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            f"Impossible de joindre LM Studio sur {url}. "
            "Vérifiez que le serveur est démarré (onglet 'Local Server' → Start Server)."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Timeout ({LLM_TIMEOUT}s) en attendant LM Studio ({url})."
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"LM Studio a retourné une erreur {response.status_code} : {response.text}"
        )

    data = response.json()
    # Trier par index pour garantir l'ordre (spec OpenAI)
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]
