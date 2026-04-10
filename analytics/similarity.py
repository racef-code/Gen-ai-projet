"""
Similarité cosinus inter-documents.

Approche :
- Pour chaque document, calcule le vecteur moyen de ses chunks (centroïde).
- Calcule la matrice de similarité cosinus N×N entre tous les centroïdes.
- Retourne un DataFrame pandas (index = noms de documents).

Utilise uniquement numpy (pas de scikit-learn) pour rester léger.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_similarity_matrix(docs: list[dict]) -> pd.DataFrame | None:
    """
    Calcule la matrice de similarité cosinus entre documents.

    Args:
        docs: Liste de métadonnées de documents [{id, name, ...}]
              (issue de metadata_store.get_all_documents()).

    Returns:
        DataFrame N×N avec les noms de documents en index/colonnes,
        valeurs dans [0, 1]. Retourne None si < 2 documents.
    """
    from persistence.vector_store import get_all_embeddings

    if len(docs) < 2:
        logger.info("Moins de 2 documents — heatmap non disponible.")
        return None

    # ── Récupération des embeddings depuis ChromaDB ──────────────────────
    raw = get_all_embeddings()
    if not raw["ids"]:
        return None

    embeddings = np.array(raw["embeddings"], dtype=np.float64)
    metadatas = raw["metadatas"]

    # ── Calcul des centroïdes par document ────────────────────────────────
    doc_id_to_name = {d["id"]: _truncate_name(d["name"]) for d in docs}
    centroids: dict[str, np.ndarray] = {}

    for i, meta in enumerate(metadatas):
        doc_id = meta.get("doc_id", "")
        if doc_id not in doc_id_to_name:
            continue
        if doc_id not in centroids:
            centroids[doc_id] = []
        centroids[doc_id].append(embeddings[i])

    if len(centroids) < 2:
        return None

    # Moyenne des vecteurs de chaque document
    centroid_matrix = []
    ordered_names = []
    for doc in docs:
        doc_id = doc["id"]
        if doc_id in centroids:
            vecs = np.array(centroids[doc_id])
            centroid_matrix.append(vecs.mean(axis=0))
            ordered_names.append(doc_id_to_name[doc_id])

    if len(centroid_matrix) < 2:
        return None

    centroid_matrix = np.array(centroid_matrix)

    # ── Similarité cosinus ────────────────────────────────────────────────
    sim = _cosine_similarity(centroid_matrix)

    return pd.DataFrame(sim, index=ordered_names, columns=ordered_names)


def get_doc_centroids(docs: list[dict]) -> dict[str, np.ndarray]:
    """
    Retourne les centroïdes (vecteur moyen) pour chaque document.
    Utile pour d'autres calculs analytiques.

    Returns:
        Dict {doc_id: centroid_vector}
    """
    from persistence.vector_store import get_all_embeddings

    raw = get_all_embeddings()
    if not raw["ids"]:
        return {}

    embeddings = np.array(raw["embeddings"], dtype=np.float64)
    metadatas = raw["metadatas"]
    doc_ids_set = {d["id"] for d in docs}

    accumulator: dict[str, list] = {}
    for i, meta in enumerate(metadatas):
        doc_id = meta.get("doc_id", "")
        if doc_id in doc_ids_set:
            accumulator.setdefault(doc_id, []).append(embeddings[i])

    return {
        doc_id: np.array(vecs).mean(axis=0)
        for doc_id, vecs in accumulator.items()
    }


# ── Calcul ────────────────────────────────────────────────────────────────────

def _cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice de similarité cosinus N×N.

    Args:
        matrix: Matrice N×D (N documents, D dimensions).

    Returns:
        Matrice N×N, valeurs clampées dans [0, 1].
    """
    # Normalisation L2 de chaque vecteur
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Éviter la division par zéro
    norms = np.where(norms == 0, 1e-10, norms)
    normalized = matrix / norms

    # Produit scalaire = cosinus (entre −1 et 1)
    sim = normalized @ normalized.T

    # Clamp dans [0, 1] (les vecteurs d'embeddings sont généralement positifs)
    return np.clip(sim, 0.0, 1.0)


def _truncate_name(name: str, max_len: int = 25) -> str:
    """Tronque les noms longs pour l'affichage."""
    if len(name) <= max_len:
        return name
    return name[:max_len - 3] + "..."
