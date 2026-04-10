"""
Interface ChromaDB — stockage et requêtes vectorielles.

Design :
- Client PersistentClient (stockage sur disque dans data/chroma_db/).
- Collection unique "documents" avec métrique cosinus.
- Singleton : une seule instance client par processus.
- Méthodes : add_chunks, query, get_all, delete_doc, stats.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.config import CHROMA_COLLECTION_NAME, CHROMA_DIR

if TYPE_CHECKING:
    from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# ── Singleton ─────────────────────────────────────────────────────────────────
_client = None
_collection = None


def _get_collection():
    """Retourne la collection ChromaDB (crée le client si besoin)."""
    global _client, _collection
    if _collection is None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("chromadb est requis : pip install chromadb") from exc

        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB prêt : collection '%s' (%d entrées existantes)",
            CHROMA_COLLECTION_NAME,
            _collection.count(),
        )
    return _collection


# ── Écriture ──────────────────────────────────────────────────────────────────

def add_chunks(chunks: "list[Chunk]", vectors: list[list[float]]) -> None:
    """
    Stocke les chunks et leurs embeddings dans ChromaDB.

    Args:
        chunks:  Liste de Chunk (produits par ingestion/chunker.py).
        vectors: Embeddings correspondants (même ordre).

    Raises:
        ValueError: Si les listes n'ont pas la même longueur.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Nombre de chunks ({len(chunks)}) ≠ nombre de vecteurs ({len(vectors)})"
        )
    if not chunks:
        return

    collection = _get_collection()

    ids = [f"{c.doc_id}__{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [_sanitize_metadata(c.metadata) for c in chunks]

    # upsert : ré-ingestion d'un doc déjà présent → mise à jour
    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info("ChromaDB : %d chunks stockés (doc_id=%s)", len(chunks), chunks[0].doc_id)


# ── Requêtes ──────────────────────────────────────────────────────────────────

def query_similar(
    query_vector: list[float],
    top_k: int = 5,
    filter_doc_ids: list[str] | None = None,
) -> list[dict]:
    """
    Récupère les top_k chunks les plus proches du vecteur requête.

    Args:
        query_vector:   Vecteur de la question utilisateur.
        top_k:          Nombre de résultats à retourner.
        filter_doc_ids: Si fourni, filtre sur ces doc_id uniquement.

    Returns:
        Liste de dicts : {id, text, metadata, distance, doc_id, chunk_index}
        Triée par distance croissante (plus similaire en premier).
    """
    collection = _get_collection()

    where = {"doc_id": {"$in": filter_doc_ids}} if filter_doc_ids else None

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        hits.append({
            "id": doc_id,
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "doc_id": results["metadatas"][0][i].get("doc_id", ""),
            "chunk_index": results["metadatas"][0][i].get("chunk_index", i),
        })

    return hits


def get_all_embeddings(doc_id: str | None = None) -> dict:
    """
    Récupère tous les embeddings stockés (utile pour le topic modeling).

    Args:
        doc_id: Si fourni, filtre sur ce document uniquement.

    Returns:
        Dict {ids, embeddings, documents, metadatas}
    """
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return {"ids": [], "embeddings": [], "documents": [], "metadatas": []}

    where = {"doc_id": {"$eq": doc_id}} if doc_id else None

    results = collection.get(
        include=["embeddings", "documents", "metadatas"],
        where=where,
    )
    return results


# ── Suppression ───────────────────────────────────────────────────────────────

def delete_doc(doc_id: str) -> int:
    """
    Supprime tous les chunks d'un document.

    Returns:
        Nombre de chunks supprimés.
    """
    collection = _get_collection()

    # Récupérer les IDs des chunks de ce document
    results = collection.get(
        where={"doc_id": {"$eq": doc_id}},
        include=[],
    )
    ids_to_delete = results["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info("ChromaDB : %d chunks supprimés pour doc_id=%s", len(ids_to_delete), doc_id)

    return len(ids_to_delete)


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """Retourne des statistiques sur la collection."""
    collection = _get_collection()
    return {
        "total_chunks": collection.count(),
        "collection_name": CHROMA_COLLECTION_NAME,
        "persist_dir": str(CHROMA_DIR),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize_metadata(metadata: dict) -> dict:
    """
    ChromaDB n'accepte que str, int, float, bool dans les métadonnées.
    Convertit tout le reste en str.
    """
    clean = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean
