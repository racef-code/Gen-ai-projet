"""
Orchestrateur de la mise à jour incrémentale.

Rôle :
1. **Déduplication** : calcule un hash SHA-256 du texte brut et vérifie
   si ce document existe déjà en base (évite les doubles ingestions).
2. **Trigger post-ingestion** : après chaque document ajouté avec succès,
   lance automatiquement la mise à jour incrémentale du topic model via
   `analytics.topic_model.fit_or_update()`.
3. **Auto-label** : si de nouveaux clusters apparaissent après la mise à
   jour, envoie une requête au LLM pour générer leurs labels.
4. **Notification** : retourne un résumé structuré des changements
   (nouveaux topics, labels) pour affichage dans l'UI.

Ce module est le "cœur" du Module D — il colle ensemble tous les autres.
"""
from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)


# ── Déduplication ─────────────────────────────────────────────────────────────

def compute_text_hash(text: str) -> str:
    """
    Calcule l'empreinte SHA-256 du texte normalisé.
    La normalisation (strip + lower) rend la détection robuste aux
    espaces de début/fin et à la casse.
    """
    normalized = text.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_duplicate(doc_hash: str) -> tuple[bool, str | None]:
    """
    Vérifie si un document avec ce hash existe déjà dans SQLite.

    Returns:
        (True, nom_du_doc)  si doublon trouvé.
        (False, None)        sinon.
    """
    from persistence.metadata_store import get_all_documents
    try:
        docs = get_all_documents()
        for doc in docs:
            if doc.get("doc_hash") == doc_hash:
                return True, doc.get("name", "?")
    except Exception as exc:
        logger.warning("Impossible de vérifier les doublons : %s", exc)
    return False, None


# ── Pipeline incrémental post-ingestion ───────────────────────────────────────

def run_incremental_update(session_state: dict) -> dict:
    """
    Déclenché automatiquement après chaque ingestion réussie.

    Séquence :
      1. fit_or_update()    → met à jour BERTopic avec les nouveaux chunks
      2. auto_label_topics() → appelle le LLM pour les nouveaux clusters

    Args:
        session_state: Le st.session_state Streamlit (dict-like).

    Returns:
        Dict de résumé :
          - success        (bool)
          - new_topics     (list[int])   : IDs des nouveaux topics
          - new_labels     (dict)        : {topic_id: label}
          - total_topics   (int)
          - error          (str | None)
    """
    result: dict = {
        "success": False,
        "new_topics": [],
        "new_labels": {},
        "total_topics": 0,
        "error": None,
    }

    try:
        from analytics.topic_model import fit_or_update, auto_label_topics
    except ImportError as exc:
        result["error"] = f"BERTopic non installé : {exc}"
        logger.warning(result["error"])
        return result

    # ── 1. Topic model update ──────────────────────────────────────────
    try:
        logger.info("Incremental : lancement de fit_or_update()...")
        state = fit_or_update(force_refit=False)
    except Exception as exc:
        result["error"] = f"Erreur topic modeling : {exc}"
        logger.error(result["error"])
        return result

    if state is None:
        result["error"] = "Pas assez de chunks pour le topic modeling."
        logger.info(result["error"])
        return result

    # Détecter les nouveaux topics (ceux absents de l'état précédent)
    previous_state = session_state.get("topic_model_state")
    old_topics = set(previous_state.topic_labels.keys()) if previous_state else set()
    current_topics = set(state.topic_labels.keys())

    try:
        if state.topic_model is not None:
            topic_info = state.topic_model.get_topic_info()
            all_topic_ids = set(
                row["Topic"] for _, row in topic_info.iterrows()
                if row["Topic"] != -1
            )
        else:
            all_topic_ids = current_topics
    except Exception:
        all_topic_ids = current_topics

    new_topic_ids = [t for t in all_topic_ids if t not in old_topics]

    # ── 2. Auto-label des nouveaux topics ─────────────────────────────
    new_labels: dict[int, str] = {}
    if new_topic_ids:
        logger.info("Incremental : %d nouveau(x) topic(s) détecté(s), auto-labelling...",
                    len(new_topic_ids))
        try:
            updated_labels = auto_label_topics(state)
            new_labels = {tid: updated_labels[tid] for tid in new_topic_ids
                         if tid in updated_labels}
        except Exception as exc:
            logger.warning("Auto-labelling partiel : %s", exc)

    # ── 3. Mise à jour du session_state ───────────────────────────────
    session_state["topic_model_state"] = state

    result.update({
        "success": True,
        "new_topics": new_topic_ids,
        "new_labels": new_labels,
        "total_topics": len(all_topic_ids),
    })

    logger.info(
        "Incremental terminé : %d topics au total, %d nouveaux.",
        result["total_topics"], len(new_topic_ids),
    )
    return result


# ── Statistiques globales ─────────────────────────────────────────────────────

def get_global_stats() -> dict:
    """
    Agrège les statistiques globales du système pour la sidebar.

    Returns:
        Dict {total_docs, total_chunks, total_chars, nb_topics}
    """
    from persistence.metadata_store import get_all_documents
    from persistence.vector_store import get_stats as chroma_stats

    stats = {
        "total_docs": 0,
        "total_chunks": 0,
        "total_chars": 0,
        "nb_topics": 0,
    }

    try:
        docs = get_all_documents()
        stats["total_docs"] = len(docs)
        stats["total_chunks"] = sum(d.get("nb_chunks", 0) for d in docs)
        stats["total_chars"] = sum(d.get("nb_chars", 0) for d in docs)
    except Exception as exc:
        logger.warning("Impossible de récupérer les stats docs : %s", exc)

    try:
        cs = chroma_stats()
        # Utiliser ChromaDB comme source de vérité pour le nombre de chunks
        stats["total_chunks"] = cs.get("total_chunks", stats["total_chunks"])
    except Exception:
        pass

    return stats
