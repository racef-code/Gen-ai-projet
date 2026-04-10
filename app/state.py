"""
Gestionnaire centralisé du session_state Streamlit.
Garantit que toutes les clés sont initialisées une seule fois.

Restauration complète au démarrage (Module D) :
  1. Documents  ← SQLite (metadata_store)
  2. Topic model ← pickle sur disque (data/topic_model/topic_state.pkl)
"""
import logging

import streamlit as st

logger = logging.getLogger(__name__)


def init_session_state() -> None:
    """
    Initialise les clés du session_state.
    Restaure l'état complet depuis les sources persistantes (SQLite + pickle)
    lors de la première exécution de la session.
    """
    if st.session_state.get("_state_initialized"):
        return

    defaults: dict = {
        "ingested_docs": [],
        "chunks_cache": [],
        "chat_history": [],
        "topic_model_state": None,
        "current_page": "ingestion",
        "last_ingestion_status": None,
        "last_ingestion_message": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # ── 1. Restauration des documents depuis SQLite ───────────────────────
    try:
        from persistence.metadata_store import get_all_documents
        persisted_docs = get_all_documents()
        if persisted_docs:
            existing_ids = {d["id"] for d in st.session_state["ingested_docs"]}
            for doc in persisted_docs:
                if doc["id"] not in existing_ids:
                    st.session_state["ingested_docs"].append(doc)
            logger.info(
                "State : %d document(s) rechargés depuis SQLite.",
                len(persisted_docs),
            )
    except Exception as exc:
        logger.warning("Impossible de recharger les docs depuis SQLite : %s", exc)

    # ── 2. Restauration du topic model depuis le pickle ───────────────────
    if st.session_state["topic_model_state"] is None:
        try:
            from analytics.topic_model import _load_state
            state = _load_state()
            if state.topic_model is not None:
                st.session_state["topic_model_state"] = state
                logger.info(
                    "State : topic model restauré depuis disque (%d topics connus).",
                    len(state.topic_labels),
                )
        except Exception as exc:
            logger.warning("Impossible de restaurer le topic model : %s", exc)

    st.session_state["_state_initialized"] = True


def reset_chat() -> None:
    """Vide l'historique du chat."""
    st.session_state["chat_history"] = []


def add_ingested_doc(doc_meta: dict) -> None:
    """Ajoute les métadonnées d'un document fraîchement ingéré."""
    st.session_state["ingested_docs"].append(doc_meta)


def get_ingested_docs() -> list:
    return st.session_state.get("ingested_docs", [])
