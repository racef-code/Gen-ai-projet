"""
Gestionnaire centralisé du session_state Streamlit.
Garantit que toutes les clés sont initialisées une seule fois.

Au démarrage, recharge les documents persistés depuis SQLite
pour que la liste soit disponible même après un redémarrage de l'app.
"""
import logging

import streamlit as st

logger = logging.getLogger(__name__)


def init_session_state() -> None:
    """
    Initialise les clés du session_state.
    Recharge les documents depuis SQLite si c'est la première exécution
    de la session (clé sentinelle "_state_initialized" absente).
    """
    if st.session_state.get("_state_initialized"):
        return

    defaults = {
        # ── Documents ingérés (rechargés depuis SQLite ci-dessous) ────────
        "ingested_docs": [],

        # ── Chunks en mémoire (cache léger) ───────────────────────────────
        "chunks_cache": [],

        # ── Historique du chat Q&A ────────────────────────────────────────
        "chat_history": [],

        # ── Topic Model ───────────────────────────────────────────────────
        "topic_model_state": None,

        # ── Page active (navigation) ──────────────────────────────────────
        "current_page": "ingestion",

        # ── Feedback ingestion ────────────────────────────────────────────
        "last_ingestion_status": None,
        "last_ingestion_message": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # ── Rechargement des docs persistés depuis SQLite ─────────────────────
    try:
        from persistence.metadata_store import get_all_documents
        persisted_docs = get_all_documents()
        if persisted_docs:
            # Fusionner : éviter les doublons si déjà en session_state
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

    st.session_state["_state_initialized"] = True


def reset_chat() -> None:
    """Vide l'historique du chat."""
    st.session_state["chat_history"] = []


def add_ingested_doc(doc_meta: dict) -> None:
    """Ajoute les métadonnées d'un document fraîchement ingéré."""
    st.session_state["ingested_docs"].append(doc_meta)


def get_ingested_docs() -> list:
    return st.session_state.get("ingested_docs", [])
