"""
Gestionnaire centralisé du session_state Streamlit.
Garantit que toutes les clés sont initialisées une seule fois.
"""
import streamlit as st


def init_session_state() -> None:
    """Initialise les clés du session_state si elles n'existent pas encore."""

    defaults = {
        # ── Documents ingérés ─────────────────────────────────────────────
        # Liste de dicts : {id, name, source, nb_chunks, ingested_at}
        "ingested_docs": [],

        # ── Chunks en mémoire (cache léger) ───────────────────────────────
        # Liste de dicts : {doc_id, chunk_index, text, embedding?}
        "chunks_cache": [],

        # ── Historique du chat Q&A ────────────────────────────────────────
        # Liste de dicts : {role: "user"|"assistant", content, sources?}
        "chat_history": [],

        # ── Topic Model ───────────────────────────────────────────────────
        # Résultats BERTopic : {topics, probs, topic_labels, embeddings_2d}
        "topic_model_state": None,

        # ── Page active (navigation) ──────────────────────────────────────
        "current_page": "ingestion",

        # ── Feedback ingestion ────────────────────────────────────────────
        "last_ingestion_status": None,   # "success" | "error" | None
        "last_ingestion_message": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_chat() -> None:
    """Vide l'historique du chat."""
    st.session_state["chat_history"] = []


def add_ingested_doc(doc_meta: dict) -> None:
    """Ajoute les métadonnées d'un document fraîchement ingéré."""
    st.session_state["ingested_docs"].append(doc_meta)


def get_ingested_docs() -> list:
    return st.session_state.get("ingested_docs", [])
