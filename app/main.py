"""
Point d'entrée de l'application Streamlit.

Lancement :
    streamlit run app/main.py

Navigation multi-pages gérée manuellement via la sidebar
(compatible Streamlit single-file, plus simple que le mode multi-pages).
"""
import sys
from pathlib import Path

# Assure que le répertoire racine du projet est dans le PYTHONPATH,
# quel que soit le répertoire de lancement.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.config import APP_TITLE
from app.state import init_session_state

# ── Configuration de la page (doit être le 1er appel Streamlit) ───────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    # Initialisation du session_state
    init_session_state()

    # ── Sidebar : navigation ──────────────────────────────────────────────
    with st.sidebar:
        st.title("📄 DocAnalysis")
        st.caption("Analyse Intelligente de Documents — 100% Local")
        st.divider()

        pages = {
            "ingestion": "Ingestion",
            "chat": "Chat & RAG",
            "analytics": "Tableau de Bord",
        }

        # Icônes par page
        icons = {
            "ingestion": "upload",
            "chat": "chat-dots",
            "analytics": "bar-chart-line",
        }

        selected = st.radio(
            "Navigation",
            options=list(pages.keys()),
            format_func=lambda k: pages[k],
            index=list(pages.keys()).index(
                st.session_state.get("current_page", "ingestion")
            ),
            key="nav_radio",
        )
        st.session_state["current_page"] = selected

        st.divider()
        st.caption(f"LLM : `{_get_config('LLM_MODEL')}`")
        st.caption(f"Embed : `{_get_config('EMBED_MODEL')}`")

        # ── Stats globales (Module D) ──────────────────────────────
        st.divider()
        try:
            from persistence.incremental import get_global_stats
            gs = get_global_stats()
            st.metric("Documents", gs["total_docs"])
            st.metric("Chunks", gs["total_chunks"])
            topic_state = st.session_state.get("topic_model_state")
            nb_topics = len(topic_state.topic_labels) if topic_state else 0
            st.metric("Topics", nb_topics)
        except Exception:
            pass

    # ── Rendu de la page active ───────────────────────────────────────────
    if selected == "ingestion":
        from ui.ingestion_ui import render_ingestion_page
        render_ingestion_page()

    elif selected == "chat":
        from ui.chat_ui import render_chat_page
        render_chat_page()

    elif selected == "analytics":
        from ui.analytics_ui import render_analytics_page
        render_analytics_page()


def _get_config(key: str) -> str:
    """Récupère une valeur de config pour l'affichage sidebar."""
    try:
        import app.config as cfg
        return getattr(cfg, key, "N/A")
    except Exception:
        return "N/A"


if __name__ == "__main__":
    main()
