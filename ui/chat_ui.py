"""
Page Chat & RAG — Streamlit.

Fonctionnalités :
- Interface de chat classique (bulles user / assistant).
- Envoi par Enter ou bouton.
- Spinner pendant la génération LLM.
- Panneau de transparence (toujours visible) : affiche les extraits de contexte
  utilisés par le LLM pour formuler sa réponse, avec leur score de similarité.
- Filtrage optionnel par document.
- Bouton de réinitialisation de la conversation.
"""
from __future__ import annotations

import uuid

import streamlit as st

from agents.query_handler import handle_query
from app.config import RETRIEVAL_TOP_K
from app.state import get_ingested_docs, reset_chat
from persistence.metadata_store import clear_chat_history


def render_chat_page() -> None:
    """Render complet de la page Chat & RAG."""
    st.title("Chat & RAG")
    st.caption("Posez vos questions — les réponses sont générées localement via LM Studio (Mistral 7B Q4).")

    # Initialisation de l'ID de session (persistant dans session_state)
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    docs = get_ingested_docs()

    if not docs:
        st.warning(
            "Aucun document ingéré. "
            "Rendez-vous dans l'onglet **Ingestion** pour ajouter des documents."
        )
        return

    # ── Mise en page : chat à gauche, transparence à droite ───────────────
    col_chat, col_context = st.columns([3, 2], gap="large")

    with col_chat:
        _render_chat_column(docs)

    with col_context:
        _render_transparency_panel()


# ── Colonne chat ──────────────────────────────────────────────────────────────

def _render_chat_column(docs: list[dict]) -> None:
    """Affiche l'historique, les options et le champ de saisie."""
    # ── Options ───────────────────────────────────────────────────────
    with st.expander("Options de recherche", expanded=False):
        top_k = st.slider(
            "Nombre de chunks (top-k)",
            min_value=1,
            max_value=10,
            value=RETRIEVAL_TOP_K,
            key="top_k_slider",
            help="Nombre d'extraits de documents fournis au LLM comme contexte.",
        )

        doc_names = {d["id"]: d["name"] for d in docs}
        selected_ids = st.multiselect(
            "Filtrer sur ces documents (vide = tous)",
            options=list(doc_names.keys()),
            format_func=lambda k: doc_names[k],
            key="doc_filter",
            help="Laissez vide pour interroger tous les documents ingérés.",
        )

    # ── Historique des messages ────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        _render_message_history()

    # ── Champ de saisie ───────────────────────────────────────────────
    st.divider()

    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            question = st.text_input(
                "Votre question",
                placeholder="Ex : Quels sont les points clés de ce document ?",
                label_visibility="collapsed",
                key="question_input",
            )
        with col_btn:
            submitted = st.form_submit_button("Envoyer", type="primary", use_container_width=True)

    if submitted and question.strip():
        _process_question(
            question=question.strip(),
            top_k=top_k,
            filter_doc_ids=selected_ids or None,
        )
        st.rerun()

    # ── Bouton reset ──────────────────────────────────────────────────
    if st.session_state.get("chat_history"):
        if st.button("Effacer la conversation", key="btn_clear_chat"):
            reset_chat()
            try:
                clear_chat_history(st.session_state["session_id"])
            except Exception:
                pass
            st.rerun()


def _render_message_history() -> None:
    """Affiche les bulles de conversation."""
    history = st.session_state.get("chat_history", [])

    if not history:
        st.info("Commencez par poser une question ci-dessous.")
        return

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Méta-info sur la réponse de l'assistant
            if msg["role"] == "assistant" and msg.get("latency"):
                st.caption(f"Généré en {msg['latency']:.1f}s")


def _process_question(
    question: str,
    top_k: int,
    filter_doc_ids: list[str] | None,
) -> None:
    """Lance le pipeline RAG et met à jour le state."""
    with st.spinner("Recherche dans les documents et génération de la réponse..."):
        handle_query(
            question=question,
            session_id=st.session_state["session_id"],
            top_k=top_k,
            filter_doc_ids=filter_doc_ids,
        )


# ── Panneau de transparence ───────────────────────────────────────────────────

def _render_transparency_panel() -> None:
    """Affiche les extraits de contexte utilisés par le dernier appel LLM."""
    st.subheader("Panneau de Transparence")
    st.caption("Extraits de documents utilisés pour la dernière réponse.")

    history = st.session_state.get("chat_history", [])

    # Trouver le dernier message assistant avec des sources
    last_sources = []
    for msg in reversed(history):
        if msg["role"] == "assistant" and msg.get("sources"):
            last_sources = msg["sources"]
            break

    if not last_sources:
        st.info("Les extraits de contexte apparaîtront ici après votre première question.")
        return

    for i, source in enumerate(last_sources, 1):
        source_name = source.get("metadata", {}).get("source", f"Document {i}")
        distance = source.get("distance", 0.0)
        similarity = max(0.0, 1.0 - distance)  # cosine distance → similarité

        with st.expander(
            f"**Extrait {i}** — {source_name} (sim. {similarity:.0%})",
            expanded=(i == 1),  # Premier extrait ouvert par défaut
        ):
            # Barre de similarité
            st.progress(similarity, text=f"Pertinence : {similarity:.0%}")

            # Texte de l'extrait
            st.text_area(
                "Contenu",
                value=source["text"],
                height=150,
                disabled=True,
                key=f"source_text_{i}_{id(source)}",
            )

            # Métadonnées de la source
            meta = source.get("metadata", {})
            cols = st.columns(2)
            if meta.get("source_type"):
                cols[0].caption(f"Type : `{meta['source_type']}`")
            if meta.get("chunk_index") is not None:
                cols[1].caption(f"Chunk : `#{meta['chunk_index']}`")
            if meta.get("url"):
                st.caption(f"URL : {meta['url']}")
            if meta.get("strategy"):
                st.caption(f"Stratégie : `{meta['strategy']}`")
