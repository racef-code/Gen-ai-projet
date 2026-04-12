"""
Orchestrateur des requêtes utilisateur.

Responsabilités :
- Valider la question avant de la passer à l'agent RAG.
- Gérer les erreurs de connexion Ollama avec un message lisible.
- Persister l'échange dans l'historique SQLite.
- Mettre à jour le session_state Streamlit.
"""
from __future__ import annotations

import logging

import streamlit as st

from agents.rag_agent import ask
from app.config import RETRIEVAL_TOP_K
from persistence.metadata_store import save_chat_message

logger = logging.getLogger(__name__)

# Longueur minimale d'une question (éviter les requêtes vides ou trop courtes)
_MIN_QUESTION_LEN = 3


def handle_query(
    question: str,
    session_id: str,
    top_k: int = RETRIEVAL_TOP_K,
    filter_doc_ids: list[str] | None = None,
) -> dict:
    """
    Pipeline complet de traitement d'une question utilisateur.

    Args:
        question:       Texte de la question.
        session_id:     Identifiant de la session Streamlit (pour l'historique).
        top_k:          Nombre de chunks à récupérer.
        filter_doc_ids: Restriction optionnelle à certains documents.

    Returns:
        Dict {answer, sources, latency, question, error?}
    """
    # ── Validation ────────────────────────────────────────────────────
    question = question.strip()
    if len(question) < _MIN_QUESTION_LEN:
        return {
            "answer": "Veuillez poser une question plus complète.",
            "sources": [],
            "latency": 0.0,
            "question": question,
            "error": "question_too_short",
        }

    # ── Appel RAG ────────────────────────────────────────────────────
    try:
        result = ask(question, top_k=top_k, filter_doc_ids=filter_doc_ids)
    except RuntimeError as exc:
        error_msg = _humanize_error(str(exc))
        logger.error("Erreur RAG : %s", exc)
        result = {
            "answer": error_msg,
            "sources": [],
            "latency": 0.0,
            "question": question,
            "error": str(exc),
        }

    # ── Persistance dans SQLite ───────────────────────────────────────
    try:
        save_chat_message(session_id, "user", question)
        save_chat_message(
            session_id,
            "assistant",
            result["answer"],
            sources=result.get("sources"),
        )
    except Exception as exc:
        logger.warning("Impossible de persister l'historique : %s", exc)

    # ── Mise à jour du session_state ──────────────────────────────────
    _update_chat_state(question, result)

    return result


def _update_chat_state(question: str, result: dict) -> None:
    """Ajoute la paire question/réponse au session_state Streamlit."""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.session_state["chat_history"].append({
        "role": "user",
        "content": question,
    })
    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "latency": result.get("latency", 0.0),
    })


def _humanize_error(error: str) -> str:
    """Traduit les erreurs techniques en messages lisibles."""
    error_lower = error.lower()

    if "connection" in error_lower or "refused" in error_lower:
        return (
            "LM Studio n'est pas accessible. "
            "Assurez-vous que LM Studio est ouvert, qu'un modèle est chargé "
            "et que le serveur local est démarré (port 1234)."
        )
    if "model" in error_lower and "not found" in error_lower:
        return (
            "Le modèle demandé n'est pas chargé dans LM Studio. "
            "Ouvrez LM Studio, chargez le modèle puis redémarrez le serveur."
        )
    if "embed" in error_lower:
        return (
            "Erreur lors de l'embedding. "
            "Vérifiez que le modèle d'embedding (nomic-embed-text-v1.5) "
            "est chargé dans LM Studio et que le serveur est actif."
        )

    return f"Une erreur est survenue : {error}"
