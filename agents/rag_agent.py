"""
Agent RAG — LangChain + ChatOpenAI (LM Studio).

Architecture de la chain :
  1. Récupération des top_k chunks via le vector_store (retriever custom).
  2. Construction du prompt avec le contexte injecté.
  3. Appel au LLM local via LM Studio (API compatible OpenAI).
  4. Retour structuré : réponse + sources (pour le panneau de transparence).

Le prompt est en français et explicitement instruit pour :
- Ne répondre QUE d'après le contexte fourni.
- Indiquer clairement si l'information n'est pas dans les documents.

Retry automatique (3 tentatives, backoff exponentiel 1s/2s/4s) sur l'appel LLM
en cas d'erreur transitoire LM Studio.
"""
from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from app.config import LLM_MODEL, LM_STUDIO_API_KEY, LM_STUDIO_BASE_URL, LLM_TIMEOUT, RETRIEVAL_TOP_K
from ingestion.embedder import embed_query
from persistence.vector_store import query_similar

logger = logging.getLogger(__name__)

# ── Prompt système ────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """Tu es un assistant expert en analyse documentaire. \
Tu réponds UNIQUEMENT à partir des extraits de documents fournis dans le contexte ci-dessous.

Règles strictes :
1. Si la réponse n'est pas dans le contexte, dis-le clairement : \
"Je ne trouve pas cette information dans les documents fournis."
2. Ne fabrique jamais d'informations.
3. Cite le document source lorsque c'est pertinent.
4. Réponds en français, de manière concise et structurée.

CONTEXTE :
{context}
"""

_HUMAN_PROMPT = "{question}"

# Nombre maximal de tentatives en cas d'erreur transitoire LM Studio
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
                    "LM Studio LLM : tentative %d/%d échouée (%s). "
                    "Nouvel essai dans %.1fs...",
                    attempt + 1, max_attempts, exc, delay,
                )
                time.sleep(delay)
    raise RuntimeError(
        f"LLM échoué après {max_attempts} tentatives : {last_exc}"
    ) from last_exc


def build_rag_chain():
    """Construit et retourne la chain LangChain (ChatOpenAI → LM Studio + prompt)."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError("langchain-openai est requis : pip install langchain-openai") from exc

    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        temperature=0.1,        # Réponses factuelles, peu créatives
        max_tokens=1024,        # Longueur max de la réponse
        timeout=LLM_TIMEOUT,    # Timeout explicite (CPU-only = plus lent)
    )

    # Mistral via LM Studio ne supporte que les rôles "user" et "assistant".
    # On fusionne le system prompt dans le message user (pas de rôle "system").
    prompt = ChatPromptTemplate.from_messages([
        ("human", _SYSTEM_PROMPT + "\n\n" + _HUMAN_PROMPT),
    ])

    return prompt | llm


# ── Point d'entrée principal ──────────────────────────────────────────────────

def ask(
    question: str,
    top_k: int = RETRIEVAL_TOP_K,
    filter_doc_ids: list[str] | None = None,
) -> dict:
    """
    Pipeline RAG complet : embed question → retrieve chunks → LLM → réponse.

    Args:
        question:       Question de l'utilisateur.
        top_k:          Nombre de chunks à récupérer.
        filter_doc_ids: Restriction optionnelle à certains documents.

    Returns:
        Dict avec :
          - answer  (str)  : réponse du LLM
          - sources (list) : chunks utilisés comme contexte
          - latency (float): temps de réponse en secondes
          - question(str)  : question originale
    """
    start = time.perf_counter()

    # ── 1. Embedding de la question ───────────────────────────────────
    logger.info("RAG : embedding de la question...")
    try:
        query_vector = embed_query(question)
    except RuntimeError as exc:
        raise RuntimeError(f"Impossible d'embedder la question : {exc}") from exc

    # ── 2. Retrieval ──────────────────────────────────────────────────
    logger.info("RAG : retrieval top-%d...", top_k)
    hits = query_similar(query_vector, top_k=top_k, filter_doc_ids=filter_doc_ids)

    if not hits:
        return {
            "answer": "Aucun document n'a encore été ingéré. "
                      "Veuillez d'abord ajouter des documents dans l'onglet **Ingestion**.",
            "sources": [],
            "latency": time.perf_counter() - start,
            "question": question,
        }

    # ── 3. Construction du contexte ───────────────────────────────────
    context_parts = []
    for i, hit in enumerate(hits, 1):
        source_label = hit["metadata"].get("source", f"Document {hit['doc_id'][:8]}")
        context_parts.append(
            f"[Extrait {i} — Source : {source_label}]\n{hit['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # ── 4. Appel LLM (avec retry) ─────────────────────────────────────
    logger.info("RAG : appel LLM '%s'...", LLM_MODEL)
    try:
        chain = build_rag_chain()
        response = _with_retry(
            lambda: chain.invoke({"context": context, "question": question})
        )
        answer = response.content if hasattr(response, "content") else str(response)
    except RuntimeError as exc:
        raise RuntimeError(f"Erreur LLM ({LLM_MODEL}) : {exc}") from exc

    latency = time.perf_counter() - start
    logger.info("RAG terminé en %.2fs", latency)

    return {
        "answer": answer,
        "sources": hits,
        "latency": latency,
        "question": question,
    }
