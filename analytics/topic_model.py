"""
Topic Modeling incrémental via BERTopic.

Stratégie incrémentale (cf. architecture décrite dans le cahier des charges) :
─────────────────────────────────────────────────────────────────────────────
1. Les embeddings sont déjà stockés dans ChromaDB — on ne les recalcule JAMAIS.
2. À chaque appel à `fit_or_update()` :
   a. On récupère TOUS les embeddings de ChromaDB (vecteurs pré-calculés).
   b. On projette avec UMAP (2D) pour la visualisation.
   c. On fit/partial_fit BERTopic sur les embeddings haute dimension.
3. Le modèle BERTopic sérialisé est sauvegardé sur disque (data/topic_model/).
4. Si le modèle existe déjà sur disque, on le recharge et on fait partial_fit
   uniquement sur les NOUVEAUX embeddings (ceux dont les doc_ids sont inconnus
   du modèle précédent).

Auto-labelling :
   Quand de nouveaux clusters apparaissent, on appelle le LLM via Ollama
   pour générer un label descriptif à partir des N premiers chunks du cluster.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from app.config import MIN_TOPIC_SIZE, NR_TOPICS, TOPIC_MODEL_DIR

logger = logging.getLogger(__name__)

_MODEL_PATH = TOPIC_MODEL_DIR / "bertopic_model.pkl"
_STATE_PATH = TOPIC_MODEL_DIR / "topic_state.pkl"


# ── Structures de données ─────────────────────────────────────────────────────

class TopicModelState:
    """Encapsule l'état complet du topic model pour la sérialisation."""

    def __init__(self) -> None:
        self.topic_model: Any = None          # objet BERTopic
        self.umap_model: Any = None           # UMAP fité
        self.known_doc_ids: set[str] = set()  # docs déjà vus par le modèle
        self.topic_labels: dict[int, str] = {}# {topic_id: label LLM}
        self.last_topics: list[int] = []      # topic assignment de chaque chunk
        self.last_embeddings_2d: list = []    # coordonnées UMAP 2D
        self.chunk_doc_map: list[str] = []    # doc_id[i] pour le chunk i


# ── API principale ────────────────────────────────────────────────────────────

def fit_or_update(force_refit: bool = False) -> TopicModelState | None:
    """
    Récupère tous les embeddings de ChromaDB, puis :
    - Si pas de modèle existant : fit complet.
    - Si modèle existant et nouveaux docs : partial_fit sur les nouveaux.
    - Si force_refit=True : refit complet.

    Returns:
        TopicModelState avec les topics, labels et coordonnées 2D,
        ou None si moins de MIN_TOPIC_SIZE * 2 chunks disponibles.
    """
    try:
        from bertopic import BERTopic
        from umap import UMAP
    except ImportError:
        logger.warning("BERTopic ou UMAP non installé — topic modeling désactivé.")
        return None

    from persistence.vector_store import get_all_embeddings

    # ── 1. Récupération des embeddings depuis ChromaDB ─────────────────
    raw = get_all_embeddings()
    if not raw["ids"]:
        logger.info("Aucun chunk dans ChromaDB — topic modeling ignoré.")
        return None

    embeddings = np.array(raw["embeddings"], dtype=np.float32)
    documents = raw["documents"]
    metadatas = raw["metadatas"]
    chunk_doc_ids = [m.get("doc_id", "") for m in metadatas]

    n_chunks = len(embeddings)
    min_required = max(MIN_TOPIC_SIZE * 2, 5)
    if n_chunks < min_required:
        logger.info("Pas assez de chunks (%d < %d) pour le topic modeling.", n_chunks, min_required)
        return None

    # ── 2. Chargement de l'état existant ──────────────────────────────
    state = _load_state()
    new_doc_ids = set(chunk_doc_ids) - state.known_doc_ids

    if state.topic_model is None or force_refit:
        logger.info("Fit complet BERTopic sur %d chunks...", n_chunks)
        state = _full_fit(embeddings, documents, chunk_doc_ids, BERTopic, UMAP)
    elif new_doc_ids:
        logger.info("Partial fit BERTopic sur %d nouveaux docs (%d chunks total)...",
                    len(new_doc_ids), n_chunks)
        state = _incremental_fit(state, embeddings, documents, chunk_doc_ids, new_doc_ids, UMAP)
    else:
        logger.info("Aucun nouveau document — état du topic model inchangé.")

    state.known_doc_ids = set(chunk_doc_ids)
    state.chunk_doc_map = chunk_doc_ids
    _save_state(state)
    return state


def get_topic_evolution(state: TopicModelState, doc_order: list[dict]) -> dict:
    """
    Calcule la distribution des topics par document (pour le timeline).

    Args:
        state:     État du topic model.
        doc_order: Liste de docs triée par date d'ingestion [{id, name, ...}].

    Returns:
        Dict {doc_name: {topic_label: count}}
    """
    if not state.last_topics or not state.chunk_doc_map:
        return {}

    topic_labels = state.topic_labels
    evolution: dict[str, dict[str, int]] = {}

    for doc in doc_order:
        doc_id = doc["id"]
        doc_name = doc["name"]
        topic_counts: dict[str, int] = {}

        for chunk_idx, (cid, topic_id) in enumerate(
            zip(state.chunk_doc_map, state.last_topics)
        ):
            if cid != doc_id:
                continue
            if topic_id == -1:
                label = "Hors-sujet"
            else:
                label = topic_labels.get(topic_id, f"Topic {topic_id}")
            topic_counts[label] = topic_counts.get(label, 0) + 1

        if topic_counts:
            evolution[doc_name] = topic_counts

    return evolution


def auto_label_topics(state: TopicModelState) -> dict[int, str]:
    """
    Génère des labels descriptifs pour les clusters via le LLM local (Ollama).
    Appelle le LLM uniquement pour les topics sans label existant.

    Returns:
        Dict mis à jour {topic_id: label_string}
    """
    if state.topic_model is None:
        return {}

    try:
        topic_info = state.topic_model.get_topic_info()
    except Exception:
        return state.topic_labels

    # Topics sans label ou avec label générique
    unlabelled = [
        row["Topic"] for _, row in topic_info.iterrows()
        if row["Topic"] != -1 and row["Topic"] not in state.topic_labels
    ]

    if not unlabelled:
        return state.topic_labels

    logger.info("Auto-labelling de %d topics via LLM...", len(unlabelled))
    new_labels = {}

    for topic_id in unlabelled:
        try:
            topic_words = state.topic_model.get_topic(topic_id)
            if not topic_words:
                continue
            top_words = [w for w, _ in topic_words[:8]]
            label = _call_llm_for_label(topic_id, top_words)
            new_labels[topic_id] = label
            logger.debug("Topic %d → '%s'", topic_id, label)
        except Exception as exc:
            logger.warning("Impossible de labelliser le topic %d : %s", topic_id, exc)
            new_labels[topic_id] = f"Topic {topic_id}"

    state.topic_labels.update(new_labels)
    return state.topic_labels


# ── Fit / Partial fit ─────────────────────────────────────────────────────────

def _full_fit(embeddings, documents, chunk_doc_ids, BERTopic, UMAP) -> TopicModelState:
    """Fit complet : UMAP + BERTopic sur tous les chunks."""
    state = TopicModelState()

    # UMAP 2D pour la visualisation
    umap_2d = UMAP(n_components=2, n_neighbors=min(15, len(embeddings) - 1),
                   min_dist=0.1, metric="cosine", random_state=42)
    coords_2d = umap_2d.fit_transform(embeddings)

    # UMAP haute dimension pour BERTopic (réduction avant clustering)
    n_components_hd = min(10, len(embeddings) - 2, embeddings.shape[1])
    umap_hd = UMAP(n_components=n_components_hd, n_neighbors=min(15, len(embeddings) - 1),
                   min_dist=0.0, metric="cosine", random_state=42)

    topic_model = BERTopic(
        umap_model=umap_hd,
        min_topic_size=MIN_TOPIC_SIZE,
        nr_topics=NR_TOPICS,
        calculate_probabilities=False,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(documents, embeddings)

    state.topic_model = topic_model
    state.umap_model = umap_2d
    state.last_topics = topics
    state.last_embeddings_2d = coords_2d.tolist()

    return state


def _incremental_fit(
    state: TopicModelState,
    embeddings: np.ndarray,
    documents: list[str],
    chunk_doc_ids: list[str],
    new_doc_ids: set[str],
    UMAP,
) -> TopicModelState:
    """
    Mise à jour incrémentale :
    - On refit UMAP 2D sur tous les embeddings (rapide, pas de re-embedding).
    - On appelle partial_fit sur les nouveaux chunks uniquement.
    """
    # Indices des nouveaux chunks
    new_indices = [i for i, did in enumerate(chunk_doc_ids) if did in new_doc_ids]
    new_embeddings = embeddings[new_indices]
    new_docs = [documents[i] for i in new_indices]

    try:
        # partial_fit BERTopic sur les nouveaux chunks
        new_topics, _ = state.topic_model.partial_fit(new_docs, new_embeddings)
        # Reconstruire la liste complète des topics
        all_topics = list(state.last_topics)
        old_count = len(all_topics)
        # Remplacer/étendre
        for i, nt in zip(new_indices, new_topics):
            if i < old_count:
                all_topics[i] = nt
            else:
                all_topics.append(nt)
        state.last_topics = all_topics
    except Exception as exc:
        logger.warning("partial_fit échoué (%s), refit complet...", exc)
        from bertopic import BERTopic as BT
        return _full_fit(embeddings, documents, chunk_doc_ids, BT, UMAP)

    # Re-projection UMAP 2D sur tous les embeddings
    try:
        coords_2d = state.umap_model.transform(embeddings)
    except Exception:
        # Si le modèle UMAP ne peut pas transformer, on le refit
        umap_2d = UMAP(n_components=2, n_neighbors=min(15, len(embeddings) - 1),
                       min_dist=0.1, metric="cosine", random_state=42)
        coords_2d = umap_2d.fit_transform(embeddings)
        state.umap_model = umap_2d

    state.last_embeddings_2d = coords_2d.tolist()
    return state


# ── Persistance ───────────────────────────────────────────────────────────────

def _save_state(state: TopicModelState) -> None:
    try:
        with open(_STATE_PATH, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("Topic model state sauvegardé : %s", _STATE_PATH)
    except Exception as exc:
        logger.warning("Impossible de sauvegarder le topic model : %s", exc)


def _load_state() -> TopicModelState:
    if _STATE_PATH.exists():
        try:
            with open(_STATE_PATH, "rb") as f:
                state = pickle.load(f)
            logger.info("Topic model state rechargé depuis disque.")
            return state
        except Exception as exc:
            logger.warning("Impossible de recharger le topic model (%s) — refit.", exc)
    return TopicModelState()


def clear_state() -> None:
    """Supprime le modèle sauvegardé (force un refit complet au prochain appel)."""
    for path in [_MODEL_PATH, _STATE_PATH]:
        if path.exists():
            path.unlink()
    logger.info("Topic model state supprimé.")


# ── LLM Auto-labelling ────────────────────────────────────────────────────────

def _call_llm_for_label(topic_id: int, top_words: list[str]) -> str:
    """
    Appelle ChatOpenAI (LM Studio) pour générer un label court pour un cluster.
    Retourne un label générique en cas d'erreur.
    """
    try:
        from langchain_openai import ChatOpenAI
        from app.config import LLM_MODEL, LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY

        llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=LM_STUDIO_BASE_URL,
            api_key=LM_STUDIO_API_KEY,
            temperature=0.3,
            max_tokens=20,
        )
        prompt = (
            f"Voici les mots-clés d'un cluster thématique : {', '.join(top_words)}.\n"
            "Génère UN label court (2-4 mots maximum) qui résume ce thème. "
            "Réponds UNIQUEMENT avec le label, sans explication."
        )
        response = llm.invoke(prompt)
        label = response.content.strip().strip('"').strip("'")
        return label[:50] if label else f"Topic {topic_id}"
    except Exception as exc:
        logger.warning("Auto-label LLM échoué pour topic %d : %s", topic_id, exc)
        return f"Topic {topic_id}"
