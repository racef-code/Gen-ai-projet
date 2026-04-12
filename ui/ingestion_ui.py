"""
Page d'ingestion Streamlit.

Fonctionnalités :
- Onglet 1 : Upload de fichiers (PDF, TXT, DOCX, MD) par drag-and-drop.
- Onglet 2 : Scraping d'une URL avec extraction de texte.
- Aperçu du texte extrait avant intégration.
- Sélection de la stratégie de chunking et affichage des stats.
- Bouton "Intégrer" pour lancer le pipeline complet (chunk + embed + store).
- Déduplication automatique par hash SHA-256 (Module D).
- Mise à jour incrémentale du topic model après chaque ingestion (Module D).
- Affichage du statut en temps réel et de la liste des documents ingérés.
"""
from __future__ import annotations

import uuid
from datetime import datetime

import streamlit as st

from app.config import (
    CHUNK_STRATEGY,
    PREVIEW_MAX_CHARS,
)
from app.state import add_ingested_doc, get_ingested_docs
from ingestion.chunker import chunk_text, get_chunks_stats
from ingestion.embedder import check_lm_studio_connection, embed_chunks
from ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from ingestion.scraper import scrape_url
from persistence.incremental import compute_text_hash, is_duplicate, run_incremental_update
from persistence.metadata_store import save_document
from persistence.vector_store import add_chunks as vs_add_chunks


def render_ingestion_page() -> None:
    """Render complet de la page d'ingestion."""
    st.title("Ingestion de Documents")
    st.caption("Ajoutez des fichiers ou des URLs pour alimenter la base de connaissances.")

    # ── Bannière de statut LM Studio ──────────────────────────────────────
    _render_lm_studio_status()

    st.divider()

    # ── Onglets : fichier vs URL ───────────────────────────────────────────
    tab_file, tab_url = st.tabs(["Fichier (PDF / TXT / DOCX / MD)", "URL Web"])

    with tab_file:
        _render_file_tab()

    with tab_url:
        _render_url_tab()

    st.divider()

    # ── Liste des documents déjà ingérés ──────────────────────────────────
    _render_ingested_docs_list()


# ── Onglet Fichier ────────────────────────────────────────────────────────────

def _render_file_tab() -> None:
    uploaded_files = st.file_uploader(
        "Glissez-déposez vos fichiers ici",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        help="Formats supportés : PDF, TXT, DOCX, Markdown",
    )

    if not uploaded_files:
        st.info("Aucun fichier sélectionné.")
        return

    for uploaded_file in uploaded_files:
        with st.expander(f"**{uploaded_file.name}** ({_fmt_size(uploaded_file.size)})", expanded=True):
            _render_file_preview_and_ingest(uploaded_file)


def _render_file_preview_and_ingest(uploaded_file) -> None:
    """Gère l'aperçu et l'ingestion pour un fichier uploadé."""
    # Clé unique par fichier pour isoler les widgets
    key = f"file_{uploaded_file.name}_{uploaded_file.size}"

    # ── Extraction du texte ────────────────────────────────────────────
    try:
        raw_bytes = uploaded_file.getvalue()
        text = load_document(raw_bytes, uploaded_file.name)
    except (ValueError, RuntimeError) as exc:
        st.error(f"Erreur d'extraction : {exc}")
        return

    if not text.strip():
        st.warning("Aucun texte extrait de ce fichier (PDF scanné ?).")
        return

    _render_preview_and_ingest_form(
        text=text,
        source_name=uploaded_file.name,
        source_type="file",
        widget_key=key,
    )


# ── Onglet URL ────────────────────────────────────────────────────────────────

def _render_url_tab() -> None:
    url_input = st.text_input(
        "URL à scraper",
        placeholder="https://exemple.com/article",
        help="La page sera téléchargée et son texte principal extrait.",
    )

    if not url_input:
        return

    scrape_btn = st.button("Analyser l'URL", key="btn_scrape_url")

    if scrape_btn:
        with st.spinner("Scraping en cours..."):
            try:
                title, text = scrape_url(url_input)
                # Stocker dans le session_state pour persister après le rerun
                st.session_state["_scraped_url_data"] = {
                    "url": url_input,
                    "title": title,
                    "text": text,
                }
            except (ValueError, RuntimeError) as exc:
                st.error(f"Erreur de scraping : {exc}")
                st.session_state.pop("_scraped_url_data", None)
                return

    # Afficher le résultat s'il existe dans le state
    scraped = st.session_state.get("_scraped_url_data")
    if scraped and scraped["url"] == url_input:
        st.success(f"Page récupérée : **{scraped['title']}**")
        _render_preview_and_ingest_form(
            text=scraped["text"],
            source_name=scraped["title"],
            source_type="url",
            widget_key=f"url_{hash(url_input)}",
            extra_metadata={"url": url_input},
        )


# ── Aperçu + Formulaire d'ingestion (partagé fichier/URL) ─────────────────────

def _render_preview_and_ingest_form(
    text: str,
    source_name: str,
    source_type: str,
    widget_key: str,
    extra_metadata: dict | None = None,
) -> None:
    """
    Affiche :
    1. L'aperçu du texte extrait.
    2. Les options de chunking.
    3. Le bouton d'intégration.
    """
    # ── Aperçu ────────────────────────────────────────────────────────────
    st.subheader("Aperçu du texte extrait")
    preview = text[:PREVIEW_MAX_CHARS]
    if len(text) > PREVIEW_MAX_CHARS:
        preview += f"\n\n... *(+{len(text) - PREVIEW_MAX_CHARS} caractères supplémentaires)*"

    st.text_area(
        "Contenu extrait",
        value=preview,
        height=220,
        disabled=True,
        key=f"preview_{widget_key}",
    )
    st.caption(f"Taille totale : **{len(text):,} caractères**")

    # ── Options de chunking ───────────────────────────────────────────────
    st.subheader("Options de découpage")
    col1, col2 = st.columns(2)

    with col1:
        strategy = st.selectbox(
            "Stratégie",
            options=["paragraph", "sliding_window"],
            index=0 if CHUNK_STRATEGY == "paragraph" else 1,
            help=(
                "**paragraph** : coupe aux double-sauts de ligne (respecte la structure).\n\n"
                "**sliding_window** : fenêtre glissante de taille fixe avec chevauchement."
            ),
            key=f"strategy_{widget_key}",
        )

    with col2:
        # Aperçu des chunks avant intégration
        preview_chunks_btn = st.button(
            "Prévisualiser les chunks",
            key=f"btn_preview_chunks_{widget_key}",
        )

    if preview_chunks_btn:
        chunks = chunk_text(text, doc_id="preview", strategy=strategy)
        stats = get_chunks_stats(chunks)
        st.info(
            f"**{stats['count']} chunks** | "
            f"Moy : {stats['avg_chars']} car. | "
            f"Min : {stats['min_chars']} | "
            f"Max : {stats['max_chars']}"
        )
        with st.expander("Voir les 3 premiers chunks"):
            for i, chunk in enumerate(chunks[:3]):
                st.markdown(f"**Chunk {i}** ({len(chunk.text)} car.)")
                st.text(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))

    # ── Bouton d'intégration ──────────────────────────────────────────────
    st.subheader("Intégration")
    ingest_btn = st.button(
        "Intégrer dans la base de connaissances",
        type="primary",
        key=f"btn_ingest_{widget_key}",
    )

    if ingest_btn:
        _run_ingestion_pipeline(
            text=text,
            source_name=source_name,
            source_type=source_type,
            strategy=strategy,
            extra_metadata=extra_metadata or {},
        )


# ── Pipeline d'ingestion ──────────────────────────────────────────────────────

def _run_ingestion_pipeline(
    text: str,
    source_name: str,
    source_type: str,
    strategy: str,
    extra_metadata: dict,
) -> None:
    """
    Exécute le pipeline complet :
    1. Déduplication (hash SHA-256)
    2. Chunking
    3. Embedding (via Ollama)
    4. Stockage dans ChromaDB
    5. Enregistrement des métadonnées (SQLite)
    6. Mise à jour incrémentale du topic model
    """
    # ── Étape 0 : Déduplication ───────────────────────────────────────
    doc_hash = compute_text_hash(text)
    duplicate, dup_name = is_duplicate(doc_hash)
    if duplicate:
        st.warning(
            f"Ce document est un doublon de **{dup_name}** (même contenu). "
            "Ingestion annulée."
        )
        return

    doc_id = str(uuid.uuid4())
    ingested_at = datetime.now().isoformat(timespec="seconds")
    progress = st.progress(0, text="Démarrage du pipeline...")

    try:
        # ── Étape 1 : Chunking ────────────────────────────────────────
        progress.progress(10, text="Découpage du texte en chunks...")
        chunks = chunk_text(
            text,
            doc_id=doc_id,
            strategy=strategy,
            extra_metadata={"source": source_name, "source_type": source_type, **extra_metadata},
        )

        if not chunks:
            st.error("Aucun chunk produit. Vérifiez le contenu du document.")
            progress.empty()
            return

        # ── Étape 2 : Embedding ───────────────────────────────────────
        progress.progress(25, text=f"Calcul des embeddings ({len(chunks)} chunks)...")
        vectors = embed_chunks(chunks)

        # ── Étape 3 : Stockage dans ChromaDB ─────────────────────────
        progress.progress(60, text="Stockage dans la base vectorielle...")
        vs_add_chunks(chunks, vectors)

        # ── Étape 4 : Métadonnées (SQLite + session_state) ────────────
        progress.progress(75, text="Enregistrement des métadonnées...")
        stats = get_chunks_stats(chunks)

        doc_meta = {
            "id": doc_id,
            "name": source_name,
            "source_type": source_type,
            "nb_chunks": stats["count"],
            "nb_chars": len(text),
            "strategy": strategy,
            "ingested_at": ingested_at,
            "doc_hash": doc_hash,
            **extra_metadata,
        }
        save_document(doc_meta)
        add_ingested_doc(doc_meta)

        # ── Étape 5 : Mise à jour incrémentale du topic model ─────────
        progress.progress(85, text="Mise à jour du topic model (incrémental)...")
        incremental_result = run_incremental_update(st.session_state)

        progress.progress(100, text="Ingestion terminée !")

        # Message de succès principal
        st.success(
            f"**{source_name}** ingéré avec succès — "
            f"{stats['count']} chunks, {len(text):,} caractères."
        )

        # Notification sur les nouveaux topics détectés
        _notify_new_topics(incremental_result)

        # Nettoyer les données temporaires de scraping
        st.session_state.pop("_scraped_url_data", None)

    except ConnectionError as exc:
        progress.empty()
        st.error(
            f"LM Studio inaccessible : {exc}. "
            "Vérifiez que LM Studio est ouvert, qu'un modèle est chargé "
            "et que le serveur local est démarré (port 1234)."
        )
    except Exception as exc:
        progress.empty()
        st.error(f"Erreur lors de l'ingestion : {exc}")


def _notify_new_topics(incremental_result: dict) -> None:
    """Affiche une notification si de nouveaux topics ont été détectés."""
    if not incremental_result.get("success"):
        if incremental_result.get("error"):
            st.info(
                f"Topic modeling ignoré : {incremental_result['error']}"
            )
        return

    new_topics = incremental_result.get("new_topics", [])
    new_labels = incremental_result.get("new_labels", {})
    total = incremental_result.get("total_topics", 0)

    if new_topics:
        labels_str = ", ".join(
            f"**{new_labels.get(t, f'Topic {t}')}**" for t in new_topics
        )
        st.info(
            f"Topic model mis à jour : {len(new_topics)} nouveau(x) thème(s) détecté(s) "
            f"→ {labels_str}  ({total} thèmes au total)"
        )
    else:
        st.caption(f"Topic model : {total} thèmes connus, aucun nouveau.")


# ── Liste des documents ingérés ───────────────────────────────────────────────

def _render_ingested_docs_list() -> None:
    """
    Affiche les documents ingérés.
    Source : SQLite (persistant) fusionné avec le session_state (session courante).
    """
    from persistence.metadata_store import get_all_documents
    from persistence.vector_store import delete_doc as vs_delete_doc
    from persistence.metadata_store import delete_document as db_delete_doc

    # Charger depuis SQLite (inclut les sessions précédentes)
    try:
        docs = get_all_documents()
    except Exception:
        docs = get_ingested_docs()  # fallback session_state

    st.subheader(f"Documents ingérés ({len(docs)})")

    if not docs:
        st.info("Aucun document ingéré.")
        return

    for doc in docs:  # SQLite trie déjà par date desc
        icon = "URL" if doc.get("source_type") == "url" else "FICHIER"
        with st.expander(f"[{icon}] {doc['name']} — {doc['ingested_at']}"):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            col1.metric("Chunks", doc["nb_chunks"])
            col2.metric("Caractères", f"{doc['nb_chars']:,}")
            col3.metric("Stratégie", doc["strategy"])
            if doc.get("url"):
                st.caption(f"Source : {doc['url']}")

            # Bouton de suppression
            if col4.button("Supprimer", key=f"del_{doc['id']}", type="secondary"):
                try:
                    vs_delete_doc(doc["id"])
                    db_delete_doc(doc["id"])
                    # Retirer du session_state aussi
                    st.session_state["ingested_docs"] = [
                        d for d in st.session_state.get("ingested_docs", [])
                        if d["id"] != doc["id"]
                    ]
                    st.success(f"Document **{doc['name']}** supprimé.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Erreur lors de la suppression : {exc}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_lm_studio_status() -> None:
    """Affiche un indicateur de connexion LM Studio dans la sidebar."""
    with st.sidebar:
        st.subheader("Statut LM Studio")
        if st.button("Vérifier la connexion", key="btn_check_lm_studio"):
            with st.spinner("Connexion..."):
                ok = check_lm_studio_connection()
            if ok:
                st.success("LM Studio connecté")
            else:
                st.error("LM Studio inaccessible")
                st.caption("Ouvrez LM Studio, chargez un modèle et démarrez le serveur local.")


def _fmt_size(size_bytes: int) -> str:
    """Formate une taille en octets de manière lisible."""
    for unit in ["o", "Ko", "Mo", "Go"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} To"
