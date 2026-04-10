"""
Page d'ingestion Streamlit.

Fonctionnalités :
- Onglet 1 : Upload de fichiers (PDF, TXT, DOCX, MD) par drag-and-drop.
- Onglet 2 : Scraping d'une URL avec extraction de texte.
- Aperçu du texte extrait avant intégration.
- Sélection de la stratégie de chunking et affichage des stats.
- Bouton "Intégrer" pour lancer le pipeline complet (chunk + embed + store).
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
from ingestion.embedder import check_ollama_connection, embed_chunks
from ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from ingestion.scraper import scrape_url


def render_ingestion_page() -> None:
    """Render complet de la page d'ingestion."""
    st.title("Ingestion de Documents")
    st.caption("Ajoutez des fichiers ou des URLs pour alimenter la base de connaissances.")

    # ── Bannière de statut Ollama ──────────────────────────────────────────
    _render_ollama_status()

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
    1. Chunking
    2. Embedding (via Ollama)
    3. Stockage dans ChromaDB
    4. Enregistrement des métadonnées
    """
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
        progress.progress(30, text=f"Calcul des embeddings ({len(chunks)} chunks)...")
        vectors = embed_chunks(chunks)

        # ── Étape 3 : Stockage dans ChromaDB ─────────────────────────
        progress.progress(70, text="Stockage dans la base vectorielle...")
        _store_in_chromadb(chunks, vectors)

        # ── Étape 4 : Métadonnées ─────────────────────────────────────
        progress.progress(90, text="Enregistrement des métadonnées...")
        stats = get_chunks_stats(chunks)

        doc_meta = {
            "id": doc_id,
            "name": source_name,
            "source_type": source_type,
            "nb_chunks": stats["count"],
            "nb_chars": len(text),
            "strategy": strategy,
            "ingested_at": ingested_at,
            **extra_metadata,
        }
        add_ingested_doc(doc_meta)

        progress.progress(100, text="Ingestion terminée !")
        st.success(
            f"**{source_name}** ingéré avec succès — "
            f"{stats['count']} chunks, {len(text):,} caractères."
        )
        # Nettoyer les données temporaires de scraping
        st.session_state.pop("_scraped_url_data", None)

    except ConnectionError as exc:
        progress.empty()
        st.error(f"Ollama inaccessible : {exc}. Vérifiez qu'Ollama tourne sur `{st.session_state.get('ollama_url', 'localhost:11434')}`.")
    except Exception as exc:
        progress.empty()
        st.error(f"Erreur lors de l'ingestion : {exc}")


def _store_in_chromadb(chunks, vectors) -> None:
    """Stocke les chunks et leurs vecteurs dans ChromaDB."""
    try:
        import chromadb
        from app.config import CHROMA_COLLECTION_NAME, CHROMA_DIR

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [f"{c.doc_id}__{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )
    except ImportError as exc:
        raise ImportError("chromadb est requis : pip install chromadb") from exc


# ── Liste des documents ingérés ───────────────────────────────────────────────

def _render_ingested_docs_list() -> None:
    docs = get_ingested_docs()
    st.subheader(f"Documents ingérés ({len(docs)})")

    if not docs:
        st.info("Aucun document ingéré dans cette session.")
        return

    for doc in reversed(docs):  # Plus récent en premier
        icon = "URL" if doc.get("source_type") == "url" else "FICHIER"
        with st.expander(f"[{icon}] {doc['name']} — {doc['ingested_at']}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Chunks", doc["nb_chunks"])
            col2.metric("Caractères", f"{doc['nb_chars']:,}")
            col3.metric("Stratégie", doc["strategy"])
            if doc.get("url"):
                st.caption(f"Source : {doc['url']}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_ollama_status() -> None:
    """Affiche un indicateur de connexion Ollama dans la sidebar."""
    with st.sidebar:
        st.subheader("Statut Ollama")
        if st.button("Vérifier la connexion", key="btn_check_ollama"):
            with st.spinner("Connexion..."):
                ok = check_ollama_connection()
            if ok:
                st.success("Ollama connecté")
            else:
                st.error("Ollama inaccessible")
                st.caption("Assurez-vous qu'Ollama tourne localement.")


def _fmt_size(size_bytes: int) -> str:
    """Formate une taille en octets de manière lisible."""
    for unit in ["o", "Ko", "Mo", "Go"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} To"
