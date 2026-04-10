"""
Page Tableau de Bord Analytique — Streamlit.

Affiche les 5 visualisations interactives organisées en onglets :
  1. Mots fréquents (Bar Chart)
  2. Nuage de mots (Word Cloud)
  3. Topic Cluster Map (UMAP 2D)
  4. Topic Evolution (Timeline)
  5. Similarité documents (Heatmap cosinus)

Logique de cache :
- Les calculs lourds (topic modeling, similarité) sont mis en cache dans
  st.session_state sous des clés dédiées.
- Le bouton "Actualiser" vide le cache et relance tous les calculs.
- Les topic labels LLM sont conservés entre les actualisations.
"""
from __future__ import annotations

import streamlit as st

from persistence.metadata_store import get_all_documents


def render_analytics_page() -> None:
    """Render complet du tableau de bord."""
    st.title("Tableau de Bord Analytique")

    docs = get_all_documents()
    if not docs:
        st.warning("Aucun document ingéré. Rendez-vous dans **Ingestion** pour commencer.")
        return

    # ── Barre de contrôle ──────────────────────────────────────────────────
    col_info, col_btn = st.columns([4, 1])
    with col_info:
        st.caption(
            f"{len(docs)} document(s) · "
            f"{sum(d['nb_chunks'] for d in docs)} chunks · "
            f"{sum(d['nb_chars'] for d in docs):,} caractères"
        )
    with col_btn:
        if st.button("Actualiser", type="secondary", use_container_width=True,
                     help="Recalcule tous les graphiques"):
            _clear_analytics_cache()
            st.rerun()

    st.divider()

    # ── Onglets ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Mots fréquents",
        "Nuage de mots",
        "Topic Clusters",
        "Topic Timeline",
        "Similarité",
    ])

    with tab1:
        _render_word_frequency(docs)

    with tab2:
        _render_wordcloud(docs)

    with tab3:
        _render_topic_clusters(docs)

    with tab4:
        _render_topic_timeline(docs)

    with tab5:
        _render_similarity_heatmap(docs)


# ── Tab 1 : Mots fréquents ────────────────────────────────────────────────────

def _render_word_frequency(docs: list[dict]) -> None:
    from analytics.text_stats import get_word_frequencies
    from analytics.visualizations import plot_word_frequency

    st.subheader("Mots les plus fréquents")
    st.caption("Stopwords FR/EN filtrés automatiquement.")

    col1, col2 = st.columns([1, 3])
    with col1:
        top_n = st.slider("Top N mots", 10, 50, 20, key="wf_top_n")
        extra_sw = st.text_input(
            "Stopwords supplémentaires",
            placeholder="mot1, mot2, ...",
            key="wf_extra_sw",
            help="Mots séparés par des virgules à exclure de l'analyse.",
        )
        extra_sw_set = {w.strip().lower() for w in extra_sw.split(",") if w.strip()} if extra_sw else set()

        doc_filter = st.multiselect(
            "Filtrer par document",
            options=[d["id"] for d in docs],
            format_func=lambda k: next(d["name"] for d in docs if d["id"] == k),
            key="wf_doc_filter",
        )

    cache_key = f"wf_{top_n}_{hash(frozenset(extra_sw_set))}_{hash(tuple(sorted(doc_filter)))}"

    if cache_key not in st.session_state:
        with st.spinner("Calcul des fréquences..."):
            word_freq = get_word_frequencies(
                doc_ids=doc_filter or None,
                top_n=top_n,
                extra_stopwords=extra_sw_set,
            )
        st.session_state[cache_key] = word_freq
    else:
        word_freq = st.session_state[cache_key]

    with col2:
        if word_freq:
            st.plotly_chart(plot_word_frequency(word_freq, top_n), use_container_width=True)
        else:
            st.info("Aucun mot trouvé avec ces paramètres.")


# ── Tab 2 : Nuage de mots ─────────────────────────────────────────────────────

def _render_wordcloud(docs: list[dict]) -> None:
    from analytics.text_stats import get_word_frequencies
    from analytics.visualizations import plot_wordcloud

    st.subheader("Nuage de Mots")

    if "wc_image" not in st.session_state:
        with st.spinner("Génération du nuage de mots..."):
            word_freq = get_word_frequencies(top_n=200)
            img = plot_wordcloud(word_freq)
        st.session_state["wc_image"] = img
        st.session_state["wc_freq"] = word_freq
    else:
        img = st.session_state["wc_image"]

    if img is None:
        st.warning(
            "La bibliothèque `wordcloud` n'est pas installée. "
            "Exécutez : `pip install wordcloud`"
        )
        # Fallback : liste des mots les plus fréquents
        if "wc_freq" in st.session_state and st.session_state["wc_freq"]:
            freq = st.session_state["wc_freq"]
            st.write("**Top 30 mots (fallback texte) :**")
            st.write(", ".join(f"`{w}` ({c})" for w, c in freq.most_common(30)))
    else:
        st.image(img, use_container_width=True, caption="Pondéré par la fréquence d'apparition")


# ── Tab 3 : Topic Cluster Map ─────────────────────────────────────────────────

def _render_topic_clusters(docs: list[dict]) -> None:
    from analytics.visualizations import plot_topic_clusters

    st.subheader("Topic Cluster Map (UMAP 2D)")
    st.caption(
        "Chaque point est un chunk de document. "
        "La couleur représente le thème détecté par BERTopic."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        force_refit = st.checkbox(
            "Forcer le re-calcul",
            key="tc_force_refit",
            help="Refit complet de BERTopic (plus lent mais plus précis après ajout de docs).",
        )
        run_autolabel = st.checkbox(
            "Auto-label LLM",
            value=True,
            key="tc_autolabel",
            help="Demande au LLM de générer des labels descriptifs pour chaque cluster.",
        )

    state_key = "topic_model_state"

    if force_refit and state_key in st.session_state:
        del st.session_state[state_key]

    if state_key not in st.session_state:
        with st.spinner("Topic modeling en cours (BERTopic + UMAP)..."):
            try:
                from analytics.topic_model import fit_or_update, auto_label_topics
                state = fit_or_update(force_refit=force_refit)
                if state and run_autolabel:
                    with st.spinner("Auto-labelling des topics via LLM..."):
                        auto_label_topics(state)
                st.session_state[state_key] = state
            except ImportError:
                st.error(
                    "BERTopic ou UMAP non installés.\n\n"
                    "```bash\npip install bertopic umap-learn\n```"
                )
                return
            except Exception as exc:
                st.error(f"Erreur topic modeling : {exc}")
                return

    state = st.session_state.get(state_key)
    if state is None:
        st.info(
            f"Pas assez de chunks pour le topic modeling "
            f"(minimum requis : {5} chunks)."
        )
        return

    from persistence.vector_store import get_all_embeddings
    raw = get_all_embeddings()
    texts = raw.get("documents", [])
    metadatas = raw.get("metadatas", [])
    chunk_doc_ids = [m.get("doc_id", "") for m in metadatas]

    with col2:
        fig = plot_topic_clusters(
            embeddings_2d=state.last_embeddings_2d,
            topics=state.last_topics,
            texts=texts,
            topic_labels=state.topic_labels,
            doc_ids=chunk_doc_ids,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Légende des topics
    if state.topic_labels:
        st.divider()
        st.subheader("Labels des thèmes")
        cols = st.columns(min(len(state.topic_labels), 3))
        for i, (tid, label) in enumerate(
            sorted(state.topic_labels.items(), key=lambda x: x[0])
        ):
            cols[i % 3].info(f"**Topic {tid}** → {label}")


# ── Tab 4 : Topic Timeline ────────────────────────────────────────────────────

def _render_topic_timeline(docs: list[dict]) -> None:
    from analytics.visualizations import plot_topic_timeline

    st.subheader("Évolution des Topics par Document")
    st.caption(
        "Distribution thématique cumulée à mesure que les documents ont été ajoutés."
    )

    state = st.session_state.get("topic_model_state")
    if state is None:
        st.info(
            "Calculez d'abord le Topic Cluster Map (onglet précédent) "
            "pour voir la timeline."
        )
        return

    from analytics.topic_model import get_topic_evolution
    evolution = get_topic_evolution(state, docs)

    if not evolution:
        st.info("Pas assez de données pour afficher le timeline.")
        return

    fig = plot_topic_timeline(evolution)
    st.plotly_chart(fig, use_container_width=True)

    # Tableau de données brutes
    with st.expander("Voir les données brutes"):
        import pandas as pd
        all_topics = sorted({t for dist in evolution.values() for t in dist})
        rows = []
        for doc_name, dist in evolution.items():
            row = {"Document": doc_name}
            row.update({t: dist.get(t, 0) for t in all_topics})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Document"), use_container_width=True)


# ── Tab 5 : Heatmap de similarité ─────────────────────────────────────────────

def _render_similarity_heatmap(docs: list[dict]) -> None:
    from analytics.visualizations import plot_similarity_heatmap

    st.subheader("Heatmap de Similarité Cosinus inter-Documents")
    st.caption(
        "Similarité calculée sur le vecteur moyen des chunks de chaque document. "
        "1.0 = identiques, 0.0 = orthogonaux."
    )

    if len(docs) < 2:
        st.info("Ajoutez au moins **2 documents** pour afficher la heatmap.")
        return

    if "sim_matrix" not in st.session_state:
        with st.spinner("Calcul de la matrice de similarité..."):
            from analytics.similarity import compute_similarity_matrix
            sim_df = compute_similarity_matrix(docs)
        st.session_state["sim_matrix"] = sim_df
    else:
        sim_df = st.session_state["sim_matrix"]

    fig = plot_similarity_heatmap(sim_df)
    st.plotly_chart(fig, use_container_width=True)

    if sim_df is not None:
        # Paires les plus similaires
        st.divider()
        st.subheader("Paires les plus similaires")
        _render_top_similar_pairs(sim_df, top_k=5)


def _render_top_similar_pairs(sim_df, top_k: int = 5) -> None:
    """Affiche les K paires de documents les plus similaires (hors diagonale)."""
    import pandas as pd
    names = list(sim_df.columns)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j], sim_df.iloc[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)
    top = pairs[:top_k]

    if not top:
        return

    df_pairs = pd.DataFrame(top, columns=["Document A", "Document B", "Similarité"])
    df_pairs["Similarité"] = df_pairs["Similarité"].round(3)
    st.dataframe(df_pairs, use_container_width=True, hide_index=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clear_analytics_cache() -> None:
    """Vide les clés de cache analytics dans le session_state."""
    keys_to_clear = [
        k for k in st.session_state
        if k.startswith("wf_") or k in ("wc_image", "wc_freq", "sim_matrix")
    ]
    # Ne pas supprimer topic_model_state (trop long à recalculer)
    for k in keys_to_clear:
        del st.session_state[k]
