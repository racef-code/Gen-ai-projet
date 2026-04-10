"""
Les 5 visualisations interactives du tableau de bord.

Toutes les fonctions retournent soit un objet plotly.graph_objects.Figure,
soit une image PIL (word cloud), utilisables directement dans Streamlit.

Visualisations :
  1. plot_word_frequency()   → Bar chart horizontal des mots fréquents
  2. plot_wordcloud()        → Nuage de mots (image PIL)
  3. plot_topic_clusters()   → Scatter UMAP 2D coloré par topic
  4. plot_topic_timeline()   → Stacked bar chart par document
  5. plot_similarity_heatmap()→ Heatmap cosinus inter-documents
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Palette de couleurs cohérente entre les graphiques
_PALETTE = px.colors.qualitative.Plotly


# ── 1. Bar Chart — Mots fréquents ─────────────────────────────────────────────

def plot_word_frequency(word_freq: Counter, top_n: int = 20) -> go.Figure:
    """
    Bar chart horizontal des mots les plus fréquents.

    Args:
        word_freq: Counter {mot: fréquence}.
        top_n:     Nombre de mots à afficher.

    Returns:
        Figure Plotly interactive.
    """
    if not word_freq:
        return _empty_figure("Aucun texte disponible pour l'analyse de fréquence.")

    items = word_freq.most_common(top_n)
    words = [w for w, _ in reversed(items)]
    counts = [c for _, c in reversed(items)]

    fig = go.Figure(go.Bar(
        x=counts,
        y=words,
        orientation="h",
        marker=dict(
            color=counts,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Fréquence"),
        ),
        hovertemplate="<b>%{y}</b><br>Fréquence : %{x}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Top {top_n} mots les plus fréquents", font=dict(size=16)),
        xaxis_title="Fréquence",
        yaxis_title="Mot",
        height=max(350, top_n * 22),
        margin=dict(l=120, r=40, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(tickfont=dict(size=12)),
    )
    return fig


# ── 2. Word Cloud ─────────────────────────────────────────────────────────────

def plot_wordcloud(word_freq: Counter, width: int = 800, height: int = 400):
    """
    Génère un nuage de mots à partir des fréquences.

    Returns:
        Image PIL si wordcloud est installé, None sinon.
    """
    if not word_freq:
        return None

    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("wordcloud non installé — pip install wordcloud")
        return None

    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        colormap="viridis",
        max_words=150,
        min_font_size=10,
        prefer_horizontal=0.85,
    ).generate_from_frequencies(word_freq)

    return wc.to_image()


# ── 3. Topic Cluster Map — Scatter UMAP 2D ────────────────────────────────────

def plot_topic_clusters(
    embeddings_2d: list[list[float]],
    topics: list[int],
    texts: list[str],
    topic_labels: dict[int, str],
    doc_ids: list[str] | None = None,
) -> go.Figure:
    """
    Scatter plot 2D (UMAP) où chaque point est un chunk, coloré par topic.

    Args:
        embeddings_2d: Coordonnées 2D [[x, y], ...].
        topics:        Assignation de topic pour chaque chunk.
        texts:         Textes des chunks (pour le tooltip).
        topic_labels:  {topic_id: label_string}.
        doc_ids:       doc_id de chaque chunk (pour le tooltip).
    """
    if not embeddings_2d or not topics:
        return _empty_figure("Pas encore assez de données pour le Topic Cluster Map.")

    coords = np.array(embeddings_2d)
    if coords.shape[1] != 2:
        return _empty_figure("Coordonnées UMAP invalides.")

    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "topic_id": topics,
        "doc_id": doc_ids if doc_ids else [""] * len(topics),
        "text_preview": [t[:120] + "..." if len(t) > 120 else t for t in texts],
    })

    # Label lisible pour chaque point
    df["topic_label"] = df["topic_id"].apply(
        lambda tid: "Hors-sujet" if tid == -1
        else topic_labels.get(tid, f"Topic {tid}")
    )

    # Ordre : outliers en dernier
    unique_labels = sorted(
        df["topic_label"].unique(),
        key=lambda l: (l == "Hors-sujet", l)
    )
    color_map = {
        label: ("lightgray" if label == "Hors-sujet" else _PALETTE[i % len(_PALETTE)])
        for i, label in enumerate(l for l in unique_labels if l != "Hors-sujet")
    }
    color_map["Hors-sujet"] = "lightgray"

    traces = []
    for label in unique_labels:
        subset = df[df["topic_label"] == label]
        traces.append(go.Scatter(
            x=subset["x"],
            y=subset["y"],
            mode="markers",
            name=label,
            marker=dict(
                size=7,
                color=color_map[label],
                opacity=0.75 if label != "Hors-sujet" else 0.35,
                line=dict(width=0.5, color="white"),
            ),
            customdata=subset[["text_preview", "doc_id"]].values,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Extrait : %{customdata[0]}<br>"
                "Doc : %{customdata[1]}<extra></extra>"
            ),
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text="Topic Cluster Map (UMAP 2D)", font=dict(size=16)),
        xaxis=dict(title="UMAP-1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP-2", showgrid=False, zeroline=False),
        height=520,
        legend=dict(title="Thèmes", itemsizing="constant"),
        plot_bgcolor="rgba(240,240,240,0.3)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


# ── 4. Topic Evolution Timeline ───────────────────────────────────────────────

def plot_topic_timeline(topic_evolution: dict[str, dict[str, int]]) -> go.Figure:
    """
    Stacked bar chart montrant la distribution des topics par document.

    Args:
        topic_evolution: {doc_name: {topic_label: count}}
                         (issu de topic_model.get_topic_evolution()).
    """
    if not topic_evolution:
        return _empty_figure("Pas encore assez de données pour le Topic Timeline.")

    # Collect tous les topics
    all_topics: set[str] = set()
    for topic_counts in topic_evolution.values():
        all_topics.update(topic_counts.keys())

    doc_names = list(topic_evolution.keys())
    topic_list = sorted(all_topics, key=lambda t: (t == "Hors-sujet", t))

    traces = []
    for i, topic in enumerate(topic_list):
        values = [topic_evolution[doc].get(topic, 0) for doc in doc_names]
        color = "lightgray" if topic == "Hors-sujet" else _PALETTE[i % len(_PALETTE)]
        traces.append(go.Bar(
            name=topic,
            x=doc_names,
            y=values,
            marker_color=color,
            hovertemplate="<b>%{x}</b><br>%{fullData.name} : %{y} chunks<extra></extra>",
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        barmode="stack",
        title=dict(text="Évolution des Topics par Document", font=dict(size=16)),
        xaxis=dict(title="Document", tickangle=-30),
        yaxis=dict(title="Nombre de chunks"),
        height=420,
        legend=dict(title="Thème", traceorder="normal"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=80),
    )
    return fig


# ── 5. Heatmap de similarité cosinus ─────────────────────────────────────────

def plot_similarity_heatmap(sim_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap de la similarité cosinus entre documents.

    Args:
        sim_df: DataFrame N×N (issu de similarity.compute_similarity_matrix()).
    """
    if sim_df is None or sim_df.empty:
        return _empty_figure(
            "Ajoutez au moins 2 documents pour afficher la heatmap de similarité."
        )

    names = list(sim_df.columns)
    values = sim_df.values

    # Annotations : valeur arrondie dans chaque cellule
    annotations = []
    for i in range(len(names)):
        for j in range(len(names)):
            annotations.append(dict(
                x=names[j],
                y=names[i],
                text=f"{values[i, j]:.2f}",
                showarrow=False,
                font=dict(size=11, color="black" if values[i, j] < 0.7 else "white"),
            ))

    fig = go.Figure(go.Heatmap(
        z=values,
        x=names,
        y=names,
        colorscale="RdYlGn",
        zmin=0.0,
        zmax=1.0,
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>Similarité : %{z:.3f}<extra></extra>",
        colorbar=dict(title="Similarité<br>cosinus"),
    ))

    fig.update_layout(
        title=dict(text="Heatmap de Similarité Cosinus inter-Documents", font=dict(size=16)),
        height=max(350, len(names) * 60 + 120),
        xaxis=dict(tickangle=-35, automargin=True),
        yaxis=dict(automargin=True),
        margin=dict(l=40, r=40, t=50, b=80),
        annotations=annotations,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Helper ────────────────────────────────────────────────────────────────────

def _empty_figure(message: str) -> go.Figure:
    """Retourne une figure vide avec un message centré."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
