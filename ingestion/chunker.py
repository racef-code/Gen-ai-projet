"""
Découpage intelligent du texte en chunks.

Deux stratégies disponibles (configurable dans app/config.py) :

1. "paragraph" (défaut) :
   Découpe aux doubles sauts de ligne. Respecte les frontières naturelles
   du texte. Idéal pour des documents bien structurés (articles, rapports).

2. "sliding_window" :
   Fenêtre glissante basée sur le nombre de mots (approximation des tokens).
   Garantit une taille de chunk uniforme avec chevauchement, au prix de
   couper parfois au milieu d'une phrase. Idéal pour les longs documents
   peu structurés.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNK_STRATEGY,
    MIN_CHUNK_CHARS,
)


@dataclass
class Chunk:
    """Représente un fragment de document prêt à être vectorisé."""
    text: str
    doc_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("Un chunk ne peut pas être vide.")


def chunk_text(
    text: str,
    doc_id: str,
    strategy: str = CHUNK_STRATEGY,
    extra_metadata: dict | None = None,
) -> list[Chunk]:
    """
    Point d'entrée principal.

    Args:
        text:           Texte brut du document.
        doc_id:         Identifiant unique du document source.
        strategy:       "paragraph" | "sliding_window"
        extra_metadata: Métadonnées supplémentaires à attacher à chaque chunk.

    Returns:
        Liste de Chunk, index séquentiel à partir de 0.

    Raises:
        ValueError: Si la stratégie est inconnue ou si le texte est vide.
    """
    if not text or not text.strip():
        raise ValueError("Le texte fourni est vide.")

    if strategy == "paragraph":
        raw_chunks = _split_by_paragraph(text)
    elif strategy == "sliding_window":
        raw_chunks = _split_by_sliding_window(text, CHUNK_SIZE, CHUNK_OVERLAP)
    else:
        raise ValueError(
            f"Stratégie de chunking inconnue : '{strategy}'. "
            "Valeurs acceptées : 'paragraph', 'sliding_window'."
        )

    # Filtrer les chunks trop courts
    raw_chunks = [c for c in raw_chunks if len(c.strip()) >= MIN_CHUNK_CHARS]

    metadata_base = extra_metadata or {}
    metadata_base["strategy"] = strategy

    return [
        Chunk(
            text=raw.strip(),
            doc_id=doc_id,
            chunk_index=idx,
            metadata={**metadata_base, "chunk_index": idx},
        )
        for idx, raw in enumerate(raw_chunks)
    ]


# ── Stratégie 1 : découpage par paragraphe ────────────────────────────────────

def _split_by_paragraph(text: str) -> list[str]:
    """
    Découpe aux doubles sauts de ligne.
    Fusionne les fragments très courts avec le paragraphe suivant
    pour éviter les micro-chunks.
    """
    # Normalisation : CRLF → LF, puis split aux doubles newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n{2,}", text)

    merged: list[str] = []
    buffer = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        buffer = (buffer + "\n\n" + para).strip() if buffer else para

        # On "flush" le buffer dès qu'il dépasse la taille minimale
        if len(buffer) >= MIN_CHUNK_CHARS:
            merged.append(buffer)
            buffer = ""

    # Flush du dernier buffer
    if buffer:
        if merged:
            # Fusionner avec le dernier chunk si trop court
            merged[-1] = merged[-1] + "\n\n" + buffer
        else:
            merged.append(buffer)

    return merged


# ── Stratégie 2 : fenêtre glissante ──────────────────────────────────────────

def _split_by_sliding_window(
    text: str, chunk_size: int, overlap: int
) -> list[str]:
    """
    Fenêtre glissante sur les mots.

    Args:
        text:       Texte source.
        chunk_size: Nombre de mots par chunk.
        overlap:    Nombre de mots de chevauchement entre deux chunks consécutifs.

    Note: on utilise les mots (split() whitespace) comme approximation des tokens.
    Ratio moyen : 1 token ≈ 0.75 mot en français/anglais.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"L'overlap ({overlap}) doit être inférieur à chunk_size ({chunk_size})."
        )

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start += step

    return chunks


# ── Utilitaire ────────────────────────────────────────────────────────────────

def get_chunks_stats(chunks: list[Chunk]) -> dict:
    """Retourne des statistiques simples sur une liste de chunks."""
    if not chunks:
        return {"count": 0, "avg_chars": 0, "min_chars": 0, "max_chars": 0}

    lengths = [len(c.text) for c in chunks]
    return {
        "count": len(chunks),
        "avg_chars": round(sum(lengths) / len(lengths)),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }
