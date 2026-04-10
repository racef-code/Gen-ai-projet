"""
Statistiques textuelles : fréquences de mots avec filtrage des stopwords.

Design :
- Stopwords FR + EN embarqués directement (pas de dépendance NLTK/download).
- Récupère les textes depuis ChromaDB (déjà stockés, pas de re-lecture).
- Filtre les tokens courts, la ponctuation et les chiffres seuls.
- Retourne un Counter trié pour faciliter la visualisation.
"""
from __future__ import annotations

import re
from collections import Counter


# ── Stopwords intégrés ────────────────────────────────────────────────────────

STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
    "en", "et", "ou", "à", "ce", "se", "sa", "son", "ses", "mon", "ma",
    "mes", "ton", "ta", "tes", "leur", "leurs", "notre", "votre", "nos",
    "vos", "il", "elle", "ils", "elles", "je", "tu", "nous", "vous", "on",
    "qui", "que", "quoi", "dont", "où", "quand", "comment", "pourquoi",
    "que", "quel", "quelle", "quels", "quelles", "si", "car", "mais",
    "par", "sur", "sous", "dans", "entre", "pour", "avec", "sans", "plus",
    "pas", "ne", "ni", "non", "oui", "très", "bien", "être", "avoir",
    "faire", "dit", "cette", "tout", "tous", "toute", "toutes", "même",
    "comme", "aussi", "alors", "donc", "ainsi", "cela", "ceci", "ceux",
    "est", "sont", "était", "ont", "a", "y", "en", "d", "l", "j", "s",
    "qu", "n", "c", "m",
}

STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "that", "this", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "i", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "which", "who", "what", "when", "where", "how",
    "if", "not", "no", "nor", "so", "yet", "both", "either", "each",
    "all", "any", "more", "most", "other", "some", "such", "than",
    "then", "there", "here", "also", "into", "about", "up", "out",
    "over", "after", "before", "between", "through", "during", "within",
}

ALL_STOPWORDS = STOPWORDS_FR | STOPWORDS_EN

# Longueur minimale d'un mot pour être comptabilisé
_MIN_WORD_LEN = 3


# ── API principale ────────────────────────────────────────────────────────────

def get_word_frequencies(
    doc_ids: list[str] | None = None,
    top_n: int = 50,
    extra_stopwords: set[str] | None = None,
) -> Counter:
    """
    Calcule les fréquences des mots dans les documents ingérés.

    Args:
        doc_ids:          Si fourni, limite l'analyse à ces documents.
        top_n:            Nombre de mots les plus fréquents à retourner.
        extra_stopwords:  Stopwords supplémentaires définis par l'utilisateur.

    Returns:
        Counter {mot: fréquence} trié par fréquence décroissante.
    """
    from persistence.vector_store import get_all_embeddings

    data = get_all_embeddings(doc_id=doc_ids[0] if doc_ids and len(doc_ids) == 1 else None)
    documents = data.get("documents", [])

    if not documents:
        return Counter()

    # Si plusieurs doc_ids, filtrer manuellement
    if doc_ids and len(doc_ids) > 1:
        metas = data.get("metadatas", [])
        documents = [
            doc for doc, meta in zip(documents, metas)
            if meta.get("doc_id") in set(doc_ids)
        ]

    stopwords = ALL_STOPWORDS.copy()
    if extra_stopwords:
        stopwords |= {w.lower() for w in extra_stopwords}

    counter = Counter()
    for text in documents:
        words = _tokenize(text)
        filtered = [w for w in words if w not in stopwords and len(w) >= _MIN_WORD_LEN]
        counter.update(filtered)

    return Counter(dict(counter.most_common(top_n)))


def get_all_text(doc_ids: list[str] | None = None) -> str:
    """Retourne le texte concaténé de tous les chunks (utile pour le word cloud)."""
    from persistence.vector_store import get_all_embeddings
    data = get_all_embeddings()
    documents = data.get("documents", [])

    if doc_ids:
        metas = data.get("metadatas", [])
        documents = [
            doc for doc, meta in zip(documents, metas)
            if meta.get("doc_id") in set(doc_ids)
        ]

    return " ".join(documents)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """
    Tokenisation simple :
    - Minuscules
    - Conserve uniquement les caractères alphabétiques (unicode)
    - Supprime les tokens purement numériques
    """
    # Remplace tout ce qui n'est pas une lettre unicode par un espace
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    words = text.lower().split()
    # Exclure les tokens numériques et ceux avec des underscores
    return [w for w in words if w.isalpha()]
