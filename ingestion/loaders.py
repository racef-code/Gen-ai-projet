"""
Loaders de documents : PDF, TXT, DOCX, Markdown → texte brut.

Chaque loader reçoit un objet file-like (BytesIO ou chemin) et retourne
une chaîne de texte nettoyée.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

from app.config import MAX_FILE_SIZE_MB

logger = logging.getLogger(__name__)

# ── Types supportés ───────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def load_document(file_bytes: bytes, filename: str) -> str:
    """
    Point d'entrée unique.

    Args:
        file_bytes: Contenu brut du fichier (issu de st.file_uploader).
        filename:   Nom original du fichier (pour détecter l'extension).

    Returns:
        Texte extrait et nettoyé.

    Raises:
        ValueError: Si l'extension n'est pas supportée ou si le fichier dépasse
                    la limite de taille configurée (MAX_FILE_SIZE_MB).
        RuntimeError: Si l'extraction échoue.
    """
    # ── Validation de la taille ───────────────────────────────────────────
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"Fichier trop volumineux ({size_mb:.1f} Mo). "
            f"Limite configurée : {MAX_FILE_SIZE_MB} Mo."
        )

    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Extension '{ext}' non supportée. "
            f"Formats acceptés : {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    loaders = {
        ".pdf": _load_pdf,
        ".txt": _load_txt,
        ".md": _load_txt,   # Markdown = texte brut
        ".docx": _load_docx,
    }

    try:
        text = loaders[ext](file_bytes)
    except Exception as exc:
        raise RuntimeError(f"Erreur lors du chargement de '{filename}': {exc}") from exc

    return _clean_text(text)


# ── Loaders internes ──────────────────────────────────────────────────────────

def _load_pdf(file_bytes: bytes) -> str:
    """Extrait le texte d'un PDF avec pypdf (100 % local, sans OCR)."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("pypdf est requis : pip install pypdf") from e

    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    if not pages_text:
        logger.warning("Aucun texte extrait du PDF (PDF scanné ?).")
        return ""

    return "\n\n".join(pages_text)


def _load_txt(file_bytes: bytes) -> str:
    """Décode un fichier texte brut ou Markdown (UTF-8 avec fallback latin-1)."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


def _load_docx(file_bytes: bytes) -> str:
    """Extrait le texte d'un fichier DOCX paragraphe par paragraphe."""
    try:
        from docx import Document
    except ImportError as e:
        raise ImportError("python-docx est requis : pip install python-docx") from e

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


# ── Nettoyage ─────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Nettoyage minimal :
    - Supprime les lignes vides multiples consécutives (> 2).
    - Retire les espaces de début/fin.
    - Normalise les sauts de ligne Windows (CRLF → LF).
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Réduire les séquences de lignes vides
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
