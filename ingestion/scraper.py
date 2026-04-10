"""
Scraper web robuste basé sur BeautifulSoup.

Stratégie :
1. Télécharge la page avec requests.
2. Supprime les balises de navigation (nav, header, footer, aside, script, style).
3. Extrait le texte du <main> ou <article> si disponible, sinon du <body>.
4. Nettoie les espaces et lignes vides consécutives.
"""
from __future__ import annotations

import logging
import re

import requests
from bs4 import BeautifulSoup

from app.config import SCRAPER_TIMEOUT, SCRAPER_USER_AGENT

logger = logging.getLogger(__name__)

# Balises à supprimer avant extraction du texte
_NOISE_TAGS = [
    "nav", "header", "footer", "aside",
    "script", "style", "noscript",
    "form", "button", "iframe", "svg",
    "advertisement", "figure",           # figure souvent = images sans texte utile
]

# Balises à privilégier pour le contenu principal
_CONTENT_CANDIDATES = ["main", "article", "[role='main']", ".content", "#content"]


def scrape_url(url: str) -> tuple[str, str]:
    """
    Télécharge et nettoie le texte d'une page web.

    Args:
        url: URL cible (doit commencer par http:// ou https://).

    Returns:
        Tuple (title, text_content).
        - title : titre de la page (balise <title>) ou l'URL si absent.
        - text_content : texte propre extrait.

    Raises:
        ValueError: Si l'URL est invalide.
        RuntimeError: Si la requête ou le parsing échoue.
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"URL invalide (doit commencer par http:// ou https://): '{url}'")

    # ── 1. Téléchargement ──────────────────────────────────────────────────
    try:
        response = requests.get(
            url,
            timeout=SCRAPER_TIMEOUT,
            headers={"User-Agent": SCRAPER_USER_AGENT},
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Timeout après {SCRAPER_TIMEOUT}s pour '{url}'")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Impossible de se connecter à '{url}'")
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Erreur HTTP {exc.response.status_code} pour '{url}'")

    # Vérification du content-type : on ne traite que du HTML
    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type.lower():
        raise RuntimeError(
            f"Le contenu de '{url}' n'est pas du HTML (Content-Type: {content_type})"
        )

    # ── 2. Parsing ─────────────────────────────────────────────────────────
    soup = BeautifulSoup(response.content, "lxml")

    # Titre de la page
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    # ── 3. Suppression des balises de bruit ────────────────────────────────
    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # ── 4. Extraction du contenu principal ─────────────────────────────────
    content_node = None
    for selector in _CONTENT_CANDIDATES:
        content_node = soup.select_one(selector)
        if content_node:
            logger.debug("Contenu extrait via sélecteur : '%s'", selector)
            break

    if content_node is None:
        content_node = soup.find("body")
        logger.debug("Fallback sur <body>")

    if content_node is None:
        raise RuntimeError(f"Impossible d'extraire le contenu de '{url}'")

    # ── 5. Extraction et nettoyage du texte ────────────────────────────────
    raw_text = content_node.get_text(separator="\n")
    clean = _clean_scraped_text(raw_text)

    if len(clean) < 100:
        logger.warning("Texte extrait très court (%d caractères) pour '%s'", len(clean), url)

    return title, clean


# ── Nettoyage ─────────────────────────────────────────────────────────────────

def _clean_scraped_text(text: str) -> str:
    """
    Nettoie le texte brut issu du scraping :
    - Supprime les lignes composées uniquement d'espaces.
    - Réduit les séquences de lignes vides (> 2).
    - Supprime les caractères de contrôle non imprimables.
    """
    # Suppression des caractères de contrôle (sauf \n et \t)
    text = re.sub(r"[^\S\n\t ]+", " ", text)

    # Supprimer les lignes composées uniquement d'espaces/tabulations
    lines = [line if line.strip() else "" for line in text.splitlines()]

    # Réduire les séquences de lignes vides consécutives
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
