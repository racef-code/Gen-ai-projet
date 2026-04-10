"""
Persistance des métadonnées via SQLite (SQLAlchemy Core).

Tables :
  - documents   : un enregistrement par document ingéré.
  - chat_history: historique des échanges (optionnel, pour la persistance inter-sessions).

Le fichier DB est créé automatiquement dans data/metadata.db.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
    delete,
)
from sqlalchemy.engine import Engine

from app.config import DB_PATH

logger = logging.getLogger(__name__)

# ── Engine SQLite ─────────────────────────────────────────────────────────────
_engine: Engine | None = None
_metadata = MetaData()

# ── Schéma ────────────────────────────────────────────────────────────────────
documents_table = Table(
    "documents",
    _metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("source_type", String, nullable=False),   # "file" | "url"
    Column("nb_chunks", Integer, nullable=False),
    Column("nb_chars", Integer, nullable=False),
    Column("strategy", String, nullable=False),
    Column("ingested_at", String, nullable=False),   # ISO 8601
    Column("extra_json", Text, nullable=True),        # JSON pour les champs optionnels (url, etc.)
)

chat_history_table = Table(
    "chat_history",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String, nullable=False),
    Column("role", String, nullable=False),           # "user" | "assistant"
    Column("content", Text, nullable=False),
    Column("sources_json", Text, nullable=True),      # JSON des chunks sources
    Column("created_at", String, nullable=False),
)


def _get_engine() -> Engine:
    """Retourne l'engine SQLite (création si absent)."""
    global _engine
    if _engine is None:
        db_url = f"sqlite:///{DB_PATH}"
        _engine = create_engine(db_url, echo=False, future=True)
        _metadata.create_all(_engine)
        logger.info("SQLite initialisé : %s", DB_PATH)
    return _engine


# ── Documents ─────────────────────────────────────────────────────────────────

def save_document(doc_meta: dict) -> None:
    """
    Insère ou met à jour un document dans la table `documents`.

    Args:
        doc_meta: Dict avec au minimum les clés :
                  id, name, source_type, nb_chunks, nb_chars, strategy, ingested_at.
                  Les champs supplémentaires sont sérialisés dans extra_json.
    """
    engine = _get_engine()

    known_keys = {"id", "name", "source_type", "nb_chunks", "nb_chars", "strategy", "ingested_at"}
    extra = {k: v for k, v in doc_meta.items() if k not in known_keys}

    row = {
        "id": doc_meta["id"],
        "name": doc_meta["name"],
        "source_type": doc_meta["source_type"],
        "nb_chunks": doc_meta["nb_chunks"],
        "nb_chars": doc_meta["nb_chars"],
        "strategy": doc_meta["strategy"],
        "ingested_at": doc_meta["ingested_at"],
        "extra_json": json.dumps(extra) if extra else None,
    }

    with engine.begin() as conn:
        # Upsert manuel : delete + insert
        conn.execute(
            delete(documents_table).where(documents_table.c.id == row["id"])
        )
        conn.execute(documents_table.insert().values(**row))

    logger.debug("Document sauvegardé : %s", row["id"])


def get_all_documents() -> list[dict]:
    """
    Retourne tous les documents ingérés, triés par date d'ingestion décroissante.
    """
    engine = _get_engine()

    with engine.connect() as conn:
        rows = conn.execute(
            select(documents_table).order_by(documents_table.c.ingested_at.desc())
        ).fetchall()

    results = []
    for row in rows:
        doc = dict(row._mapping)
        if doc.get("extra_json"):
            extra = json.loads(doc.pop("extra_json"))
            doc.update(extra)
        else:
            doc.pop("extra_json", None)
        results.append(doc)

    return results


def delete_document(doc_id: str) -> bool:
    """Supprime un document des métadonnées. Retourne True si trouvé."""
    engine = _get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            delete(documents_table).where(documents_table.c.id == doc_id)
        )
    return result.rowcount > 0


def get_document_count() -> int:
    """Retourne le nombre de documents ingérés."""
    engine = _get_engine()
    with engine.connect() as conn:
        return conn.execute(select(documents_table)).fetchall().__len__()


# ── Historique du chat ────────────────────────────────────────────────────────

def save_chat_message(
    session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> None:
    """Persiste un message du chat (user ou assistant)."""
    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(chat_history_table.insert().values(
            session_id=session_id,
            role=role,
            content=content,
            sources_json=json.dumps(sources) if sources else None,
            created_at=datetime.now().isoformat(timespec="seconds"),
        ))


def get_chat_history(session_id: str) -> list[dict]:
    """Retourne l'historique du chat pour une session donnée."""
    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(chat_history_table)
            .where(chat_history_table.c.session_id == session_id)
            .order_by(chat_history_table.c.id)
        ).fetchall()

    history = []
    for row in rows:
        msg = dict(row._mapping)
        if msg.get("sources_json"):
            msg["sources"] = json.loads(msg.pop("sources_json"))
        else:
            msg.pop("sources_json", None)
        history.append(msg)
    return history


def clear_chat_history(session_id: str) -> None:
    """Supprime l'historique du chat pour une session."""
    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(
            delete(chat_history_table).where(
                chat_history_table.c.session_id == session_id
            )
        )
