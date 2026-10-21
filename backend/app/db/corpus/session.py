"""Corpus engine. Import ONLY from app/corpus/ and app/api/corpus.py (SPEC §5b, §6b)."""
from __future__ import annotations

import os
from collections.abc import Iterator

from sqlalchemy import text
from sqlmodel import Session, create_engine

from app.config import settings
from app.db.corpus.models import CORPUS_DDL, corpus_registry

_engine = None


def get_corpus_engine():
    global _engine
    if _engine is None:
        settings.corpus_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(settings.corpus_dir, 0o700)
        _engine = create_engine(settings.corpus_url, connect_args={"check_same_thread": False})
    return _engine


def reset_corpus_engine() -> None:
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def init_corpus() -> None:
    eng = get_corpus_engine()
    corpus_registry.metadata.create_all(eng)
    with eng.begin() as conn:
        for stmt in CORPUS_DDL:
            conn.execute(text(stmt))
    db_file = settings.corpus_dir / "corpus.db"
    if db_file.exists():
        os.chmod(db_file, 0o600)


def get_corpus_session() -> Iterator[Session]:
    with Session(get_corpus_engine()) as s:
        yield s
