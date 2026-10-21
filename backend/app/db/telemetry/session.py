from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from app.config import settings
from app.db.telemetry import models  # noqa: F401  (register tables)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.telemetry_url, connect_args={"check_same_thread": False})

        @event.listens_for(_engine, "connect")
        def _pragmas(dbapi_conn, _):  # WAL so analytics reads never block writes
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.close()

    return _engine


def reset_engine() -> None:
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


# Columns added after the first public release. `create_all` never ALTERs an existing table, so an
# existing telemetry.db gets them via an idempotent ADD COLUMN (SQLite has no IF NOT EXISTS for columns).
_ADDED_COLUMNS = {"attempts": {"family": "VARCHAR", "category": "VARCHAR", "channel": "VARCHAR", "delivery": "VARCHAR",
                              "technique": "VARCHAR", "generated_by": "VARCHAR"}}


def _migrate(engine) -> None:
    with engine.begin() as c:
        for table, cols in _ADDED_COLUMNS.items():
            have = {r[1] for r in c.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()}
            for name, typ in cols.items():
                if name not in have:
                    c.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")


def init_telemetry() -> None:
    SQLModel.metadata.create_all(get_engine())
    _migrate(get_engine())


def get_telemetry_session() -> Iterator[Session]:
    with Session(get_engine()) as s:
        yield s
