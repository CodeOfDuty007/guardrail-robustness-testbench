"""DuckDB read replica over telemetry.db (SPEC §3). Analytics reads ONLY this — never the corpus.

The flat attempt table is rebuilt from SQLite (one join) whenever the source version changes, so
heavy aggregations run in DuckDB and never contend with the orchestrator's writes.
"""
from __future__ import annotations

import contextlib
import contextvars
import re
import threading
from dataclasses import dataclass

import duckdb
from sqlalchemy import text

from app.db.telemetry.session import get_engine

_FLAT_SQL = """
SELECT a.id AS attempt_id, a.run_id, a.iteration, a.objective_id, a.strategy_name, a.cipher_type,
       COALESCE(a.defense_name, 'none') AS defense, a.status, a.created_at,
       r.target_model, r.latency_ms, r.cost_usd, r.prompt_tokens, r.completion_tokens,
       p.plaintext_size_bytes, p.ciphertext_size_bytes, p.encryption_time_ms, p.decryption_time_ms,
       p.avalanche_score, p.confusion_chi2, p.diffusion_score, p.shannon_entropy,
       j.filter_evaded, j.semantic_complied, (j.filter_evaded AND j.semantic_complied) AS bypass,
       j.regex_verdict, j.judge_lm_verdict, j.llamaguard_verdict, j.score_source,
       u.owasp_tag, u.atlas_tag, u.dataset_name, u.seed, u.name AS run_name
FROM attempts a
JOIN payloads p ON p.attempt_id = a.id
JOIN responses r ON r.attempt_id = a.id
JOIN judgments j ON j.attempt_id = a.id
JOIN runs u ON u.id = a.run_id
"""
_lock = threading.RLock()
_con: duckdb.DuckDBPyConnection | None = None
_db_key: str | None = None
_pinned: contextvars.ContextVar[duckdb.DuckDBPyConnection | None] = contextvars.ContextVar("pinned", default=None)
_TYPES = {"attempt_id": "BIGINT", "run_id": "BIGINT", "iteration": "INTEGER", "created_at": "TIMESTAMP",
          "filter_evaded": "BOOLEAN", "semantic_complied": "BOOLEAN", "bypass": "BOOLEAN",
          "regex_verdict": "BOOLEAN", "judge_lm_verdict": "BOOLEAN", "llamaguard_verdict": "BOOLEAN"}
_TEXT = {"objective_id", "strategy_name", "cipher_type", "defense", "status", "target_model", "score_source",
         "owasp_tag", "atlas_tag", "dataset_name", "run_name"}
_BOOLS = ("filter_evaded", "semantic_complied", "bypass", "regex_verdict", "judge_lm_verdict", "llamaguard_verdict")
# aggregate over columns that can change in place; compared on the SAME id set in source and replica
_AGG_SRC = ("SELECT COUNT(*), COALESCE(SUM(j.filter_evaded*1000003 + j.semantic_complied),0), COALESCE(SUM(r.cost_usd),0), "
            "COALESCE(SUM(LENGTH(COALESCE(u.owasp_tag,''))+LENGTH(COALESCE(u.atlas_tag,''))+LENGTH(u.name)),0) "
            "FROM attempts a JOIN payloads p ON p.attempt_id=a.id JOIN responses r ON r.attempt_id=a.id "
            "JOIN judgments j ON j.attempt_id=a.id JOIN runs u ON u.id=a.run_id WHERE a.id <= :mx")
_AGG_REP = ("SELECT COUNT(*), COALESCE(SUM(CAST(filter_evaded AS INT)*1000003 + CAST(semantic_complied AS INT)),0), "
            "COALESCE(SUM(cost_usd),0), COALESCE(SUM(LENGTH(COALESCE(owasp_tag,''))+LENGTH(COALESCE(atlas_tag,''))+LENGTH(run_name)),0) "
            "FROM flat WHERE attempt_id <= ?")


@dataclass
class Filters:
    model: str | None = None
    cipher: str | None = None
    strategy: str | None = None
    run_id: int | None = None
    defense: str | None = None
    date_from: str | None = None
    date_to: str | None = None

    def where(self) -> tuple[str, list]:
        c, a = [], []
        for col, v in (("target_model", self.model), ("cipher_type", self.cipher), ("strategy_name", self.strategy),
                       ("run_id", self.run_id), ("defense", self.defense)):
            if v not in (None, ""):
                c.append(f"{col} = ?"); a.append(v)
        if self.date_from:
            c.append("created_at >= CAST(? AS TIMESTAMP)"); a.append(self.date_from)
        if self.date_to:
            c.append("created_at < CAST(? AS TIMESTAMP) + INTERVAL 1 DAY"); a.append(self.date_to)
        return (" WHERE " + " AND ".join(c)) if c else "", a


def _fetch(where: str = "", params: dict | None = None) -> tuple[list[str], list[tuple]]:
    with get_engine().connect() as c:
        res = c.execute(text(_FLAT_SQL + where), params or {})
        return list(res.keys()), [tuple(r) for r in res.fetchall()]


def _insert(con: duckdb.DuckDBPyConnection, cols: list[str], rows: list[tuple]) -> None:
    if not rows:
        return
    fixed = [tuple(bool(v) if n in _BOOLS and v is not None else v for n, v in zip(cols, r)) for r in rows]
    con.executemany(f"INSERT INTO flat VALUES ({','.join('?' * len(cols))})", fixed)


def _full_rebuild() -> duckdb.DuckDBPyConnection:
    cols, rows = _fetch()
    con = duckdb.connect(":memory:")
    ddl = ", ".join(f'"{n}" {_TYPES.get(n, "VARCHAR" if n in _TEXT else "DOUBLE")}' for n in cols)
    con.execute(f"CREATE TABLE flat ({ddl})")
    _insert(con, cols, rows)
    return con


def _refresh(force: bool = False) -> duckdb.DuckDBPyConnection:
    """Keep the replica current: append only NEW completed attempts; rebuild fully if existing rows were edited.

    Old connections are not closed explicitly — a pinned report may still be reading from one.
    """
    global _con, _db_key
    key = str(get_engine().url)
    if force or _con is None or key != _db_key:
        _con, _db_key = _full_rebuild(), key
        return _con
    have = {r[0] for r in _con.execute("SELECT attempt_id FROM flat").fetchall()}
    mx = max(have, default=0)
    with get_engine().connect() as c:  # aggregate over ids the replica already holds
        src = tuple(c.execute(text(_AGG_SRC), {"mx": mx}).one())
        done = {r[0] for r in c.execute(text("SELECT a.id FROM attempts a JOIN payloads p ON p.attempt_id=a.id "
                                             "JOIN responses r ON r.attempt_id=a.id JOIN judgments j ON j.attempt_id=a.id")).all()}
    rep = tuple(_con.execute(_AGG_REP, [mx]).fetchone())
    if not have <= done or tuple(round(float(x), 9) for x in src) != tuple(round(float(x), 9) for x in rep):
        _con = _full_rebuild()  # rows were edited or deleted in place
        return _con
    new = sorted(done - have)  # ids can finish out of order, so diff the sets rather than using max(id)
    for i in range(0, len(new), 500):
        ids = new[i:i + 500]
        cols, rows = _fetch(f" WHERE a.id IN ({','.join(str(int(x)) for x in ids)})")
        _insert(_con, cols, rows)
    return _con


@contextlib.contextmanager
def snapshot():
    """Pin every query inside the block to ONE replica state (consistent multi-section reports)."""
    with _lock:
        con = _refresh()
    token = _pinned.set(con)
    try:
        yield
    finally:
        _pinned.reset(token)


def query(sql: str, params: list | None = None) -> list[dict]:
    """Run SQL against the replica; `flat` is the table. Returns list of dicts."""
    con = _pinned.get()
    if con is None:
        with _lock:
            con = _refresh()
    cur = con.cursor().execute(sql, params or [])  # own cursor: safe alongside other readers
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def flat_rows(f: Filters, columns: str = "*", extra: str = "") -> list[dict]:
    w, a = f.where()
    return query(f"SELECT {columns} FROM flat{w} {extra}", a)


_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")


def model_size_b(name: str) -> float | None:
    """Parameter count (billions) parsed from names like llama3:8b / tinyllama:1.1b / mock-weak-1b."""
    m = _SIZE_RE.search(name.replace("-", " ").replace(":", " "))
    return float(m.group(1)) if m else None
