"""Providers, runs, explorer, compute, meta + WebSocket endpoints."""
from __future__ import annotations

import asyncio
import json
import re
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import text
from sqlmodel import Session, select

from app.compute import nodes as compute
from app.config import settings
from app.datasets.loaders import list_datasets
from app.db.telemetry import models as m
from app.db.telemetry.session import get_engine, get_telemetry_session
from app.defenses import DEFENSE_REGISTRY
from app.engine import pyrit_bridge
from app.engine.converters.ciphers import CIPHER_REGISTRY
from app.engine.orchestrator import MOCK_PROFILES, RunConfig, create_run, execute_run
from app.engine.strategies import STRATEGY_REGISTRY
from app.providers.key_manager import key_manager
from app.providers.redaction import redact
from app.ws.hub import hub

router = APIRouter()
_tasks: set[asyncio.Task] = set()


# ---- meta -------------------------------------------------------------------------------------
@router.get("/api/meta")
def meta():
    return {
        "ciphers": [{"name": n, "label": c.label} for n, c in CIPHER_REGISTRY.items()],
        "strategies": [{"name": n, "adaptive": s.adaptive, "pyrit_native": s.pyrit_native}
                       for n, s in STRATEGY_REGISTRY.items()],
        "defenses": list(DEFENSE_REGISTRY), "datasets": list_datasets(),
        "mock_models": list(MOCK_PROFILES), "pyrit": pyrit_bridge.status(),
        "operator": settings.operator_name, "default_seed": settings.default_seed,
    }


# ---- providers (keys never persisted here) -----------------------------------------------------
class ProviderIn(BaseModel):
    label: str
    kind: str
    model_name: str
    base_url: str | None = None
    role_hint: str = "target"
    compute_node_id: int | None = None
    cost_per_1k_prompt: float = 0.0
    cost_per_1k_completion: float = 0.0


class KeyIn(BaseModel):
    key: str
    remember: bool = False


def _prov(p: m.Provider) -> dict:
    return p.model_dump() | {"has_key": key_manager.has(p.id)}


@router.get("/api/providers")
def list_providers(s: Session = Depends(get_telemetry_session)):
    return [_prov(p) for p in s.exec(select(m.Provider)).all()]


@router.post("/api/providers")
def add_provider(body: ProviderIn, s: Session = Depends(get_telemetry_session)):
    p = m.Provider(**body.model_dump()); s.add(p); s.commit(); s.refresh(p)
    return _prov(p)


@router.put("/api/providers/{pid}/key")
def set_key(pid: int, body: KeyIn, s: Session = Depends(get_telemetry_session)):
    if not s.get(m.Provider, pid):
        raise HTTPException(404)
    key_manager.set(pid, body.key, body.remember)
    return {"ok": True, "has_key": True}  # never echo the key


@router.delete("/api/providers/{pid}")
def del_provider(pid: int, s: Session = Depends(get_telemetry_session)):
    p = s.get(m.Provider, pid)
    if p:
        s.delete(p); s.commit()
    key_manager.delete(pid)
    return {"ok": True}


@router.post("/api/providers/{pid}/validate")
async def validate_provider(pid: int, s: Session = Depends(get_telemetry_session)):
    from app.engine.orchestrator import resolve_target
    p = s.get(m.Provider, pid)
    if not p:
        raise HTTPException(404)
    try:
        r = await asyncio.wait_for(resolve_target(p, 0).send("", "ping"), 15)
        return {"ok": True, "latency_ms": r.latency_ms}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": redact(f"{type(e).__name__}: {e}")[:200]}


# ---- compute nodes (GPU connectivity) ----------------------------------------------------------
class NodeIn(BaseModel):
    label: str
    base_url: str
    kind: str = "ollama"
    token: str | None = None


def _node(n: m.ComputeNode) -> dict:
    return n.model_dump(exclude={"models_json"}) | {"models": json.loads(n.models_json)}


@router.get("/api/compute/nodes")
def list_nodes(s: Session = Depends(get_telemetry_session)):
    return [_node(n) for n in s.exec(select(m.ComputeNode)).all()]


@router.post("/api/compute/nodes")
async def add_node(body: NodeIn, s: Session = Depends(get_telemetry_session)):
    if not re.match(r"^https?://", body.base_url):
        raise HTTPException(422, "base_url must start with http:// or https://")
    host = (urlparse(body.base_url).hostname or "")
    if host.startswith("169.254.") or host in ("metadata.google.internal", "0.0.0.0"):
        raise HTTPException(422, "link-local / metadata hosts are not allowed")
    n = m.ComputeNode(label=body.label, base_url=body.base_url, kind=body.kind)
    s.add(n); s.commit(); s.refresh(n)
    if body.token:
        key_manager.set(-n.id, body.token)
    return _node(await compute.refresh(s, n))


@router.post("/api/compute/nodes/{nid}/probe")
async def probe_node(nid: int, s: Session = Depends(get_telemetry_session)):
    n = s.get(m.ComputeNode, nid)
    if not n:
        raise HTTPException(404)
    return _node(await compute.refresh(s, n))


@router.post("/api/compute/nodes/{nid}/adopt")
def adopt_models(nid: int, models: list[str], role: str = "target", s: Session = Depends(get_telemetry_session)):
    """Register selected models from a node as Providers (so they appear in New Experiment)."""
    n = s.get(m.ComputeNode, nid)
    if not n:
        raise HTTPException(404)
    made = []
    for name in models:
        p = m.Provider(label=f"{n.label}:{name}", kind=n.kind if n.kind != "ollama" else "ollama",
                       model_name=name, base_url=n.base_url, role_hint=role, compute_node_id=n.id)
        s.add(p); made.append(p)
    s.commit()
    token = key_manager.get(-n.id)
    for p in made:
        s.refresh(p)
        if token:
            key_manager.set(p.id, token)
    return [_prov(p) for p in made]


@router.delete("/api/compute/nodes/{nid}")
def del_node(nid: int, s: Session = Depends(get_telemetry_session)):
    n = s.get(m.ComputeNode, nid)
    if n:
        s.delete(n); s.commit()
    key_manager.delete(-nid)
    return {"ok": True}


# ---- runs --------------------------------------------------------------------------------------
@router.post("/api/runs")
async def start_run(cfg: RunConfig):
    rid = create_run(cfg)
    t = asyncio.create_task(execute_run(rid, cfg))
    _tasks.add(t); t.add_done_callback(_tasks.discard)
    return {"id": rid}


@router.get("/api/runs")
def list_runs(s: Session = Depends(get_telemetry_session)):
    return s.exec(select(m.Run).order_by(m.Run.id.desc())).all()


@router.get("/api/runs/{rid}")
def get_run(rid: int, s: Session = Depends(get_telemetry_session)):
    r = s.get(m.Run, rid)
    if not r:
        raise HTTPException(404)
    return r


@router.get("/api/runs/{rid}/attempts")
def run_attempts(rid: int, s: Session = Depends(get_telemetry_session)):
    return s.exec(select(m.Attempt).where(m.Attempt.run_id == rid).order_by(m.Attempt.id)).all()


@router.get("/api/runs/{rid}/config.yaml")
def run_yaml(rid: int, s: Session = Depends(get_telemetry_session)):
    import yaml
    from fastapi.responses import PlainTextResponse
    r = s.get(m.Run, rid)
    if not r:
        raise HTTPException(404)
    return PlainTextResponse(yaml.safe_dump(json.loads(r.config_json), sort_keys=False))


@router.websocket("/ws/live")
async def ws_live(ws: WebSocket, run_id: int | None = None):
    origin = ws.headers.get("origin")
    if origin and origin not in settings.cors_origins and origin.split("://", 1)[-1] != ws.headers.get("host"):
        await ws.close(code=1008)  # cross-site WebSocket hijacking guard
        return
    await ws.accept()
    q = hub.subscribe(run_id)
    try:
        while True:
            await ws.send_json(await q.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.unsubscribe(run_id, q)


# ---- DB explorer (telemetry only) --------------------------------------------------------------
TABLES = {"runs": m.Run, "attempts": m.Attempt, "payloads": m.Payload, "responses": m.Response,
          "judgments": m.Judgment, "defenses": m.DefenseApplied, "disclosures": m.Disclosure}


EXPORT_CAP = 100_000
REDACT_COLS = {"payloads": ["plaintext_prompt", "ciphertext_payload", "decode_instruction"],
               "responses": ["target_response"], "judgments": ["judge_rationale"], "defenses": ["transformed_payload"]}


def _scope_of(table: str, rows: list[dict], s: Session) -> dict:
    """Which runs / models / datasets / seeds the exported rows came from (provenance, SPEC §1)."""
    if table == "runs":
        run_ids = {r["id"] for r in rows}
    elif table == "attempts":
        run_ids = {r["run_id"] for r in rows}
    elif table == "disclosures":
        return {"models": sorted({r["target_model"] for r in rows})}
    else:
        att_ids = {r["attempt_id"] for r in rows if "attempt_id" in r}
        run_ids = {a.run_id for a in s.exec(select(m.Attempt).where(m.Attempt.id.in_(att_ids))).all()} if att_ids else set()
    runs = s.exec(select(m.Run).where(m.Run.id.in_(run_ids))).all() if run_ids else []
    models = {r.target_model for r in s.exec(select(m.Response).join(m.Attempt, m.Attempt.id == m.Response.attempt_id)
                                             .where(m.Attempt.run_id.in_(run_ids))).all()} if run_ids else set()
    return {"runs": sorted(run_ids), "models": sorted(models), "datasets": sorted({r.dataset_name for r in runs}),
            "seeds": sorted({r.seed for r in runs})}


def _explorer_query(model, session, q: str | None, eligible: bool, sort: str | None, desc: bool):
    from sqlalchemy import String, cast, func, or_
    stmt = select(model)
    if eligible and model is m.Judgment:
        stmt = stmt.where(m.Judgment.filter_evaded == True, m.Judgment.semantic_complied == True)  # noqa: E712
    if q:
        stmt = stmt.where(or_(*[cast(c, String).ilike(f"%{q}%") for c in model.__table__.c]))
    total = session.exec(select(func.count()).select_from(stmt.subquery())).one()
    col = model.__table__.c.get(sort) if sort else None
    col = col if col is not None else model.__table__.c.id
    return stmt.order_by(col.desc() if desc else col.asc()), total


@router.get("/api/explorer/{table}")
def explorer(table: str, limit: int = 100, offset: int = 0, q: str | None = None, eligible: bool = False,
             sort: str | None = None, desc: bool = True, s: Session = Depends(get_telemetry_session)):
    model = TABLES.get(table)
    if not model:
        raise HTTPException(404, "unknown table")
    stmt, total = _explorer_query(model, s, q, eligible, sort, desc)
    rows = [r.model_dump() for r in s.exec(stmt.offset(max(0, offset)).limit(min(max(1, limit), 1000))).all()]
    return {"total": total, "rows": rows, "offset": offset}


@router.get("/api/explorer/{table}/export")
def explorer_export(table: str, format: str = "csv", redact: bool = True, q: str | None = None,
                    eligible: bool = False, s: Session = Depends(get_telemetry_session)):
    """Exports carry a provenance header; raw payloads/responses are redacted unless redact=false (SPEC §1)."""
    import csv
    import io
    from datetime import datetime, timezone

    from fastapi.responses import Response
    model = TABLES.get(table)
    if not model:
        raise HTTPException(404, "unknown table")
    stmt, total = _explorer_query(model, s, q, eligible, None, False)
    rows = [r.model_dump(mode="json") for r in s.exec(stmt.limit(EXPORT_CAP)).all()]
    if redact:
        for r in rows:
            for c in REDACT_COLS.get(table, []):
                if c in r:
                    r[c] = "[redacted]"
    exported = len(rows)
    prov = {"operator": settings.operator_name, "generated_at": datetime.now(timezone.utc).isoformat(),
            "table": table, "rows_exported": exported, "rows_matching": total, "truncated": total > exported,
            "search": q or None, "promote_eligible_only": eligible, "redacted": redact, "license": "CC-BY-4.0",
            **_scope_of(table, rows, s), "note": "keys never stored; raw text redacted by default"}
    if format == "json":
        return Response(json.dumps({"provenance": prov, "rows": rows}, indent=1, default=str), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{table}.json"'})
    buf = io.StringIO()
    buf.write("".join(f"# {k}: {v}\n" for k, v in prov.items()))
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{table}.csv"'})


class SqlIn(BaseModel):
    sql: str


SQL_DEADLINE_S = 3.0


@router.post("/api/explorer-sql")
def explorer_sql(body: SqlIn):
    import time
    sql = body.sql.strip().rstrip(";")
    if len(sql) > 4000:
        raise HTTPException(400, "query too long (4000 characters max)")
    if not re.match(r"(?is)^(select|with)\b", sql) or ";" in sql or "--" in sql or "/*" in sql or re.search(r"(?i)\b(attach|pragma|insert|update|delete|drop|alter|create|replace)\b", sql):
        raise HTTPException(400, "single read-only SELECT (or WITH … SELECT) statements only")
    with get_engine().connect() as c:
        raw = c.connection.driver_connection
        end = time.monotonic() + SQL_DEADLINE_S
        c.execute(text("PRAGMA query_only=ON"))
        raw.set_progress_handler(lambda: 1 if time.monotonic() > end else 0, 10_000)  # abort runaway queries
        try:
            res = c.execute(text(sql))
            return {"columns": list(res.keys()), "rows": [list(r) for r in res.fetchmany(500)]}
        except Exception as e:  # noqa: BLE001
            if "interrupted" in str(e).lower():
                raise HTTPException(408, f"query stopped after {SQL_DEADLINE_S:.0f}s — add filters or a LIMIT") from None
            raise HTTPException(400, "query failed: " + str(e).split("\n")[0][:200]) from None
        finally:  # the pooled connection must go back clean
            raw.set_progress_handler(None, 0)
            c.execute(text("PRAGMA query_only=OFF"))
