from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api import analytics, core, corpus
from app.config import ROOT, settings
from app.corpus import init_corpus
from app.db.telemetry.session import init_telemetry
from app.providers.redaction import install_log_redaction


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(level=logging.INFO)
    install_log_redaction()
    init_telemetry()
    init_corpus()
    ui = "built UI served here" if (DIST / "index.html").exists() else "UI not built: run `cd frontend && npm run dev` -> http://localhost:5173"
    bar = "=" * 64
    print(f"\n{bar}\n  Red-Team Arcade is running locally\n\n  Open:  http://localhost:8000\n  API:   http://localhost:8000/docs\n  ({ui})\n{bar}\n", flush=True)
    yield


app = FastAPI(title="Red-Team Testbench", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
app.include_router(core.router)
app.include_router(analytics.router)
app.include_router(corpus.router)


@app.get("/api/health")
def health():
    return {"ok": True}


# ---- serve the built UI from the same origin (http://localhost:8000) ------------------------------
DIST = ROOT / "frontend" / "dist"

_LANDING = """<!doctype html><meta charset=utf-8><title>Red-Team Arcade API</title>
<style>body{font:16px/1.6 system-ui;max-width:640px;margin:3rem auto;padding:0 1rem;background:#0b0a1f;color:#f5f5ff}
a{color:#ffda14}code{background:#1c1a3d;padding:2px 6px}h1{color:#ffda14}</style>
<h1>Red-Team Arcade — backend is running</h1>
<p>This is the API server. The web interface is not built yet, so there is nothing to show at this address.</p>
<ul><li>Start the interface: <code>cd frontend &amp;&amp; npm run dev</code>, then open <a href="http://localhost:5173">localhost:5173</a></li>
<li>Or build it once (<code>cd frontend &amp;&amp; npm run build</code>) and reload this page to use it from here</li>
<li><a href="/docs">API documentation</a> · <a href="/api/health">health check</a> · <a href="/api/meta">settings and options</a></li></ul>"""


@app.get("/", include_in_schema=False)
def index():
    if (DIST / "index.html").exists():
        return FileResponse(DIST / "index.html")
    return HTMLResponse(_LANDING)


if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa_fallback(path: str):
    """Client-side routes (/analytics, /db, ...) fall back to the UI; unknown API paths stay 404."""
    if path.startswith(("api/", "ws/", "docs", "openapi")) or not (DIST / "index.html").exists():
        raise HTTPException(404)
    f = (DIST / path).resolve()
    if DIST.resolve() in f.parents and f.is_file():
        return FileResponse(f)
    return FileResponse(DIST / "index.html")
