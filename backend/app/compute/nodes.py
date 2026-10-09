"""GPU / remote compute connectivity. A node is any host serving an OpenAI-compatible API
(Ollama, vLLM, llama.cpp). Probing lists its models + latency; models become Providers in one click.
No credentials are stored on the node row — tokens go through the BYOK key manager.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import httpx
from sqlmodel import Session

from app.db.telemetry import models as m
from app.providers.key_manager import key_manager

PROBE_TIMEOUT_S = 6.0


async def probe(node: m.ComputeNode, token: str | None = None) -> dict:
    base = node.base_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    t0 = time.perf_counter()
    info: dict = {"status": "offline", "models": [], "gpu_info": None, "error": None}
    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S, headers=headers) as c:
        try:
            r = await c.get(f"{root}/v1/models")
            r.raise_for_status()
            info["models"] = [x["id"] for x in r.json().get("data", [])]
            info["status"] = "online"
        except Exception as e:  # noqa: BLE001
            info["error"] = f"{type(e).__name__}: {e}"[:200]
            return info | {"latency_ms": None}
        info["latency_ms"] = (time.perf_counter() - t0) * 1000
        # Ollama exposes /api/ps (loaded models + VRAM); vLLM exposes nothing GPU-specific here.
        try:
            ps = await c.get(f"{root}/api/ps")
            if ps.status_code == 200:
                loaded = ps.json().get("models", [])
                info["gpu_info"] = ", ".join(
                    f"{x.get('name')} ({x.get('size_vram', 0) / 1e9:.1f} GB VRAM)" for x in loaded) or "idle"
        except Exception:
            pass
    return info


async def refresh(session: Session, node: m.ComputeNode) -> m.ComputeNode:
    info = await probe(node, key_manager.get(-node.id) if node.id else None)
    node.status, node.latency_ms = info["status"], info.get("latency_ms")
    node.models_json = json.dumps(info["models"])
    node.gpu_info = info["gpu_info"] or node.gpu_info
    if info["status"] == "online":
        node.last_seen = datetime.now(timezone.utc).replace(tzinfo=None)
    session.add(node); session.commit(); session.refresh(node)
    return node
