"""Billing/auth failures must surface as ProviderError, never as a fake timeout (and never leak keys)."""
import asyncio

import httpx
import pytest

from app.engine.targets.base import OpenAICompatTarget, ProviderError


def _target(status: int):
    t = OpenAICompatTarget("http://x/v1", "m", retries=2)

    class Client:
        calls = 0

        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

        async def post(self, url, json=None, headers=None):
            Client.calls += 1
            return httpx.Response(status, request=httpx.Request("POST", url), json={"error": "secret-body"})
    return t, Client


def test_402_is_provider_error_without_retry(monkeypatch):
    t, C = _target(402)
    monkeypatch.setattr(httpx, "AsyncClient", C)
    with pytest.raises(ProviderError) as e:
        asyncio.run(t.send("", "hi"))
    assert "402" in str(e.value) and "secret-body" not in str(e.value) and C.calls == 1


def test_exhausted_5xx_is_provider_error_not_timeout(monkeypatch):
    t, C = _target(503)
    monkeypatch.setattr(httpx, "AsyncClient", C)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *_: real_sleep(0))
    with pytest.raises(ProviderError):
        asyncio.run(t.send("", "hi"))
