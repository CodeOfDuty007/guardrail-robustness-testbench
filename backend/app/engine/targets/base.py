"""Target abstraction (LiteLLMTarget analogue). Providers never touch orchestration code."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.providers.key_manager import key_manager


class ProviderError(RuntimeError):
    """Auth/billing/bad-model/transport failure: NOT a model timeout and must never be scored as a refusal."""


@dataclass
class TargetResult:
    text: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class Target(Protocol):
    model_name: str
    provider_kind: str

    async def send(self, system: str, user: str) -> TargetResult: ...


class OpenAICompatTarget:
    """Any OpenAI-compatible endpoint: Ollama (/v1), vLLM, llama.cpp server, LM Studio, cloud.

    This is the path used for GPU nodes. 30s hard timeout, exponential-backoff retries (SPEC §15).
    """

    def __init__(self, base_url: str, model_name: str, provider_id: int | None = None,
                 provider_kind: str = "openai_compat", timeout_s: float = 30.0,
                 retries: int = 2, seed: int | None = None,
                 price_prompt_1k: float = 0.0, price_completion_1k: float = 0.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name, self.provider_kind = model_name, provider_kind
        self.provider_id, self.timeout_s, self.retries, self.seed = provider_id, timeout_s, retries, seed
        self.price_p, self.price_c = price_prompt_1k, price_completion_1k

    async def send(self, system: str, user: str) -> TargetResult:
        import time
        headers = {}
        if (key := key_manager.get(self.provider_id)):
            headers["Authorization"] = f"Bearer {key}"
        url = self.base_url + ("/chat/completions" if self.base_url.endswith(("/v1", "/openai")) else "/v1/chat/completions")
        body = {"model": self.model_name, "temperature": 0.0, "messages": [
            *([{"role": "system", "content": system}] if system else []),
            {"role": "user", "content": user}]}
        if self.seed is not None and "googleapis.com" not in self.base_url:  # Gemini's OpenAI layer rejects `seed`
            body["seed"] = self.seed
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=self.timeout_s) as c:
                    r = await c.post(url, json=body, headers=headers)
                    r.raise_for_status()
                data = r.json()
                usage = data.get("usage") or {}
                pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
                return TargetResult(
                    text=data["choices"][0]["message"]["content"] or "",
                    latency_ms=(time.perf_counter() - t0) * 1000, prompt_tokens=pt, completion_tokens=ct,
                    cost_usd=pt / 1000 * self.price_p + ct / 1000 * self.price_c,
                )
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if 400 <= code < 500 and code not in (408, 429):  # auth/billing/bad model: retrying cannot help
                    raise ProviderError(f"target {self.model_name} rejected the request (HTTP {code})") from None
                last = e
                wait = 0.5 * 2 ** attempt
                if code == 429:  # honour the provider's own retry hint (Retry-After / Gemini retryDelay), capped
                    import re
                    hint = e.response.headers.get("retry-after") or (re.search(r'"retryDelay":\s*"([\d.]+)s"', e.response.text) or [None, None])[1]
                    try:
                        wait = min(float(hint) + 0.5, 15.0)
                    except (TypeError, ValueError):
                        pass
                await asyncio.sleep(wait)
            except (httpx.HTTPError, KeyError, ValueError) as e:
                last = e
                await asyncio.sleep(0.5 * 2 ** attempt)
        if isinstance(last, httpx.TimeoutException):
            raise TimeoutError(f"target {self.model_name} timed out after {self.retries + 1} tries")
        raise ProviderError(f"target {self.model_name} failed after {self.retries + 1} tries: {type(last).__name__}")


class LiteLLMTarget:
    """Optional: routes through LiteLLM for first-party clouds (needs `pip install .[providers]`)."""

    def __init__(self, model_name: str, provider_id: int | None = None, base_url: str | None = None,
                 provider_kind: str = "litellm", seed: int | None = None) -> None:
        self.model_name, self.provider_id, self.base_url = model_name, provider_id, base_url
        self.provider_kind, self.seed = provider_kind, seed

    async def send(self, system: str, user: str) -> TargetResult:
        import time
        try:
            import litellm
        except ImportError as e:
            raise RuntimeError("litellm not installed; run `uv pip install -e .[providers]`") from e
        t0 = time.perf_counter()
        resp = await litellm.acompletion(
            model=self.model_name, api_key=key_manager.get(self.provider_id), api_base=self.base_url,
            timeout=30, seed=self.seed, temperature=0,
            messages=[*([{"role": "system", "content": system}] if system else []),
                      {"role": "user", "content": user}])
        u = resp.usage
        return TargetResult(
            text=resp.choices[0].message.content or "", latency_ms=(time.perf_counter() - t0) * 1000,
            prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
            cost_usd=float(litellm.completion_cost(completion_response=resp) or 0.0))
