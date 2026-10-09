"""Pluggable defenses (SPEC §13.6). Drop a class + @register to add one."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

DEFENSE_REGISTRY: dict[str, "Defense"] = {}


@dataclass
class DefenseResult:
    blocked: bool
    transformed: str


class Defense:
    name = ""
    def apply(self, payload: str) -> DefenseResult: raise NotImplementedError


def register(cls):
    DEFENSE_REGISTRY[cls.name] = cls()
    return cls


def _char_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


@register
class PerplexityFilter(Defense):
    """Proxy for a perplexity filter: block payload bodies with near-random char entropy."""
    name = "perplexity_filter"
    def apply(self, payload: str) -> DefenseResult:
        body = payload.split("]", 1)[-1].strip()
        return DefenseResult(len(body) > 24 and _char_entropy(body) > 5.2, payload)


@register
class EncodingSignatureGate(Defense):
    name = "encoding_signature_gate"
    _b64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
    def apply(self, payload: str) -> DefenseResult:
        return DefenseResult(bool(self._b64.search(payload)), payload)
