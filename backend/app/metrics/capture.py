"""MetricCapture — the novel layer (SPEC §10). Pure functions + a wrapper around a Cipher."""
from __future__ import annotations

import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass

from app.engine.converters.ciphers import Cipher


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in Counter(data).values())


def chi2_uniform(data: bytes) -> float:
    """Chi-square of byte histogram vs uniform over 256 symbols (≈255 for random data)."""
    if not data:
        return 0.0
    exp = len(data) / 256
    counts = Counter(data)
    return sum((counts.get(i, 0) - exp) ** 2 / exp for i in range(256))


def _bit_diff(a: bytes, b: bytes) -> tuple[int, int]:
    n = min(len(a), len(b))
    diff = sum(bin(x ^ y).count("1") for x, y in zip(a[:n], b[:n]))
    return diff, n * 8


def avalanche(encode, plaintext: bytes, trials: int = 10, rng: random.Random | None = None) -> float:
    """Mean fraction of ciphertext bits changed when ONE plaintext bit is flipped."""
    if not plaintext:
        return 0.0
    rng = rng or random.Random(0)
    c1 = encode(plaintext)
    ratios = []
    for _ in range(trials):
        p = bytearray(plaintext)
        p[rng.randrange(len(p))] ^= 1 << rng.randrange(8)
        d, total = _bit_diff(c1, encode(bytes(p)))
        if total:
            ratios.append(d / total)
    return sum(ratios) / len(ratios) if ratios else 0.0


def diffusion(encode, plaintext: bytes, trials: int = 10, rng: random.Random | None = None) -> float:
    """Mean fraction of OUTPUT BYTES that change when ONE input byte changes (spread of change)."""
    if not plaintext:
        return 0.0
    rng = rng or random.Random(0)
    c1 = encode(plaintext)
    vals = []
    for _ in range(trials):
        p = bytearray(plaintext)
        i = rng.randrange(len(p))
        p[i] = (p[i] + 1 + rng.randrange(255)) % 256
        c2 = encode(bytes(p))
        n = min(len(c1), len(c2))
        if n:
            vals.append(sum(x != y for x, y in zip(c1[:n], c2[:n])) / n)
    return sum(vals) / len(vals) if vals else 0.0


@dataclass
class CryptoMetrics:
    encryption_time_ms: float
    decryption_time_ms: float
    avalanche_score: float
    confusion_chi2: float
    diffusion_score: float
    shannon_entropy: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class MetricCapture:
    """Wraps a Cipher: times encode/decode and records the crypto metrics for each call."""

    def __init__(self, cipher: Cipher, trials: int = 10) -> None:
        self.cipher, self.trials = cipher, trials
        self.last: CryptoMetrics | None = None

    def record(self, plaintext: str) -> tuple[str, CryptoMetrics]:
        raw = plaintext.encode()
        t0 = time.perf_counter()
        ct = self.cipher.encode_bytes(raw)
        t1 = time.perf_counter()
        self.cipher.decode_bytes(ct)
        t2 = time.perf_counter()
        rng = random.Random(self.cipher.seed)  # deterministic metrics given seed
        m = CryptoMetrics(
            encryption_time_ms=(t1 - t0) * 1000,
            decryption_time_ms=(t2 - t1) * 1000,
            avalanche_score=avalanche(self.cipher.metric_encode_bytes, raw, self.trials, rng),
            confusion_chi2=chi2_uniform(ct),
            diffusion_score=diffusion(self.cipher.metric_encode_bytes, raw, self.trials, rng),
            shannon_entropy=shannon_entropy(ct),
        )
        self.last = m
        return ct.decode("latin-1"), m


def random_bytes(n: int) -> bytes:
    return os.urandom(n)
