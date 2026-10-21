"""Tiny Mixture-of-Experts causal LM for the REX toy reproduction (plan §B.1).

d_model~128, 2 layers, 8 experts, top-2 routing, ~1M params, single-process CPU.

* Experts live in an ``nn.ModuleList`` per layer and the router is a ``(n_experts, d_model)`` parameter,
  so experts can be **added** (expansion) or **physically removed** (excision) together with their
  router rows (see ``relocate.py``).
* ``forward(..., active=mask)`` hides experts from routing by setting their router logits to ``-inf``
  before top-k. The physically excised model is numerically identical to the full model run with the
  target experts masked out (tested), which is how relocation trains against the "excised view".
* Experts are computed densely (tiny model) and combined with the sparse top-k gates; this keeps the
  code short and deterministic. Dropout is 0 everywhere.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class MoEConfig:
    vocab_size: int
    max_len: int = 16
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    n_experts: int = 8
    top_k: int = 2

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.experts = nn.ModuleList(Expert(d_model, d_ff) for _ in range(n_experts))
        self.router = nn.Parameter(torch.randn(n_experts, d_model) * 0.02)

    @property
    def n_experts(self) -> int:
        return len(self.experts)

    def forward(self, x: torch.Tensor, active: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, router_probs over all experts). ``active``: bool (n_experts,) mask."""
        logits = x @ self.router.t()                                   # (B,T,E)
        if active is not None:
            logits = logits.masked_fill(~active, float("-inf"))
        probs = logits.softmax(-1)                                     # router distribution (for losses)
        top_val, top_idx = logits.topk(self.top_k, dim=-1)
        gates = torch.zeros_like(logits).scatter(-1, top_idx, top_val.softmax(-1))  # renormalised top-k
        outs = torch.stack([e(x) for e in self.experts], dim=-2)      # (B,T,E,D) dense, tiny
        return (gates.unsqueeze(-1) * outs).sum(-2), probs


class Block(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=0.0, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.moe = MoELayer(cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.top_k)

    def forward(self, x: torch.Tensor, causal: torch.Tensor,
                active: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.ln1(x)
        # right-padded batches + causal mask: real tokens never attend to padding, so no pad mask
        a, _ = self.attn(h, h, h, attn_mask=causal, need_weights=False)
        x = x + a
        m, probs = self.moe(self.ln2(x), active)
        return x + m, probs


class ToyMoE(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    @property
    def n_experts(self) -> int:
        return self.moe_layers()[0].n_experts

    def moe_layers(self) -> list[MoELayer]:
        return [cast(Block, b).moe for b in self.blocks]

    def forward(self, ids: torch.Tensor, active: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """ids (B,T) -> (logits (B,T,V), [router_probs (B,T,E) per layer])."""
        T = ids.shape[1]
        if T > self.cfg.max_len:
            raise ValueError(f"sequence length {T} > max_len {self.cfg.max_len}")
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        all_probs = []
        for b in self.blocks:
            x, p = b(x, causal, active)
            all_probs.append(p)
        return self.head(self.ln_f(x)), all_probs


def build_model(cfg: MoEConfig, seed: int) -> ToyMoE:
    """Seeded init that does not depend on (or disturb) the global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ToyMoE(cfg)
    return model


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
