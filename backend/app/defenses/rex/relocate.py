"""REX at toy scale: expert expansion, selective gradient masking + routing differentiation, excision.

Reimplementation (not vendored) of the three mechanisms described in
  Jin et al., "Towards Architectural-Level Safety Control: Relocating Hazardous Knowledge into
  Removable Experts in Mixture-of-Experts Models", arXiv:2610.25791 (Shanghai AI Lab), CC BY 4.0.
  Reference code: github.com/AI45Lab/REX (Apache-2.0; targets 26-35B MoE, so not vendored).

Pipeline (:func:`run_rex_pipeline`):
1. **base**: train the toy MoE on forget + retain data; it learns both (the "pretrained" model).
2. **expand** (:func:`expand_experts`): add ``k`` target experts per layer, each a copy of an
   existing expert; router rows are extended with copies (+ tiny seeded noise to break ties).
3. **relocate** (:func:`relocate`): each step combines four losses, each routed by an explicit
   gradient mask (``torch.autograd.grad`` restricted to an allowed parameter set):
   * forget LM loss on the full model            -> **target experts only**
   * routing loss (forget tokens -> targets, retain tokens -> originals) -> **router only**
   * retain LM loss on full + excised view       -> original experts + shared (attention/embeddings)
   * push-out loss on forget data, excised view (targets masked): pushes the original path's
     prediction of the forget answer toward uniform  -> **original experts only**. This is our toy
     reading of "relocates hazardous knowledge *out of* the original experts"; it can be disabled
     (``push_out_weight=0``) and the pipeline reports that ablation honestly.
4. **excise** (:func:`excise_experts`): physically delete target experts and their router rows ->
   a strictly smaller model, numerically identical to the full model with targets masked.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from app.defenses.rex.data import Corpora, DataConfig, Tokenizer, generate_corpora
from app.defenses.rex.evaluate import (
    IGNORE, Batch, accuracy, answer_prob, encode_examples, route_to_experts, seq_len,
)
from app.defenses.rex.toy_moe import MoEConfig, ToyMoE, build_model, param_count

PAPER = ("Jin et al., Towards Architectural-Level Safety Control: Relocating Hazardous Knowledge into "
         "Removable Experts in Mixture-of-Experts Models, arXiv:2610.25791 (CC BY 4.0)")
DEFAULT_TMP = Path(__file__).resolve().parents[3] / ".tmp" / "rex"   # backend/.tmp (git-ignored)


@dataclass(frozen=True)
class RexConfig:
    data: DataConfig = field(default_factory=DataConfig)
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    n_experts: int = 8
    top_k: int = 2
    k_target: int = 2
    base_steps: int = 600
    relocate_steps: int = 400
    forget_batch: int = 32
    retain_batch: int = 64
    base_lr: float = 3e-3
    relocate_lr: float = 2e-3
    balance_weight: float = 0.01
    routing_weight: float = 1.0
    push_out_weight: float = 1.0
    num_threads: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fast_config() -> RexConfig:
    """Seconds-scale config used by unit tests."""
    return RexConfig(data=DataConfig(n_forget=24, n_retain_train=800, n_retain_test=200, copy_len=3,
                                     n_symbols=12),
                     d_model=64, d_ff=128, base_steps=300, relocate_steps=200)


def default_config() -> RexConfig:
    """Minutes-scale CPU config (d_model 128, 2 layers, 8 experts, top-2)."""
    return RexConfig()


# ---------------------------------------------------------------- architecture surgery

def expand_experts(model: ToyMoE, k: int, gen: torch.Generator) -> ToyMoE:
    """Return a copy with ``k`` target experts appended per layer (copies of existing experts)."""
    new = copy.deepcopy(model)
    n = model.n_experts
    for layer in new.moe_layers():
        for j in range(k):
            layer.experts.append(copy.deepcopy(layer.experts[j % n]))
        rows = layer.router.data[[j % n for j in range(k)]].clone()
        rows += 1e-3 * torch.randn(rows.shape, generator=gen)
        layer.router = nn.Parameter(torch.cat([layer.router.data, rows], 0))
    new.cfg = replace(model.cfg, n_experts=n + k)
    return new


def excise_experts(model: ToyMoE, remove: list[int]) -> ToyMoE:
    """Physically delete experts ``remove`` (and their router rows) from every layer."""
    new = copy.deepcopy(model)
    keep = [i for i in range(model.n_experts) if i not in set(remove)]
    if len(keep) < model.cfg.top_k:
        raise ValueError("cannot excise below top_k experts")
    for layer in new.moe_layers():
        layer.experts = nn.ModuleList(layer.experts[i] for i in keep)
        layer.router = nn.Parameter(layer.router.data[keep].clone())
    new.cfg = replace(model.cfg, n_experts=len(keep))
    return new


def active_mask(n_experts: int, disabled: list[int]) -> torch.Tensor:
    m = torch.ones(n_experts, dtype=torch.bool)
    m[disabled] = False
    return m


# ---------------------------------------------------------------- training helpers

def _batches(n: int, size: int, gen: torch.Generator) -> Iterator[torch.Tensor]:
    """Endless seeded epochs of shuffled index batches (``0 < size <= n``)."""
    if not 0 < size <= n:
        raise ValueError(f"need 0 < batch size {size} <= dataset size {n}")
    while True:
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n - size + 1, size):
            yield perm[s: s + size]


def _lm_loss(logits: torch.Tensor, b: Batch) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b.targets.reshape(-1), ignore_index=IGNORE)


def _balance_loss(probs: list[torch.Tensor], real: torch.Tensor, top_k: int) -> torch.Tensor:
    """Switch-style load-balancing loss (keeps all experts in use during base training)."""
    total = torch.zeros(())
    for p in probs:
        pr = p[real]                                                  # (N,E)
        E = pr.shape[-1]
        frac = torch.zeros(E).scatter_add(0, pr.topk(top_k, -1).indices.reshape(-1),
                                          torch.ones(pr.shape[0] * top_k)) / (pr.shape[0] * top_k)
        total = total + E * (frac * pr.mean(0)).sum()
    return total / len(probs)


def _routing_loss(probs: list[torch.Tensor], b: Batch, experts: list[int]) -> torch.Tensor:
    """-log(router mass on ``experts``) over real tokens, skipping BOS (shared by both domains)."""
    sel = b.real.clone(); sel[:, 0] = False
    vals = [-(p[..., experts].sum(-1).clamp_min(1e-9).log())[sel].mean() for p in probs]
    return torch.stack(vals).mean()


def _push_out_loss(logits: torch.Tensor, b: Batch) -> torch.Tensor:
    """Unlikelihood -log(1 - p(answer)) on forget-answer positions (-> 0 as the answer is dropped)."""
    lg = logits[b.acc_mask]                                           # (N,V)
    tgt = b.targets[b.acc_mask]
    others = lg.scatter(-1, tgt.unsqueeze(-1), float("-inf"))
    return (lg.logsumexp(-1) - others.logsumexp(-1)).mean()


def _accumulate(loss: torch.Tensor, params: list[nn.Parameter]) -> None:
    """Selective gradient masking: add d(loss)/d(params) to ``params`` only; nothing else gets grad."""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    for p, g in zip(params, grads):
        if g is None:
            continue
        p.grad = g if p.grad is None else p.grad + g


def train_base(model: ToyMoE, forget: Batch, retain: Batch, cfg: RexConfig, gen: torch.Generator) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=cfg.base_lr)
    fb = _batches(len(forget), min(cfg.forget_batch, len(forget)), gen)
    rb = _batches(len(retain), min(cfg.retain_batch, len(retain)), gen)
    model.train()
    for _ in range(cfg.base_steps):
        f, r = forget.take(next(fb)), retain.take(next(rb))
        loss = torch.zeros(())
        for b in (f, r):
            logits, probs = model(b.ids)
            loss = loss + _lm_loss(logits, b) + cfg.balance_weight * _balance_loss(probs, b.real, cfg.top_k)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def param_groups(model: ToyMoE, n_orig: int) -> dict[str, list[nn.Parameter]]:
    target: list[nn.Parameter] = []
    orig: list[nn.Parameter] = []
    router: list[nn.Parameter] = []
    for layer in model.moe_layers():
        router.append(layer.router)
        for i, e in enumerate(layer.experts):
            (orig if i < n_orig else target).extend(e.parameters())
    special = {id(p) for p in (*target, *orig, *router)}
    shared = [p for p in model.parameters() if id(p) not in special]
    return {"target": target, "orig": orig, "router": router, "shared": shared}


def relocate(model: ToyMoE, n_orig: int, forget: Batch, retain: Batch, cfg: RexConfig,
             gen: torch.Generator) -> None:
    """REX step 2: selective gradient masking + routing differentiation (see module docstring)."""
    groups = param_groups(model, n_orig)
    targets = list(range(n_orig, model.n_experts))
    originals = list(range(n_orig))
    excised_view = active_mask(model.n_experts, targets)
    # weight_decay=0: unselected dense experts get zero (not None) grads; decay would move masked params
    opt = torch.optim.Adam(model.parameters(), lr=cfg.relocate_lr, weight_decay=0.0)
    fb = _batches(len(forget), min(cfg.forget_batch, len(forget)), gen)
    rb = _batches(len(retain), min(cfg.retain_batch, len(retain)), gen)
    model.train()
    for _ in range(cfg.relocate_steps):
        f, r = forget.take(next(fb)), retain.take(next(rb))
        opt.zero_grad(set_to_none=True)
        f_logits, f_probs = model(f.ids)
        r_logits, r_probs = model(r.ids)
        r_logits_ex, _ = model(r.ids, active=excised_view)
        # forget knowledge -> target experts only
        _accumulate(_lm_loss(f_logits, f), groups["target"])
        # routing differentiation -> router only
        route = _routing_loss(f_probs, f, targets) + _routing_loss(r_probs, r, originals)
        _accumulate(cfg.routing_weight * route, groups["router"])
        # general capability -> original experts + shared, on both the full and the excised view
        _accumulate(_lm_loss(r_logits, r) + _lm_loss(r_logits_ex, r), groups["orig"] + groups["shared"])
        if cfg.push_out_weight > 0:
            f_logits_ex, _ = model(f.ids, active=excised_view)
            _accumulate(cfg.push_out_weight * _push_out_loss(f_logits_ex, f), groups["orig"])
        opt.step()


# ---------------------------------------------------------------- pipeline

def state_sha256(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def moe_config(cfg: RexConfig, tok: Tokenizer, corpora: Corpora) -> MoEConfig:
    return MoEConfig(vocab_size=tok.size, max_len=seq_len(corpora.all_examples()), d_model=cfg.d_model,
                     n_layers=cfg.n_layers, n_heads=cfg.n_heads, d_ff=cfg.d_ff, n_experts=cfg.n_experts,
                     top_k=cfg.top_k)


@dataclass
class RexArtifacts:
    tokenizer: Tokenizer
    base: ToyMoE
    full: ToyMoE        # expanded + relocated (targets present)
    excised: ToyMoE     # targets physically removed ("public" model)
    metrics: dict[str, Any]


def rex_pipeline(seed: int, config: RexConfig | None = None) -> RexArtifacts:
    """Train base -> expand -> relocate -> excise -> evaluate. Deterministic from ``seed``."""
    cfg = config or default_config()
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(cfg.num_threads)
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return _run(seed, cfg)
    finally:
        torch.set_num_threads(prev_threads)
        torch.use_deterministic_algorithms(prev_det)


def _run(seed: int, cfg: RexConfig) -> RexArtifacts:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    corpora = generate_corpora(seed, cfg.data)
    tok = Tokenizer(corpora.vocab)
    T = seq_len(corpora.all_examples())
    forget = encode_examples(tok, corpora.forget, T)
    retain_tr = encode_examples(tok, corpora.retain_train, T)
    retain_te = encode_examples(tok, corpora.retain_test, T)
    gen = torch.Generator().manual_seed(seed)
    timings["data_s"] = time.perf_counter() - t0

    t = time.perf_counter()
    base = build_model(moe_config(cfg, tok, corpora), seed)
    train_base(base, forget, retain_tr, cfg, gen)
    timings["base_train_s"] = time.perf_counter() - t

    t = time.perf_counter()
    n_orig = base.n_experts
    full = expand_experts(base, cfg.k_target, gen)
    targets = list(range(n_orig, full.n_experts))
    relocate(full, n_orig, forget, retain_tr, cfg, gen)
    timings["relocate_s"] = time.perf_counter() - t

    t = time.perf_counter()
    excised = excise_experts(full, targets)
    timings["excise_s"] = time.perf_counter() - t

    t = time.perf_counter()
    m: dict[str, Any] = {
        "seed": seed,
        "paper": PAPER,
        "config": cfg.to_dict(),
        "params": {"base": param_count(base), "full": param_count(full), "excised": param_count(excised)},
        "experts_per_layer": {"base": n_orig, "full": full.n_experts, "excised": excised.n_experts},
        "target_experts": targets,
        **{name: {"forget_acc": accuracy(mdl, forget), "retain_acc": accuracy(mdl, retain_te),
                  "forget_answer_prob": answer_prob(mdl, forget)}
           for name, mdl in (("base", base), ("full", full), ("excised", excised))},
        "routing": {"forget_mass_on_targets": route_to_experts(full, forget, targets),
                    "retain_mass_on_targets": route_to_experts(full, retain_te, targets)},
        "n_forget": len(corpora.forget), "n_retain_test": len(corpora.retain_test),
        "excised_sha256": state_sha256(excised),
    }
    m["retain_drop"] = m["base"]["retain_acc"] - m["excised"]["retain_acc"]
    m["forget_drop"] = m["base"]["forget_acc"] - m["excised"]["forget_acc"]
    timings["eval_s"] = time.perf_counter() - t
    timings["total_s"] = time.perf_counter() - t0
    m["timings_s"] = {k: round(v, 3) for k, v in timings.items()}
    return RexArtifacts(tok, base, full, excised, m)


def run_rex_pipeline(seed: int = 0, config: RexConfig | None = None, *,
                     save_dir: str | Path | None = None) -> dict[str, Any]:
    """Run the pipeline and return a JSON-able metrics dict; optionally save models to ``save_dir``."""
    art = rex_pipeline(seed, config)
    if save_dir is not None:
        save_artifacts(art, save_dir)
        art.metrics["save_dir"] = str(save_dir)
    return art.metrics


def default_save_dir(seed: int) -> Path:
    return DEFAULT_TMP / f"seed-{seed}"


def save_artifacts(art: RexArtifacts, out: str | Path) -> Path:
    """State dicts -> ``.pt`` (weights only); config/vocab/metrics -> JSON."""
    d = Path(out); d.mkdir(parents=True, exist_ok=True)
    for name in ("full", "excised"):
        model: ToyMoE = getattr(art, name)
        torch.save(model.state_dict(), d / f"{name}.pt")
        (d / f"{name}.config.json").write_text(json.dumps(model.cfg.to_dict(), indent=2))
    (d / "vocab.json").write_text(json.dumps(art.tokenizer.vocab))
    (d / "metrics.json").write_text(json.dumps(art.metrics, indent=2))
    return d


def load_model(d: str | Path, name: str = "excised") -> tuple[ToyMoE, Tokenizer]:
    """Load ``full`` or ``excised`` model + tokenizer saved by :func:`save_artifacts`."""
    d = Path(d)
    if name not in ("full", "excised"):
        raise ValueError("name must be 'full' or 'excised'")
    cfg = MoEConfig(**json.loads((d / f"{name}.config.json").read_text()))
    model = ToyMoE(cfg)
    model.load_state_dict(torch.load(d / f"{name}.pt", weights_only=True))
    model.eval()
    return model, Tokenizer(json.loads((d / "vocab.json").read_text()))
