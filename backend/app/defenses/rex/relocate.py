"""REX at toy scale: expansion, relocation, excision, residual cleanup (+ the matched RMU-only control).

Reimplementation (not vendored) of
  Jin et al., "Towards Architectural-Level Safety Control: Relocating Hazardous Knowledge into
  Removable Experts in Mixture-of-Experts Models", arXiv:2610.25791 (Shanghai AI Lab), CC BY 4.0.
  Reference code: github.com/AI45Lab/REX (Apache-2.0; targets 26-35B MoE, so not vendored).

``variant="paper"`` (default) follows the paper's recipe (§3.2-3.5, App. A.1-A.2, Tables 6-7, the
Intern-S2/Bio column for setting-specific values):

1. **B**: a *prelearned* base: the toy MoE is trained on forget + retain data, so the hazardous
   knowledge is already spread over its original experts (the paper starts from a pretrained model).
2. **F0 = expand(B)** (§3.2): per layer, count top-K selection frequency of each expert on the forget
   and the general *training* data, take the M experts with the largest ``f_harm - f_std`` as the source
   set S (differential-activation expansion), clone each (expert + router row, 1e-5 Gaussian noise) as a
   target expert, then scale the source experts' output by ``s = 0.5``.
3. **F1 = relocate(F0)** (§3.3-3.4): shared params (attention, embeddings, norms, LM head) are frozen.
   Per-token gradient masking: the loss on target tokens updates target experts + routers; the loss on
   general tokens updates original experts + routers. General tokens are dispatched with the pre-top-K
   activation mask (target logits -> -inf). Objective ``L_LM + L_mass + lambda_r * L_rank`` (eq. 5-7),
   with the mass/rank losses on the pre-mask router logits. AdamW (0.9, 0.999), wd 0, clip 1, no warmup,
   cosine to 0, per-bank learning-rate multipliers.
4. **A1 = excise(F1)** (§3.5): physically delete the target experts and their router rows.
5. **A2 = RMU(A1)** residual cleanup (§3.5, eq. 8; ``cleanup.py``).
6. **RR = RMU(B)**: the paper's matched control (§4.5): same RMU operator, hyperparameters, budget and
   seed applied to B. ``A2 vs RR`` is what relocation + excision add over cleanup alone.

``variant="prelearned_pushout"`` is an **ablation that is NOT from the paper**: our earlier toy recipe
(clone experts 0..k-1, train shared params, a -log router-mass routing loss, and an unlikelihood
"push-out" loss on the excised view that drives the original path away from the forget answer).
``push_out_weight=0`` is its sub-ablation. Stages 4-6 are identical for both variants.

Post-training (paper §3.6, F2/F3/A3) is not reproduced.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from app.defenses.rex.cleanup import reference_hidden_norm, rmu_cleanup
from app.defenses.rex.data import Corpora, DataConfig, Tokenizer, generate_corpora
from app.defenses.rex.evaluate import (
    IGNORE, Batch, accuracy, answer_prob, encode_examples, route_to_experts, seq_len,
)
from app.defenses.rex.toy_moe import Expert, MoEConfig, ToyMoE, build_model, param_count

PAPER = ("Jin et al., Towards Architectural-Level Safety Control: Relocating Hazardous Knowledge into "
         "Removable Experts in Mixture-of-Experts Models, arXiv:2610.25791 (CC BY 4.0)")
DEFAULT_TMP = Path(__file__).resolve().parents[3] / ".tmp" / "rex"   # backend/.tmp (git-ignored)


VARIANTS = ("paper", "prelearned_pushout")
VARIANT_PROVENANCE = {
    "paper": "paper recipe (arXiv:2610.25791 §3.2-3.5), toy-scaled",
    "prelearned_pushout": "ablation NOT from the paper (toy push-out recipe)",
}


@dataclass(frozen=True)
class RexConfig:
    data: DataConfig = field(default_factory=DataConfig)
    variant: str = "paper"
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    n_experts: int = 8
    top_k: int = 2
    k_target: int = 2                 # M target experts per layer (paper: 8 of 128-256)
    base_steps: int = 600
    relocate_steps: int = 400
    forget_batch: int = 32
    retain_batch: int = 64
    base_lr: float = 3e-3
    balance_weight: float = 0.01
    grad_clip: float = 1.0
    # --- paper relocation (Table 6 + Table 7 Intern-S2/Bio; learning rates x100 for the toy scale)
    source_scale: float = 0.5         # s
    clone_noise: float = 1e-5
    paper_lr: float = 1e-4            # paper 1e-6
    orig_expert_mult: float = 1.0     # expert multipliers original / target = 1 / 40
    target_expert_mult: float = 40.0
    router_mult: float = 40.0         # paper 40 (original rows) / 60 (target rows); toy router is one matrix
    mass_floor: float = 0.75          # tau_h (weight lambda_h)
    mass_floor_weight: float = 6.0
    mass_ceiling: float = 0.02        # tau_s (weight lambda_s)
    mass_ceiling_weight: float = 1.0
    rank_margin: float = 0.1          # delta (weight lambda_r)
    rank_weight: float = 30.0
    # --- RMU residual cleanup, shared by A2 and RR (Table 7 Intern-S2/Bio; see cleanup.py)
    rmu_monitor_layer: int = 0        # "early" monitor layer; updates expert MLPs of layers 0..monitor
    rmu_steer: float = 4.0            # c, in units of the reference mean hidden norm (paper c=300 absolute)
    rmu_alpha: float = 350.0
    rmu_lr: float = 1e-3              # paper 1e-5, constant
    rmu_steps: int = 100
    # --- prelearned_pushout ablation (NOT from the paper)
    pushout_lr: float = 2e-3
    routing_weight: float = 1.0
    push_out_weight: float = 1.0
    num_threads: int = 1

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")

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

def expand_experts(model: ToyMoE, k: int, gen: torch.Generator, *,
                   sources: list[list[int]] | None = None, scale: float = 1.0,
                   noise: float = 1e-3, expert_noise: bool = False) -> ToyMoE:
    """Return a copy with ``k`` target experts appended per layer, cloned from source experts.

    ``sources[l]``: the k source experts of layer l (default: experts 0..k-1). Each clone copies the
    expert and its router row; ``noise`` is added to the cloned router rows (and to the cloned expert
    weights if ``expert_noise``). Afterwards the *source* experts' outputs are scaled by ``scale``
    (paper §3.2: weakens the hazardous knowledge they hold while keeping most general computation).
    """
    new = copy.deepcopy(model)
    n = model.n_experts
    for li, layer in enumerate(new.moe_layers()):
        src = sources[li] if sources is not None else [j % n for j in range(k)]
        if len(src) != k:
            raise ValueError(f"layer {li}: need {k} sources, got {len(src)}")
        for j in src:
            clone = copy.deepcopy(layer.experts[j])
            if expert_noise and noise > 0:
                with torch.no_grad():
                    for p in clone.parameters():
                        p.add_(noise * torch.randn(p.shape, generator=gen))
            layer.experts.append(clone)
        rows = layer.router.data[src].clone()
        rows += noise * torch.randn(rows.shape, generator=gen)
        layer.router = nn.Parameter(torch.cat([layer.router.data, rows], 0))
        if scale != 1.0:
            with torch.no_grad():
                for j in set(src):
                    fc2 = cast(Expert, layer.experts[j]).fc2
                    fc2.weight.mul_(scale)
                    fc2.bias.mul_(scale)
    new.cfg = replace(model.cfg, n_experts=n + k)
    return new


def _route_sel(b: Batch) -> torch.Tensor:
    """Tokens the routing statistics/losses use: real input tokens except BOS (shared by both domains)."""
    sel = b.real.clone()
    sel[:, 0] = False
    return sel


@torch.no_grad()
def selection_frequency(model: ToyMoE, b: Batch, chunk: int = 512) -> list[torch.Tensor]:
    """Per layer: fraction of routed tokens for which each expert is in the top-K."""
    model.eval()
    E, K = model.n_experts, model.cfg.top_k
    counts = [torch.zeros(E) for _ in model.moe_layers()]
    n_tok = 0
    for s in range(0, len(b), chunk):
        sub = b.take(torch.arange(s, min(s + chunk, len(b))))
        _, _, raw = model(sub.ids, return_router_logits=True)
        sel = _route_sel(sub)
        n_tok += int(sel.sum())
        for li, r in enumerate(raw):
            idx = r[sel].topk(K, -1).indices.reshape(-1)
            counts[li] += torch.bincount(idx, minlength=E).float()
    return [c / max(1, n_tok) for c in counts]


def differential_activation(model: ToyMoE, forget: Batch, retain_train: Batch, k: int
                            ) -> tuple[list[list[int]], list[list[float]]]:
    """Paper §3.2: per layer, the k experts with the largest ``f_harm - f_std`` (training data only)."""
    fh = selection_frequency(model, forget)
    fs = selection_frequency(model, retain_train)
    sources, deltas = [], []
    for h, s in zip(fh, fs):
        d = h - s
        order = sorted(range(len(d)), key=lambda i: (-float(d[i]), i))   # stable tie-break
        sources.append(order[:k])
        deltas.append([round(float(x), 6) for x in d])
    return sources, deltas


def excise_experts(model: ToyMoE, remove: list[int]) -> ToyMoE:
    """Physically delete experts ``remove`` (and their router rows) from every layer."""
    bad = [i for i in remove if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < model.n_experts]
    if bad:
        raise ValueError(f"expert indices out of range [0, {model.n_experts}): {bad}")
    if len(set(remove)) != len(remove):
        raise ValueError(f"duplicate expert indices: {remove}")
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


def relocate_pushout(model: ToyMoE, n_orig: int, forget: Batch, retain: Batch, cfg: RexConfig,
                     gen: torch.Generator) -> None:
    """``prelearned_pushout`` ablation (NOT from the paper): per-loss gradient masking, shared params
    trained, -log router-mass routing loss, optional unlikelihood push-out on the excised view."""
    groups = param_groups(model, n_orig)
    targets = list(range(n_orig, model.n_experts))
    originals = list(range(n_orig))
    excised_view = active_mask(model.n_experts, targets)
    # weight_decay=0: unselected dense experts get zero (not None) grads; decay would move masked params
    opt = torch.optim.Adam(model.parameters(), lr=cfg.pushout_lr, weight_decay=0.0)
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


def _target_gate_mass(raw: torch.Tensor, n_orig: int, top_k: int) -> torch.Tensor:
    """z_t: normalised top-K gate weight on the target experts, from *pre-mask* logits (paper §3.4)."""
    top_val, top_idx = raw.topk(top_k, dim=-1)
    return (top_val.softmax(-1) * (top_idx >= n_orig).float()).sum(-1)


def _rank_violation(raw: torch.Tensor, n_orig: int, top_k: int, margin: float) -> torch.Tensor:
    """max(0, delta - (s_K - h_max)) per token: s_K = K-th largest original logit, h_max = max target."""
    s_k = raw[..., :n_orig].topk(top_k, dim=-1).values[..., -1]
    h_max = raw[..., n_orig:].max(-1).values
    return F.relu(margin - (s_k - h_max))


def mixed_batch(f: Batch, r: Batch, n_experts: int, n_orig: int
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate target (f) and general (r) sequences -> ids, targets, routed-token mask, per-token
    domain label ``harm``, and the per-token pre-top-K activation mask (targets hidden on general tokens).

    Routing-loss tokens: target rows skip BOS (shared by both domains, so it must not be pulled to the
    targets); general rows *include* BOS. Otherwise the BOS router is never trained: the activation mask
    hides the targets at BOS during relocation, but at inference the unmasked router sent general BOS to
    the targets (QA S1-B finding). The paper packs without BOS (App. A.1); the toy keeps BOS."""
    ids = torch.cat([f.ids, r.ids])
    targets = torch.cat([f.targets, r.targets])
    route = torch.cat([_route_sel(f), r.real])
    harm = torch.zeros_like(ids, dtype=torch.bool)
    harm[: len(f)] = True
    active = torch.ones((*ids.shape, n_experts), dtype=torch.bool)
    active[~harm] = torch.arange(n_experts) < n_orig
    return ids, targets, route, harm, active


def paper_losses(model: ToyMoE, n_orig: int, f: Batch, r: Batch, cfg: RexConfig
                 ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """One mixed batch -> (loss on target tokens, loss on general tokens, diagnostics).

    Per-token domain label ``harm``; the pre-top-K activation mask hides the target experts from
    general tokens only. Each toy sequence is single-domain, so attention never mixes domains."""
    ids, targets, route, harm, active = mixed_batch(f, r, model.n_experts, n_orig)
    logits, _, raw = model(ids, active=active, return_router_logits=True)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE,
                         reduction="none").reshape(ids.shape)
    lab = targets != IGNORE
    # one L_LM = mean over *all* labelled tokens of the mixed batch (as on a packed paper batch); its
    # target-token and general-token parts are split only to route their gradients (sum = L_LM)
    n_lab = lab.sum().clamp_min(1)
    lm_h, lm_s = ce[lab & harm].sum() / n_lab, ce[lab & ~harm].sum() / n_lab
    K = cfg.top_k
    z = [_target_gate_mass(x, n_orig, K) for x in raw]
    z_h = torch.stack([zz[route & harm].mean() for zz in z]).mean()
    z_s = torch.stack([zz[route & ~harm].mean() for zz in z]).mean()
    mass_h = cfg.mass_floor_weight * F.relu(cfg.mass_floor - z_h)
    mass_s = cfg.mass_ceiling_weight * F.relu(z_s - cfg.mass_ceiling)
    rank = torch.stack([_rank_violation(x, n_orig, K, cfg.rank_margin)[route & ~harm].mean()
                        for x in raw]).mean()
    diag = {k: float(v.detach()) for k, v in
            (("lm_harm", lm_h), ("lm_std", lm_s), ("z_harm", z_h), ("z_std", z_s), ("rank", rank))}
    return lm_h + mass_h, lm_s + mass_s + cfg.rank_weight * rank, diag


def relocate_paper(model: ToyMoE, n_orig: int, forget: Batch, retain: Batch, cfg: RexConfig,
                   gen: torch.Generator) -> list[dict[str, float]]:
    """Paper §3.3-3.4 relocation F0 -> F1 (in place). Returns per-step diagnostics (first/last kept)."""
    groups = param_groups(model, n_orig)
    for p in groups["shared"]:
        p.requires_grad_(False)                                       # theta_shared frozen
    trainable = groups["orig"] + groups["target"] + groups["router"]
    lr = cfg.paper_lr
    opt = torch.optim.AdamW([
        {"params": groups["orig"], "lr": lr * cfg.orig_expert_mult},
        {"params": groups["target"], "lr": lr * cfg.target_expert_mult},
        {"params": groups["router"], "lr": lr * cfg.router_mult},
    ], betas=(0.9, 0.999), weight_decay=0.0)
    steps = cfg.relocate_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    fb = _batches(len(forget), min(cfg.forget_batch, len(forget)), gen)
    rb = _batches(len(retain), min(cfg.retain_batch, len(retain)), gen)
    log: list[dict[str, float]] = []
    model.train()
    try:
        for step in range(steps):
            f, r = forget.take(next(fb)), retain.take(next(rb))
            opt.zero_grad(set_to_none=True)
            loss_h, loss_s, diag = paper_losses(model, n_orig, f, r, cfg)
            _accumulate(loss_h, groups["target"] + groups["router"])  # D_harm -> theta_harm + routers
            _accumulate(loss_s, groups["orig"] + groups["router"])    # D_std  -> theta_std + routers
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            opt.step()
            sched.step()
            if step in (0, steps - 1):
                log.append({"step": step, **{k: round(v, 5) for k, v in diag.items()}})
    finally:
        for p in groups["shared"]:
            p.requires_grad_(True)
    return log


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


STAGES = ("B", "F0", "F1", "A1", "A2", "RR")


@dataclass
class RexArtifacts:
    tokenizer: Tokenizer
    base: ToyMoE        # B: prelearned base (knows forget + retain)
    expanded: ToyMoE    # F0: B + target experts (before relocation)
    full: ToyMoE        # F1: relocated, targets present (authorised-user model)
    excised: ToyMoE     # A1: targets physically removed
    delivered: ToyMoE   # A2: A1 after RMU residual cleanup (delivered base model)
    rmu_only: ToyMoE    # RR: matched control, the same RMU applied to B
    metrics: dict[str, Any]

    def stage(self, name: str) -> ToyMoE:
        return {"B": self.base, "F0": self.expanded, "F1": self.full, "A1": self.excised,
                "A2": self.delivered, "RR": self.rmu_only}[name]


def rex_pipeline(seed: int, config: RexConfig | None = None, *, base: ToyMoE | None = None) -> RexArtifacts:
    """B -> F0 -> F1 -> A1 -> A2, plus RR. Deterministic from ``seed``.

    ``base``: reuse an already trained B from :func:`rex_pipeline` with the same seed, data and base
    settings (each stage has its own seeded generator, so results are identical to training it here)."""
    cfg = config or default_config()
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(cfg.num_threads)
    torch.use_deterministic_algorithms(True)
    try:
        # CPU only: fork and seed the CPU generator, never accelerator RNGs (torch.manual_seed would)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            return _run(seed, cfg, base)
    finally:
        torch.set_num_threads(prev_threads)
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)


def _stage_metrics(model: ToyMoE, forget: Batch, forget_ho: Batch, retain_te: Batch,
                   active: torch.Tensor | None = None) -> dict[str, float]:
    """``*_train``: the trained forget prompts. ``*_heldout_prompt``: never-trained phrasings of the same
    facts (does the association survive, or only the memorised prompt string?)."""
    return {"forget_acc_train": accuracy(model, forget, active),
            "forget_acc_heldout_prompt": accuracy(model, forget_ho, active),
            "retain_acc": accuracy(model, retain_te, active),
            "forget_answer_prob_train": answer_prob(model, forget, active),
            "forget_answer_prob_heldout_prompt": answer_prob(model, forget_ho, active)}


@torch.no_grad()
def inference_routing(model: ToyMoE, b: Batch, n_orig: int) -> dict[str, Any]:
    """INFERENCE-time (unmasked) routing on ``b``'s real tokens, BOS included, per layer and mean:
    ``selection``: fraction of tokens whose top-K contains any target expert; ``z``: mean normalised
    top-K gate mass on the targets (the paper's z_t, compare tau_s on general data)."""
    model.eval()
    _, _, raw = model(b.ids, return_router_logits=True)
    sel = b.real
    K = model.cfg.top_k
    s_l = [float((r.topk(K, -1).indices >= n_orig).any(-1)[sel].float().mean()) for r in raw]
    z_l = [float(_target_gate_mass(r, n_orig, K)[sel].mean()) for r in raw]
    return {"selection_rate": sum(s_l) / len(s_l), "selection_rate_per_layer": s_l,
            "z": sum(z_l) / len(z_l), "z_per_layer": z_l}


def _run(seed: int, cfg: RexConfig, base: ToyMoE | None) -> RexArtifacts:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    corpora = generate_corpora(seed, cfg.data)
    tok = Tokenizer(corpora.vocab)
    T = seq_len(corpora.all_examples())
    forget = encode_examples(tok, corpora.forget, T)
    forget_ho = encode_examples(tok, corpora.forget_heldout, T)   # evaluation only, never trained on
    retain_tr = encode_examples(tok, corpora.retain_train, T)
    retain_te = encode_examples(tok, corpora.retain_test, T)
    timings["data_s"] = time.perf_counter() - t0

    t = time.perf_counter()
    if base is None:
        base = build_model(moe_config(cfg, tok, corpora), seed)
        train_base(base, forget, retain_tr, cfg, torch.Generator().manual_seed(seed))
    base = base.eval()
    timings["base_train_s"] = time.perf_counter() - t

    t = time.perf_counter()
    gen = torch.Generator().manual_seed(seed * 7919 + 1)
    n_orig = base.n_experts
    expansion: dict[str, Any]
    reloc_log: list[dict[str, float]] = []
    if cfg.variant == "paper":
        sources, deltas = differential_activation(base, forget, retain_tr, cfg.k_target)
        expanded = expand_experts(base, cfg.k_target, gen, sources=sources, scale=cfg.source_scale,
                                  noise=cfg.clone_noise, expert_noise=True)
        expansion = {"method": "differential_activation", "sources": sources, "delta": deltas,
                     "source_scale": cfg.source_scale, "clone_noise": cfg.clone_noise}
        full = copy.deepcopy(expanded)
        reloc_log = relocate_paper(full, n_orig, forget, retain_tr, cfg, gen)
    else:
        expanded = expand_experts(base, cfg.k_target, gen)
        expansion = {"method": "copy_first_k", "sources": [list(range(cfg.k_target))] * cfg.n_layers,
                     "source_scale": 1.0, "clone_noise": 1e-3}
        full = copy.deepcopy(expanded)
        relocate_pushout(full, n_orig, forget, retain_tr, cfg, gen)
    full = full.eval()
    targets = list(range(n_orig, full.n_experts))
    timings["relocate_s"] = time.perf_counter() - t

    t = time.perf_counter()
    excised = excise_experts(full, targets).eval()
    timings["excise_s"] = time.perf_counter() - t

    t = time.perf_counter()
    # one absolute steering magnitude for both, from the common reference B (matched control)
    ref_norm = reference_hidden_norm(base, retain_tr, cfg.rmu_monitor_layer)
    steer = cfg.rmu_steer * ref_norm
    delivered = rmu_cleanup(excised, forget, retain_tr, cfg, seed, steer)
    rmu_only = rmu_cleanup(base, forget, retain_tr, cfg, seed, steer)  # same seed/budget/c/u
    timings["cleanup_s"] = time.perf_counter() - t

    t = time.perf_counter()
    models = {"B": base, "F0": expanded, "F1": full, "A1": excised, "A2": delivered, "RR": rmu_only}
    m: dict[str, Any] = {
        "seed": seed,
        "paper": PAPER,
        "variant": cfg.variant,
        "provenance": VARIANT_PROVENANCE[cfg.variant],
        "config": cfg.to_dict(),
        "stages": {k: _stage_metrics(v, forget, forget_ho, retain_te) for k, v in models.items()},
        # diagnostic: F0 with targets masked == B with its source experts scaled by s (before relocation)
        "F0_excised_view": _stage_metrics(expanded, forget, forget_ho, retain_te,
                                          active_mask(full.n_experts, targets)),
        "rmu": {"steer_reference": "B", "reference_hidden_norm": round(ref_norm, 6),
                "steering_coefficient_c": round(steer, 6), "monitor_layer": cfg.rmu_monitor_layer},
        "params": {k: param_count(v) for k, v in models.items()},
        "experts_per_layer": {k: v.n_experts for k, v in models.items()},
        "target_experts": targets,
        "expansion": expansion,
        "relocation_log": reloc_log,
        "routing": {"forget_mass_on_targets": route_to_experts(full, forget, targets),
                    "retain_mass_on_targets": route_to_experts(full, retain_te, targets),
                    # F1 at inference, router unmasked, all real tokens incl. BOS
                    "inference_general": inference_routing(full, retain_te, n_orig),
                    "inference_forget": inference_routing(full, forget, n_orig)},
        "n_forget": len(corpora.forget), "n_forget_heldout_prompt": len(corpora.forget_heldout),
        "n_retain_test": len(corpora.retain_test),
        "excised_sha256": state_sha256(excised),
        "delivered_sha256": state_sha256(delivered),
    }
    st = m["stages"]
    m["summary"] = {
        **{key: {k: st[k][key] for k in ("B", "F1", "A1", "A2", "RR")}
           for key in ("forget_acc_train", "forget_acc_heldout_prompt", "retain_acc")},
        "excision_forget_drop": st["F1"]["forget_acc_train"] - st["A1"]["forget_acc_train"],  # Delta WMDP
        "a2_minus_rr_forget": st["A2"]["forget_acc_train"] - st["RR"]["forget_acc_train"],    # <0: REX wins
        "a2_minus_rr_forget_heldout_prompt": (st["A2"]["forget_acc_heldout_prompt"]
                                              - st["RR"]["forget_acc_heldout_prompt"]),
        "a2_minus_rr_retain": st["A2"]["retain_acc"] - st["RR"]["retain_acc"],
    }
    timings["eval_s"] = time.perf_counter() - t
    timings["total_s"] = time.perf_counter() - t0
    m["timings_s"] = {k: round(v, 3) for k, v in timings.items()}
    return RexArtifacts(tok, base, expanded, full, excised, delivered, rmu_only, m)


def run_rex_pipeline(seed: int = 0, config: RexConfig | None = None, *,
                     save_dir: str | Path | None = None) -> dict[str, Any]:
    """Run the pipeline and return a JSON-able metrics dict; optionally save models to ``save_dir``."""
    art = rex_pipeline(seed, config)
    if save_dir is not None:
        save_artifacts(art, save_dir)
        art.metrics["save_dir"] = str(save_dir)
    return art.metrics


SAVED = ("full", "excised", "delivered")


def default_save_dir(seed: int) -> Path:
    return DEFAULT_TMP / f"seed-{seed}"


def save_artifacts(art: RexArtifacts, out: str | Path) -> Path:
    """State dicts -> ``.pt`` (weights only); config/vocab/metrics -> JSON."""
    d = Path(out); d.mkdir(parents=True, exist_ok=True)
    for name in SAVED:
        model: ToyMoE = getattr(art, name)
        torch.save(model.state_dict(), d / f"{name}.pt")
        (d / f"{name}.config.json").write_text(json.dumps(model.cfg.to_dict(), indent=2))
    (d / "vocab.json").write_text(json.dumps(art.tokenizer.vocab))
    (d / "metrics.json").write_text(json.dumps(art.metrics, indent=2))
    return d


def load_model(d: str | Path, name: str = "excised") -> tuple[ToyMoE, Tokenizer]:
    """Load ``full`` (F1), ``excised`` (A1) or ``delivered`` (A2) + tokenizer saved by :func:`save_artifacts`."""
    d = Path(d)
    if name not in SAVED:
        raise ValueError(f"name must be one of {SAVED}")
    cfg = MoEConfig(**json.loads((d / f"{name}.config.json").read_text()))
    model = ToyMoE(cfg)
    model.load_state_dict(torch.load(d / f"{name}.pt", weights_only=True))
    model.eval()
    return model, Tokenizer(json.loads((d / "vocab.json").read_text()))
