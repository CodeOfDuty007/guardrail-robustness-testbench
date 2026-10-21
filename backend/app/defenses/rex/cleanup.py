"""REX stage 2b: RMU residual cleanup (paper §3.5, eq. 8, App. A.2), toy scale.

Applied to the excised model A1 it gives the delivered base model A2. Applied with the *same* operator,
hyperparameters, budget, random direction and data order to the original model B it gives the matched
control RR (paper §4.5, Table 5), whose only difference from A2 is the starting point.

    L_RMU = E_{t~D_harm} ||phi(t) - c*u||^2 + alpha * E_{t~D_std} ||phi(t) - phi_ref(t)||^2

* ``phi`` is the residual-stream hidden state after the monitor layer; ``phi_ref`` is the frozen model
  before cleanup; ``u`` a fixed random unit vector from a fixed seed.
* Trainable: the routed-expert MLPs of layers ``0..monitor`` (the toy has no shared-expert MLP); routers,
  attention, embeddings, norms and the LM head stay frozen.
* Both terms use labelled tokens only, averaged per sequence, squared error averaged over the hidden dim.

Toy adaptations (documented in README): the steering coefficient is expressed in units of B's mean
monitor-layer hidden-state norm on general data (``c = rmu_steer * norm(B)``, one value for both A2 and
RR) because the paper's absolute c=300 is tied to its models' activation scale; the budget is a fixed
number of steps, shared by A2 and RR.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from torch import nn

from app.defenses.rex.evaluate import IGNORE, Batch
from app.defenses.rex.toy_moe import ToyMoE

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from app.defenses.rex.relocate import RexConfig


def _per_seq_mean(sq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """sq (B,T) per-token errors -> mean over masked tokens of each sequence, then over sequences."""
    m = mask.float()
    n = m.sum(-1)
    keep = n > 0
    return ((sq * m).sum(-1)[keep] / n[keep]).mean()


def rmu_trainable(model: ToyMoE, monitor_layer: int) -> list[nn.Parameter]:
    return [p for layer in model.moe_layers()[: monitor_layer + 1] for e in layer.experts for p in e.parameters()]


@torch.no_grad()
def reference_hidden_norm(model: ToyMoE, retain: Batch, layer: int) -> float:
    """Mean L2 norm of ``model``'s monitor-layer hidden states on labelled general tokens."""
    model.eval()
    lab = (retain.targets != IGNORE) & retain.real
    return float(model.hidden(retain.ids, layer)[lab].norm(dim=-1).mean())


def rmu_cleanup(model: ToyMoE, forget: Batch, retain: Batch, cfg: RexConfig, seed: int,
                steer: float) -> ToyMoE:
    """Return a cleaned copy of ``model``, steering forget states to ``steer * u``.

    ``steer`` is the *absolute* steering coefficient c. The pipeline derives it once from the common
    reference B (``rmu_steer * reference_hidden_norm(B)``) and passes the same value to A2 and RR.
    Deterministic from ``seed`` (direction ``u`` and data order)."""
    from app.defenses.rex.relocate import _batches  # local: relocate imports this module

    L = cfg.rmu_monitor_layer
    gen = torch.Generator().manual_seed(seed * 1_000_003 + 17)
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    m = copy.deepcopy(model)
    trainable = rmu_trainable(m, L)
    train_ids = {id(p) for p in trainable}
    for p in m.parameters():
        p.requires_grad_(id(p) in train_ids)

    d = m.cfg.d_model
    u = torch.randn(d, generator=gen)
    u = u / u.norm()
    target = steer * u

    opt = torch.optim.AdamW(trainable, lr=cfg.rmu_lr, betas=(0.9, 0.999), weight_decay=0.0)
    fb = _batches(len(forget), min(cfg.forget_batch, len(forget)), gen)
    rb = _batches(len(retain), min(cfg.retain_batch, len(retain)), gen)
    m.eval()  # no dropout anywhere; eval() just documents that no train-time behaviour is wanted
    for _ in range(cfg.rmu_steps):
        f, r = forget.take(next(fb)), retain.take(next(rb))
        hf = m.hidden(f.ids, L)
        hr = m.hidden(r.ids, L)
        with torch.no_grad():
            hr_ref = ref.hidden(r.ids, L)
        lf = _per_seq_mean(((hf - target) ** 2).mean(-1), (f.targets != IGNORE) & f.real)
        lr = _per_seq_mean(((hr - hr_ref) ** 2).mean(-1), (r.targets != IGNORE) & r.real)
        loss = lf + cfg.rmu_alpha * lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        opt.step()
    for p in m.parameters():
        p.requires_grad_(True)
    return m.eval()
