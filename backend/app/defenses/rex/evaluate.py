"""Forget / retain accuracy and plain-Python inference helpers (plan §B.2 metrics).

* **forget accuracy** = fraction of synthetic facts whose answer token the model recalls (greedy).
* **retain accuracy** = exact-match on held-out copy-task sequences (greedy).

Teacher-forced argmax over all answer positions equals greedy decoding for exact match (if every
step is right given the right prefix, greedy follows that prefix), so accuracy is one batched forward.

:func:`generate` / :func:`answer` take and return plain ``str`` so a later ``ToyMoETarget`` (S3-B3)
can wrap a model without touching tensors.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from app.defenses.rex.data import BOS, EOS, Example, Tokenizer
from app.defenses.rex.toy_moe import ToyMoE

IGNORE = -100


@dataclass
class Batch:
    ids: torch.Tensor        # (N,T) inputs
    targets: torch.Tensor    # (N,T) next-token targets on answer+EOS positions, IGNORE elsewhere
    acc_mask: torch.Tensor   # (N,T) bool: answer positions only (EOS excluded)
    real: torch.Tensor       # (N,T) bool: non-padding input positions

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def take(self, idx: torch.Tensor) -> Batch:
        return Batch(self.ids[idx], self.targets[idx], self.acc_mask[idx], self.real[idx])


def seq_len(examples: list[Example]) -> int:
    """Input length needed: prompt + answer + EOS, minus one (next-token shift)."""
    return max(len(e.prompt) + len(e.answer) for e in examples)


def encode_examples(tok: Tokenizer, examples: list[Example], T: int) -> Batch:
    n = len(examples)
    ids = torch.full((n, T), tok.pad_id, dtype=torch.long)
    targets = torch.full((n, T), IGNORE, dtype=torch.long)
    acc = torch.zeros((n, T), dtype=torch.bool)
    real = torch.zeros((n, T), dtype=torch.bool)
    for i, ex in enumerate(examples):
        full = tok.encode([*ex.prompt, *ex.answer, EOS])
        if len(full) - 1 > T:
            raise ValueError(f"example longer than T={T}")
        inp, tgt = full[:-1], full[1:]
        ids[i, : len(inp)] = torch.tensor(inp)
        real[i, : len(inp)] = True
        start = len(ex.prompt) - 1          # position whose next token is the first answer token
        targets[i, start: len(inp)] = torch.tensor(tgt[start:])
        acc[i, start: start + len(ex.answer)] = True
    return Batch(ids, targets, acc, real)


@torch.no_grad()
def accuracy(model: ToyMoE, batch: Batch, active: torch.Tensor | None = None, chunk: int = 512) -> float:
    """Exact-match accuracy over all answer tokens of each example."""
    model.eval()
    correct = 0
    for s in range(0, len(batch), chunk):
        b = batch.take(torch.arange(s, min(s + chunk, len(batch))))
        logits, _ = model(b.ids, active=active)
        ok = (logits.argmax(-1) == b.targets) | ~b.acc_mask
        correct += int(ok.all(-1).sum())
    return correct / max(1, len(batch))


@torch.no_grad()
def answer_prob(model: ToyMoE, batch: Batch, active: torch.Tensor | None = None) -> float:
    """Mean probability assigned to the correct answer tokens (leak check beyond argmax accuracy)."""
    model.eval()
    logits, _ = model(batch.ids, active=active)
    p = logits.softmax(-1).gather(-1, batch.targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return float(p[batch.acc_mask].mean())


@torch.no_grad()
def route_to_experts(model: ToyMoE, batch: Batch, experts: list[int]) -> float:
    """Mean router probability mass on ``experts`` over real (non-BOS, non-pad) tokens, all layers."""
    model.eval()
    _, probs = model(batch.ids)
    sel = batch.real.clone(); sel[:, 0] = False
    vals = [p[..., experts].sum(-1)[sel].mean() for p in probs]
    return float(torch.stack(vals).mean())


@torch.no_grad()
def generate(model: ToyMoE, tok: Tokenizer, prompt: str, max_new_tokens: int = 8) -> str:
    """Greedy continuation of a whitespace-tokenised prompt; returns the generated words (no EOS)."""
    model.eval()
    ids = [tok.bos_id, *tok.encode(tok.tokenize(prompt))]
    out: list[int] = []
    for _ in range(max_new_tokens):
        window = ids[-model.cfg.max_len:]
        logits, _ = model(torch.tensor([window], dtype=torch.long))
        nxt = int(logits[0, -1].argmax())
        if nxt == tok.eos_id:
            break
        out.append(nxt); ids.append(nxt)
    return " ".join(w for w in tok.decode(out) if w != BOS)


def answer(model: ToyMoE, tok: Tokenizer, prompt: str) -> str:
    """First generated word (the toy's 'answer' to e.g. ``the access rune for VAULT-123 is``)."""
    words = generate(model, tok, prompt, max_new_tokens=1).split()
    return words[0] if words else ""
