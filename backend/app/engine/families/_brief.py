"""Shared plumbing for single-turn "brief-only" jailbreak families (S1-A1). Registers nothing.

Each such family ships metadata plus a neutral one-line `technique_brief` naming a published method.
The ATTACKER LLM writes the attack text at runtime via `ask_attacker`; this repo authors none. With no
attacker (or an empty attacker reply) the bare objective text is sent unchanged.
"""
from __future__ import annotations

import random

from app.datasets.loaders import Objective
from app.engine.families.base import CandidatePrompt, Category, Kind, Turn, ask_attacker
from app.engine.targets.base import Target, TargetResult


class _Recorder:
    """Wraps the attacker so we can tell a real (non-empty) attacker reply from the fallback."""

    def __init__(self, inner: Target) -> None:
        self.inner, self.model_name = inner, getattr(inner, "model_name", "attacker")
        self.provider_kind: str = getattr(inner, "provider_kind", "")
        self.raw: str | None = None

    async def send(self, system: str, user: str) -> TargetResult:
        r = await self.inner.send(system, user)
        self.raw = r.text
        return r


class BriefFamily:
    """Base for families whose only content is citation + tags + a neutral technique brief."""
    name: str
    citation: str
    owasp_tag = "LLM01:Prompt Injection"
    atlas_tag = "AML.T0054"
    category: Category = "jailbreak"
    kind: Kind = "single"
    wrap_with_cipher = False
    technique_brief: str          # neutral one-liner; may contain "{technique}" for a named sub-technique
    sub_techniques: tuple[str, ...] = ()   # paper-defined named set (plain labels), chosen by rng
    default_technique: str = ""

    def pick_technique(self, rng: random.Random) -> str:
        return rng.choice(self.sub_techniques) if self.sub_techniques else self.default_technique

    def brief_for(self, technique: str) -> str:
        return self.technique_brief.format(technique=technique)

    async def generate(self, *, objective: Objective, attacker: Target | None,
                       history: list[Turn] | None, rng: random.Random) -> CandidatePrompt:
        technique = self.pick_technique(rng)
        brief = self.brief_for(technique)
        rec = _Recorder(attacker) if attacker is not None else None
        text = await ask_attacker(rec, brief, objective, objective.text, history)
        by = "attacker" if rec is not None and (rec.raw or "").strip() else "fallback"
        if by == "fallback":
            text = objective.text
        return CandidatePrompt(user=text, plaintext=text, family=self.name, technique=technique,
                               meta={"generated_by": by, "technique_brief": brief})
