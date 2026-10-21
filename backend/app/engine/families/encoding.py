"""`encoding` — obfuscation family (reference implementation for every other family).

Published method: Wei, Haghtalab & Steinhardt 2023, "Jailbroken: How Does LLM Safety Training Fail?"
(competing objectives / mismatched generalisation via base64 etc.), and Yuan et al., ICLR 2024,
"GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher" (CipherChat). The framing is the
objective itself; the orchestrator applies the run's cipher converter + decode-instruction wrapper
(`apply_cipher`), byte-identical to the legacy `DirectCipherWrap` strategy (golden-tested).
"""
from __future__ import annotations

import random

from app.datasets.loaders import Objective
from app.engine.families.base import CandidatePrompt, Turn, register_family
from app.engine.targets.base import Target


@register_family
class Encoding:
    name = "encoding"
    citation = "Wei et al. 2023 (arXiv:2307.02483); Yuan et al. ICLR 2024 CipherChat (arXiv:2308.06463)"
    owasp_tag = "LLM01:Prompt Injection"
    atlas_tag = "AML.T0051"
    category = "jailbreak"
    kind = "single"
    wrap_with_cipher = True

    async def generate(self, *, objective: Objective, attacker: Target | None,
                       history: list[Turn] | None, rng: random.Random) -> CandidatePrompt:
        return CandidatePrompt(user=objective.text, plaintext=objective.text, family=self.name, technique="cipher")
