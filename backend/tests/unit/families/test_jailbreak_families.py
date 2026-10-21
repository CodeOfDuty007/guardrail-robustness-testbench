"""S1-A1 single-turn jailbreak families: structure only (never potency).

Families ship citation + tags + a neutral brief; the attacker LLM writes all attack text. These tests
use a fake attacker that returns a fixed marker string, so no attack text appears in this file.
"""
import sys

import pytest

from app.datasets.loaders import canary_objectives
from app.engine.families import FAMILY_REGISTRY, family_rng, get_family, load_all
from app.engine.families.low_resource_lang import LANGUAGES
from app.engine.families.many_shot import SHOT_COUNTS
from app.engine.families.persuasion import PAP_TECHNIQUES
from app.engine.targets.base import TargetResult
from app.engine.targets.mock import SIM_FAMILY_BOOST

S1_A1 = ["roleplay_persona", "authority_context", "persuasion", "refusal_suppression",
         "hypothetical_framing", "many_shot", "low_resource_lang", "payload_splitting"]
NAMED_SETS = {"persuasion": PAP_TECHNIQUES, "low_resource_lang": LANGUAGES, "many_shot": SHOT_COUNTS}
OBJ = canary_objectives(1337)[0]


class FakeAttacker:
    model_name = "fake-attacker"

    def __init__(self, reply: str = "ATTACKER-OUTPUT-MARKER") -> None:
        self.reply, self.calls = reply, []

    async def send(self, system: str, user: str) -> TargetResult:
        self.calls.append((system, user))
        return TargetResult(text=self.reply)


def _rng(name, seed=1337, oid=OBJ.id):
    return family_rng(seed, name, oid, "NONE", 0)


@pytest.mark.parametrize("name", S1_A1)
def test_registered_with_contract_metadata(name):
    load_all()
    f = FAMILY_REGISTRY[name]
    assert f.citation and f.owasp_tag == "LLM01:Prompt Injection" and f.atlas_tag == "AML.T0054"
    assert (f.category, f.kind, f.wrap_with_cipher) == ("jailbreak", "single", False)
    assert f.technique_brief.startswith("Apply the ") and "\n" not in f.technique_brief
    doc = sys.modules[type(f).__module__].__doc__ or ""
    assert "Published method:" in doc and len(doc) > 200  # one-paragraph docstring citing the paper
    assert all(tok in doc for tok in f.citation.replace(";", " ").split() if tok.startswith("(arXiv:"))
    assert 0.05 <= SIM_FAMILY_BOOST[name] <= 0.35


def test_sim_boosts_distinct():
    vals = [SIM_FAMILY_BOOST[n] for n in S1_A1]
    assert len(set(vals)) == len(vals)


@pytest.mark.parametrize("name", S1_A1)
async def test_fallback_returns_bare_objective(name):
    f = get_family(name)
    c = await f.generate(objective=OBJ, attacker=None, history=None, rng=_rng(name))
    assert c.user == c.plaintext == OBJ.text
    assert c.family == name and c.cipher == "NONE" and c.channel == "direct"
    assert c.meta["generated_by"] == "fallback"


@pytest.mark.parametrize("name", S1_A1)
async def test_attacker_receives_brief_and_output_is_returned(name):
    f, att = get_family(name), FakeAttacker()
    c = await f.generate(objective=OBJ, attacker=att, history=None, rng=_rng(name))
    assert len(att.calls) == 1
    _, user = att.calls[0]
    assert c.meta["technique_brief"] in user and OBJ.text in user
    assert c.user == c.plaintext == "ATTACKER-OUTPUT-MARKER"
    assert c.meta["generated_by"] == "attacker"


@pytest.mark.parametrize("name", S1_A1)
async def test_empty_attacker_output_falls_back(name):
    c = await get_family(name).generate(objective=OBJ, attacker=FakeAttacker("   "), history=None, rng=_rng(name))
    assert c.user == OBJ.text and c.meta["generated_by"] == "fallback"


@pytest.mark.parametrize("name", S1_A1)
async def test_deterministic_under_fixed_seed(name):
    f = get_family(name)
    a = await f.generate(objective=OBJ, attacker=None, history=None, rng=_rng(name))
    b = await f.generate(objective=OBJ, attacker=None, history=None, rng=_rng(name))
    assert (a.user, a.technique, a.meta) == (b.user, b.technique, b.meta)


@pytest.mark.parametrize("name", sorted(NAMED_SETS))
async def test_sub_technique_recorded_and_reproducible(name):
    f, labels = get_family(name), NAMED_SETS[name]
    picks = []
    for seed in range(40):
        att = FakeAttacker()
        c = await f.generate(objective=OBJ, attacker=att, history=None, rng=_rng(name, seed))
        assert c.technique in labels and c.technique in c.meta["technique_brief"] and c.technique in att.calls[0][1]
        again = await f.generate(objective=OBJ, attacker=None, history=None, rng=_rng(name, seed))
        assert again.technique == c.technique
        picks.append(c.technique)
    assert len(set(picks)) > 1  # rng actually drives the choice


@pytest.mark.parametrize("name", sorted(set(S1_A1) - set(NAMED_SETS)))
async def test_fixed_technique_families_do_not_vary(name):
    f = get_family(name)
    techs = {(await f.generate(objective=OBJ, attacker=None, history=None, rng=_rng(name, s))).technique
             for s in range(10)}
    assert len(techs) == 1 and techs != {""}
