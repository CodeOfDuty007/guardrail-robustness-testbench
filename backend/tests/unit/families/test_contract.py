"""Sprint-0 contract tests: registry, rng determinism, encoding == legacy DirectCipherWrap (golden)."""
import json
from pathlib import Path

from app.datasets.loaders import canary_objectives
from app.engine.converters.ciphers import CIPHER_REGISTRY, get_cipher
from app.engine.families import FAMILY_REGISTRY, AttackFamily, family_rng, get_family, list_families, load_all
from app.engine.orchestrator import RunConfig, apply_cipher, family_cells
from app.metrics.capture import MetricCapture

GOLDEN = Path(__file__).parents[2] / "golden" / "encoding_direct_cipher_wrap.json"


def test_every_family_satisfies_contract():
    load_all()
    assert "encoding" in FAMILY_REGISTRY
    for f in FAMILY_REGISTRY.values():
        assert isinstance(f, AttackFamily)
        assert f.citation and f.owasp_tag and f.atlas_tag
        assert f.category in ("jailbreak", "injection") and f.kind in ("single", "meta")
    assert [f["name"] for f in list_families()]  # plain JSON view


def test_family_rng_is_stable_and_keyed():
    a = family_rng(1337, "encoding", "o1", "NONE", 0).random()
    assert a == family_rng(1337, "encoding", "o1", "NONE", 0).random()
    assert a != family_rng(1337, "encoding", "o2", "NONE", 0).random()
    assert a != family_rng(1338, "encoding", "o1", "NONE", 0).random()


async def test_encoding_family_matches_legacy_direct_cipher_wrap_golden():
    golden = json.loads(GOLDEN.read_text())
    fam = get_family("encoding")
    for cn in sorted(CIPHER_REGISTRY):
        for o in canary_objectives(1337):
            c = get_cipher(cn, 1337)
            cand = await fam.generate(objective=o, attacker=None, history=None, rng=family_rng(1337, "encoding", o.id, cn))
            cand = apply_cipher(cand, c, MetricCapture(c))
            assert cand.user == golden[f"{cn}|{o.id}"], cn


def test_family_cells_cross_product():
    cfg = RunConfig(families=["encoding"], ciphers=["NONE", "STANDARD_BASE64"])
    assert family_cells(cfg) == [("encoding", "NONE"), ("encoding", "STANDARD_BASE64")]


async def test_family_replay_sends_exact_stored_request():
    import pytest
    from app.engine.orchestrator import replay
    from app.engine.targets.base import TargetResult

    class Spy:
        model_name, provider_kind, sent = "spy", "mock", []
        async def send(self, system, user):
            self.sent.append(user); return TargetResult("Okay.")

    o = canary_objectives(1337)[1]
    stored = "ATTACKER-GENERATED FRAMING\n" + o.text  # differs from plaintext on purpose
    spec = {"seed": 1337, "objective_id": o.id, "cipher_type": "STANDARD_BASE64", "strategy_name": "encoding",
            "plaintext_prompt": o.text, "full_request_sent": stored}
    spy = Spy()
    await replay(spec, spy)
    assert spy.sent == [stored]
    with pytest.raises(ValueError, match="full_request_sent"):
        await replay({**spec, "full_request_sent": ""}, spy)


async def test_replay_reapplies_simulated_family_conditions(monkeypatch):
    """A mock success must re-verify as a success: replay sets the same sim_family the run did."""
    from app.engine import orchestrator
    from app.engine.targets import mock
    seen = []

    class Spy:
        model_name, provider_kind = "spy", "mock"
        async def send(self, system, user):
            seen.append(mock.sim_family.get()); return orchestrator.TargetResult("Okay.")

    o = canary_objectives(1337)[1]
    await orchestrator.replay({"seed": 1337, "objective_id": o.id, "cipher_type": "NONE", "strategy_name": "encoding",
                               "plaintext_prompt": o.text, "full_request_sent": o.text}, Spy())
    assert seen == ["encoding"] and mock.sim_family.get() is None
