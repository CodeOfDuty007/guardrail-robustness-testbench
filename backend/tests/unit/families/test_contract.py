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
