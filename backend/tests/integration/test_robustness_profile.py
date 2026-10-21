"""S2-A3: Robustness Profile analytics (plan §A.5 + §D.5) on real mock-run data."""
import pytest
from sqlalchemy import text

from app import analytics_store
from app.db.telemetry.session import get_engine
from tests.integration.test_run import wait_done

JB = ["roleplay_persona", "persuasion"]
INJ = ["inj_context_ignore", "inj_authority"]
LEGACY = {"DirectCipherWrap", "PAIR"}
MODELS = ("mock-guarded-1b", "mock-weak-1b")


def _run(c, **body) -> int:
    rid = c.post("/api/runs", json={"seed": 42, **body}).json()["id"]
    assert wait_done(c, rid)["status"] == "done"
    return rid


def _sql(stmt: str, **p) -> None:
    """Edit telemetry in place. NO forced rebuild: the ordinary replica refresh must notice the edit."""
    with get_engine().begin() as con:
        con.execute(text(stmt), p)


@pytest.fixture()
def prof(client):
    c = client
    weak = c.post("/api/providers", json={"label": "weak", "kind": "mock", "model_name": "mock-weak-1b"}).json()["id"]
    third = c.post("/api/providers", json={"label": "third", "kind": "mock", "model_name": "mock-third-7b"}).json()["id"]
    for pid in (None, weak):  # jailbreak families on two models; persuasion also cipher-wrapped
        _run(c, target_provider_id=pid, families=JB, wrap_with_cipher=["persuasion"],
             ciphers=["NONE", "STANDARD_BASE64", "ROT_N"])
    # a bare-only cell on a third model: must NOT enter the compounding comparison
    _run(c, target_provider_id=third, families=["persuasion"], ciphers=["NONE"])
    _run(c, strategies=sorted(LEGACY), ciphers=["STANDARD_BASE64", "NONE"])  # legacy rows: excluded from profile
    _run(c, dataset="injection_canary", families=INJ, ciphers=["NONE"])
    return c


def _ci_ok(x):
    assert 0.0 <= x["ci_low"] <= x["rate"] <= x["ci_high"] <= 1.0 and 0 <= x["k"] <= x["n"]


def test_heatmap_shape_cis_and_legacy_excluded(prof):
    h = prof.get("/api/analytics/family-heatmap").json()
    fams = {f["family"]: f for f in h["families"]}
    assert set(fams) == set(JB) | set(INJ)  # no DirectCipherWrap / PAIR rows
    assert not LEGACY & {c["family"] for c in h["cells"]}
    assert set(h["models"]) >= set(MODELS)
    for c in h["cells"]:
        _ci_ok(c)
    cell = {(c["family"], c["model"]): c for c in h["cells"]}
    assert cell[("roleplay_persona", "mock-weak-1b")]["n"] == 3        # 3 canary objectives, bare only
    assert cell[("persuasion", "mock-weak-1b")]["n"] == 9              # bare + 2 ciphers
    assert fams["inj_authority"]["category"] == "injection" and fams["inj_authority"]["channel"] == "direct"
    assert fams["persuasion"]["category"] == "jailbreak" and fams["persuasion"]["citation"]
    # ATLAS rolled up by PARENT technique, sub-id kept as detail
    assert fams["inj_authority"]["atlas_tag"] == "AML.T0051.000" and fams["inj_authority"]["atlas_parent"] == "AML.T0051"
    atlas = {a["atlas"]: a for a in h["by_atlas"]}
    assert set(atlas) == {"AML.T0051", "AML.T0054"}
    assert "AML.T0051.000" in atlas["AML.T0051"]["sub_ids"] and set(INJ) <= set(atlas["AML.T0051"]["families"])
    assert h["attacker_refused"] == 0


def test_model_filter_and_radar_overlay(prof):
    one = prof.get("/api/analytics/family-heatmap?model=mock-weak-1b").json()
    assert one["models"] == ["mock-weak-1b"] and all(c["model"] == "mock-weak-1b" for c in one["cells"])
    r = prof.get("/api/analytics/family-radar").json()
    per = {m["model"]: m["families"] for m in r["models"]}
    assert set(per) >= set(MODELS)
    for mo in MODELS:
        assert set(JB) <= set(per[mo])
        for v in per[mo].values():
            _ci_ok(v)
    assert set(per["mock-third-7b"]) == {"persuasion"}  # untested families are absent, not zero
    assert not LEGACY & set(r["families"])


def test_compounding_uses_matched_cells_only(prof):
    # QA cross-run case: a second, bare-only persuasion run on the SAME model must not join the first run's
    # cells (each run generates its own framing; cells are keyed on run + iteration)
    weak2 = prof.post("/api/providers", json={"label": "weak2", "kind": "mock", "model_name": "mock-weak-1b"}).json()["id"]
    _run(prof, target_provider_id=weak2, families=["persuasion"], ciphers=["NONE"])
    out = prof.get("/api/analytics/compounding?cipher=ROT_N").json()
    assert out["scope"]["cipher_filter_ignored"] is True and "run" in out["scope"]["comparison"]
    d = {x["family"]: x for x in out["families"]}
    p = d["persuasion"]
    # 2 models x 3 objectives matched; bare-only cells (third model, second weak run) are excluded
    assert p["matched_cells"] == p["n_cells"] == 6 and p["bare"]["n"] == 6 and p["ciphered"]["n"] == 12
    by = {x["cipher"]: x for x in p["by_cipher"]}
    assert set(by) == {"STANDARD_BASE64", "ROT_N"}
    for x in by.values():
        assert x["matched_cells"] == 6 and x["bare"]["n"] == x["ciphered"]["n"] == 6
        _ci_ok(x["bare"]); _ci_ok(x["ciphered"])
    assert -1 <= p["delta_ci_low"] <= p["delta"] <= p["delta_ci_high"] <= 1
    rp = d["roleplay_persona"]  # never ciphered -> note, not a fabricated zero
    assert rp["matched_cells"] == 0 and rp["delta"] is None and rp["delta_ci_low"] is None and "wrap_with_cipher" in rp["note"]
    assert not LEGACY & set(d)
    assert prof.get("/api/analytics/compounding?cipher=ROT_N").json() == out  # bootstrap is deterministic


def _rows(*bits):
    return [{"bypass": bool(b)} for b in bits]


def test_delta_is_within_cell_equal_weight():
    """Codex example: unequal attempt counts per cell must not manufacture a delta when every cell has none."""
    from app.api.analytics import matched_delta
    cells = {"c1": {"NONE": _rows(1), "B64": _rows(1, 1, 1)},     # bare 100%, cipher 100%
             "c2": {"NONE": _rows(0, 0, 0), "B64": _rows(0)}}     # bare 0%,   cipher 0%
    r = matched_delta(cells, lambda arms: arms.get("B64") or [])
    assert r["n_cells"] == 2 and r["delta"] == 0.0
    assert r["bare"]["n"] == 4 and r["ciphered"]["n"] == 4  # pooled arms would show 25% vs 75% (a fake +50 pp)
    assert r["delta_ci_low"] == r["delta_ci_high"] == 0.0


def test_bootstrap_ci_null_below_two_cells_and_deterministic():
    from app.api.analytics import cell_bootstrap, matched_delta
    one = matched_delta({"c1": {"NONE": _rows(0), "B64": _rows(1)}}, lambda arms: arms.get("B64") or [])
    assert one["n_cells"] == 1 and one["delta"] == 1.0 and one["delta_ci_low"] is None and one["delta_ci_high"] is None
    assert one["delta_note"]
    none = cell_bootstrap([])
    assert none["delta"] is None and none["delta_ci_low"] is None
    ds = [0.0, 1.0, 0.5, -0.5, 1.0, 0.0, 0.0]
    a, b = cell_bootstrap(ds), cell_bootstrap(ds)
    assert a == b and a["delta_ci_low"] < a["delta"] < a["delta_ci_high"]
    assert -1 <= a["delta_ci_low"] and a["delta_ci_high"] <= 1


def test_ordinary_refresh_sees_in_place_attribute_edits(prof):
    """Replica staleness (Codex #1): these columns are written/edited after a row is visible. No forced rebuild."""
    q = "SELECT family_raw, category, channel, delivery, generated_by, technique, cipher_type FROM flat WHERE attempt_id = ?"
    aid, gen = analytics_store.query("SELECT attempt_id, generated_by FROM flat WHERE family_raw = 'persuasion' "
                                     "ORDER BY attempt_id LIMIT 1")[0].values()
    same_len = "attacker" if gen == "fallback" else "fallback"  # same length: a LENGTH() checksum would miss it
    _sql("UPDATE attempts SET family='many_shot', category='injection', channel='indirect', delivery='visible', "
         "generated_by=:g, technique='edited', cipher_type='ROT_N' WHERE id=:i", g=same_len, i=aid)
    assert analytics_store.query(q, [aid])[0] == {
        "family_raw": "many_shot", "category": "injection", "channel": "indirect", "delivery": "visible",
        "generated_by": same_len, "technique": "edited", "cipher_type": "ROT_N"}
    _sql("UPDATE attempts SET generated_by=:g WHERE id=:i", g=gen, i=aid)  # same-length swap back
    assert analytics_store.query(q, [aid])[0]["generated_by"] == gen


def test_injection_breakdown_direct_then_indirect(prof):
    b = prof.get("/api/analytics/injection-breakdown").json()
    assert b["has_direct"] and not b["has_indirect"] and b["by_delivery"] == [] and b["note"]
    assert [x["channel"] for x in b["by_channel"]] == ["direct"] and b["by_channel"][0]["n"] == 6
    # indirect data comes from S2-A5; simulate its shape on existing injection rows
    _sql("UPDATE attempts SET channel='indirect', delivery='html_comment' WHERE family='inj_authority'")
    b = prof.get("/api/analytics/injection-breakdown").json()
    assert b["has_indirect"] and b["note"] is None
    ch = {x["channel"]: x for x in b["by_channel"]}
    assert ch["direct"]["n"] == 3 and ch["indirect"]["n"] == 3 and ch["indirect"]["families"] == ["inj_authority"]
    assert [x["delivery"] for x in b["by_delivery"]] == ["html_comment"] and b["by_delivery"][0]["per_model"]
    _ci_ok(b["by_delivery"][0])


def test_attacker_refused_rows_excluded_and_counted(prof):
    before = {(c["family"], c["model"]): c["n"] for c in prof.get("/api/analytics/family-heatmap").json()["cells"]}
    _sql("UPDATE attempts SET generated_by='attacker_refused' WHERE id IN (SELECT a.id FROM attempts a JOIN responses r "
         "ON r.attempt_id=a.id WHERE a.family='roleplay_persona' AND r.target_model='mock-weak-1b' LIMIT 2)")
    h = prof.get("/api/analytics/family-heatmap").json()
    after = {(c["family"], c["model"]): c["n"] for c in h["cells"]}
    assert after[("roleplay_persona", "mock-weak-1b")] == before[("roleplay_persona", "mock-weak-1b")] - 2
    assert h["attacker_refused"] == 2
    assert prof.get("/api/analytics/family-radar").json()["attacker_refused"] == 2
    assert prof.get("/api/analytics/injection-breakdown").json()["attacker_refused"] == 0


def test_correlation_family_scope_and_backcompat(prof):
    allr = prof.get("/api/analytics/correlation?metric=shannon_entropy").json()
    assert "scope" not in allr and allr["n"] > 0
    _run(prof, families=["encoding"], ciphers=["STANDARD_BASE64", "ROT_N"])
    enc = prof.get("/api/analytics/correlation?metric=shannon_entropy&scope=encoding_family").json()
    assert enc["scope"] == "encoding_family" and enc["n"] == 6  # 3 objectives x 2 ciphers, encoding only
    assert {x["cipher"] for x in enc["by_cipher"]} == {"STANDARD_BASE64", "ROT_N"}
    assert "error" in prof.get("/api/analytics/correlation?scope=bogus").json()


def test_report_has_profile_section_with_citations(prof):
    rep = prof.get("/api/analytics/report").json()
    rp = rep["robustness_profile"]
    assert {f["family"] for f in rp["families"]} == set(JB) | set(INJ)
    page = prof.get("/api/analytics/report?format=html").text
    assert "Robustness Profile" in page and "<ol class=refs>" in page
    for f in rp["families"]:
        assert f["family"] in page
    assert "DirectCipherWrap" not in page.split("Robustness Profile")[1].split("<h2>")[0]
    assert "CANARY-" not in page and "INJECTION-OK" not in page


def test_profile_empty_state(client):
    _run(client, strategies=["DirectCipherWrap"], ciphers=["NONE"])
    h = client.get("/api/analytics/family-heatmap").json()
    assert h["families"] == [] and h["cells"] == [] and h["models"] == []
    assert client.get("/api/analytics/compounding").json()["families"] == []
    assert client.get("/api/analytics/injection-breakdown").json()["by_channel"] == []
    assert "No attack-family attempts" in client.get("/api/analytics/report?format=html").text
