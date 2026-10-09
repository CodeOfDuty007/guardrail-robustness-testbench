
import pytest

from tests.integration.test_run import wait_done


def _confirmed_attempt(client):
    rid = client.post("/api/runs", json={
        "ciphers": ["CUSTOM_BASE64", "STANDARD_BASE64", "ROT_N", "NONE"], "strategies": ["DirectCipherWrap", "PAIR", "PersonaRoleplay"],
        "max_iterations": 5, "owasp_tag": "LLM01:Prompt Injection", "atlas_tag": "AML.T0051",
        "target_provider_id": None}).json()["id"]
    wait_done(client, rid)
    j = client.get("/api/explorer/judgments?eligible=true").json()["rows"]
    assert j, "mock should yield at least one confirmed jailbreak"
    return j[0]["attempt_id"]


def test_promote_requires_eligibility(client):
    rid = client.post("/api/runs", json={"ciphers": ["XOR_MASK"]}).json()["id"]
    wait_done(client, rid)
    aid = client.get(f"/api/runs/{rid}/attempts").json()[0]["id"]
    assert client.post("/api/corpus/promote", json={"source_attempt_id": aid}).status_code == 409
    assert client.post("/api/corpus/promote", json={"source_attempt_id": 99999}).status_code == 404


def test_promote_verify_disclose_flow(client):
    aid = _confirmed_attempt(client)
    item = client.post("/api/corpus/promote", json={"source_attempt_id": aid}).json()
    assert item["status"] == "unverified" and item["tags"].startswith("LLM01")
    assert client.post("/api/corpus/promote", json={"source_attempt_id": aid}).json()["id"] == item["id"]  # idempotent
    v = client.post(f"/api/corpus/{item['id']}/verify").json()
    assert v["status"] == "working" and v["last_verified_at"]
    models = client.get("/api/corpus/models").json()
    assert models[0]["target_model"] == item["target_model"] and models[0]["working"] == 1
    rep = client.post("/api/corpus/disclosure", json={"item_ids": [item["id"]], "vendor_name": "Acme AI"}).json()
    assert len(rep["sha256"]) == 64 and "Acme AI" in rep["html"] and rep["report"]["provenance"]["hold_until"]
    audit = client.get("/api/explorer/disclosures").json()["rows"]
    assert audit[0]["report_hash"] == rep["sha256"] and "plaintext" not in str(audit[0]).lower()
    assert client.get(f"/api/corpus?target_model={item['target_model']}").json()[0]["disclosure_id"] == rep["disclosure_id"]


def test_patch_lifecycle_working_to_patched(client, monkeypatch):
    aid = _confirmed_attempt(client)
    item = client.post("/api/corpus/promote", json={"source_attempt_id": aid}).json()
    client.post(f"/api/corpus/{item['id']}/verify")
    # simulate a vendor patch: the same target now always refuses
    from app.engine.targets import mock
    monkeypatch.setattr(mock.MockTarget, "_respond", lambda self, s, u: mock.REFUSAL)
    out = client.post("/api/corpus/verify-all", params={"target_model": item["target_model"]}).json()
    assert out == {"checked": 1, "working": 0, "patched": 1}


def test_corpus_is_physically_separate(client, env):
    import sqlite3
    t = sqlite3.connect(env / "t.db"); c = sqlite3.connect(env / "corpus" / "corpus.db")
    tt = {r[0] for r in t.execute("select name from sqlite_master")}
    ct = {r[0] for r in c.execute("select name from sqlite_master")}
    assert "successful_attacks" in ct and "successful_attacks" not in tt
    assert "runs" in tt and "runs" not in ct
    import os, stat
    assert stat.S_IMODE(os.stat(env / "corpus" / "corpus.db").st_mode) == 0o600
    assert stat.S_IMODE(os.stat(env / "corpus").st_mode) == 0o700


def test_one_report_one_model(client):
    aid = _confirmed_attempt(client)
    item = client.post("/api/corpus/promote", json={"source_attempt_id": aid}).json()
    assert client.post("/api/corpus/disclosure", json={"item_ids": ["nope"], "vendor_name": "x"}).status_code == 404
    assert item


def test_only_corpus_modules_import_corpus_engine():
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parents[2] / "app"
    allowed = {"corpus", "api/corpus.py", "db/corpus"}
    for f in root.rglob("*.py"):
        rel = f.relative_to(root).as_posix()
        if re.search(r"db\.corpus", f.read_text()) and not any(rel.startswith(a) for a in allowed):
            pytest.fail(f"{rel} imports the corpus engine")


def test_pyrit_only_imported_in_engine():
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parents[2] / "app"
    for f in root.rglob("*.py"):
        rel = f.relative_to(root).as_posix()
        if re.search(r"^\s*(import|from)\s+pyrit\b", f.read_text(), re.M) and not rel.startswith("engine/"):
            pytest.fail(f"{rel} imports pyrit outside engine/")
