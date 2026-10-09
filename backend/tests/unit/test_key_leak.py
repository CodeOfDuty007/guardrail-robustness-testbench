"""SPEC §16: serialise telemetry, corpus rows and disclosure artifacts; fail if any key-shaped string."""
import json
import sqlite3

from app.providers.redaction import find_keys, redact
from tests.integration.test_run import wait_done

FAKE_KEYS = ["sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWX1234567890", "sk-proj-1234567890abcdefghijklmnop",
             "AIzaSyA-1234567890abcdefghijklmnopqrstu", "gsk_abcdefghijklmnopqrstuvwx", "hf_abcdefghijklmnopqrstuvwxyz"]


def test_pattern_detection():
    for k in FAKE_KEYS:
        assert find_keys(f"x {k} y"), k
        assert k not in redact(f"x {k} y")
    assert not find_keys("The secret token is CANARY-0000053a.")


def test_no_key_in_any_store_or_report(client, env):
    pid = client.post("/api/providers", json={"label": "m", "kind": "mock", "model_name": "mock-weak-1b"}).json()["id"]
    for k in FAKE_KEYS:
        r = client.put(f"/api/providers/{pid}/key", json={"key": k})
        assert r.json() == {"ok": True, "has_key": True}
    assert client.get("/api/providers").json()[0]["has_key"] is True
    rid = client.post("/api/runs", json={"target_provider_id": pid, "ciphers": ["CUSTOM_BASE64", "ROT_N", "NONE", "STANDARD_BASE64"],
                                         "strategies": ["DirectCipherWrap", "PAIR", "PersonaRoleplay"], "max_iterations": 5}).json()["id"]
    wait_done(client, rid)
    el = client.get("/api/explorer/judgments?eligible=true").json()["rows"]
    item = client.post("/api/corpus/promote", json={"source_attempt_id": el[0]["attempt_id"]}).json()
    rep = client.post("/api/corpus/disclosure", json={"item_ids": [item["id"]], "vendor_name": "V"}).json()

    blobs = [json.dumps(rep), rep["html"], client.get(f"/api/runs/{rid}/config.yaml").text]
    for db in (env / "t.db", env / "corpus" / "corpus.db"):
        con = sqlite3.connect(db)
        for (tbl,) in con.execute("select name from sqlite_master where type='table'"):
            blobs.append(json.dumps(con.execute(f"select * from {tbl}").fetchall(), default=str))
    for b in blobs:
        assert not find_keys(b), find_keys(b)
        assert not any(k in b for k in FAKE_KEYS)
    # raw bytes of the db files too
    for f in (env / "t.db", env / "corpus" / "corpus.db"):
        raw = f.read_bytes()
        assert not any(k.encode() in raw for k in FAKE_KEYS)
