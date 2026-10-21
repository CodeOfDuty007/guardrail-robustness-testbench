"""Sprint-0: a family run persists family/category columns; legacy DB gets migrated columns."""
import sqlite3

from tests.integration.test_run import wait_done


def test_family_run_persists_family_columns(client):
    rid = client.post("/api/runs", json={"families": ["encoding"], "ciphers": ["NONE", "STANDARD_BASE64"]}).json()["id"]
    assert wait_done(client, rid)["status"] == "done"
    atts = client.get(f"/api/runs/{rid}/attempts").json()
    assert len(atts) == 6  # 3 canary objectives x 2 ciphers; legacy strategies NOT also run
    assert {a["family"] for a in atts} == {"encoding"} and {a["category"] for a in atts} == {"jailbreak"}
    assert {a["channel"] for a in atts} == {"direct"}
    assert client.get("/api/families").json()[0]["name"]


def test_unknown_family_errors(client):
    rid = client.post("/api/runs", json={"families": ["nope"]}).json()["id"]
    assert wait_done(client, rid)["status"] == "error"


def test_migration_adds_columns_to_legacy_db(tmp_path, monkeypatch):
    from app.config import settings
    from app.db.telemetry.session import init_telemetry, reset_engine
    db = tmp_path / "legacy.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE attempts (id INTEGER PRIMARY KEY, run_id INT, iteration INT, objective_id TEXT, "
                "strategy_name TEXT, cipher_type TEXT, defense_name TEXT, status TEXT, created_at TEXT)")
    con.commit(); con.close()
    monkeypatch.setattr(settings, "telemetry_url", f"sqlite:///{db}")
    reset_engine()
    init_telemetry(); init_telemetry()  # idempotent
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(attempts)")}
    assert {"family", "category", "channel", "delivery"} <= cols
    reset_engine()
