import time


def wait_done(client, rid, t=30):
    end = time.time() + t
    while time.time() < end:
        r = client.get(f"/api/runs/{rid}").json()
        if r["status"] in ("done", "error"):
            return r
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def test_full_run_persists_everything(client):
    rid = client.post("/api/runs", json={
        "ciphers": ["STANDARD_BASE64", "CUSTOM_BASE64", "AES_128_GCM", "XOR_MASK"],
        "strategies": ["DirectCipherWrap", "PAIR"], "max_iterations": 3,
        "owasp_tag": "LLM01:Prompt Injection"}).json()["id"]
    assert wait_done(client, rid)["status"] == "done"
    for table in ("attempts", "payloads", "responses", "judgments"):
        assert client.get(f"/api/explorer/{table}").json()["total"] >= 12
    pay = client.get("/api/explorer/payloads").json()["rows"][0]
    assert pay["shannon_entropy"] is not None and pay["avalanche_score"] is not None
    assert client.get(f"/api/runs/{rid}/config.yaml").status_code == 200


def test_run_is_reproducible_from_seed(client):
    body = {"ciphers": ["CUSTOM_BASE64", "ROT_N", "XOR_MASK", "AES_128_GCM"], "strategies": ["DirectCipherWrap", "PAIR"],
            "max_iterations": 3, "seed": 99}
    ids = []
    for _ in range(2):
        rid = client.post("/api/runs", json=body).json()["id"]
        wait_done(client, rid); ids.append(rid)

    def sig(rid):
        atts = client.get(f"/api/runs/{rid}/attempts").json()
        return sorted((a["objective_id"], a["strategy_name"], a["cipher_type"], a["iteration"], a["status"]) for a in atts)
    assert sig(ids[0]) == sig(ids[1]) and len(sig(ids[0])) >= 24


def test_budget_halts_run(client):
    rid = client.post("/api/runs", json={"budget_usd": 0.0, "ciphers": ["NONE", "ROT_N"]}).json()["id"]
    wait_done(client, rid)
    assert client.get(f"/api/runs/{rid}/attempts").json() == []


def test_defense_blocks_base64(client):
    rid = client.post("/api/runs", json={"ciphers": ["STANDARD_BASE64"], "defenses": ["encoding_signature_gate"]}).json()["id"]
    wait_done(client, rid)
    assert client.get("/api/explorer/defenses").json()["rows"][0]["blocked"] is True


def test_analytics_endpoints(client):
    rid = client.post("/api/runs", json={"ciphers": ["STANDARD_BASE64", "CUSTOM_BASE64", "AES_128_GCM", "XOR_MASK", "ROT_N", "NONE"],
                                         "strategies": ["DirectCipherWrap", "PAIR", "PersonaRoleplay"], "max_iterations": 4}).json()["id"]
    wait_done(client, rid)
    assert client.get("/api/analytics/overview").json()["total_attempts"] > 20
    c = client.get("/api/analytics/correlation?metric=avalanche_score").json()
    assert c["n"] > 20 and len(c["by_cipher"]) >= 5
    assert client.get("/api/analytics/leaderboard").json()[0]["model"]
    assert client.get("/api/analytics/pareto").status_code == 200


def test_sql_box_is_read_only(client):
    assert client.post("/api/explorer-sql", json={"sql": "select 1 as x"}).json()["rows"] == [[1]]
    assert client.post("/api/explorer-sql", json={"sql": "drop table runs"}).status_code == 400
    assert client.post("/api/explorer-sql", json={"sql": "select 1; drop table runs"}).status_code == 400


def test_sql_console_does_not_poison_pool(client):
    for _ in range(3):
        assert client.post("/api/explorer-sql", json={"sql": "select 1"}).status_code == 200
    assert client.post("/api/runs", json={"ciphers": ["NONE"]}).status_code == 200
    assert client.post("/api/explorer-sql", json={"sql": "select 1 -- x"}).status_code == 400


def test_budget_never_exceeded_with_concurrency(client):
    rid = client.post("/api/runs", json={"budget_usd": 1e-9, "concurrency": 1, "ciphers": ["NONE", "ROT_N", "CUSTOM_BASE64", "XOR_MASK"]}).json()["id"]
    wait_done(client, rid)
    assert len(client.get(f"/api/runs/{rid}/attempts").json()) <= 1


def test_disclosure_preview_does_not_persist(client):
    from tests.unit.test_corpus import _confirmed_attempt
    item = client.post("/api/corpus/promote", json={"source_attempt_id": _confirmed_attempt(client)}).json()
    client.post("/api/corpus/disclosure/preview", json={"item_ids": [item["id"]], "vendor_name": "V"})
    assert client.get("/api/explorer/disclosures").json()["total"] == 0


def test_verify_refuses_missing_provider(client):
    from tests.unit.test_corpus import _confirmed_attempt
    item = client.post("/api/corpus/promote", json={"source_attempt_id": _confirmed_attempt(client)}).json()
    # simulate a deleted provider by patching repro config through a fake id
    import sqlite3, json
    from app.config import settings
    con = sqlite3.connect(settings.corpus_dir / "corpus.db")
    cfg = json.loads(con.execute("select repro_config_json from successful_attacks").fetchone()[0]); cfg["target_provider_id"] = 4242
    con.execute("update successful_attacks set repro_config_json=?", (json.dumps(cfg),)); con.commit(); con.close()
    assert client.post(f"/api/corpus/{item['id']}/verify").status_code == 409


def test_explorer_pagination_sort_search_and_redacted_export(client):
    rid = client.post("/api/runs", json={"ciphers": ["ROT_N", "CUSTOM_BASE64", "NONE"], "strategies": ["DirectCipherWrap", "PAIR"], "max_iterations": 3}).json()["id"]
    wait_done(client, rid)
    p1 = client.get("/api/explorer/attempts?limit=5&offset=0&sort=id&desc=false").json()
    p2 = client.get("/api/explorer/attempts?limit=5&offset=5&sort=id&desc=false").json()
    assert p1["total"] > 10 and len(p1["rows"]) == 5 and p1["rows"][-1]["id"] < p2["rows"][0]["id"]
    assert all(r["cipher_type"] == "ROT_N" for r in client.get("/api/explorer/attempts?q=ROT_N").json()["rows"])
    red = client.get("/api/explorer/payloads/export?format=json").json()
    assert red["provenance"]["redacted"] is True and red["rows"][0]["plaintext_prompt"] == "[redacted]"
    csv_text = client.get("/api/explorer/responses/export?format=csv").text
    assert csv_text.startswith("# operator:") and "[redacted]" in csv_text
    raw = client.get("/api/explorer/responses/export?format=json&redact=false").json()
    assert raw["provenance"]["redacted"] is False and raw["rows"][0]["target_response"] != "[redacted]"


def test_root_and_spa_routes_are_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert client.get("/api/does-not-exist").status_code == 404
    assert client.get("/analytics").status_code in (200, 404)  # 200 once the UI is built


def test_sql_console_stops_runaway_queries(client, monkeypatch):
    import app.api.core as core
    monkeypatch.setattr(core, "SQL_DEADLINE_S", 0.3)
    heavy = ("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x < 50000000) SELECT count(*) FROM n")
    r = client.post("/api/explorer-sql", json={"sql": heavy})
    assert r.status_code == 408 and "stopped" in r.json()["detail"]
    assert client.post("/api/explorer-sql", json={"sql": "select 1"}).status_code == 200      # pool is healthy afterwards
    assert client.post("/api/runs", json={"ciphers": ["NONE"]}).status_code == 200            # and writable
    assert client.post("/api/explorer-sql", json={"sql": "select " + "1+" * 3000 + "1"}).status_code == 400


def test_export_provenance_has_scope_and_truncation(client, monkeypatch):
    import app.api.core as core
    wait_done(client, client.post("/api/runs", json={"ciphers": ["ROT_N", "NONE"], "seed": 5}).json()["id"])
    j = client.get("/api/explorer/attempts/export?format=json").json()["provenance"]
    assert j["runs"] == [1] and j["seeds"] == [5] and j["datasets"] == ["canary"] and j["models"] == ["mock-guarded-1b"]
    assert j["truncated"] is False and j["rows_matching"] == j["rows_exported"] == 6
    monkeypatch.setattr(core, "EXPORT_CAP", 2)
    t = client.get("/api/explorer/attempts/export?format=json").json()["provenance"]
    assert t["truncated"] is True and t["rows_exported"] == 2 and t["rows_matching"] == 6
    csv = client.get("/api/explorer/judgments/export?format=csv&q=ROT").text
    assert "# search: ROT" in csv and "# models:" in csv
