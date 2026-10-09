import pytest

from tests.integration.test_run import wait_done


@pytest.fixture()
def populated(client):
    for model_pid in (None,):
        rid = client.post("/api/runs", json={
            "ciphers": ["STANDARD_BASE64", "CUSTOM_BASE64", "AES_128_GCM", "XOR_MASK", "ROT_N", "NONE"],
            "strategies": ["DirectCipherWrap", "PAIR", "PersonaRoleplay"], "max_iterations": 4,
            "owasp_tag": "LLM01:Prompt Injection", "atlas_tag": "AML.T0051"}).json()["id"]
        wait_done(client, rid)
    rid2 = client.post("/api/runs", json={"ciphers": ["STANDARD_BASE64", "ROT_N"], "defenses": ["encoding_signature_gate"]}).json()["id"]
    wait_done(client, rid2)
    return client


def test_filters_narrow_every_endpoint(populated):
    c = populated
    allr = c.get("/api/analytics/overview").json()["total_attempts"]
    one = c.get("/api/analytics/overview?cipher=ROT_N").json()["total_attempts"]
    assert 0 < one < allr
    corr = c.get("/api/analytics/correlation?metric=shannon_entropy&cipher=ROT_N").json()
    assert {x["cipher"] for x in corr["by_cipher"]} == {"ROT_N"}
    assert all(r["cipher"] == "ROT_N" for r in c.get("/api/analytics/heatmap?cipher=ROT_N").json())
    assert c.get("/api/analytics/overview?date_from=2999-01-01").json()["total_attempts"] == 0
    opts = c.get("/api/analytics/filters").json()
    assert "ROT_N" in opts["ciphers"] and opts["models"] and len(opts["runs"]) == 2


def test_defense_ablation_shows_delta(populated):
    rows = populated.get("/api/analytics/defense-ablation").json()
    d = {r["defense"]: r for r in rows}
    assert "none" in d and "encoding_signature_gate" in d and d["encoding_signature_gate"]["delta_vs_none"] is not None
    b64 = {r["defense"]: r for r in populated.get("/api/analytics/defense-ablation?cipher=STANDARD_BASE64").json()}
    assert b64["encoding_signature_gate"]["k"] == 0   # the gate blocks every base64-shaped payload


def test_metric_table_heatmap_scaling_pareto_modelsize_timeline(populated):
    c = populated
    mt = c.get("/api/analytics/metric-table").json()
    assert {m["metric"] for m in mt} == {"avalanche_score", "shannon_entropy", "diffusion_score", "confusion_chi2"}
    assert c.get("/api/analytics/heatmap").json()
    sc = c.get("/api/analytics/scaling").json()
    assert sc["points"] and "AES_128_GCM" in sc["fits"] and sc["fits"]["AES_128_GCM"]["expansion"] > 1
    assert any(p["frontier"] for p in c.get("/api/analytics/pareto").json())
    ms = c.get("/api/analytics/model-size").json()
    assert ms[0]["model"] == "mock-guarded-1b" and ms[0]["size_b"] == 1.0
    assert len(c.get("/api/analytics/timeline").json()) == 2


def test_judge_agreement_has_three_signal_pairs(populated):
    row = populated.get("/api/analytics/judge-agreement").json()[0]
    assert set(row) >= {"regex_judge", "regex_guard", "judge_guard"}
    assert row["regex_judge"]["agreement"] is None      # no judge configured → no fabricated agreement


def test_report_is_aggregate_only_and_flags_simulation(populated):
    c = populated
    rep = c.get("/api/analytics/report").json()
    assert rep["provenance"]["simulated_data"] is True and rep["provenance"]["license"] == "CC-BY-4.0"
    blob = str(rep) + c.get("/api/analytics/report?format=html").text
    row = c.get("/api/explorer/payloads?limit=1").json()["rows"][0]
    assert row["plaintext_prompt"] not in blob and row["ciphertext_payload"] not in blob
    assert "CANARY-" not in blob and "SECRET_TOKEN" not in blob


def test_logistic_fit_recovers_slope():
    import numpy as np

    from app.api.analytics import logistic_fit
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 4000)
    y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-(-1 + 4 * x)))).astype(float)
    b0, b1 = logistic_fit(x, y)
    assert abs(b1 - 4) < 0.6 and abs(b0 + 1) < 0.4


def test_defense_export_redacts_transformed_payload(populated):
    raw = populated.get("/api/explorer/defenses/export?format=json&redact=true").json()
    assert all(r["transformed_payload"] in (None, "[redacted]") for r in raw["rows"])


def test_replica_refreshes_on_in_place_updates(populated):
    from sqlalchemy import text

    from app.db.telemetry.session import get_engine
    before = populated.get("/api/analytics/overview").json()["confirmed"]
    with get_engine().begin() as c:  # same row counts, different verdicts
        c.execute(text("UPDATE judgments SET semantic_complied = 0"))
    assert populated.get("/api/analytics/overview").json()["confirmed"] == 0 != before


def test_constant_timings_do_not_break_scaling_json(env):
    from app.api.analytics import _ols
    r = _ols([1, 2, 3, 4], [0.5, 0.5, 0.5, 0.5])
    assert r is not None and (r["r2"] is None or r["r2"] == r["r2"])  # never NaN


def test_overview_run_count_is_filter_scoped(populated):
    assert populated.get("/api/analytics/overview?run_id=1").json()["total_runs"] == 1


def test_ablation_compares_matched_cells_only(client):
    """The defended run covers only 2 ciphers; the baseline covers 6. Only the shared cells may be compared."""
    ALL = ["STANDARD_BASE64", "CUSTOM_BASE64", "AES_128_GCM", "XOR_MASK", "ROT_N", "NONE"]
    for body in ({"ciphers": ALL}, {"ciphers": ["STANDARD_BASE64", "ROT_N"], "defenses": ["encoding_signature_gate"]}):
        wait_done(client, client.post("/api/runs", json=body).json()["id"])
    rows = {r["defense"]: r for r in client.get("/api/analytics/defense-ablation").json()}
    d = rows["encoding_signature_gate"]
    assert d["matched_cells"] == 2 and d["n"] == 6        # 2 shared cipher cells x 3 objectives
    assert d["baseline_n"] == 6                            # NOT the 18 undefended attempts
    assert d["delta_ci_low"] <= d["delta_vs_none"] <= d["delta_ci_high"]
    assert rows["none"]["scope"]["comparison"].startswith("matched")


def test_ablation_without_overlap_says_so(client):
    wait_done(client, client.post("/api/runs", json={"ciphers": ["ROT_N"], "defenses": ["perplexity_filter"]}).json()["id"])
    wait_done(client, client.post("/api/runs", json={"ciphers": ["NONE"]}).json()["id"])
    d = {r["defense"]: r for r in client.get("/api/analytics/defense-ablation").json()}["perplexity_filter"]
    assert d["matched_cells"] == 0 and d["delta_vs_none"] is None and "no cells" in d["note"]


def test_replica_is_incremental_and_rebuilds_on_edit(populated, monkeypatch):
    from sqlalchemy import text

    from app import analytics_store as store
    from app.db.telemetry.session import get_engine
    populated.get("/api/analytics/overview")
    builds = []
    orig = store._full_rebuild
    monkeypatch.setattr(store, "_full_rebuild", lambda: (builds.append(1), orig())[1])
    wait_done(populated, populated.post("/api/runs", json={"ciphers": ["ROT_N"]}).json()["id"])
    n = populated.get("/api/analytics/overview").json()["total_attempts"]
    assert builds == [] and n > 0                           # appended, no rebuild
    with get_engine().begin() as c:
        c.execute(text("UPDATE judgments SET semantic_complied = 0"))
    assert populated.get("/api/analytics/overview").json()["confirmed"] == 0 and builds == [1]


def test_report_sections_complete_and_from_one_snapshot(populated):
    rep = populated.get("/api/analytics/report").json()
    assert {"pareto", "scaling_fits", "defense_ablation", "model_size", "standards", "judge_agreement"} <= set(rep)
    assert rep["provenance"]["attempts"] == rep["overview"]["total_attempts"] == sum(m["n"] for m in rep["model_size"])
    html = populated.get("/api/analytics/report?format=html").text
    for h in ("Defense ablation", "Model size", "OWASP", "Judge agreement", "Cost frontier", "Print / save as PDF"):
        assert h in html


def test_filtered_html_report_renders(populated):
    r = populated.get("/api/analytics/report?format=html&cipher=ROT_N&model=mock-guarded-1b")
    assert r.status_code == 200 and "cipher" in r.text and "ROT_N" in r.text
