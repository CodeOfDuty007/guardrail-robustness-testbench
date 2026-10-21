"""S1-A4: direct prompt-injection families run end-to-end against the mock target."""
from sqlmodel import Session, select

from app.db.telemetry import models as m
from app.db.telemetry.session import get_engine
from tests.integration.test_run import wait_done

INJ = ["inj_context_ignore", "inj_authority", "inj_conditional_trigger", "inj_identity_rewrite", "inj_output_control"]


def _run(client, **extra):
    body = {"dataset": "injection_canary", "families": INJ, "ciphers": ["NONE"], "seed": 1337, **extra}
    rid = client.post("/api/runs", json=body).json()["id"]
    assert wait_done(client, rid)["status"] == "done"
    atts = client.get(f"/api/runs/{rid}/attempts").json()
    with Session(get_engine()) as s:
        judg = {j.attempt_id: j for j in s.exec(select(m.Judgment).where(
            m.Judgment.attempt_id.in_([a["id"] for a in atts]))).all()}  # type: ignore[attr-defined]
    return atts, judg


def test_injection_run_on_default_mock(client):
    atts, judg = _run(client)
    assert len(atts) == 15  # 3 objectives x 5 families, bare only (wrap_with_cipher=False)
    assert {a["family"] for a in atts} == set(INJ)
    assert {a["category"] for a in atts} == {"injection"} and {a["channel"] for a in atts} == {"direct"}
    assert {a["cipher_type"] for a in atts} == {"NONE"}
    assert len(judg) == 15 and all(j.regex_verdict is not None for j in judg.values())
    assert "injection_canary" in {d["name"] for d in client.get("/api/meta").json()["datasets"]}


def test_injection_detection_exercised_both_ways_on_weak_mock(client):
    pid = client.post("/api/providers", json={"label": "weak", "kind": "mock", "model_name": "mock-weak-1b"}).json()["id"]
    atts, judg = _run(client, target_provider_id=pid)
    verdicts = [j.regex_verdict for j in judg.values()]
    assert True in verdicts and False in verdicts
    for j in judg.values():  # regex arm is authoritative for injection objectives
        assert j.semantic_complied == j.regex_verdict


def test_injection_run_is_reproducible(client):
    def sig():
        atts, judg = _run(client)
        return sorted((a["objective_id"], a["family"], judg[a["id"]].regex_verdict) for a in atts)
    assert sig() == sig()
