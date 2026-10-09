import asyncio

import pytest

from tests.integration.test_run import wait_done

pytest.importorskip("pyrit")


def test_pyrit_engine_runs_all_native_attacks(client):
    rid = client.post("/api/runs", json={
        "engine": "pyrit", "ciphers": ["ROT_N", "CUSTOM_BASE64"], "max_iterations": 3, "target_provider_id": None,
        "strategies": ["DirectCipherWrap", "Crescendo", "TAP", "PAIR"], "owasp_tag": "LLM01:Prompt Injection"}).json()["id"]
    run = wait_done(client, rid, 120)
    assert run["status"] == "done", run
    atts = client.get(f"/api/runs/{rid}/attempts").json()
    assert {a["strategy_name"] for a in atts} == {"DirectCipherWrap", "Crescendo", "TAP", "PAIR"}
    assert len(atts) == 3 * 2 * 4
    pay = client.get("/api/explorer/payloads?limit=500").json()["rows"]
    assert all(p["shannon_entropy"] is not None for p in pay)       # crypto metrics flowed through PyRIT converter
    assert any(a["status"] == "success" for a in atts)


def test_pyrit_engine_rejects_defenses_and_unknown(client):
    rid = client.post("/api/runs", json={"engine": "pyrit", "defenses": ["perplexity_filter"]}).json()["id"]
    r = wait_done(client, rid)
    assert r["status"] == "error" and "defenses" in r["error"]


def test_pyrit_types_do_not_leak_into_api(client):
    rid = client.post("/api/runs", json={"engine": "pyrit", "ciphers": ["ROT_N"]}).json()["id"]
    wait_done(client, rid, 60)
    body = client.get("/api/explorer/responses").text + client.get(f"/api/runs/{rid}").text
    assert "pyrit." not in body and "ComponentIdentifier" not in body


def test_attacker_receives_refusal_feedback():
    from app.engine.converters.ciphers import get_cipher
    from app.engine.strategies import STRATEGY_REGISTRY
    seen = []

    async def attacker(obj, i, last):
        seen.append(last); return obj
    asyncio.run(STRATEGY_REGISTRY["PAIR"].build("x", get_cipher("NONE"), 1, attacker, lambda t: t, "I'm sorry, no."))
    assert seen == ["I'm sorry, no."]


def test_all_model_roles_are_metered(env):
    """Attacker/judge/guard calls share the semaphore, the deadline and the budget (SPEC §15)."""
    from app.engine.orchestrator import Metered, RunConfig, RunState, _BudgetHalt
    from app.engine.targets.base import TargetResult

    class Paid:
        model_name = "paid"
        async def send(self, system, user): return TargetResult("ok", cost_usd=0.4)

    st = RunState(1, RunConfig(budget_usd=0.5))
    judge = Metered(Paid(), st, "judge")

    async def go():
        await judge.send("", "a"); await judge.send("", "b")
        with pytest.raises(_BudgetHalt):
            await judge.send("", "c")
    asyncio.run(go())
    assert st.spent == pytest.approx(0.8)


def test_pyrit_init_is_single_flight():
    from app.engine import pyrit_native as pn
    calls = []

    async def fake(*a, **k):
        calls.append(1); await asyncio.sleep(0.05)

    async def go():
        pn._initialised = False
        orig = pn.initialize_pyrit_async
        pn.initialize_pyrit_async = fake
        try:
            await asyncio.gather(*[pn.ensure_init() for _ in range(8)])
        finally:
            pn.initialize_pyrit_async = orig
            pn._initialised = True
    asyncio.run(go())
    assert len(calls) == 1


def test_cancelled_run_is_finalised(env):
    from sqlmodel import Session

    from app.db.telemetry import models as m
    from app.db.telemetry.session import get_engine
    from app.engine.orchestrator import RunConfig, create_run, execute_run

    async def slow(self, system, user):
        await asyncio.sleep(30)

    from app.engine.targets.mock import MockTarget
    orig = MockTarget.send
    MockTarget.send = slow
    try:
        cfg = RunConfig(ciphers=["NONE", "ROT_N"])
        rid = create_run(cfg)

        async def go():
            t = asyncio.create_task(execute_run(rid, cfg))
            await asyncio.sleep(0.3)
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        asyncio.run(go())
    finally:
        MockTarget.send = orig
    with Session(get_engine()) as s:
        r = s.get(m.Run, rid)
        assert r.status == "error" and r.error == "Cancelled"
