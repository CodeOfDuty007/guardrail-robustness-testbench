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


def test_scripted_attacker_sends_bare_objective():
    """Invariant 1: offline fallback must not contain any authored framing template text."""
    import json
    from app.engine import pyrit_native as pn
    from pyrit.models import Message

    obj = "CANARY-SCRIPTED-CHECK"
    sa = pn.ScriptedAttacker(objective=obj)

    async def go():
        pieces = await sa._send_prompt_to_target_async(normalized_conversation=[
            Message.from_prompt(prompt="system", role="system"),
            Message.from_prompt(prompt="user", role="user"),
        ])
        return json.loads(pieces[0].get_value())["next_message"]

    msg = asyncio.run(go())
    assert msg == obj, f"ScriptedAttacker must send the bare objective; got {msg!r}"
    assert "authorized" not in msg.lower()
    assert "hypothetically" not in msg.lower()
    assert "manager" not in msg.lower()


def test_run_attack_exposes_crescendo_backtracks():
    """max_backtracks is wired into CrescendoAttack, not silently discarded."""
    import inspect
    from pyrit.executor.attack import CrescendoAttack
    from app.engine import pyrit_native as pn
    sig = inspect.signature(pn.run_attack)
    assert "max_backtracks" in sig.parameters
    assert "tree_width" in sig.parameters
    assert "branching_factor" in sig.parameters
    ctor = inspect.signature(CrescendoAttack.__init__)
    assert "max_backtracks" in ctor.parameters


def test_offline_adaptive_fallback_sends_bare_objective():
    """strategies.Strategy.build() with no attacker must send the raw objective (invariant 1)."""
    from app.engine.converters.ciphers import get_cipher
    from app.engine.strategies import STRATEGY_REGISTRY

    async def go():
        return await STRATEGY_REGISTRY["PAIR"].build("BARE-OBJ", get_cipher("NONE"), 0, None, lambda t: t, None)

    built = asyncio.run(go())
    assert built.plaintext == "BARE-OBJ"
    assert "authorized" not in built.user.lower()
    assert "hypothetically" not in built.user.lower()


def test_pyrit_runconfig_rejects_out_of_range_search_params():
    """RunConfig must reject search parameters outside their allowed bounds (Codex finding)."""
    import pytest
    from pydantic import ValidationError
    from app.engine.orchestrator import RunConfig
    with pytest.raises(ValidationError):
        RunConfig(pyrit_max_backtracks=-1)
    with pytest.raises(ValidationError):
        RunConfig(pyrit_max_backtracks=51)
    with pytest.raises(ValidationError):
        RunConfig(pyrit_tree_width=0)
    with pytest.raises(ValidationError):
        RunConfig(pyrit_tree_width=9)
    with pytest.raises(ValidationError):
        RunConfig(pyrit_branching_factor=0)
    with pytest.raises(ValidationError):
        RunConfig(pyrit_branching_factor=9)
    # valid boundary values must not raise
    RunConfig(pyrit_max_backtracks=1, pyrit_tree_width=1, pyrit_branching_factor=1)
    RunConfig(pyrit_max_backtracks=50, pyrit_tree_width=8, pyrit_branching_factor=8)


def test_scripted_attacker_caps_to_1_backtrack_when_no_llm(env):
    """Without a real attacker LLM, Crescendo must not waste calls on identical retries."""
    import tempfile
    from pathlib import Path
    from app.config import settings
    d = Path(tempfile.mkdtemp())
    settings.telemetry_url = f"sqlite:///{d/'t.db'}"; settings.corpus_dir = d/"c"; settings.datasets_dir = d/"ds"
    from app.db.telemetry.session import init_telemetry
    from app.db.corpus.session import init_corpus
    from app.engine.orchestrator import RunConfig, create_run, execute_run
    init_telemetry(); init_corpus()
    # large backtracks + no attacker: must not multiply target calls
    cfg = RunConfig(engine="pyrit", strategies=["Crescendo"], ciphers=["NONE"], max_iterations=2,
                    pyrit_max_backtracks=20, limit=1)
    calls = []
    from app.engine.targets import mock as mock_mod
    orig = mock_mod.MockTarget.send
    async def counting_send(self, system, user):
        calls.append(1); return await orig(self, system, user)
    mock_mod.MockTarget.send = counting_send
    try:
        rid = create_run(cfg)
        asyncio.run(execute_run(rid, cfg))
    finally:
        mock_mod.MockTarget.send = orig
    # with eff_backtracks=0, Crescendo makes exactly 1 target call per turn per objective (no retries)
    # 1 objective, 2 turns => exactly 2 target calls
    assert len(calls) == 2, f"expected 2 target calls (1 obj × 2 turns × 0 backtracks), got {len(calls)}"


def test_offline_pair_capped_to_width_1(env):
    """Without an attacker LLM, PAIR must send the same target exactly once (eff_width=1), not three times."""
    import tempfile
    from pathlib import Path
    from app.config import settings
    d = Path(tempfile.mkdtemp())
    settings.telemetry_url = f"sqlite:///{d/'t.db'}"; settings.corpus_dir = d/"c"; settings.datasets_dir = d/"ds"
    from app.db.telemetry.session import init_telemetry
    from app.db.corpus.session import init_corpus
    from app.engine.orchestrator import RunConfig, create_run, execute_run
    init_telemetry(); init_corpus()
    cfg = RunConfig(engine="pyrit", strategies=["PAIR"], ciphers=["NONE"], max_iterations=1,
                    pyrit_tree_width=3, limit=1)  # configured width=3 but no attacker → must cap to 1
    calls = []
    from app.engine.targets import mock as mock_mod
    orig = mock_mod.MockTarget.send
    async def counting_send(self, system, user):
        calls.append(1); return await orig(self, system, user)
    mock_mod.MockTarget.send = counting_send
    try:
        rid = create_run(cfg)
        asyncio.run(execute_run(rid, cfg))
    finally:
        mock_mod.MockTarget.send = orig
    # PAIR with eff_width=1 and depth=1 sends exactly 1 target prompt per objective (1 objective → 1 call).
    assert len(calls) == 1, f"expected exactly 1 target call for offline PAIR (width capped to 1), got {len(calls)}"


def test_winning_request_lookup_hits_uuid():
    """Empirically verify that run_attack returns a non-None winning_request via UUID lookup.

    Uses a stub target that always returns identical text (the worst-case collision scenario for
    a text-keyed map) and a ScriptedAttacker.  If PyRIT rebuilds response pieces from memory
    with new UUIDs, the lookup would miss and winning_request would be None — that failure mode
    is caught here before it reaches the orchestrator.
    """
    from app.engine.targets.base import TargetResult
    from app.engine import pyrit_native as pn

    calls: list[str] = []

    class StubTarget:
        model_name = "stub"

        async def send(self, system: str, user: str) -> TargetResult:
            calls.append(user)
            return TargetResult("same reply every time")  # identical text — kills text-keyed maps

    async def go() -> dict:
        await pn.ensure_init()
        sa = pn.ScriptedAttacker(objective="test-obj")
        adapter = await pn.make_adapter(StubTarget())
        scorer = pn.substring_scorer("NEVER_MATCHES_CANARY_XYZ")
        return await pn.run_attack(
            strategy="Crescendo", objective="test-obj",
            target=adapter, converter=None, scorer=scorer,
            adversarial=sa, max_turns=2, max_backtracks=0,
        )

    out = asyncio.run(go())
    assert out["winning_request"] is not None, (
        "winning_request is None — UUID not preserved through PyRIT memory (regression)")
    assert out["winning_request"] in calls, "winning_request must be one of the actual calls made"
    assert out["winning_plaintext"] is not None, (
        "winning_plaintext is None — original_value not extracted from request piece")


def test_winning_ciphertext_and_metrics_are_branch_accurate():
    """CipherConverter._last_by_task + TargetAdapter attribution gives branch-accurate ct/metrics.

    Each asyncio task gets its own slot in _last_by_task; parallel PAIR/TAP branches (different
    tasks) cannot overwrite each other even when the attacker generates the same text.
    """
    import asyncio
    from app.engine.converters.ciphers import get_cipher
    from app.engine import pyrit_native as pn

    cipher = get_cipher("ROT_N")
    conv = pn.CipherConverter(cipher=cipher)

    async def branch(text: str) -> tuple:
        await conv.convert_async(prompt=text)
        task = asyncio.current_task()
        return conv._last_by_task.get(task)

    async def go():
        # Run two branches concurrently (different tasks) with different plaintexts
        a, b = await asyncio.gather(
            asyncio.create_task(branch("first plaintext")),
            asyncio.create_task(branch("second plaintext")),
        )
        return a, b

    a, b = asyncio.run(go())
    assert a is not None and b is not None, "_last_by_task must have entries for each task"
    plain_a, ct_a, m_a = a
    plain_b, ct_b, m_b = b
    assert plain_a == "first plaintext" and plain_b == "second plaintext", "each task sees its own plaintext"
    assert ct_a != ct_b, "ROT_N produces distinct ciphertexts for distinct inputs"
    assert isinstance(m_a, dict) and m_a, "metrics must be populated for each branch"
    assert plain_a != ct_a and plain_b != ct_b, "ROT_N should transform the text"


def test_none_cipher_parallel_branches_no_plaintext_race():
    """NONE-cipher parallel branches: TargetAdapter._send_prompt_to_target_async must
    record the plaintext from THAT branch's convert, not a later branch's convert.

    Simulates the actual TAP/PAIR race: branch A converts, then yields (sleep(0)) to
    let branch B's convert overwrite a shared-attribute design, then branch A sends.
    With _last_by_task (per-asyncio.Task), A's send still reads A's slot.
    With the old CipherConverter.last_plaintext (shared), B's convert would overwrite
    A's value before A's send reads it, recording B's text under A's response UUID.

    Goes through TargetAdapter._send_prompt_to_target_async (the full attribution path)
    and checks _plaintext_by_resp_id / _ciphertext_by_resp_id by response piece UUID.
    """
    import asyncio
    from pyrit.models import Message
    from app.engine.converters.ciphers import get_cipher
    from app.engine import pyrit_native as pn
    from app.engine.targets.base import TargetResult

    class StubTarget:
        model_name = "stub"
        async def send(self, system: str, user: str) -> TargetResult:
            return TargetResult(f"reply:{user}")

    async def branch_task(label: str, conv: pn.CipherConverter, adapter: pn.TargetAdapter,
                          yield_after_convert: bool = False) -> str:
        """Simulate one TAP branch: convert → (optional yield) → send.

        The yield forces the interleaving: if yield_after_convert=True, another branch's
        convert_async can run in the gap, which would corrupt last_plaintext in the old design.
        """
        await conv.convert_async(prompt=label)
        if yield_after_convert:
            await asyncio.sleep(0)  # yield: let the other branch's convert run now
        msgs = await adapter._send_prompt_to_target_async(
            normalized_conversation=[
                Message.from_prompt(prompt="sys", role="system"),
                Message.from_prompt(prompt=label, role="user"),
            ]
        )
        return str(msgs[0].message_pieces[0].id)

    async def go() -> dict:
        await pn.ensure_init()
        conv = pn.CipherConverter(cipher=get_cipher("NONE"))
        adapter = pn.TargetAdapter(target=StubTarget())
        adapter._conv = conv
        # Branch A yields after convert so branch B's convert can overwrite a shared attribute.
        # Branch B does NOT yield — its convert runs inside A's yield window.
        pid_a, pid_b = await asyncio.gather(
            asyncio.create_task(branch_task("plain-A", conv, adapter, yield_after_convert=True)),
            asyncio.create_task(branch_task("plain-B", conv, adapter, yield_after_convert=False)),
        )
        return {"pid_a": pid_a, "pid_b": pid_b,
                "plain_a": adapter._plaintext_by_resp_id.get(pid_a),
                "ct_a": adapter._ciphertext_by_resp_id.get(pid_a),
                "plain_b": adapter._plaintext_by_resp_id.get(pid_b),
                "ct_b": adapter._ciphertext_by_resp_id.get(pid_b)}

    r = asyncio.run(go())
    assert r["plain_a"] == "plain-A", (
        f"NONE-cipher race: branch A got plaintext {r['plain_a']!r} instead of 'plain-A' "
        f"— B's convert overwrote before A's send read it")
    assert r["plain_b"] == "plain-B", (
        f"NONE-cipher race: branch B got plaintext {r['plain_b']!r}")
    # NONE cipher: ct == plain
    assert r["ct_a"] == "plain-A", f"ct mismatch for branch A: {r['ct_a']!r}"
    assert r["ct_b"] == "plain-B", f"ct mismatch for branch B: {r['ct_b']!r}"


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
