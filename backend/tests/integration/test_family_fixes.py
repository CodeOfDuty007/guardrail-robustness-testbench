"""Merge-time fixes after S1-A1 QA/Codex review: non-flat default profile, compounding compares identical
framings, sub-technique + provenance persisted, attacker refusals reported separately."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.datasets.loaders import canary_objectives
from app.engine.families import family_rng, get_family
from app.engine.families.base import ask_attacker_ex
from app.engine.targets.base import TargetResult
from tests.integration.test_run import wait_done

JAILBREAK = ["encoding", "roleplay_persona", "authority_context", "persuasion", "refusal_suppression",
             "hypothetical_framing", "many_shot", "low_resource_lang", "payload_splitting"]


class FakeAttacker:
    model_name, provider_kind = "fake-attacker", "mock"

    def __init__(self, reply: str | None = "ATTACKER-OUTPUT-MARKER") -> None:
        self.reply, self.calls = reply, 0

    async def send(self, system, user):
        self.calls += 1
        return TargetResult(self.reply)  # type: ignore[arg-type]


@pytest.mark.parametrize("profile", ["mock-weak-1b", "mock-guarded-1b", "mock-hardened-8b"])
def test_default_seed_profile_is_not_flat(client, profile):
    pid = client.post("/api/providers", json={"label": profile, "kind": "mock", "model_name": profile}).json()["id"]
    rid = client.post("/api/runs", json={"families": JAILBREAK, "ciphers": ["NONE"], "target_provider_id": pid}).json()["id"]
    assert wait_done(client, rid)["status"] == "done"
    by: dict[str, list[str]] = {}
    for a in client.get(f"/api/runs/{rid}/attempts").json():
        by.setdefault(a["family"], []).append(a["status"])
    rates = {f: v.count("success") / len(v) for f, v in by.items()}
    assert len(set(rates.values())) >= 2, rates


def test_technique_and_provenance_persisted(client):
    rid = client.post("/api/runs", json={"families": ["persuasion"], "ciphers": ["NONE"]}).json()["id"]
    wait_done(client, rid)
    atts = client.get(f"/api/runs/{rid}/attempts").json()
    assert all(a["technique"] and a["generated_by"] == "fallback" for a in atts)
    steps = client.get("/api/explorer/payloads").json()["rows"][0]["reproduction_steps"]
    assert "Attack family: persuasion" in steps and "verbatim" in steps


async def test_compounding_cells_share_one_framing(env, monkeypatch):
    """bare and ciphered cells must carry the SAME attacker text and sub-technique; one attacker call."""
    from sqlmodel import Session, select

    from app.db.telemetry import models as m
    from app.db.telemetry.session import get_engine
    from app.engine import orchestrator as orch
    att = FakeAttacker()
    monkeypatch.setattr(orch, "Metered", lambda inner, st, role: att if role == "attacker" else inner)
    with Session(get_engine()) as s:
        p = m.Provider(label="a", kind="mock", model_name="mock-guarded-1b"); s.add(p); s.commit(); s.refresh(p)
        pid = p.id
    cfg = orch.RunConfig(families=["persuasion"], wrap_with_cipher=["persuasion"], ciphers=["NONE", "STANDARD_BASE64", "ROT_N"],
                         attacker_provider_id=pid, limit=1)
    rid = orch.create_run(cfg)
    await orch.execute_run(rid, cfg)
    with Session(get_engine()) as s:
        atts = s.exec(select(m.Attempt).where(m.Attempt.run_id == rid)).all()
        plains = {s.exec(select(m.Payload).where(m.Payload.attempt_id == a.id)).one().plaintext_prompt for a in atts}
    assert len(atts) == 3 and att.calls == 1
    assert len({a.technique for a in atts}) == 1 and plains == {"ATTACKER-OUTPUT-MARKER"}


@pytest.mark.parametrize("reply,expected", [
    ("ATTACKER-OUTPUT-MARKER", "attacker"),
    ("I'm sorry, but I can't help with that.", "attacker_refused"),
    ("   ", "fallback"),
    ("​", "fallback"),
    (None, "fallback"),
])
async def test_attacker_provenance(reply, expected):
    o = canary_objectives(1337)[0]
    text, by = await ask_attacker_ex(FakeAttacker(reply), "brief", o, o.text)
    assert by == expected
    assert text == (reply if expected == "attacker" else o.text)


async def test_attacker_refusal_recorded_by_family():
    o = canary_objectives(1337)[0]
    cand = await get_family("persuasion").generate(objective=o, attacker=FakeAttacker("I cannot help with that."),
                                                   history=None, rng=family_rng(1337, "persuasion", o.id))
    assert cand.meta["generated_by"] == "attacker_refused" and cand.user == o.text


_PROBE = r"""
import asyncio, json, sys
sys.path.insert(0, ".")
from app.datasets.loaders import canary_objectives
from app.engine.families import FAMILY_REGISTRY, family_rng, load_all
from app.engine.targets.base import TargetResult
class A:
    model_name = provider_kind = "x"
    async def send(self, s, u): return TargetResult("ATTACKER-OUTPUT-MARKER|" + u[-40:])
async def main():
    load_all(); out = {}
    for name, f in sorted(FAMILY_REGISTRY.items()):
        for o in canary_objectives(7):
            for att in (None, A()):
                c = await f.generate(objective=o, attacker=att, history=None, rng=family_rng(7, name, o.id))
                out[f"{name}|{o.id}|{att is None}"] = [c.user, c.technique, c.meta.get("generated_by")]
    print(json.dumps(out, sort_keys=True))
asyncio.run(main())
"""


def test_generation_is_identical_across_processes():
    backend = Path(__file__).parents[2]
    outs = [subprocess.run([sys.executable, "-c", _PROBE], cwd=backend, capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": hs}).stdout for hs in ("1", "2")]
    assert outs[0] == outs[1] and len(json.loads(outs[0])) >= 9 * 3 * 2


async def test_attacker_refusal_skips_target_and_is_excluded_from_rates(env, monkeypatch):
    """Codex stop-review: refusals by the ATTACKER must never reach the target nor count toward robustness."""
    from sqlmodel import Session, select

    from app.analytics_store import Filters, flat_rows
    from app.api.analytics import overview
    from app.db.telemetry import models as m
    from app.db.telemetry.session import get_engine
    from app.engine import orchestrator as orch
    refusing, target_calls = FakeAttacker("I'm sorry, but I can't help with that."), []

    class CountingTarget:
        model_name, provider_kind = "mock-guarded-1b", "mock"
        async def send(self, system, user):
            target_calls.append(user); return TargetResult("Okay.")

    monkeypatch.setattr(orch, "Metered", lambda inner, st, role: refusing if role == "attacker" else CountingTarget())
    with Session(get_engine()) as s:
        p = m.Provider(label="a", kind="mock", model_name="mock-guarded-1b"); s.add(p); s.commit(); s.refresh(p)
        pid = p.id
    cfg = orch.RunConfig(families=["persuasion", "encoding"], ciphers=["NONE"], attacker_provider_id=pid)
    rid = orch.create_run(cfg)
    await orch.execute_run(rid, cfg)
    with Session(get_engine()) as s:
        atts = s.exec(select(m.Attempt).where(m.Attempt.run_id == rid)).all()
    refused = [a for a in atts if a.family == "persuasion"]
    assert refused and all(a.status == "attacker_refused" and a.generated_by == "attacker_refused" for a in refused)
    assert len(target_calls) == 3  # only the 3 encoding attempts reached the target
    assert {r["family"] for r in flat_rows(Filters(run_id=rid))} == {"encoding"}
    ov = overview(Filters(run_id=rid))
    assert ov["total_attempts"] == 3 and ov["attacker_refused"] == 3


async def test_refusals_are_never_scored_as_target_responses(env, monkeypatch):
    """Codex stop-review #2: family AND legacy-PAIR attacker refusals get no target call and no real judgment."""
    from sqlmodel import Session, select

    from app.db.telemetry import models as m
    from app.db.telemetry.session import get_engine
    from app.engine import orchestrator as orch
    refusing, target_calls, judge_calls = FakeAttacker("I'm sorry, but I can't help with that."), [], []

    class CountingTarget:
        model_name, provider_kind = "mock-guarded-1b", "mock"
        async def send(self, system, user):
            (judge_calls if system.startswith("You are a strict evaluator") else target_calls).append(user)
            return TargetResult("YES")

    monkeypatch.setattr(orch, "Metered", lambda inner, st, role: refusing if role == "attacker" else CountingTarget())
    with Session(get_engine()) as s:
        p = m.Provider(label="a", kind="mock", model_name="mock-guarded-1b"); s.add(p); s.commit(); s.refresh(p)
        pid = p.id
    cfg = orch.RunConfig(families=["persuasion"], strategies=["PAIR"], ciphers=["NONE"], max_iterations=2,
                         attacker_provider_id=pid, judge_provider_id=pid)
    rid = orch.create_run(cfg)
    await orch.execute_run(rid, cfg)
    assert target_calls == [] and judge_calls == []
    with Session(get_engine()) as s:
        atts = s.exec(select(m.Attempt).where(m.Attempt.run_id == rid)).all()
        js = [s.exec(select(m.Judgment).where(m.Judgment.attempt_id == a.id)).one() for a in atts]
    assert {a.strategy_name for a in atts} == {"persuasion", "PAIR"}
    assert all(a.status == "attacker_refused" and a.generated_by == "attacker_refused" for a in atts)
    assert all(j.score_source == "attacker_refused" and not j.filter_evaded and not j.semantic_complied for j in js)
