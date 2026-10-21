"""Corpus service: promote, re-verify, disclosure. ONLY module (with api/corpus.py) that opens corpus.db."""
from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlmodel import Session, select

from app.config import settings
from app.db.corpus.models import SuccessfulAttack
from app.db.telemetry import models as m
from app.db.telemetry.session import get_engine
from app.engine.orchestrator import Deadline, replay, resolve_target
from app.providers.redaction import find_keys, redact


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def promote(cs: Session, attempt_id: int, curator: str | None = None, tags: str = "", notes: str = "") -> SuccessfulAttack:
    """Human-curated promotion. Eligible only if filter_evaded AND semantic_complied (SPEC §6b)."""
    with Session(get_engine()) as ts:
        att = ts.get(m.Attempt, attempt_id)
        if att is None:
            raise LookupError("attempt not found")
        j = ts.exec(select(m.Judgment).where(m.Judgment.attempt_id == attempt_id)).first()
        if not (j and j.filter_evaded and j.semantic_complied):
            raise ValueError("attempt is not promote-eligible (needs filter_evaded AND semantic_complied)")
        p = ts.exec(select(m.Payload).where(m.Payload.attempt_id == attempt_id)).one()
        r = ts.exec(select(m.Response).where(m.Response.attempt_id == attempt_id)).one()
        run = ts.get(m.Run, att.run_id)
        prov = ts.get(m.Provider, run.target_provider_id) if run.target_provider_id else None
        existing = cs.exec(select(SuccessfulAttack).where(
            SuccessfulAttack.source_attempt_id == attempt_id)).first()
        if existing:
            return existing
        auto_tags = ",".join(t for t in (run.owasp_tag, run.atlas_tag) if t)
        item = SuccessfulAttack(
            source_run_id=run.id, source_attempt_id=attempt_id, target_model=r.target_model,
            target_provider=prov.kind if prov else "mock", objective_id=att.objective_id,
            strategy_name=att.strategy_name, cipher_type=att.cipher_type,
            plaintext_prompt=p.plaintext_prompt, ciphertext_payload=p.ciphertext_payload,
            decode_instruction=p.decode_instruction,
            full_request_sent=p.full_request_sent, reproduction_steps=p.reproduction_steps,
            harm_score_0_1=j.harm_score_0_1,
            judge_rationale=j.judge_rationale, classifier_label=j.classifier_label,
            metrics_json={k: getattr(p, k) for k in (
                "encryption_time_ms", "decryption_time_ms", "avalanche_score", "confusion_chi2",
                "diffusion_score", "shannon_entropy")} | {"latency_ms": r.latency_ms, "cost_usd": r.cost_usd},
            seed=run.seed, repro_config_json=json.loads(run.config_json) | {"dataset": run.dataset_name},
            curated_by=curator or settings.operator_name, tags=",".join(filter(None, [auto_tags, tags])), notes=notes)
        cs.add(item); cs.commit(); cs.refresh(item)
        return item


async def verify(cs: Session, item: SuccessfulAttack, judge=None) -> SuccessfulAttack:
    """Re-run the stored payload against the *current* target; flip working/patched."""
    with Session(get_engine()) as ts:
        pid = item.repro_config_json.get("target_provider_id")
        prov = ts.get(m.Provider, pid) if pid else None
        if pid and prov is None:
            raise LookupError(f"target provider {pid} no longer exists; cannot re-verify")
        target = resolve_target(prov, item.seed)
        if target.model_name != item.target_model:
            raise ValueError(f"provider now serves {target.model_name!r}, item is for {item.target_model!r}")
    spec = {"seed": item.seed, "cipher_type": item.cipher_type, "strategy_name": item.strategy_name,
            "objective_id": item.objective_id, "plaintext_prompt": item.plaintext_prompt,
            "full_request_sent": item.full_request_sent,
            "dataset": item.repro_config_json.get("dataset", "canary")}
    item_id = item.id
    cs.commit()  # end the read transaction: never hold a corpus connection across a model call
    score = await replay(spec, target, judge)
    item = cs.get(SuccessfulAttack, item_id) or item
    item.status = "working" if score.confirmed else "patched"
    item.last_verified_at = _now()
    cs.add(item); cs.commit(); cs.refresh(item)
    return item


async def verify_all(cs: Session, target_model: str, judge=None) -> dict:
    ids = list(cs.exec(select(SuccessfulAttack.id).where(
        SuccessfulAttack.target_model == target_model, SuccessfulAttack.status != "patched")).all())
    cs.commit()
    out = {"checked": 0, "working": 0, "patched": 0}
    for iid in ids:
        it = cs.get(SuccessfulAttack, iid)
        if it is None:
            continue
        try:
            await verify(cs, it, judge)
        except (LookupError, ValueError, RuntimeError):
            out["skipped"] = out.get("skipped", 0) + 1
            continue
        out["checked"] += 1
        out[it.status] += 1
    return out


def make_judge(judge_provider_id: int | None):
    """Judge LLM for BYO-dataset re-verification (None → regex canary only)."""
    if not judge_provider_id:
        return None
    from app.engine.scorers.composite import make_llm_judge
    with Session(get_engine()) as ts:
        p = ts.get(m.Provider, judge_provider_id)
        if p is None:
            raise LookupError(f"judge provider {judge_provider_id} not found")
        j = make_llm_judge(Deadline(resolve_target(p, 0)))
        j.model = p.model_name  # type: ignore[attr-defined]
        return j


def by_model(cs: Session) -> list[dict]:
    rows = cs.execute(text("SELECT * FROM known_jailbreaks_by_model")).mappings().all()
    return [dict(r) for r in rows]


# ---- disclosure (SPEC §6a) --------------------------------------------------------------------
def _signing_key() -> bytes:
    p = settings.corpus_dir / ".signing_key"
    if not p.exists():
        p.write_text(secrets.token_hex(32))
        os.chmod(p, 0o600)
    return p.read_text().strip().encode()


def build_disclosure(cs: Session, item_ids: list[str], vendor_name: str, vendor_contact: str = "",
                     hold_days: int | None = None, notes: str = "", persist: bool = True) -> dict:
    items = [cs.get(SuccessfulAttack, i) for i in item_ids]
    if not items or any(i is None for i in items):
        raise LookupError("one or more corpus items not found")
    models = {i.target_model for i in items}
    if len(models) != 1:
        raise ValueError("one report = one target model (SPEC §6a scope rule)")
    now = _now()
    hold = now + timedelta(days=hold_days if hold_days is not None else settings.hold_days)
    body = {
        "provenance": {"target_model": next(iter(models)), "operator": settings.operator_name,
                       "generated_at": now.isoformat(), "hold_until": hold.isoformat(),
                       "vendor": vendor_name, "license": "CC-BY-4.0", "notes": notes,
                       "test_window": [min(i.first_seen_at for i in items).isoformat(),
                                       max(i.last_verified_at or i.first_seen_at for i in items).isoformat()]},
        "items": [{
            "id": i.id, "status": i.status, "strategy": i.strategy_name, "cipher": i.cipher_type,
            "objective": i.objective_id, "plaintext_prompt": i.plaintext_prompt,
            "ciphertext_payload": i.ciphertext_payload, "decode_instruction": i.decode_instruction,
            "full_request_sent": i.full_request_sent, "reproduction_steps": i.reproduction_steps,
            "metrics": i.metrics_json, "judge_rationale": i.judge_rationale,
            "repro": {"seed": i.seed, "config": i.repro_config_json},
            "tags": i.tags} for i in items],
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    if find_keys(canonical):
        raise RuntimeError("key-shaped material found in report; refusing to generate")
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    signature = hmac.new(_signing_key(), canonical.encode(), hashlib.sha256).hexdigest()
    report = {"report": body, "sha256": digest, "hmac_sha256": signature}
    if not persist:
        report["html"] = render_html(report)
        return report
    with Session(get_engine()) as ts:  # audit row lives in telemetry.db (no payload text)
        d = m.Disclosure(target_model=next(iter(models)), vendor_name=vendor_name, vendor_contact=vendor_contact,
                         report_hash=digest, corpus_item_ids_json=json.dumps(item_ids),
                         item_count=len(items), hold_until=hold, notes=notes)
        ts.add(d); ts.commit(); ts.refresh(d)
        disclosure_id = str(d.id)
    for i in items:
        i.disclosure_id = disclosure_id
        cs.add(i)
    cs.commit()
    report["disclosure_id"] = disclosure_id
    report["html"] = render_html(report)
    return report


def render_html(report: dict) -> str:
    b, e = report["report"], html.escape
    rows = "".join(
        f"<section><h3>{e(i['strategy'])} × {e(i['cipher'])} — <em>{e(i['status'])}</em></h3>"
        f"<p><b>Prompt (plaintext):</b> <code>{e(i['plaintext_prompt'])}</code></p>"
        f"<p><b>Payload:</b> <code>{e(i['ciphertext_payload'])}</code></p>"
        f"<p><b>Decode instruction:</b> {e(i['decode_instruction'])}</p>"
        f"<p><b>Exact request sent to model:</b></p><pre>{e(i.get('full_request_sent', ''))}</pre>"
        f"<p><b>How to reproduce:</b></p><pre>{e(i.get('reproduction_steps', ''))}</pre>"
        f"<p><b>Judge:</b> {e(i['judge_rationale'])}</p>"
        f"<p><b>Repro:</b> seed={i['repro']['seed']}</p>"
        f"<pre>{e(json.dumps(i['metrics'], indent=1))}</pre></section>" for i in b["items"])
    p = b["provenance"]
    return redact(f"""<!doctype html><meta charset=utf-8><title>Disclosure — {e(p['target_model'])}</title>
<style>body{{font:15px/1.5 system-ui;max-width:860px;margin:2rem auto;padding:0 1rem}}code,pre{{background:#f3f3f8;padding:2px 4px;word-break:break-all}}</style>
<h1>Guardrail disclosure report — {e(p['target_model'])}</h1>
<p>Vendor: {e(p['vendor'])} · Operator: {e(p['operator'])} · Generated: {e(p['generated_at'])}<br>
<b>Hold until: {e(p['hold_until'])}</b> · Licence {e(p['license'])}</p>
<p>SHA-256 <code>{report['sha256']}</code><br>HMAC <code>{report['hmac_sha256']}</code></p>{rows}""")
