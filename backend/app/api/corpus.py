"""Corpus endpoints (SPEC §6b). The only API module allowed to open corpus.db."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from app.corpus import service
from app.db.corpus.models import SuccessfulAttack
from app.db.corpus.session import get_corpus_session
from app.providers.redaction import redact

router = APIRouter(prefix="/api/corpus", tags=["corpus"])


class PromoteIn(BaseModel):
    source_attempt_id: int
    tags: str = ""
    notes: str = ""
    curated_by: str | None = None


class PatchIn(BaseModel):
    tags: str | None = None
    notes: str | None = None
    status: str | None = None


class DisclosureIn(BaseModel):
    item_ids: list[str]
    vendor_name: str
    vendor_contact: str = ""
    hold_days: int | None = None
    notes: str = ""


@router.get("/models")
def models(cs: Session = Depends(get_corpus_session)):
    return service.by_model(cs)


@router.get("")
def list_items(target_model: str | None = None, status: str | None = None, strategy: str | None = None,
               cipher: str | None = None, tag: str | None = None, cs: Session = Depends(get_corpus_session)):
    q = select(SuccessfulAttack).order_by(SuccessfulAttack.curated_at.desc())
    if target_model: q = q.where(SuccessfulAttack.target_model == target_model)
    if status: q = q.where(SuccessfulAttack.status == status)
    if strategy: q = q.where(SuccessfulAttack.strategy_name == strategy)
    if cipher: q = q.where(SuccessfulAttack.cipher_type == cipher)
    items = cs.exec(q).all()
    if tag:
        items = [i for i in items if tag in i.tags]
    return items


@router.post("/promote")
def promote(body: PromoteIn, cs: Session = Depends(get_corpus_session)):
    try:
        return service.promote(cs, body.source_attempt_id, body.curated_by, body.tags, body.notes)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))


@router.patch("/{item_id}")
def patch(item_id: str, body: PatchIn, cs: Session = Depends(get_corpus_session)):
    it = cs.get(SuccessfulAttack, item_id)
    if not it:
        raise HTTPException(404)
    if body.status and body.status not in ("unverified", "working", "patched"):
        raise HTTPException(422, "bad status")
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(it, k, v)
    cs.add(it); cs.commit(); cs.refresh(it)
    return it


@router.delete("/{item_id}")
def delete(item_id: str, cs: Session = Depends(get_corpus_session)):
    it = cs.get(SuccessfulAttack, item_id)
    if it:
        cs.delete(it); cs.commit()
    return {"ok": True}


@router.post("/verify-all")
async def verify_all(target_model: str, judge_provider_id: int | None = None, cs: Session = Depends(get_corpus_session)):
    try:
        return await service.verify_all(cs, target_model, service.make_judge(judge_provider_id))
    except LookupError as e:
        raise HTTPException(404, str(e))


@router.post("/disclosure")
def disclosure(body: DisclosureIn, cs: Session = Depends(get_corpus_session)):
    try:
        return service.build_disclosure(cs, body.item_ids, body.vendor_name, body.vendor_contact,
                                        body.hold_days, body.notes)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.post("/disclosure/preview", response_class=HTMLResponse)
def disclosure_html(body: DisclosureIn, cs: Session = Depends(get_corpus_session)):
    try:
        r = service.build_disclosure(cs, body.item_ids, body.vendor_name, body.vendor_contact, body.hold_days, body.notes, persist=False)
    except (LookupError, ValueError) as e:
        raise HTTPException(422, str(e))
    return HTMLResponse(r["html"])


@router.post("/{item_id}/verify")
async def verify(item_id: str, judge_provider_id: int | None = None, cs: Session = Depends(get_corpus_session)):
    it = cs.get(SuccessfulAttack, item_id)
    if not it:
        raise HTTPException(404)
    try:
        return await service.verify(cs, it, service.make_judge(judge_provider_id))
    except (LookupError, ValueError) as e:
        raise HTTPException(409, str(e))
    except RuntimeError as e:  # ProviderError: target unreachable/billing/auth — corpus status untouched
        raise HTTPException(502, redact(str(e)))
