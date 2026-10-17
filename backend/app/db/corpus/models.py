"""Corpus DB schema (SPEC §5b). Own registry/metadata so tables can never land in telemetry.db."""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import JSON, Column
from sqlalchemy.orm import registry
from sqlmodel import Field, SQLModel

corpus_registry = registry()


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class CorpusModel(SQLModel, registry=corpus_registry):
    pass


class SuccessfulAttack(CorpusModel, table=True):
    __tablename__ = "successful_attacks"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    source_run_id: int
    source_attempt_id: int
    target_model: str = Field(index=True)  # PARTITION KEY
    target_provider: str = ""
    model_snapshot_date: date = Field(default_factory=lambda: _now().date())
    objective_id: str = ""
    strategy_name: str = ""
    cipher_type: str = ""
    plaintext_prompt: str = ""
    ciphertext_payload: str = ""
    decode_instruction: str = ""
    filter_evaded: bool = True
    semantic_complied: bool = True
    harm_score_0_1: float = 0.0
    judge_rationale: str = ""
    classifier_label: Optional[str] = None
    # crypto/timing metrics materialised so vendor reports render offline
    metrics_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    seed: int = 0
    repro_config_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    # Exact message that was sent to the target (decode hint + cipher marker).
    full_request_sent: str = ""
    # Step-by-step guide for an independent researcher to reconstruct and resend this attack.
    reproduction_steps: str = ""
    status: str = "unverified"  # unverified | working | patched
    first_seen_at: datetime = Field(default_factory=_now)
    last_verified_at: Optional[datetime] = None
    curated_by: str = ""
    curated_at: datetime = Field(default_factory=_now)
    tags: str = ""
    notes: str = ""
    disclosure_id: Optional[str] = None


CORPUS_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_corpus_model_status ON successful_attacks(target_model, status)",
    "CREATE INDEX IF NOT EXISTS idx_corpus_strategy ON successful_attacks(strategy_name)",
    "CREATE INDEX IF NOT EXISTS idx_corpus_cipher ON successful_attacks(cipher_type)",
    "DROP VIEW IF EXISTS known_jailbreaks_by_model",
    """CREATE VIEW known_jailbreaks_by_model AS
SELECT
    target_model,
    target_provider,
    COUNT(*)                                          AS total_items,
    SUM(CASE WHEN status = 'working'    THEN 1 ELSE 0 END) AS working,
    SUM(CASE WHEN status = 'patched'    THEN 1 ELSE 0 END) AS patched,
    SUM(CASE WHEN status = 'unverified' THEN 1 ELSE 0 END) AS unverified,
    MAX(last_verified_at)                             AS last_verified,
    GROUP_CONCAT(DISTINCT strategy_name)              AS strategies_that_work,
    GROUP_CONCAT(DISTINCT cipher_type)                AS ciphers_that_work
FROM successful_attacks
GROUP BY target_model, target_provider
ORDER BY working DESC""",
]
