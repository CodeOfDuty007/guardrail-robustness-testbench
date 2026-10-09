"""Telemetry DB schema (SPEC §5a). API keys are never stored here."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Provider(SQLModel, table=True):
    __tablename__ = "providers"
    id: Optional[int] = Field(default=None, primary_key=True)
    label: str
    kind: str  # openai|anthropic|gemini|groq|ollama|vllm|azureopenai|openai_compat|mock
    base_url: Optional[str] = None
    model_name: str
    role_hint: str = "target"
    compute_node_id: Optional[int] = None
    cost_per_1k_prompt: float = 0.0      # USD, user-supplied pricing so budget caps work for any provider
    cost_per_1k_completion: float = 0.0
    created_at: datetime = Field(default_factory=utcnow)


class ComputeNode(SQLModel, table=True):
    """A remote (GPU) inference host the orchestrator can reach. No credentials stored."""
    __tablename__ = "compute_nodes"
    id: Optional[int] = Field(default=None, primary_key=True)
    label: str
    base_url: str
    kind: str = "ollama"  # ollama|vllm|openai_compat
    status: str = "unknown"  # unknown|online|offline
    gpu_info: Optional[str] = None
    models_json: str = "[]"
    latency_ms: Optional[float] = None
    last_seen: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)


class Run(SQLModel, table=True):
    __tablename__ = "runs"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=utcnow)
    config_json: str = "{}"
    attacker_provider_id: Optional[int] = None
    target_provider_id: Optional[int] = None
    judge_provider_id: Optional[int] = None
    dataset_name: str = "canary"
    seed: int = 1337
    status: str = "pending"  # pending|running|done|error
    max_iterations: int = 1
    budget_usd: Optional[float] = None
    spent_usd: float = 0.0
    owasp_tag: Optional[str] = None
    atlas_tag: Optional[str] = None
    error: Optional[str] = None


class Attempt(SQLModel, table=True):
    __tablename__ = "attempts"
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(index=True)
    iteration: int = 0
    objective_id: str
    strategy_name: str
    cipher_type: str
    defense_name: Optional[str] = None
    status: str = "running"  # running|success|failed|timeout
    created_at: datetime = Field(default_factory=utcnow)


class Payload(SQLModel, table=True):
    __tablename__ = "payloads"
    id: Optional[int] = Field(default=None, primary_key=True)
    attempt_id: int = Field(index=True)
    plaintext_prompt: str
    plaintext_size_bytes: int
    ciphertext_payload: str
    ciphertext_size_bytes: int
    decode_instruction: str = ""
    cipher_type: str
    cipher_key_id: Optional[str] = None
    encryption_time_ms: float = 0.0
    decryption_time_ms: float = 0.0
    avalanche_score: Optional[float] = None
    confusion_chi2: Optional[float] = None
    diffusion_score: Optional[float] = None
    shannon_entropy: Optional[float] = None
    # The exact message transmitted to the model (decode hint + cipher marker), and a human-readable
    # step-by-step guide for an independent researcher to replicate and resend this attack.
    full_request_sent: str = ""
    reproduction_steps: str = ""


class Response(SQLModel, table=True):
    __tablename__ = "responses"
    id: Optional[int] = Field(default=None, primary_key=True)
    attempt_id: int = Field(index=True)
    target_model: str
    target_response: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class Judgment(SQLModel, table=True):
    __tablename__ = "judgments"
    id: Optional[int] = Field(default=None, primary_key=True)
    attempt_id: int = Field(index=True)
    filter_evaded: bool
    semantic_complied: bool
    harm_score_0_1: float = 0.0
    judge_rationale: str = ""
    judge_model: str = ""
    classifier_label: Optional[str] = None
    judge_lm_verdict: Optional[bool] = None
    llamaguard_verdict: Optional[bool] = None
    regex_verdict: Optional[bool] = None
    score_source: str = "ensemble"


class DefenseApplied(SQLModel, table=True):
    __tablename__ = "defenses_applied"
    id: Optional[int] = Field(default=None, primary_key=True)
    attempt_id: int = Field(index=True)
    defense_name: str
    blocked: bool
    transformed_payload: Optional[str] = None


class Disclosure(SQLModel, table=True):
    """Audit log only — payload text lives in the corpus DB."""
    __tablename__ = "disclosures"
    id: Optional[int] = Field(default=None, primary_key=True)
    target_model: str
    vendor_name: str
    vendor_contact: str = ""
    report_hash: str
    corpus_item_ids_json: str
    item_count: int
    generated_at: datetime = Field(default_factory=utcnow)
    sent_at: Optional[datetime] = None
    hold_until: datetime
    notes: str = ""


TELEMETRY_TABLES = [
    "providers", "compute_nodes", "runs", "attempts", "payloads",
    "responses", "judgments", "defenses_applied", "disclosures",
]
