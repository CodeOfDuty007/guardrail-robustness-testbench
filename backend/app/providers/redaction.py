"""Key-shaped pattern scrubbing (SPEC §6.4). Used by the log filter, exports and the leak test."""
from __future__ import annotations

import logging
import re

KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),            # OpenAI / Anthropic style
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),            # Google
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),               # Groq
    re.compile(r"hf_[A-Za-z0-9]{20,}"),                # HuggingFace
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|authorization)\"?\s*[:=]\s*\"?[A-Za-z0-9._\-]{16,}"),
]


def find_keys(text: str) -> list[str]:
    return [m.group(0) for p in KEY_PATTERNS for m in p.finditer(text)]


def redact(text: str) -> str:
    for p in KEY_PATTERNS:
        text = p.sub("[REDACTED]", text)
    return text


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(str(record.getMessage()))
        record.args = ()
        if record.exc_info:  # fold the traceback into msg so it is scrubbed too
            import traceback
            record.msg = redact(f"{record.msg}\n{''.join(traceback.format_exception(*record.exc_info))}")
            record.exc_info = None
        record.exc_text = None
        return True


def install_log_redaction() -> None:
    f = RedactionFilter()
    for name in ("", "uvicorn", "uvicorn.access", "uvicorn.error", "httpx"):
        logging.getLogger(name).addFilter(f)
    for h in logging.getLogger().handlers:
        h.addFilter(f)
