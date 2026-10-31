"""Key-shaped pattern scrubbing (SPEC §6.4). Used by the log filter, exports and the leak test."""
from __future__ import annotations

import logging
import re

KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),            # OpenAI / Anthropic style
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),            # Google
    re.compile(r"AQ\.[0-9A-Za-z_\-]{30,}"),          # Google (newer AQ. key format)
    re.compile(r"(?i)[?&](key|api_key|token)=[^&\s\"']+"),  # key smuggled in a URL query string
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


def _safe_str(o: object) -> str:
    try:
        return str(o)
    except Exception:  # a hostile/broken __str__ must not break logging or escape scrubbing
        return "<unprintable>"


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered, malformed = record.getMessage(), False
        except Exception:  # malformed template/args: scrub a best-effort rendering, never raise
            # args cannot be paired with their template, so a secret's context may be split from its value:
            # fail closed and withhold the args entirely rather than guess
            rendered, malformed = f"{_safe_str(record.msg)} [args withheld: malformed log record]", True
        clean = redact(rendered)
        # uvicorn.access's formatter unpacks exactly five positional args, so collapsing them would drop every line
        if (record.name == "uvicorn.access" and isinstance(record.args, tuple) and len(record.args) == 5
                and not malformed and not record.exc_info):
            if clean == rendered:
                return True
            # scrub each arg (the request path carries any ?key=); keep the shape only if the result verifies clean
            new_args = tuple(a if isinstance(a, (int, float)) else redact(_safe_str(a)) for a in record.args)
            try:
                out = _safe_str(record.msg) % new_args
            except Exception:
                out = None
            if out is not None and redact(out) == out:
                record.args, record.exc_text = new_args, None
                return True
        record.msg = clean  # render first, then scrub: a key's context may sit in the template, not the arg
        record.args = ()
        if record.exc_info:  # fold the traceback into msg so it is scrubbed too
            import traceback
            try:
                tb = "".join(traceback.format_exception(*record.exc_info))
            except Exception:
                tb = "<traceback unavailable>"
            record.msg = redact(f"{record.msg}\n{tb}")
            record.exc_info = None
        record.exc_text = None
        return True


def install_log_redaction() -> None:
    f = RedactionFilter()
    for name in ("", "uvicorn", "uvicorn.access", "uvicorn.error", "httpx"):
        logging.getLogger(name).addFilter(f)
    for h in logging.getLogger().handlers:
        h.addFilter(f)
