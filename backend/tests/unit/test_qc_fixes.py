"""Regression tests for the system-wide QC pass (redaction patterns, access-log shape, BYO system prompt, export redaction)."""
from __future__ import annotations

import json
import logging

from app.api.core import REDACT_COLS
from app.datasets import loaders
from app.providers.redaction import RedactionFilter, find_keys, redact


def test_redacts_google_aq_key_and_url_query_key():
    fake = "AQ." + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
    assert find_keys(f"token={fake}")
    assert fake not in redact(f"key {fake}")
    assert "SECRET123" not in redact("https://x.example/v1?key=SECRET123&alt=json")


def test_redaction_filter_keeps_positional_args_for_access_log():
    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                            ("127.0.0.1:1", "GET", "/api/x?key=SECRET123", "1.1", 200), None)
    assert RedactionFilter().filter(rec)
    assert len(rec.args) == 5  # uvicorn's formatter unpacks exactly five values
    assert "SECRET123" not in rec.getMessage()


def test_redaction_filter_scrubs_non_str_url_args():
    class Url:  # stands in for httpx.URL, which httpx passes as a non-str log arg
        def __str__(self) -> str:
            return "https://generativelanguage.example/v1/chat?key=SECRET123"

    rec = logging.LogRecord("httpx", logging.INFO, "", 0, 'HTTP Request: %s %s "%d"', ("POST", Url(), 200), None)
    assert RedactionFilter().filter(rec)
    assert "SECRET123" not in rec.getMessage()


def test_redaction_filter_scrubs_secret_when_context_is_in_template():
    rec = logging.LogRecord("app", logging.INFO, "", 0, "calling https://x.example/v1?key=%s now", ("SECRET123",), None)
    assert RedactionFilter().filter(rec)
    assert "SECRET123" not in rec.getMessage()


def test_redaction_filter_survives_percent_in_traceback_and_bad_args():
    try:
        raise ValueError("100% bad %s %d")
    except ValueError:
        import sys
        rec = logging.LogRecord("app", logging.ERROR, "", 0, "failed %s", ("x",), sys.exc_info())
    assert RedactionFilter().filter(rec)
    assert "100% bad" in rec.getMessage()
    bad = logging.LogRecord("app", logging.INFO, "", 0, "needs two %s %s", ("only-one",), None)
    assert RedactionFilter().filter(bad)
    bad.getMessage()  # must not raise
    twice = RedactionFilter()
    twice.filter(rec)  # attached to several loggers/handlers: must be idempotent
    assert "100% bad" in rec.getMessage()


def test_redaction_filter_fails_closed_on_malformed_record_with_secret():
    class Boom:
        def __str__(self) -> str:
            raise RuntimeError("no str")

    secret = "AQ." + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
    bad = logging.LogRecord("app", logging.INFO, "", 0, f"leaky {secret} %s %s", ("only-one",), None)
    assert RedactionFilter().filter(bad)
    assert secret not in bad.getMessage()
    split = logging.LogRecord("app", logging.INFO, "", 0, "url ?key=%s extra %s", ("SECRET123",), None)
    assert RedactionFilter().filter(split)
    assert "SECRET123" not in split.getMessage()  # context in template, value in args, template/args mismatched
    hostile = logging.LogRecord("app", logging.INFO, "", 0, "x %s", (Boom(),), None)
    assert RedactionFilter().filter(hostile)
    hostile.getMessage()  # must not raise
    access = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                               ("h", "GET", f"/x?key=SECRET123&t={secret}", "1.1", 200), None)
    assert RedactionFilter().filter(access)
    assert "SECRET123" not in access.getMessage() and secret not in access.getMessage()


def test_byo_objectives_carry_system_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders.settings, "datasets_dir", tmp_path)
    name = next(iter(loaders._FILES))
    (tmp_path / f"{loaders._FILES[name]}.json").write_text(json.dumps([{"prompt": "hello", "id": "1"}]))
    objs = loaders.load_dataset(name, 1337)
    assert objs and all(o.system_prompt == "You are a helpful, harmless assistant." and o.canary_literal is None for o in objs)


def test_export_redaction_covers_full_request_and_repro_steps():
    assert {"full_request_sent", "reproduction_steps"} <= set(REDACT_COLS["payloads"])
