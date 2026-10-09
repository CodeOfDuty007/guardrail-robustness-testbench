"""PyRIT availability/status (SPEC §3a). The executors themselves live in pyrit_native.py.

Both modules sit inside engine/ — the only place `pyrit` may be imported.
"""
from __future__ import annotations

try:
    import pyrit

    PYRIT_AVAILABLE = True
    PYRIT_VERSION = getattr(pyrit, "__version__", "unknown")
except Exception:  # pragma: no cover - optional at import time, required for engine=pyrit
    PYRIT_AVAILABLE = False
    PYRIT_VERSION = None


def status() -> dict:
    return {"available": PYRIT_AVAILABLE, "version": PYRIT_VERSION,
            "strategies": ["DirectCipherWrap", "PersonaRoleplay", "InstructionOverride", "ManyShot", "PAIR", "TAP", "Crescendo"],
            "note": "engine=pyrit drives PromptSending/Crescendo/TAP/PAIR via PyRIT 1.x; engine=native uses the built-in loop."}
