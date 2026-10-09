"""BYOK key manager (SPEC §6). Memory-only by default; OS keyring only on explicit opt-in.

Keys are NEVER written to either database, logs, exports or reports.
"""
from __future__ import annotations

import threading

_SERVICE = "redteam-testbench"


class KeyManager:
    def __init__(self) -> None:
        self._keys: dict[int, str] = {}
        self._lock = threading.Lock()

    def set(self, provider_id: int, key: str, remember: bool = False) -> None:
        with self._lock:
            self._keys[provider_id] = key
        if remember:
            try:
                import keyring
                keyring.set_password(_SERVICE, str(provider_id), key)
            except Exception:  # keyring backend may be missing — memory-only is the safe fallback
                pass

    def get(self, provider_id: int | None) -> str | None:
        if provider_id is None:
            return None
        with self._lock:
            if provider_id in self._keys:
                return self._keys[provider_id]
        try:
            import keyring
            return keyring.get_password(_SERVICE, str(provider_id))
        except Exception:
            return None

    def has(self, provider_id: int) -> bool:
        return self.get(provider_id) is not None

    def delete(self, provider_id: int) -> None:
        with self._lock:
            self._keys.pop(provider_id, None)
        try:
            import keyring
            keyring.delete_password(_SERVICE, str(provider_id))
        except Exception:
            pass


key_manager = KeyManager()
