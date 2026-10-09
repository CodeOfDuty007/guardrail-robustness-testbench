"""Settings. Paths are resolved relative to the repo root so `uvicorn` can run from anywhere."""
from __future__ import annotations

import getpass
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RT_", env_file=".env", extra="ignore")

    telemetry_url: str = f"sqlite:///{ROOT / 'telemetry.db'}"
    corpus_dir: Path = ROOT / "corpus"
    datasets_dir: Path = ROOT / "datasets"
    operator_name: str = getpass.getuser()
    default_seed: int = 1337
    target_timeout_s: float = 30.0
    max_concurrency: int = 4
    hold_days: int = 90
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @property
    def corpus_url(self) -> str:
        return f"sqlite:///{self.corpus_dir / 'corpus.db'}"


settings = Settings()
