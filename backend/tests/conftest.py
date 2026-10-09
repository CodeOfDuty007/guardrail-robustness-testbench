import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.db.corpus.session import init_corpus, reset_corpus_engine
from app.db.telemetry.session import init_telemetry, reset_engine


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "telemetry_url", f"sqlite:///{tmp_path / 't.db'}")
    monkeypatch.setattr(settings, "corpus_dir", tmp_path / "corpus")
    monkeypatch.setattr(settings, "datasets_dir", tmp_path / "datasets")
    reset_engine(); reset_corpus_engine()
    init_telemetry(); init_corpus()
    yield tmp_path
    reset_engine(); reset_corpus_engine()


@pytest.fixture()
def client(env):
    from app.main import app
    with TestClient(app) as c:
        yield c
