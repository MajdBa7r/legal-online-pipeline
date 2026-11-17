"""Startup and request behaviour when the configured name model is (un)available."""
import functools
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import Settings
from app.pii.factory import build_masker
from app.pii.masker import PIIMasker
from app.pii.name_model import NameModelUnavailable, load_name_model

FIX = Path(__file__).parent / "fixtures"


class OkModel:
    name = "ok-model"

    def detect_names(self, text):
        return []


def ok_loader(kind, **kw):
    return OkModel()


def failing_loader(kind, **kw):
    raise NameModelUnavailable("weights missing")


def test_none_is_rules_only():
    m, st = build_masker(Settings(pii_name_model="none", app_env="production"))
    assert m.name_model is None and st.serving and not st.degraded and st.label() == "none"


def test_camelbert_without_ml_libraries_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(NameModelUnavailable):
        load_name_model("camelbert-da", model_id="x", min_score=0.0)


def test_development_falls_back_to_rules_with_warning(caplog):
    m, st = build_masker(Settings(pii_name_model="camelbert-da", app_env="development"), loader=failing_loader)
    assert m.name_model is None and st.serving and st.degraded
    assert "unavailable, rules only" in st.label()
    assert "RULES ONLY" in caplog.text


def test_production_blocks_requests(caplog):
    m, st = build_masker(Settings(pii_name_model="camelbert-da", app_env="production"), loader=failing_loader)
    assert not st.serving and st.degraded and "requests blocked" in st.label()
    assert "BLOCKED" in caplog.text


def test_real_loader_failure_takes_the_failure_path(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)   # as on a machine without requirements-ml.txt
    _, dev = build_masker(Settings(pii_name_model="camelbert-da", app_env="development"))
    _, prod = build_masker(Settings(pii_name_model="camelbert-da", app_env="production"))
    assert dev.serving and dev.degraded and not prod.serving


def test_loaded_model_policy_follows_environment():
    m, st = build_masker(Settings(pii_name_model="camelbert-da", app_env="development"), loader=ok_loader)
    assert st.active == "rules+camelbert-da" and m.model_error_policy == "fallback"
    m, st = build_masker(Settings(pii_name_model="camelbert-da", app_env="production"), loader=ok_loader)
    assert m.model_error_policy == "fail_closed" and st.serving and not st.degraded


@pytest.mark.parametrize("bad", [{"app_env": "staging"}, {"pii_name_model": "gliner"}])
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        build_masker(Settings(**bad))


# ---- the running app ---------------------------------------------------------
def _settings(tmp_path, **kw):
    return Settings(local_data_dir=str(FIX / "local_laws"), local_traces_path=str(tmp_path / "t.jsonl"),
                    local_embeddings_dir=str(tmp_path / "none"), **kw)


def _app_loader(monkeypatch, loader):
    """Make app startup use `loader`, so the result never depends on what is installed or downloaded."""
    monkeypatch.setattr(main, "build_masker", functools.partial(build_masker, loader=loader))


def test_app_production_returns_503_and_never_runs_the_pipeline(tmp_path, monkeypatch):
    _app_loader(monkeypatch, failing_loader)
    monkeypatch.setattr(main, "settings", _settings(tmp_path, app_env="production", pii_name_model="camelbert-da"))
    with TestClient(main.app) as c:
        called = []
        monkeypatch.setattr(main.app.state.pipeline, "handle", lambda *a, **k: called.append(1))
        r = c.post("/api/ask", json={"question": "اسمي براء وبدي أعرف حقوقي"})
        assert r.status_code == 503 and "camelbert-da" in r.json()["pii_name_model"]
        assert called == []
        h = c.get("/api/health").json()
        assert h["status"] == "unavailable" and "requests blocked" in h["pii_name_model"]


def test_app_development_serves_with_rules_only(tmp_path, monkeypatch):
    _app_loader(monkeypatch, failing_loader)
    monkeypatch.setattr(main, "settings", _settings(tmp_path, app_env="development", pii_name_model="camelbert-da"))
    with TestClient(main.app) as c:
        assert c.post("/api/ask", json={"question": "سؤال عن الإجازة"}).status_code == 200
        h = c.get("/api/health").json()
        assert h["status"] == "degraded" and "rules only" in h["pii_name_model"]


def test_app_with_a_loaded_model_serves_normally(tmp_path, monkeypatch):
    _app_loader(monkeypatch, ok_loader)
    monkeypatch.setattr(main, "settings", _settings(tmp_path, app_env="production", pii_name_model="camelbert-da"))
    with TestClient(main.app) as c:
        assert c.post("/api/ask", json={"question": "سؤال عن الإجازة"}).status_code == 200
        h = c.get("/api/health").json()
        assert h["status"] != "unavailable" and h["pii_name_model"] == "camelbert-da"
        assert main.app.state.pipeline.masker.name_model is not None


def test_app_default_reports_none(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "settings", _settings(tmp_path))
    with TestClient(main.app) as c:
        assert c.get("/api/health").json()["pii_name_model"] == "none"
        assert isinstance(main.app.state.pipeline.masker, PIIMasker)
        assert main.app.state.pipeline.masker.name_model is None
