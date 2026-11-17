"""CORS for a website hosted on another origin (CORS_ORIGINS)."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import configure_cors


def _app(origins):
    a = FastAPI()

    @a.post("/api/ask")
    def ask():
        return {"ok": True}

    configure_cors(a, origins)
    return a


def test_allowed_origin_gets_cors_headers():
    c = TestClient(_app("https://web.example.com, https://other.example.com/"))
    r = c.options("/api/ask", headers={"Origin": "https://web.example.com", "Access-Control-Request-Method": "POST"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "https://web.example.com"


def test_other_origins_and_default_get_none():
    c = TestClient(_app("https://web.example.com"))
    r = c.post("/api/ask", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in r.headers
    assert configure_cors(FastAPI(), "") == []
