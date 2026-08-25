from __future__ import annotations

from fastapi.testclient import TestClient

from quantum_earth.api.app import app

client = TestClient(app)


def test_dashboard_serves():
    res = client.get("/")
    assert res.status_code == 200
    assert "QUANTUM" in res.text


def test_sources_and_models_endpoints():
    sources = client.get("/api/sources")
    assert sources.status_code == 200
    assert any(s["source_id"] == "open-meteo-forecast" for s in sources.json())
    models = client.get("/api/models")
    assert models.status_code == 200
    assert len(models.json()) >= 3


def test_unimplemented_endpoints_are_explicit():
    for path in ("/api/events", "/api/phenology", "/api/extremes", "/api/season", "/api/hazards"):
        res = client.get(path)
        assert res.status_code == 200
        body = res.json()
        assert body["integrity"] == "UNKNOWN"
