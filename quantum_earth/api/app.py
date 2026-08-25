from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from quantum_earth import __version__
from quantum_earth.config import resolve_location
from quantum_earth.core.jsonutil import json_safe
from quantum_earth.data.registry import DataSourceRegistry
from quantum_earth.models.baselines import ModelRegistry
from quantum_earth.orchestration.loop import OperatingLoop

app = FastAPI(
    title="QUANTUM EARTH",
    description="Autonomous Earth-System Prediction Intelligence Platform",
    version=__version__,
)

loop = OperatingLoop()
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    index = DASHBOARD_DIR / "index.html"
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.get("/api/current")
def current(location: str = Query("Birmingham")):
    try:
        return json_safe(loop.ingest_and_state(location))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/forecast")
def forecast(
    location: str = Query("Birmingham"),
    variable: str = Query("temperature_2m"),
    hours: int = Query(48, ge=1, le=168),
):
    try:
        return json_safe(loop.forecast(location, variable=variable, hours=hours))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/ensemble")
def ensemble(
    location: str = Query("Birmingham"),
    variable: str = Query("temperature_2m"),
    hours: int = Query(48, ge=1, le=168),
):
    fc = forecast(location=location, variable=variable, hours=hours)
    return json_safe(
        {
            "location": fc["location_name"],
            "variable": fc["variable"],
            "assurance": fc["assurance"],
            "members": fc["members"],
            "mean": fc["mean"],
            "q10": fc["q10"],
            "q90": fc["q90"],
            "threshold_probabilities": fc["threshold_probabilities"],
            "integrity": fc["integrity"],
        }
    )


@app.get("/api/earth-state")
def earth_state(location: str = Query("Birmingham")):
    return loop.ingest_and_state(location)["earth_state"]


@app.get("/api/models")
def models():
    reg = ModelRegistry()
    return [m.model_dump(mode="json") for m in reg.list_models()]


@app.get("/api/models/leaderboard")
def leaderboard():
    reg = ModelRegistry()
    return [m.model_dump(mode="json") for m in reg.leaderboard("mae")]


@app.get("/api/sources")
def sources():
    reg = DataSourceRegistry()
    return [s.model_dump(mode="json") for s in reg.list_sources()]


@app.get("/api/verification")
def verification(
    location: str = Query("Birmingham"),
    variable: str = Query("temperature_2m"),
    hours: int = Query(24, ge=6, le=168),
):
    return json_safe(loop.verify_recent(location, variable=variable, hours=hours))


@app.get("/api/system-health")
def system_health():
    return loop.status()


@app.get("/api/events")
def events():
    return {
        "integrity": "UNKNOWN",
        "status": "BLOCKED_EXTERNAL_DEPENDENCY",
        "message": "Event engine scaffolding deferred; Stage 1 vertical slice focuses on continuous variables.",
        "events": [],
    }


@app.get("/api/phenology")
def phenology():
    return {
        "integrity": "UNKNOWN",
        "status": "BLOCKED_EXTERNAL_DEPENDENCY",
        "message": "Phenology engine not yet built. See Stage 6 plan.",
    }


@app.get("/api/extremes")
def extremes():
    return {
        "integrity": "UNKNOWN",
        "status": "NOT_IMPLEMENTED",
        "message": "Extreme-value engine planned; not claimed operational.",
    }


@app.get("/api/season")
def season():
    return {
        "integrity": "UNKNOWN",
        "status": "NOT_IMPLEMENTED",
        "message": "Season engine planned; seasons will be threshold-based, not calendar-fixed.",
    }


@app.get("/api/hazards")
def hazards():
    return {
        "integrity": "UNKNOWN",
        "status": "NOT_IMPLEMENTED",
        "message": "Hazard/impact engines deferred. No deterministic geohazard claims.",
    }


@app.get("/api/resolve")
def resolve(location: str = Query(...)):
    name, lat, lon = resolve_location(location)
    return {"name": name, "latitude": lat, "longitude": lon}


if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR / "static")), name="static")
