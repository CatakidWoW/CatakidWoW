from __future__ import annotations

import pytest

from quantum_earth.assurance.engine import QuantumAssurance
from quantum_earth.core.schemas import AssuranceLevel
from quantum_earth.data.open_meteo import OpenMeteoArchiveConnector, OpenMeteoForecastConnector
from quantum_earth.forecast.engine import ForecastEngine
from quantum_earth.orchestration.loop import OperatingLoop


@pytest.mark.integration
def test_open_meteo_current_real():
    conn = OpenMeteoForecastConnector()
    obs = conn.fetch_current(52.4862, -1.8904)
    assert len(obs) >= 1
    assert all(o.quantity.estimate is not None for o in obs)
    assert all(o.quantity.provenance.source_id == "open-meteo-forecast" for o in obs)


@pytest.mark.integration
def test_open_meteo_archive_real():
    conn = OpenMeteoArchiveConnector()
    series = conn.fetch_hourly_series(52.4862, -1.8904, "2024-08-01", "2024-08-03")
    assert len(series["timestamps"]) > 24
    assert "temperature_2m" in series["variables"]


@pytest.mark.integration
def test_forecast_ensemble_real():
    eng = ForecastEngine()
    fc = eng.forecast("Birmingham", 52.4862, -1.8904, variable="temperature_2m", hours=24)
    assert fc.integrity.value == "FORECAST"
    assert len(fc.members) >= 1
    assert fc.assurance in ("GREEN", "YELLOW", "RED")
    assert any(v is not None for v in fc.mean)


@pytest.mark.integration
def test_operating_loop_verify_real():
    loop = OperatingLoop()
    report = loop.verify_recent("Birmingham", variable="temperature_2m", hours=24)
    assert report["status"] == "OK"
    assert report["scores"]["persistence"]["mae"] >= 0
    assert report["integrity"] == "OBSERVED"


def test_assurance_red_without_members():
    level, reasons = QuantumAssurance().evaluate_forecast(
        n_members=0, mean_spread=None, missing_fraction=1.0
    )
    assert level == AssuranceLevel.RED
    assert reasons
