from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from quantum_earth.core.schemas import (
    IntegrityClass,
    Observation,
    Provenance,
    QualityScore,
    SpatiotemporalRef,
    ValuedQuantity,
)
from quantum_earth.data.validation import ObservationValidator
from quantum_earth.models.baselines import ar1_forecast, persistence_forecast
from quantum_earth.state.engine import EarthStateEngine
from quantum_earth.verification.metrics import bias, crps_ensemble, mae, rmse


def _obs(variable: str, value: float, ts: datetime | None = None) -> Observation:
    ts = ts or datetime(2024, 8, 1, 12, tzinfo=timezone.utc)
    return Observation(
        quantity=ValuedQuantity(
            variable=variable,
            unit="°C" if "temp" in variable else "mm",
            estimate=value,
            uncertainty=0.5,
            integrity=IntegrityClass.OBSERVED,
            when=SpatiotemporalRef(latitude=52.5, longitude=-1.9, timestamp=ts),
            provenance=Provenance(
                source_id="test",
                source_name="test",
                licence="test",
                method="unit",
            ),
            quality=QualityScore(score=0.9),
        )
    )


def test_observation_never_stores_forecast_integrity():
    obs = _obs("temperature_2m", 18.0)
    assert obs.quantity.integrity == IntegrityClass.OBSERVED


def test_validator_rejects_corruption_and_duplicates():
    v = ObservationValidator()
    good = [_obs("temperature_2m", 18.0), _obs("precipitation", 1.2)]
    ok, errs = v.validate_batch(good)
    assert len(ok) == 2 and not errs

    corrupt = v.inject_corruption(good, "nan_magnitude")
    ok2, errs2 = v.validate_batch(corrupt)
    assert any("corrupt" in e or "range" in e for e in errs2)

    neg = v.inject_corruption(good, "negative_precip")
    ok3, errs3 = v.validate_batch(neg)
    assert any("negative precipitation" in e for e in errs3)

    dup = v.inject_corruption(good, "duplicate")
    ok4, errs4 = v.validate_batch(dup)
    assert any("duplicate" in e for e in errs4)


def test_earth_state_is_inferred_not_observed():
    engine = EarthStateEngine()
    state = engine.assimilate_point([_obs("temperature_2m", 10.0), _obs("temperature_2m", 12.0)])
    assert state.integrity == IntegrityClass.INFERRED
    cell = state.cells[0]
    assert cell.variables["temperature_2m"].integrity == IntegrityClass.INFERRED
    assert cell.variables["temperature_2m"].estimate == pytest.approx(11.0)


def test_empty_earth_state_is_unknown():
    state = EarthStateEngine().assimilate_point([])
    assert state.integrity == IntegrityClass.UNKNOWN
    assert state.quality.score == 0.0


def test_persistence_and_ar1():
    means, uncs = persistence_forecast(20.0, 5, 1.0)
    assert means == [20.0] * 5
    assert uncs[-1] > uncs[0]
    series = [10, 11, 12, 11, 10, 9, 10, 11]
    m, u, phi = ar1_forecast(series, 6)
    assert len(m) == 6 and len(u) == 6
    assert -1 < phi < 1


def test_verification_metrics():
    truth = [1.0, 2.0, 3.0, 4.0]
    pred = [1.5, 2.5, 2.5, 3.5]
    assert mae(pred, truth) == pytest.approx(0.5)
    assert rmse(pred, truth) == pytest.approx(np.sqrt(0.25))
    assert bias(pred, truth) == pytest.approx(0.0)
    members = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]], dtype=float)
    score = crps_ensemble(members, truth)
    assert score >= 0


def test_data_source_registry_seeded():
    from quantum_earth.data.registry import DataSourceRegistry
    from quantum_earth.data.base import ConnectorStatus

    reg = DataSourceRegistry()
    sources = reg.list_sources()
    assert any(s.source_id == "open-meteo-forecast" for s in sources)
    blocked = [s for s in sources if s.status == ConnectorStatus.BLOCKED_EXTERNAL_DEPENDENCY]
    assert len(blocked) >= 1
    # Every source has required metadata fields
    for s in sources:
        assert s.licence and s.coverage and s.resolution and s.update_frequency


def test_model_registry_statuses():
    from quantum_earth.models.baselines import ModelRegistry
    from quantum_earth.core.schemas import ModelStatus

    reg = ModelRegistry()
    models = {m.model_id: m for m in reg.list_models()}
    assert models["baseline.persistence"].status == ModelStatus.PRODUCTION
    assert models["baseline.ar1"].status == ModelStatus.CHALLENGER
