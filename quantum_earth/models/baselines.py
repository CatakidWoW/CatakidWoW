from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np

from quantum_earth.core.schemas import ModelCard, ModelStatus
from quantum_earth.storage import MetadataDB


def register_baseline_models(db: MetadataDB | None = None) -> list[ModelCard]:
    db = db or MetadataDB()
    cards = [
        ModelCard(
            model_id="baseline.persistence",
            version="1.0.0",
            target_variables=["temperature_2m", "precipitation", "wind_speed_10m"],
            input_variables=["temperature_2m", "precipitation", "wind_speed_10m"],
            spatial_resolution_m=25000.0,
            temporal_resolution_s=3600.0,
            forecast_horizon_hours=48,
            training_data="none (persistence)",
            training_period="n/a",
            validation_period="rolling",
            uncertainty_method="climatological_spread_scaled_by_lead",
            known_failure_modes=["fails in frontal passages", "diurnal cycle ignored"],
            compute_requirements="CPU negligible",
            status=ModelStatus.PRODUCTION,
            family="baseline",
        ),
        ModelCard(
            model_id="baseline.climatology",
            version="1.0.0",
            target_variables=["temperature_2m", "precipitation", "wind_speed_10m"],
            input_variables=["historical_hourly"],
            spatial_resolution_m=25000.0,
            temporal_resolution_s=3600.0,
            forecast_horizon_hours=48,
            training_data="open-meteo-archive same DOY hours",
            training_period="prior years same calendar window",
            validation_period="rolling-origin",
            uncertainty_method="interannual_hourly_std",
            known_failure_modes=["regime shifts", "climate trends not modelled"],
            compute_requirements="CPU low",
            status=ModelStatus.PRODUCTION,
            family="baseline",
        ),
        ModelCard(
            model_id="baseline.ar1",
            version="1.0.0",
            target_variables=["temperature_2m", "wind_speed_10m"],
            input_variables=["temperature_2m", "wind_speed_10m"],
            spatial_resolution_m=25000.0,
            temporal_resolution_s=3600.0,
            forecast_horizon_hours=48,
            training_data="recent local hourly series",
            training_period="last 14–30 days",
            validation_period="holdout last 48h",
            uncertainty_method="innovation_variance_growth",
            known_failure_modes=["non-stationary periods", "precipitation not suited"],
            compute_requirements="CPU low",
            status=ModelStatus.CHALLENGER,
            family="classical",
        ),
        ModelCard(
            model_id="openmeteo.multimodel",
            version="1.0.0",
            target_variables=["temperature_2m", "precipitation", "wind_speed_10m"],
            input_variables=["earth_state"],
            spatial_resolution_m=25000.0,
            temporal_resolution_s=3600.0,
            forecast_horizon_hours=168,
            training_data="upstream NWP centres via Open-Meteo",
            training_period="operational NWP",
            validation_period="continuous",
            uncertainty_method="multi_model_ensemble_spread",
            known_failure_modes=["shared upstream bias", "API outage"],
            compute_requirements="network + CPU",
            status=ModelStatus.PRODUCTION,
            family="physics_informed",
        ),
    ]
    for card in cards:
        db.upsert("models", "model_id", card.model_id, card.model_dump(mode="json"))
    return cards


class ModelRegistry:
    def __init__(self, db: MetadataDB | None = None) -> None:
        self.db = db or MetadataDB()
        register_baseline_models(self.db)

    def list_models(self) -> list[ModelCard]:
        return [ModelCard.model_validate(p) for p in self.db.list_all("models")]

    def get(self, model_id: str) -> ModelCard | None:
        payload = self.db.get("models", "model_id", model_id)
        return ModelCard.model_validate(payload) if payload else None

    def leaderboard(self, metric: str = "mae") -> list[ModelCard]:
        models = self.list_models()
        return sorted(models, key=lambda m: m.metrics.get(metric, float("inf")))


def persistence_forecast(last_value: float, hours: int, base_unc: float) -> tuple[list[float], list[float]]:
    means = [last_value] * hours
    # Uncertainty grows with lead time
    uncs = [base_unc * (1.0 + 0.05 * h) for h in range(1, hours + 1)]
    return means, uncs


def climatology_forecast(
    historical_by_hour: dict[int, Sequence[float]],
    start: datetime,
    hours: int,
) -> tuple[list[float | None], list[float | None]]:
    means: list[float | None] = []
    uncs: list[float | None] = []
    for h in range(hours):
        ts = start + timedelta(hours=h)
        samples = list(historical_by_hour.get(ts.hour, []))
        if len(samples) < 2:
            means.append(None)
            uncs.append(None)
            continue
        arr = np.asarray(samples, dtype=float)
        means.append(float(np.mean(arr)))
        uncs.append(float(np.std(arr)))
    return means, uncs


def ar1_forecast(
    series: Sequence[float],
    hours: int,
    phi: float | None = None,
) -> tuple[list[float], list[float], float]:
    arr = np.asarray(series, dtype=float)
    if len(arr) < 5:
        last = float(arr[-1]) if len(arr) else 0.0
        return [last] * hours, [1.0] * hours, 0.0
    x0 = arr[:-1]
    x1 = arr[1:]
    phi_hat = float(np.dot(x0, x1) / max(np.dot(x0, x0), 1e-9)) if phi is None else phi
    phi_hat = float(np.clip(phi_hat, -0.99, 0.99))
    resid = x1 - phi_hat * x0
    sigma = float(np.std(resid)) if len(resid) else 1.0
    mean_level = float(np.mean(arr))
    # mean-reverting AR1 around sample mean
    level = float(arr[-1])
    means: list[float] = []
    uncs: list[float] = []
    for lead in range(1, hours + 1):
        level = mean_level + phi_hat * (level - mean_level)
        means.append(level)
        # analytic variance growth for AR1
        var = sigma**2 * (1 - phi_hat ** (2 * lead)) / max(1 - phi_hat**2, 1e-9)
        uncs.append(float(np.sqrt(max(var, 1e-9))))
    return means, uncs, phi_hat


def hourly_climatology_buckets(
    timestamps: Sequence[datetime],
    values: Sequence[float | None],
) -> dict[int, list[float]]:
    buckets: dict[int, list[float]] = {h: [] for h in range(24)}
    for ts, val in zip(timestamps, values):
        if val is None:
            continue
        buckets[ts.hour].append(float(val))
    return buckets
