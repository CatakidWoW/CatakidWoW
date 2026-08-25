from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from quantum_earth.assurance.engine import QuantumAssurance
from quantum_earth.core.schemas import (
    EarthState,
    EnsembleMember,
    IntegrityClass,
    ProbabilisticForecast,
    Provenance,
)
from quantum_earth.data.open_meteo import OpenMeteoArchiveConnector, OpenMeteoForecastConnector
from quantum_earth.models.baselines import (
    ar1_forecast,
    climatology_forecast,
    hourly_climatology_buckets,
    persistence_forecast,
)
from quantum_earth.storage import StorageLayout

MULTI_MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]


def _safe_quantile(rows: np.ndarray, q: float) -> list[float | None]:
    out: list[float | None] = []
    for col in range(rows.shape[1]):
        col_vals = rows[:, col]
        col_vals = col_vals[~np.isnan(col_vals)]
        if col_vals.size == 0:
            out.append(None)
        else:
            out.append(float(np.quantile(col_vals, q)))
    return out


def _threshold_probs(rows: np.ndarray, thresholds: list[float]) -> dict[str, list[float | None]]:
    result: dict[str, list[float | None]] = {}
    for thr in thresholds:
        probs: list[float | None] = []
        for col in range(rows.shape[1]):
            col_vals = rows[:, col]
            col_vals = col_vals[~np.isnan(col_vals)]
            if col_vals.size == 0:
                probs.append(None)
            else:
                probs.append(float(np.mean(col_vals > thr)))
        result[f"P(>{thr})"] = probs
    return result


class ForecastEngine:
    """Multi-model + baseline ensemble forecasting for Stage 1 vertical slice."""

    def __init__(self) -> None:
        self.forecast_connector = OpenMeteoForecastConnector()
        self.archive_connector = OpenMeteoArchiveConnector()
        self.assurance = QuantumAssurance()
        self.storage = StorageLayout()

    def forecast(
        self,
        location_name: str,
        latitude: float,
        longitude: float,
        variable: str = "temperature_2m",
        hours: int = 48,
        earth_state: EarthState | None = None,
    ) -> ProbabilisticForecast:
        issued_at = datetime.now(timezone.utc)
        hours = max(1, min(int(hours), 168))
        unit = {
            "temperature_2m": "°C",
            "precipitation": "mm",
            "wind_speed_10m": "km/h",
        }.get(variable, "unknown")

        members: list[EnsembleMember] = []
        model_ids: list[str] = []
        provenance: list[Provenance] = []
        reasons: list[str] = []

        # --- Multi-model NWP via Open-Meteo ---
        try:
            raw = self.forecast_connector.fetch_hourly_forecast(
                latitude, longitude, hours=hours, models=MULTI_MODELS
            )
            hourly = raw.get("hourly") or {}
            times = [
                datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
                for t in hourly.get("time", [])[:hours]
            ]
            for model in MULTI_MODELS:
                key = f"{variable}_{model}"
                if key not in hourly:
                    # single-model response uses bare variable name when models omitted;
                    # with models=, Open-Meteo suffixes variable names.
                    continue
                vals = [float(v) if v is not None else float("nan") for v in hourly[key][:hours]]
                if len(vals) < hours:
                    vals.extend([float("nan")] * (hours - len(vals)))
                members.append(
                    EnsembleMember(
                        member_id=f"openmeteo:{model}",
                        model_id=f"openmeteo.{model}",
                        values=vals,
                        timestamps=times,
                    )
                )
                model_ids.append(f"openmeteo.{model}")
            provenance.append(
                Provenance(
                    source_id="open-meteo-forecast",
                    source_name="Open-Meteo multi-model",
                    licence="Open-Meteo / upstream NWP",
                    method="multi_model_ensemble",
                )
            )
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            times = [issued_at + timedelta(hours=i + 1) for i in range(hours)]
            reasons.append(f"open-meteo multimodel unavailable: {exc}")
            times = [issued_at + timedelta(hours=i) for i in range(hours)]

        if not members:
            times = [issued_at + timedelta(hours=i) for i in range(hours)]

        # --- Persistence baseline from EarthState or current observation ---
        last_val = None
        if earth_state and earth_state.cells:
            qty = earth_state.cells[0].variables.get(variable)
            if qty and qty.estimate is not None:
                last_val = float(qty.estimate)
        if last_val is None:
            try:
                current = self.forecast_connector.fetch_current(latitude, longitude)
                for obs in current:
                    if obs.quantity.variable == variable and obs.quantity.estimate is not None:
                        last_val = float(obs.quantity.estimate)
                        break
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"current obs unavailable for persistence: {exc}")

        if last_val is not None:
            means_p, _ = persistence_forecast(last_val, hours, base_unc=1.0)
            members.append(
                EnsembleMember(
                    member_id="baseline.persistence",
                    model_id="baseline.persistence",
                    values=means_p,
                    timestamps=times,
                )
            )
            model_ids.append("baseline.persistence")

        # --- Climatology from archive (prior-year seasonal window) ---
        try:
            # Use a ~21-day window around DOY from prior year for hourly buckets
            window_start = (issued_at - timedelta(days=370)).date()
            window_end = (issued_at - timedelta(days=350)).date()
            series = self.archive_connector.fetch_hourly_series(
                latitude,
                longitude,
                window_start.isoformat(),
                window_end.isoformat(),
            )
            buckets = hourly_climatology_buckets(
                series["timestamps"], series["variables"].get(variable, [])
            )
            clim_means, _ = climatology_forecast(buckets, times[0] if times else issued_at, hours)
            clim_vals = [float(v) if v is not None else float("nan") for v in clim_means]
            if any(not np.isnan(v) for v in clim_vals):
                members.append(
                    EnsembleMember(
                        member_id="baseline.climatology",
                        model_id="baseline.climatology",
                        values=clim_vals,
                        timestamps=times,
                    )
                )
                model_ids.append("baseline.climatology")
                provenance.append(
                    Provenance(
                        source_id="open-meteo-archive",
                        source_name="Open-Meteo Archive climatology",
                        licence="ERA5 / Open-Meteo",
                        method="hourly_doy_climatology",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"climatology unavailable: {exc}")

        # --- AR1 for continuous vars ---
        if variable in ("temperature_2m", "wind_speed_10m"):
            try:
                end = (issued_at - timedelta(days=5)).date()
                start = end - timedelta(days=14)
                series = self.archive_connector.fetch_hourly_series(
                    latitude, longitude, start.isoformat(), end.isoformat()
                )
                hist = [v for v in series["variables"].get(variable, []) if v is not None]
                if hist:
                    means_ar, _, _ = ar1_forecast(hist, hours)
                    members.append(
                        EnsembleMember(
                            member_id="baseline.ar1",
                            model_id="baseline.ar1",
                            values=means_ar,
                            timestamps=times,
                        )
                    )
                    model_ids.append("baseline.ar1")
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"ar1 unavailable: {exc}")

        if not members:
            # Explicit UNKNOWN — do not fabricate
            empty = ProbabilisticForecast(
                variable=variable,
                unit=unit,
                location_name=location_name,
                latitude=latitude,
                longitude=longitude,
                issued_at=issued_at,
                horizon_hours=hours,
                timestamps=times,
                mean=[None] * hours,
                median=[None] * hours,
                q10=[None] * hours,
                q90=[None] * hours,
                members=[],
                model_ids=[],
                provenance=provenance,
                assurance="RED",
                assurance_reasons=reasons + ["no ensemble members available"],
                integrity=IntegrityClass.UNKNOWN,
            )
            return empty

        # Align member lengths
        n = min(len(m.values) for m in members)
        n = min(n, len(times), hours)
        times = times[:n]
        for m in members:
            m.values = m.values[:n]
            m.timestamps = times

        rows = np.array([m.values for m in members], dtype=float)
        mean = _safe_quantile(rows, 0.5)  # use median-like; also compute mean
        mean = []
        for col in range(rows.shape[1]):
            col_vals = rows[:, col]
            col_vals = col_vals[~np.isnan(col_vals)]
            mean.append(float(np.mean(col_vals)) if col_vals.size else None)

        median = _safe_quantile(rows, 0.5)
        q10 = _safe_quantile(rows, 0.1)
        q90 = _safe_quantile(rows, 0.9)

        thresholds = [1.0, 10.0, 25.0] if variable == "precipitation" else []
        thr = _threshold_probs(rows, thresholds) if thresholds else {}

        # Model disagreement
        spreads = []
        for col in range(rows.shape[1]):
            col_vals = rows[:, col]
            col_vals = col_vals[~np.isnan(col_vals)]
            if col_vals.size:
                spreads.append(float(np.std(col_vals)))
        mean_spread = float(np.mean(spreads)) if spreads else None

        level, assurance_reasons = self.assurance.evaluate_forecast(
            n_members=len(members),
            mean_spread=mean_spread,
            missing_fraction=float(np.mean(np.isnan(rows))),
            extra_reasons=reasons,
            variable=variable,
        )

        forecast = ProbabilisticForecast(
            variable=variable,
            unit=unit,
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            issued_at=issued_at,
            horizon_hours=n,
            timestamps=times,
            mean=mean,
            median=median,
            q10=q10,
            q90=q90,
            members=members,
            threshold_probabilities=thr,
            model_ids=model_ids,
            provenance=provenance,
            assurance=level.value,
            assurance_reasons=assurance_reasons,
            integrity=IntegrityClass.FORECAST,
        )
        self.storage.write_json(
            "FORECAST",
            f"{forecast.id}.json",
            forecast.model_dump(mode="json"),
        )
        return forecast
