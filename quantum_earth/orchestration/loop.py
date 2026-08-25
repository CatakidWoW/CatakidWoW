from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np

from quantum_earth.assurance.engine import QuantumAssurance
from quantum_earth.config import resolve_location
from quantum_earth.core.schemas import HealthState, IntegrityClass
from quantum_earth.data.open_meteo import OpenMeteoArchiveConnector, OpenMeteoForecastConnector
from quantum_earth.data.registry import DataSourceRegistry
from quantum_earth.data.validation import ObservationValidator
from quantum_earth.forecast.engine import ForecastEngine
from quantum_earth.health import HealthMonitor
from quantum_earth.models.baselines import ModelRegistry
from quantum_earth.state.engine import EarthStateEngine
from quantum_earth.storage import MetadataDB, StorageLayout
from quantum_earth.verification.metrics import VerificationEngine


class OperatingLoop:
    """Stage-1 autonomous operating loop skeleton for the vertical slice.

    DISCOVER → INGEST → VALIDATE → ASSIMILATE → UPDATE EARTH STATE
    → FORECAST → ENSEMBLE → ASSURANCE → PUBLISH → VERIFY (when truth available)
    """

    def __init__(self) -> None:
        self.db = MetadataDB()
        self.storage = StorageLayout()
        self.sources = DataSourceRegistry(self.db)
        self.models = ModelRegistry(self.db)
        self.validator = ObservationValidator()
        self.state_engine = EarthStateEngine()
        self.forecast_engine = ForecastEngine()
        self.verification = VerificationEngine(self.db)
        self.health = HealthMonitor(self.db)
        self.assurance = QuantumAssurance()
        self.forecast_connector = OpenMeteoForecastConnector()
        self.archive_connector = OpenMeteoArchiveConnector()

    def discover(self) -> dict:
        operational = self.sources.operational()
        blocked = [s for s in self.sources.list_sources() if s.status.value == "BLOCKED_EXTERNAL_DEPENDENCY"]
        return {
            "operational_sources": [s.source_id for s in operational],
            "blocked_external": [s.source_id for s in blocked],
            "models": [m.model_id for m in self.models.list_models()],
        }

    def ingest_and_state(self, location: str = "Birmingham") -> dict:
        name, lat, lon = resolve_location(location)
        t0 = time.perf_counter()
        try:
            obs = self.forecast_connector.fetch_current(lat, lon)
            good, errors = self.validator.validate_batch(obs)
            state = self.state_engine.assimilate_point(good)
            latency = (time.perf_counter() - t0) * 1000
            self.health.report("ingest.open-meteo", HealthState.HEALTHY, latency_ms=latency)
            self.health.report("earth-state", HealthState.HEALTHY, latency_ms=latency)
            return {
                "location": name,
                "latitude": lat,
                "longitude": lon,
                "observations": len(good),
                "validation_errors": errors,
                "earth_state": state.model_dump(mode="json"),
                "data_assurance": self.assurance.evaluate_data_trust(
                    state.quality.score, len(good)
                ).value,
            }
        except Exception as exc:  # noqa: BLE001
            self.health.report(
                "ingest.open-meteo",
                HealthState.FAILED,
                message=str(exc),
                fallback="baseline.climatology if archive available",
            )
            raise

    def forecast(
        self,
        location: str = "Birmingham",
        variable: str = "temperature_2m",
        hours: int = 48,
    ) -> dict:
        name, lat, lon = resolve_location(location)
        state_payload = self.ingest_and_state(location)
        from quantum_earth.core.schemas import EarthState

        state = EarthState.model_validate(state_payload["earth_state"])
        fc = self.forecast_engine.forecast(name, lat, lon, variable=variable, hours=hours, earth_state=state)
        return fc.model_dump(mode="json")

    def verify_recent(
        self,
        location: str = "Birmingham",
        variable: str = "temperature_2m",
        hours: int = 24,
    ) -> dict:
        """Pseudo-operational verification using archive: climatology/persistence vs truth.

        Uses strict temporal split: train/climatology from earlier window, test on later window.
        """
        name, lat, lon = resolve_location(location)
        # Archive lag ~5 days
        end = datetime.now(timezone.utc).date() - timedelta(days=5)
        start = end - timedelta(days=max(hours // 24 + 1, 2))
        series = self.archive_connector.fetch_hourly_series(
            lat, lon, start.isoformat(), end.isoformat()
        )
        timestamps = series["timestamps"]
        values = series["variables"].get(variable, [])
        if len(values) < hours + 2:
            return {
                "status": "INSUFFICIENT_DATA",
                "integrity": IntegrityClass.UNKNOWN.value,
                "location": name,
            }

        truth = [float(v) if v is not None else float("nan") for v in values[-hours:]]
        # Persistence: last value before test window
        prior = values[-hours - 1]
        if prior is None:
            prior = next((v for v in reversed(values[: -hours]) if v is not None), 0.0)
        persistence = [float(prior)] * hours

        # Climatology from earlier part of series
        from quantum_earth.models.baselines import climatology_forecast, hourly_climatology_buckets

        buckets = hourly_climatology_buckets(timestamps[:-hours], values[:-hours])
        clim_means, _ = climatology_forecast(buckets, timestamps[-hours], hours)
        climatology = [float(v) if v is not None else float("nan") for v in clim_means]

        members = np.vstack(
            [
                np.asarray(persistence, dtype=float),
                np.asarray(climatology, dtype=float),
            ]
        )
        ensemble_mean = np.nanmean(members, axis=0)

        scores = {
            "persistence": self.verification.score_continuous(
                "baseline.persistence", variable, persistence, truth, meta={"location": name, "hours": hours}
            ),
            "climatology": self.verification.score_continuous(
                "baseline.climatology", variable, climatology, truth, meta={"location": name, "hours": hours}
            ),
            "ensemble_mean": self.verification.score_continuous(
                "baseline.ensemble_mean",
                variable,
                ensemble_mean.tolist(),
                truth,
                members=members,
                meta={"location": name, "hours": hours},
            ),
        }

        # Update model metrics on registry
        for model_id, score in (
            ("baseline.persistence", scores["persistence"]),
            ("baseline.climatology", scores["climatology"]),
        ):
            card = self.models.get(model_id)
            if card:
                card.metrics = {
                    "mae": score["mae"],
                    "rmse": score["rmse"],
                    "bias": score["bias"],
                    "correlation": score["correlation"],
                }
                self.db.upsert("models", "model_id", model_id, card.model_dump(mode="json"))

        report = {
            "status": "OK",
            "location": name,
            "variable": variable,
            "hours": hours,
            "period": {"start": timestamps[-hours].isoformat(), "end": timestamps[-1].isoformat()},
            "scores": scores,
            "integrity": IntegrityClass.OBSERVED.value,
            "notes": "Ground truth from Open-Meteo archive (reanalysis-based). Not synthetic.",
        }
        self.storage.write_json("METRICS", f"verify_{name}_{variable}_{hours}.json", report)
        return report

    def status(self) -> dict:
        return {
            "platform": "QUANTUM EARTH",
            "version": "0.1.0",
            "discover": self.discover(),
            "health": [h.model_dump(mode="json") for h in self.health.snapshot()],
            "reduced_capability": True,
            "gpu": False,
        }
