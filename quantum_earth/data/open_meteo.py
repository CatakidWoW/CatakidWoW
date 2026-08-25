from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from quantum_earth.config import settings
from quantum_earth.core.schemas import (
    IntegrityClass,
    Observation,
    Provenance,
    QualityScore,
    SpatiotemporalRef,
    ValuedQuantity,
)
from quantum_earth.data.base import BaseConnector, ConnectorStatus, DataSourceDescriptor
from quantum_earth.data.registry import FREE_SOURCES
from quantum_earth.storage import StorageLayout

VARIABLE_META = {
    "temperature_2m": ("°C", 0.5),
    "precipitation": ("mm", 0.2),
    "wind_speed_10m": ("km/h", 1.0),
    "relative_humidity_2m": ("%", 3.0),
    "pressure_msl": ("hPa", 0.5),
}


def _parse_ts(value: str) -> datetime:
    # Open-Meteo returns ISO8601 without Z; treat as UTC
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class OpenMeteoForecastConnector(BaseConnector):
    def __init__(self, client: httpx.Client | None = None) -> None:
        self.descriptor = next(s for s in FREE_SOURCES if s.source_id == "open-meteo-forecast")
        self.client = client or httpx.Client(timeout=settings.request_timeout_s)
        self.storage = StorageLayout()

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        resp = self.client.get(settings.open_meteo_forecast_url, params=params)
        resp.raise_for_status()
        raw_path = self.storage.path(
            "RAW",
            "open-meteo-forecast",
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
        )
        raw_path.write_bytes(resp.content)
        return resp.json()

    def fetch_current(self, latitude: float, longitude: float) -> list[Observation]:
        data = self._get(
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": ",".join(VARIABLE_META.keys()),
                "timezone": "UTC",
            }
        )
        current = data.get("current") or {}
        ts = _parse_ts(current["time"]) if "time" in current else datetime.now(timezone.utc)
        checksum = StorageLayout.checksum_bytes(str(current).encode())
        provenance = Provenance(
            source_id=self.descriptor.source_id,
            source_name=self.descriptor.name,
            licence=self.descriptor.licence,
            method="open-meteo-current",
            checksum=checksum,
        )
        quality = QualityScore(score=0.9, completeness=1.0, latency_s=0.0)
        obs: list[Observation] = []
        for var, (unit, unc) in VARIABLE_META.items():
            if var not in current or current[var] is None:
                continue
            qty = ValuedQuantity(
                variable=var,
                unit=unit,
                estimate=float(current[var]),
                uncertainty=unc,
                integrity=IntegrityClass.OBSERVED,
                when=SpatiotemporalRef(
                    latitude=float(data.get("latitude", latitude)),
                    longitude=float(data.get("longitude", longitude)),
                    altitude_m=float(data["elevation"]) if "elevation" in data else None,
                    timestamp=ts,
                    spatial_resolution_m=25000.0,
                    temporal_resolution_s=3600.0,
                ),
                provenance=provenance,
                quality=quality,
            )
            obs.append(Observation(quantity=qty, integrity=IntegrityClass.OBSERVED))
        return obs

    def fetch_hourly_forecast(
        self,
        latitude: float,
        longitude: float,
        hours: int = 48,
        models: list[str] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "forecast_hours": max(1, min(hours, 384)),
            "timezone": "UTC",
        }
        if models:
            params["models"] = ",".join(models)
        return self._get(params)

    def fetch_history(
        self,
        latitude: float,
        longitude: float,
        start: datetime,
        end: datetime,
    ) -> list[Observation]:
        # Forecast connector does not provide deep history; defer to archive connector.
        raise NotImplementedError("Use OpenMeteoArchiveConnector for history")


class OpenMeteoArchiveConnector(BaseConnector):
    def __init__(self, client: httpx.Client | None = None) -> None:
        self.descriptor = next(s for s in FREE_SOURCES if s.source_id == "open-meteo-archive")
        self.client = client or httpx.Client(timeout=settings.request_timeout_s)
        self.storage = StorageLayout()

    def fetch_current(self, latitude: float, longitude: float) -> list[Observation]:
        end = datetime.now(timezone.utc).date()
        start = end
        series = self.fetch_hourly_series(latitude, longitude, start.isoformat(), end.isoformat())
        # Take last available hour as current proxy — integrity remains OBSERVED (reanalysis product)
        if not series["timestamps"]:
            return []
        idx = -1
        obs: list[Observation] = []
        for var in ("temperature_2m", "precipitation", "wind_speed_10m"):
            values = series["variables"].get(var, [])
            if not values or values[idx] is None:
                continue
            unit, unc = VARIABLE_META[var]
            provenance = Provenance(
                source_id=self.descriptor.source_id,
                source_name=self.descriptor.name,
                licence=self.descriptor.licence,
                method="open-meteo-archive-latest",
            )
            qty = ValuedQuantity(
                variable=var,
                unit=unit,
                estimate=float(values[idx]),
                uncertainty=unc,
                integrity=IntegrityClass.OBSERVED,
                when=SpatiotemporalRef(
                    latitude=latitude,
                    longitude=longitude,
                    timestamp=series["timestamps"][idx],
                    spatial_resolution_m=25000.0,
                    temporal_resolution_s=3600.0,
                ),
                provenance=provenance,
                quality=QualityScore(score=0.85, completeness=1.0),
            )
            obs.append(Observation(quantity=qty))
        return obs

    def fetch_hourly_series(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "timezone": "UTC",
        }
        resp = self.client.get(settings.open_meteo_archive_url, params=params)
        resp.raise_for_status()
        raw_path = self.storage.path(
            "RAW",
            "open-meteo-archive",
            f"{start_date}_{end_date}_{latitude}_{longitude}.json",
        )
        raw_path.write_bytes(resp.content)
        data = resp.json()
        hourly = data.get("hourly") or {}
        timestamps = [_parse_ts(t) for t in hourly.get("time", [])]
        variables = {
            k: hourly.get(k, [])
            for k in ("temperature_2m", "precipitation", "wind_speed_10m")
        }
        return {"timestamps": timestamps, "variables": variables, "raw": data}

    def fetch_history(
        self,
        latitude: float,
        longitude: float,
        start: datetime,
        end: datetime,
    ) -> list[Observation]:
        series = self.fetch_hourly_series(
            latitude,
            longitude,
            start.date().isoformat(),
            end.date().isoformat(),
        )
        provenance = Provenance(
            source_id=self.descriptor.source_id,
            source_name=self.descriptor.name,
            licence=self.descriptor.licence,
            method="open-meteo-archive",
        )
        out: list[Observation] = []
        for i, ts in enumerate(series["timestamps"]):
            for var, values in series["variables"].items():
                if i >= len(values) or values[i] is None:
                    continue
                unit, unc = VARIABLE_META[var]
                qty = ValuedQuantity(
                    variable=var,
                    unit=unit,
                    estimate=float(values[i]),
                    uncertainty=unc,
                    integrity=IntegrityClass.OBSERVED,
                    when=SpatiotemporalRef(
                        latitude=latitude,
                        longitude=longitude,
                        timestamp=ts,
                        spatial_resolution_m=25000.0,
                        temporal_resolution_s=3600.0,
                    ),
                    provenance=provenance,
                    quality=QualityScore(score=0.9, completeness=1.0),
                )
                out.append(Observation(quantity=qty))
        return out


def blocked_descriptor(source_id: str) -> DataSourceDescriptor:
    from quantum_earth.data.registry import DataSourceRegistry

    reg = DataSourceRegistry()
    src = reg.get(source_id)
    if src is None:
        raise KeyError(source_id)
    assert src.status == ConnectorStatus.BLOCKED_EXTERNAL_DEPENDENCY
    return src
