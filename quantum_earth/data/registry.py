from __future__ import annotations

from quantum_earth.core.schemas import Domain
from quantum_earth.data.base import ConnectorStatus, DataSourceDescriptor
from quantum_earth.storage import MetadataDB

# Free-first registry. Never fabricate an API.
# Status BLOCKED_EXTERNAL_DEPENDENCY means connector scaffolding exists but live access is not claimed.

FREE_SOURCES: list[DataSourceDescriptor] = [
    DataSourceDescriptor(
        source_id="open-meteo-forecast",
        name="Open-Meteo Forecast API",
        domain=Domain.ATMOSPHERE,
        source="https://open-meteo.com/",
        licence="Open-Meteo non-commercial / attribution; upstream model licences apply",
        coverage="Global",
        resolution="~0.1–0.25° depending on upstream model",
        update_frequency="Hourly",
        historical_depth="Forecast horizon up to 16 days (model-dependent)",
        reliability=0.9,
        latency="Minutes to ~1 hour",
        authentication="None",
        status=ConnectorStatus.OPERATIONAL,
        endpoint="https://api.open-meteo.com/v1/forecast",
        variables=["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m", "pressure_msl"],
        notes="Primary Stage-1 free weather source. Multi-model ensemble members available.",
        free=True,
    ),
    DataSourceDescriptor(
        source_id="open-meteo-archive",
        name="Open-Meteo Historical Weather API",
        domain=Domain.ATMOSPHERE,
        source="https://open-meteo.com/",
        licence="Open-Meteo non-commercial / attribution; ERA5 and other reanalysis licences apply",
        coverage="Global",
        resolution="~0.1–0.25°",
        update_frequency="Daily backfill",
        historical_depth="1940–present (ERA5-based)",
        reliability=0.92,
        latency="Typically 2–5 days for recent reanalysis",
        authentication="None",
        status=ConnectorStatus.OPERATIONAL,
        endpoint="https://archive-api.open-meteo.com/v1/archive",
        variables=["temperature_2m", "precipitation", "wind_speed_10m"],
        notes="Used for climatology baselines and verification ground truth.",
        free=True,
    ),
    DataSourceDescriptor(
        source_id="noaa-gfs-direct",
        name="NOAA GFS (direct NOMADS)",
        domain=Domain.ATMOSPHERE,
        source="https://nomads.ncep.noaa.gov/",
        licence="US Government public domain",
        coverage="Global",
        resolution="0.25°",
        update_frequency="4× daily",
        historical_depth="Model cycles retained on NOMADS (days)",
        reliability=0.85,
        latency="Hours",
        authentication="None",
        status=ConnectorStatus.BLOCKED_EXTERNAL_DEPENDENCY,
        endpoint="https://nomads.ncep.noaa.gov/",
        variables=["temperature", "wind", "precipitation"],
        notes="Reachable, but Stage-1 uses Open-Meteo aggregation rather than raw GRIB ingest.",
        free=True,
    ),
    DataSourceDescriptor(
        source_id="dwd-opendata",
        name="DWD Open Data",
        domain=Domain.ATMOSPHERE,
        source="https://opendata.dwd.de/",
        licence="DWD open data licence",
        coverage="Germany / Europe-focused products",
        resolution="Product-dependent",
        update_frequency="Product-dependent",
        historical_depth="Product-dependent",
        reliability=0.88,
        latency="Minutes to hours",
        authentication="None",
        status=ConnectorStatus.BLOCKED_EXTERNAL_DEPENDENCY,
        endpoint="https://opendata.dwd.de/",
        notes="Scaffolded for Stage 2+. Not claimed as operational ingest in Stage 1.",
        free=True,
    ),
    DataSourceDescriptor(
        source_id="goes-geostationary",
        name="GOES Geostationary Imagery",
        domain=Domain.ATMOSPHERE,
        source="NOAA / AWS Open Data",
        licence="US Government public domain",
        coverage="Americas",
        resolution="0.5–2 km",
        update_frequency="Minutes",
        historical_depth="Multi-year archives",
        reliability=0.8,
        latency="Minutes",
        authentication="None (public buckets)",
        status=ConnectorStatus.BLOCKED_EXTERNAL_DEPENDENCY,
        notes="No fake satellite integration. Connector deferred until Stage 2+.",
        free=True,
    ),
]


class DataSourceRegistry:
    def __init__(self, db: MetadataDB | None = None) -> None:
        self.db = db or MetadataDB()
        self._seed()

    def _seed(self) -> None:
        for src in FREE_SOURCES:
            self.db.upsert("data_sources", "source_id", src.source_id, src.model_dump(mode="json"))

    def list_sources(self) -> list[DataSourceDescriptor]:
        return [DataSourceDescriptor.model_validate(p) for p in self.db.list_all("data_sources")]

    def get(self, source_id: str) -> DataSourceDescriptor | None:
        payload = self.db.get("data_sources", "source_id", source_id)
        return DataSourceDescriptor.model_validate(payload) if payload else None

    def operational(self) -> list[DataSourceDescriptor]:
        return [s for s in self.list_sources() if s.status == ConnectorStatus.OPERATIONAL]
