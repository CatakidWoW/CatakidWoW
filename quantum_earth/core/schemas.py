from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class IntegrityClass(str, Enum):
    """Scientific integrity — never blur these categories."""

    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    FORECAST = "FORECAST"
    SCENARIO = "SCENARIO"
    HYPOTHESIS = "HYPOTHESIS"
    UNKNOWN = "UNKNOWN"


class Domain(str, Enum):
    ATMOSPHERE = "atmosphere"
    OCEAN = "ocean"
    LAND = "land"
    HYDROLOGY = "hydrology"
    CRYOSPHERE = "cryosphere"
    ECOLOGY = "ecology"
    AIR_QUALITY = "air_quality"
    SPACE_WEATHER = "space_weather"
    GEOHAZARD = "geohazard"


class Provenance(BaseModel):
    source_id: str
    source_name: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    licence: str
    method: str
    checksum: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class QualityScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0, default=1.0)
    latency_s: float | None = None
    flags: list[str] = Field(default_factory=list)


class SpatiotemporalRef(BaseModel):
    latitude: float
    longitude: float
    altitude_m: float | None = None
    depth_m: float | None = None
    timestamp: datetime
    spatial_resolution_m: float | None = None
    temporal_resolution_s: float | None = None


class ValuedQuantity(BaseModel):
    """Every value carries estimate + uncertainty + provenance + quality."""

    variable: str
    unit: str
    estimate: float | None
    uncertainty: float | None
    integrity: IntegrityClass
    when: SpatiotemporalRef
    provenance: Provenance
    quality: QualityScore
    distribution: dict[str, float] | None = None  # e.g. quantiles
    id: str = Field(default_factory=lambda: str(uuid4()))


class Observation(BaseModel):
    quantity: ValuedQuantity
    raw_uri: str | None = None
    integrity: IntegrityClass = IntegrityClass.OBSERVED

    def model_post_init(self, __context: Any) -> None:
        if self.quantity.integrity != IntegrityClass.OBSERVED:
            # Observations must never be stored as forecasts
            object.__setattr__(
                self.quantity,
                "integrity",
                IntegrityClass.OBSERVED,
            )


class EarthStateCell(BaseModel):
    latitude: float
    longitude: float
    altitude_m: float | None = None
    timestamp: datetime
    variables: dict[str, ValuedQuantity]


class EarthState(BaseModel):
    """Central estimated current state of Earth (never stores forecasts as observations)."""

    as_of: datetime
    cells: list[EarthStateCell]
    integrity: IntegrityClass = IntegrityClass.INFERRED
    assimilation_method: str
    provenance: list[Provenance] = Field(default_factory=list)
    quality: QualityScore
    id: str = Field(default_factory=lambda: str(uuid4()))


class EnsembleMember(BaseModel):
    member_id: str
    model_id: str
    values: list[float]
    timestamps: list[datetime]


class ProbabilisticForecast(BaseModel):
    variable: str
    unit: str
    location_name: str
    latitude: float
    longitude: float
    issued_at: datetime
    horizon_hours: int
    timestamps: list[datetime]
    mean: list[float | None]
    median: list[float | None]
    q10: list[float | None]
    q90: list[float | None]
    members: list[EnsembleMember]
    threshold_probabilities: dict[str, list[float | None]] = Field(default_factory=dict)
    integrity: IntegrityClass = IntegrityClass.FORECAST
    model_ids: list[str]
    provenance: list[Provenance]
    assurance: str  # GREEN | YELLOW | RED
    assurance_reasons: list[str] = Field(default_factory=list)
    id: str = Field(default_factory=lambda: str(uuid4()))


class ModelStatus(str, Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    CHALLENGER = "CHALLENGER"
    CANDIDATE = "CANDIDATE"
    PRODUCTION = "PRODUCTION"
    DEGRADED = "DEGRADED"
    RETIRED = "RETIRED"


class ModelCard(BaseModel):
    model_id: str
    version: str
    target_variables: list[str]
    input_variables: list[str]
    spatial_resolution_m: float | None
    temporal_resolution_s: float | None
    forecast_horizon_hours: int
    training_data: str
    training_period: str
    validation_period: str
    test_period: str | None = None
    uncertainty_method: str
    known_failure_modes: list[str] = Field(default_factory=list)
    compute_requirements: str
    status: ModelStatus = ModelStatus.EXPERIMENTAL
    metrics: dict[str, float] = Field(default_factory=dict)
    calibration: dict[str, float] = Field(default_factory=dict)
    family: str


class ExperimentRecord(BaseModel):
    experiment_id: str = Field(default_factory=lambda: str(uuid4()))
    hypothesis: str
    model_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "PLANNED"
    metrics: dict[str, float] = Field(default_factory=dict)
    decision: str | None = None  # PROMOTE | REJECT | KEEP_CHALLENGER
    provenance: Provenance | None = None


class EventLifecycle(str, Enum):
    WATCH = "WATCH"
    DEVELOPING = "DEVELOPING"
    ACTIVE = "ACTIVE"
    PEAK = "PEAK"
    DECAYING = "DECAYING"
    ENDED = "ENDED"
    VERIFIED = "VERIFIED"


class EventRecord(BaseModel):
    event_type: str
    location_name: str
    latitude: float
    longitude: float
    start_probability: float
    expected_start: datetime | None
    expected_duration_hours: float | None
    severity: float | None
    affected_area_km2: float | None
    probability: float
    uncertainty: float
    precursor_conditions: list[str] = Field(default_factory=list)
    confidence: float
    observed_confirmation: bool = False
    lifecycle: EventLifecycle = EventLifecycle.WATCH
    integrity: IntegrityClass = IntegrityClass.FORECAST


class HealthState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class ComponentHealth(BaseModel):
    component: str
    state: HealthState
    last_success: datetime | None = None
    error_count: int = 0
    latency_ms: float | None = None
    message: str | None = None
    fallback: str | None = None


class AssuranceLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
