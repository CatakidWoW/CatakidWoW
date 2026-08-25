from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from quantum_earth.core.schemas import Domain, Observation


class ConnectorStatus(str, Enum):
    OPERATIONAL = "OPERATIONAL"
    DEGRADED = "DEGRADED"
    BLOCKED_EXTERNAL_DEPENDENCY = "BLOCKED_EXTERNAL_DEPENDENCY"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class DataSourceDescriptor(BaseModel):
    source_id: str
    name: str
    domain: Domain
    source: str
    licence: str
    coverage: str
    resolution: str
    update_frequency: str
    historical_depth: str
    reliability: float = Field(ge=0.0, le=1.0)
    latency: str
    authentication: str
    status: ConnectorStatus
    endpoint: str | None = None
    variables: list[str] = Field(default_factory=list)
    notes: str | None = None
    free: bool = True


class BaseConnector(ABC):
    descriptor: DataSourceDescriptor

    @abstractmethod
    def fetch_current(self, latitude: float, longitude: float) -> list[Observation]:
        raise NotImplementedError

    @abstractmethod
    def fetch_history(
        self,
        latitude: float,
        longitude: float,
        start: datetime,
        end: datetime,
    ) -> list[Observation]:
        raise NotImplementedError

    def health_probe(self) -> dict[str, Any]:
        return {"source_id": self.descriptor.source_id, "status": self.descriptor.status.value}
