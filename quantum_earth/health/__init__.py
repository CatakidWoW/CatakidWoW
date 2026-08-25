from __future__ import annotations

from datetime import datetime, timezone

from quantum_earth.core.schemas import ComponentHealth, HealthState
from quantum_earth.storage import MetadataDB


class HealthMonitor:
    def __init__(self, db: MetadataDB | None = None) -> None:
        self.db = db or MetadataDB()

    def report(
        self,
        component: str,
        state: HealthState,
        latency_ms: float | None = None,
        message: str | None = None,
        fallback: str | None = None,
        error_count: int = 0,
    ) -> ComponentHealth:
        existing = self.db.get("health", "component", component)
        prev_errors = existing.get("error_count", 0) if existing else 0
        health = ComponentHealth(
            component=component,
            state=state,
            last_success=datetime.now(timezone.utc) if state == HealthState.HEALTHY else (
                datetime.fromisoformat(existing["last_success"]) if existing and existing.get("last_success") else None
            ),
            error_count=error_count if error_count else (prev_errors + (0 if state == HealthState.HEALTHY else 1)),
            latency_ms=latency_ms,
            message=message,
            fallback=fallback,
        )
        if state == HealthState.HEALTHY:
            health.error_count = 0
            health.last_success = datetime.now(timezone.utc)
        self.db.upsert("health", "component", component, health.model_dump(mode="json"))
        return health

    def snapshot(self) -> list[ComponentHealth]:
        return [ComponentHealth.model_validate(p) for p in self.db.list_all("health")]
