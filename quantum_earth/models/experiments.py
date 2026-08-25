from __future__ import annotations

from quantum_earth.core.schemas import ExperimentRecord, Provenance
from quantum_earth.storage import MetadataDB


class ExperimentRegistry:
    def __init__(self, db: MetadataDB | None = None) -> None:
        self.db = db or MetadataDB()

    def create(self, hypothesis: str, model_id: str) -> ExperimentRecord:
        rec = ExperimentRecord(
            hypothesis=hypothesis,
            model_id=model_id,
            status="PLANNED",
            provenance=Provenance(
                source_id="experiment-registry",
                source_name="ExperimentRegistry",
                licence="internal",
                method="manual_or_autonomous_research",
            ),
        )
        self.db.upsert("experiments", "experiment_id", rec.experiment_id, rec.model_dump(mode="json"))
        return rec

    def decide(self, experiment_id: str, decision: str, metrics: dict[str, float]) -> ExperimentRecord:
        if decision not in {"PROMOTE", "REJECT", "KEEP_CHALLENGER"}:
            raise ValueError("decision must be PROMOTE | REJECT | KEEP_CHALLENGER")
        payload = self.db.get("experiments", "experiment_id", experiment_id)
        if not payload:
            raise KeyError(experiment_id)
        rec = ExperimentRecord.model_validate(payload)
        # Promotion is metric-driven only — never LLM opinion
        rec.metrics = metrics
        rec.decision = decision
        rec.status = "COMPLETED"
        self.db.upsert("experiments", "experiment_id", rec.experiment_id, rec.model_dump(mode="json"))
        return rec

    def list_experiments(self) -> list[ExperimentRecord]:
        return [ExperimentRecord.model_validate(p) for p in self.db.list_all("experiments")]
