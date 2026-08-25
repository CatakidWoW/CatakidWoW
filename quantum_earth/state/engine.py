from __future__ import annotations

from datetime import datetime, timezone

from quantum_earth.core.schemas import (
    EarthState,
    EarthStateCell,
    IntegrityClass,
    Observation,
    Provenance,
    QualityScore,
    ValuedQuantity,
)
from quantum_earth.storage import StorageLayout


class EarthStateEngine:
    """Estimate current Earth state from validated observations.

    Stage 1 method: per-variable quality-weighted blend at a point location.
    Future: Kalman / EnKF / variational / particle / neural DA via AssimilationInterface.
    """

    def __init__(self) -> None:
        self.storage = StorageLayout()

    def assimilate_point(
        self,
        observations: list[Observation],
        method: str = "quality_weighted_blend_v1",
    ) -> EarthState:
        if not observations:
            now = datetime.now(timezone.utc)
            return EarthState(
                as_of=now,
                cells=[],
                integrity=IntegrityClass.UNKNOWN,
                assimilation_method=method,
                provenance=[],
                quality=QualityScore(score=0.0, completeness=0.0, flags=["no_observations"]),
            )

        by_var: dict[str, list[Observation]] = {}
        for obs in observations:
            by_var.setdefault(obs.quantity.variable, []).append(obs)

        # Use first observation location/time as cell anchor
        anchor = observations[0].quantity.when
        variables: dict[str, ValuedQuantity] = {}
        provenances: list[Provenance] = []
        qualities: list[float] = []

        for var, obs_list in by_var.items():
            weights = [max(o.quantity.quality.score, 1e-6) for o in obs_list]
            estimates = [o.quantity.estimate for o in obs_list if o.quantity.estimate is not None]
            if not estimates:
                continue
            wsum = sum(weights[: len(estimates)])
            blend = sum(e * w for e, w in zip(estimates, weights)) / wsum
            # Uncertainty: max reported unc + disagreement across sources
            uncs = [o.quantity.uncertainty or 0.0 for o in obs_list]
            disagreement = float(max(estimates) - min(estimates)) if len(estimates) > 1 else 0.0
            unc = max(uncs) + 0.5 * disagreement
            best = max(obs_list, key=lambda o: o.quantity.quality.score)
            provenances.append(best.quantity.provenance)
            qualities.append(best.quantity.quality.score)
            variables[var] = ValuedQuantity(
                variable=var,
                unit=best.quantity.unit,
                estimate=float(blend),
                uncertainty=float(unc),
                integrity=IntegrityClass.INFERRED,
                when=best.quantity.when,
                provenance=Provenance(
                    source_id="earth-state-engine",
                    source_name="EarthStateEngine",
                    licence="internal",
                    method=method,
                    parent_ids=[o.quantity.id for o in obs_list],
                ),
                quality=QualityScore(
                    score=sum(qualities) / len(qualities) if qualities else 0.0,
                    completeness=1.0,
                ),
            )

        state = EarthState(
            as_of=anchor.timestamp if anchor.timestamp.tzinfo else anchor.timestamp.replace(tzinfo=timezone.utc),
            cells=[
                EarthStateCell(
                    latitude=anchor.latitude,
                    longitude=anchor.longitude,
                    altitude_m=anchor.altitude_m,
                    timestamp=anchor.timestamp,
                    variables=variables,
                )
            ],
            integrity=IntegrityClass.INFERRED,
            assimilation_method=method,
            provenance=provenances,
            quality=QualityScore(
                score=(sum(qualities) / len(qualities)) if qualities else 0.0,
                completeness=min(1.0, len(variables) / 3.0),
            ),
        )
        self.storage.write_json(
            "ASSIMILATED",
            f"earth_state/{state.id}.json",
            state.model_dump(mode="json"),
        )
        return state
