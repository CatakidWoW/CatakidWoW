from __future__ import annotations

from quantum_earth.core.schemas import Observation


class ObservationValidator:
    """Validate, detect corruption, and score observation batches."""

    REQUIRED_FIELDS = ("variable", "estimate", "unit")

    def validate_batch(self, observations: list[Observation]) -> tuple[list[Observation], list[str]]:
        good: list[Observation] = []
        errors: list[str] = []
        seen: set[tuple[str, str]] = set()
        for obs in observations:
            q = obs.quantity
            if q.estimate is None:
                errors.append(f"missing estimate for {q.variable}")
                continue
            if abs(q.estimate) > 1e6:
                errors.append(f"corrupt magnitude for {q.variable}: {q.estimate}")
                continue
            if q.variable == "temperature_2m" and not (-90 <= q.estimate <= 60):
                errors.append(f"temperature out of physical range: {q.estimate}")
                continue
            if q.variable == "precipitation" and q.estimate < 0:
                errors.append(f"negative precipitation: {q.estimate}")
                continue
            if q.variable == "wind_speed_10m" and q.estimate < 0:
                errors.append(f"negative wind: {q.estimate}")
                continue
            key = (q.variable, q.when.timestamp.isoformat())
            if key in seen:
                errors.append(f"duplicate observation {key}")
                continue
            seen.add(key)
            good.append(obs)
        return good, errors

    @staticmethod
    def inject_corruption(observations: list[Observation], mode: str) -> list[Observation]:
        """Adversarial helper — intentionally corrupt copies for testing."""
        import copy

        clones = [copy.deepcopy(o) for o in observations]
        if not clones:
            return clones
        if mode == "nan_magnitude":
            clones[0].quantity.estimate = 1e12
        elif mode == "negative_precip":
            for c in clones:
                if c.quantity.variable == "precipitation":
                    c.quantity.estimate = -5.0
                    break
        elif mode == "duplicate":
            clones.append(copy.deepcopy(clones[0]))
        return clones
