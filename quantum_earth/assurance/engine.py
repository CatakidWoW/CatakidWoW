from __future__ import annotations

from quantum_earth.core.schemas import AssuranceLevel


class QuantumAssurance:
    """Continuously answers: is there enough evidence to publish this forecast?"""

    def evaluate_forecast(
        self,
        n_members: int,
        mean_spread: float | None,
        missing_fraction: float,
        extra_reasons: list[str] | None = None,
        variable: str = "temperature_2m",
    ) -> tuple[AssuranceLevel, list[str]]:
        reasons = list(extra_reasons or [])
        if n_members <= 0 or missing_fraction >= 0.9:
            reasons.append("insufficient ensemble evidence")
            return AssuranceLevel.RED, reasons

        # Overconfidence / disagreement heuristics
        if mean_spread is not None:
            if variable == "temperature_2m" and mean_spread > 4.0:
                reasons.append(f"high model disagreement (spread={mean_spread:.2f}°C)")
            if variable == "precipitation" and mean_spread > 3.0:
                reasons.append(f"high precip disagreement (spread={mean_spread:.2f}mm)")
            if variable == "wind_speed_10m" and mean_spread > 8.0:
                reasons.append(f"high wind disagreement (spread={mean_spread:.2f}km/h)")

        if n_members < 3:
            reasons.append("few ensemble members")
            return AssuranceLevel.YELLOW, reasons

        if reasons:
            return AssuranceLevel.YELLOW, reasons
        return AssuranceLevel.GREEN, ["multi-model ensemble available", "uncertainty quantified"]

    def evaluate_data_trust(self, quality_score: float, source_count: int) -> AssuranceLevel:
        if source_count == 0 or quality_score < 0.4:
            return AssuranceLevel.RED
        if source_count == 1 or quality_score < 0.7:
            return AssuranceLevel.YELLOW
        return AssuranceLevel.GREEN
