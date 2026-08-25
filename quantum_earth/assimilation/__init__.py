from __future__ import annotations

from abc import ABC, abstractmethod

from quantum_earth.core.schemas import EarthState, Observation
from quantum_earth.state.engine import EarthStateEngine


class AssimilationInterface(ABC):
    """Future methods: Kalman, EnKF, 4D-Var, particle filters, neural DA."""

    name: str

    @abstractmethod
    def assimilate(self, observations: list[Observation]) -> EarthState:
        raise NotImplementedError


class QualityWeightedBlend(AssimilationInterface):
    name = "quality_weighted_blend_v1"

    def __init__(self) -> None:
        self.engine = EarthStateEngine()

    def assimilate(self, observations: list[Observation]) -> EarthState:
        return self.engine.assimilate_point(observations, method=self.name)


# Explicit stubs — interfaces first, no over-engineering
class KalmanFilterAssimilation(AssimilationInterface):
    name = "kalman_filter"

    def assimilate(self, observations: list[Observation]) -> EarthState:
        raise NotImplementedError("Kalman filtering planned for later stage")


class EnsembleKalmanAssimilation(AssimilationInterface):
    name = "ensemble_kalman"

    def assimilate(self, observations: list[Observation]) -> EarthState:
        raise NotImplementedError("EnKF planned for later stage")
