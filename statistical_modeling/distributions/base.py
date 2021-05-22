from statistical_modeling.sample import Sample
from abc import ABC, abstractmethod
from enum import Enum

from functools import cached_property

__all__ = ['DistributionAlgorithm', 'Distribution']


class DistributionAlgorithm(Enum):
    pass


class Distribution(ABC):
    class Algorithm(DistributionAlgorithm):
        pass

    @abstractmethod
    def value(self, algorithm: Algorithm = 1) -> float:
        pass

    def sample(self, n: int, algorithm: Algorithm = 1) -> Sample:
        return Sample([self.value(algorithm) for _ in range(n)])

    @cached_property
    @abstractmethod
    def E(self) -> float:
        pass

    @cached_property
    @abstractmethod
    def D(self) -> float:
        pass
