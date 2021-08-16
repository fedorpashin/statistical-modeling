from statistical_modeling.sample import Sample
from abc import ABC, abstractmethod
from enum import Enum

from functools import cached_property


class Distribution(ABC):
    class Algorithm(Enum):
        pass

    @property
    @abstractmethod
    def default_algorithm(self) -> Algorithm:
        pass

    @abstractmethod
    def value(self, algorithm: Algorithm) -> float:
        pass

    def sample(self, n: int, algorithm: Algorithm = default_algorithm) -> Sample:  # type: ignore
        return Sample([self.value(algorithm) for _ in range(n)])

    @cached_property
    @abstractmethod
    def E(self) -> float:
        pass

    @cached_property
    @abstractmethod
    def D(self) -> float:
        pass
