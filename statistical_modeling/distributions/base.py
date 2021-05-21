from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum

from functools import cached_property

__all__ = ['DistributionAlgorithm', 'Distribution']


class DistributionAlgorithm(Enum):
    pass


class Distribution(ABC):
    class Algorithm(DistributionAlgorithm):
        pass

    @abstractmethod
    def generate(self, algorithm: Optional[Algorithm]) -> float:
        pass

    @cached_property
    @abstractmethod
    def E(self) -> float:
        pass

    @cached_property
    @abstractmethod
    def D(self) -> float:
        pass
