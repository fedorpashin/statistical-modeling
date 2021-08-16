from .base import ContinuousDistribution, ContinuousDistributionAlgorithm
from abc import abstractmethod

from math import log
from numpy import random

from functools import cached_property


class Distribution(ContinuousDistribution):
    _β: float

    def __init__(self, β: float):
        self._β = β

    @property
    def β(self) -> float:
        return self._β


class Algorithm(ContinuousDistributionAlgorithm['Algorithm']):
    @abstractmethod
    def value(self, distribution: Distribution) -> float:
        pass

    @classmethod
    def default(cls, distribution: Distribution) -> 'Algorithm':
        return StandardAlgorithm()


class StandardAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> float:
        β = distribution.β
        return β * log(random.uniform())


class Mean:
    distribution: Distribution

    @cached_property
    def Mean(self) -> float:
        β = self.distribution.β
        return β


class Variance:
    distribution: Distribution

    @cached_property
    def Variance(self) -> float:
        β = self.distribution.β
        return β**2
