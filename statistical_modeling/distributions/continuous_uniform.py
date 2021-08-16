from .base import ContinuousDistribution, ContinuousDistributionAlgorithm
from abc import abstractmethod

from numpy import random

from functools import cached_property


class Distribution(ContinuousDistribution):
    _a: int
    _b: int

    def __init__(self, a: int = 0, b: int = 1):
        self._a = a
        self._b = b

    @property
    def a(self) -> int:
        return self._a

    @property
    def b(self) -> int:
        return self._b


class Algorithm(ContinuousDistributionAlgorithm['Algorithm']):
    @abstractmethod
    def value(self, distribution: Distribution) -> float:
        pass

    @classmethod
    def default(cls, distribution: Distribution) -> 'Algorithm':
        return StandardAlgorithm()


class StandardAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> float:
        a = distribution.a
        b = distribution.b
        return (b - a) * random.uniform() + a


class Mean:
    distribution: Distribution

    @cached_property
    def _value(self) -> float:
        a = self.distribution.a
        b = self.distribution.b
        return (a + b) / 2


class Variance:
    distribution: Distribution

    @cached_property
    def _value(self) -> float:
        a = self.distribution.a
        b = self.distribution.b
        return ((b - a)**2) / 12
