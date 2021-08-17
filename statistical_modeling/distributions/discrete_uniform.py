from .base import DiscreteDistribution, DiscreteDistributionAlgorithm
from abc import abstractmethod

from math import floor
from numpy import random

from functools import cached_property


class Distribution(DiscreteDistribution):
    _a: int
    _b: int

    def __init__(self, a: int, b: int):
        self._a = a
        self._b = b

    @property
    def a(self) -> int:
        return self._a

    @property
    def b(self) -> int:
        return self._b

    @cached_property
    def n(self) -> int:
        return self.b - self.a + 1


class Algorithm(DiscreteDistributionAlgorithm):
    @classmethod
    def default(cls, distribution: Distribution) -> 'Algorithm':
        pass

    @abstractmethod
    def value(self, distribution: Distribution) -> int:
        pass


class StandardAlgorithm(Algorithm):

    def value(self, distribution: Distribution) -> int:
        n = distribution.n
        a = distribution.a
        return floor(n * random.uniform() + a)


class Mean(float):
    distribution: Distribution

    @cached_property
    def _value(self) -> float:
        a = self.distribution.a
        b = self.distribution.b
        return (a + b) / 2


class Variance(float):
    distribution: Distribution

    @cached_property
    def _value(self) -> float:
        n = self.distribution.n
        return (n**2 - 1) / 12
