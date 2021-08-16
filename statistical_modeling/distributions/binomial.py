from .base import DiscreteDistribution, DiscreteDistributionAlgorithm
from abc import abstractmethod
from generic.multimethod import has_multimethods, multimethod

from .cumulative import cumulative
from math import sqrt
from numpy import random

from functools import cached_property, lru_cache


class Distribution(DiscreteDistribution):
    _n: int
    _p: float

    def __init__(self, n: int, p: float):
        assert n > 0
        assert 0 <= p <= 1
        self._n = n
        self._p = p

    @property
    def n(self) -> int:
        return self._n

    @property
    def p(self) -> float:
        return self._p

    @cached_property
    def q(self) -> float:
        return 1 - self.p


@has_multimethods
class Algorithm(DiscreteDistributionAlgorithm['Algorithm']):
    @classmethod
    @multimethod()
    def default(cls) -> 'Algorithm':
        return cls.default(100)

    @classmethod # noqa
    @default.register(int)
    def default(cls, threshold: int) -> 'Algorithm':
        if threshold < 100:
            return CumulativeAlgorithm()
        else:
            return NormalApproximationAlgorithm()

    @abstractmethod
    def value(self, distribution: Distribution) -> int:
        pass


class CumulativeAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> int:
        n = distribution.n
        p = distribution.p
        q = distribution.q

        @lru_cache()
        def p_of(r):
            if r == 0:
                return q**n
            else:
                return p_of(r - 1) * p * (n - r + 1) / (r * q)

        return cumulative(p_of)


class NormalApproximationAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> int:
        n = distribution.n
        p = distribution.p
        q = distribution.q
        return round(random.normal(n * p, sqrt(n * p * q)) + 0.5)


class Mean:
    distribution: Distribution

    @cached_property
    def _value(self) -> float:
        n = self.distribution.p
        p = self.distribution.p
        return n * p


class Variance:
    distribution: Distribution

    @cached_property
    def _value(self) -> float:
        n = self.distribution.p
        p = self.distribution.p
        q = self.distribution.q
        return n * p * q