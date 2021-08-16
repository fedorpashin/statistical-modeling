from .base import DiscreteDistribution, DiscreteDistributionAlgorithm
from abc import abstractmethod

from .cumulative import cumulative
from math import log

from functools import cached_property, lru_cache


class Distribution(DiscreteDistribution):
    _q: float

    def __init__(self, q: float):
        assert 0 <= q <= 1
        self._q = q

    @cached_property
    def p(self) -> float:
        return 1 - self.q

    @property
    def q(self) -> float:
        return self._q

    @cached_property
    def α(self) -> float:
        return 1 / log(self.p)


class Algorithm(DiscreteDistributionAlgorithm):
    @classmethod
    def default(cls, distribution: Distribution) -> 'Algorithm':
        return CumulativeAlgorithm()

    @abstractmethod
    def value(self, distribution: Distribution) -> int:
        pass


class CumulativeAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> int:
        α = distribution.α
        q = distribution.q

        @lru_cache
        def p_of(x):
            if x == 1:
                return -α * q
            else:
                return q * (x - 1) / x

        return cumulative(p_of, x=1)


class Mean:
    distribution: Distribution

    @cached_property
    def _value(self) -> float:
        α = self.distribution.α
        p = self.distribution.p
        q = self.distribution.q
        return -α * q / p


class Variance:
    distribution: Distribution

    @cached_property
    def _value(self) -> float:
        α = self.distribution.α
        p = self.distribution.p
        q = self.distribution.q
        return -α * q * (1 + α * q) / p**2
