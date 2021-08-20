from .base import DiscreteDistribution, DiscreteDistributionAlgorithm, DistributionMean, DistributionVariance

from .cumulative import cumulative
from math import log

from functools import cached_property, lru_cache


class Distribution(DiscreteDistribution):
    __q: float

    def __init__(self, q: float):
        assert 0 <= q <= 1
        self.__q = q

    @cached_property
    def p(self) -> float:
        return 1 - self.q

    @property
    def q(self) -> float:
        return self.__q

    @cached_property
    def α(self) -> float:
        return 1 / log(self.p)


class Algorithm(DiscreteDistributionAlgorithm[Distribution]):
    @staticmethod
    def default(distribution: Distribution) -> 'Algorithm':
        return CumulativeAlgorithm()


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


class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        α = distribution.α
        p = distribution.p
        q = distribution.q
        return -α * q / p


class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        α = distribution.α
        p = distribution.p
        q = distribution.q
        return -α * q * (1 + α * q) / p**2
