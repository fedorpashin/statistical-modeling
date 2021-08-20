from .base import DiscreteDistribution, DiscreteDistributionAlgorithm, DistributionMean, DistributionVariance

from .cumulative import cumulative
from math import log
from numpy import random

from functools import cached_property, lru_cache


class Distribution(DiscreteDistribution):
    __p: float

    def __init__(self, p: float):
        assert 0 <= p <= 1
        self.__p = p

    @property
    def p(self) -> float:
        return self.__p

    @cached_property
    def q(self) -> float:
        return 1 - self.p


class Algorithm(DiscreteDistributionAlgorithm[Distribution]):
    @staticmethod
    def default(distribution: Distribution) -> 'Algorithm':
        return CumulativeAlgorithm()


class CumulativeAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> int:
        p = distribution.p
        q = distribution.q

        @lru_cache
        def p_of(x):
            if x == 0:
                return p
            else:
                return p_of(x - 1) * q

        return cumulative(p_of)


class ForwardAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> int:
        p = distribution.p
        x = 0
        while random.uniform() > p:
            x += 1
        return x


class ImprovedCumulativeAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> int:
        q = distribution.q
        M = random.uniform()
        return round(log(M) / log(q)) + 1


class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        p = distribution.p
        return 1 / p


class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        p = distribution.p
        q = distribution.q
        return q / p**2
