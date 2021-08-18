from .base import DiscreteDistribution, DiscreteDistributionAlgorithm, DistributionMean, DistributionVariance
from abc import abstractmethod

from .cumulative import cumulative
from math import sqrt
from numpy import random

from functools import cached_property, lru_cache


class Distribution(DiscreteDistribution):
    __n: int
    __p: float

    def __init__(self, n: int, p: float):
        assert n > 0
        assert 0 <= p <= 1
        self.__n = n
        self.__p = p

    @property
    def n(self) -> int:
        return self.__n

    @property
    def p(self) -> float:
        return self.__p

    @cached_property
    def q(self) -> float:
        return 1 - self.p


class Algorithm(DiscreteDistributionAlgorithm['Algorithm']):
    @staticmethod
    def default(distribution: Distribution, threshold: int = 100) -> 'Algorithm':
        if distribution.n < threshold:
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


class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        n = distribution.p
        p = distribution.p
        return n * p


class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        n = distribution.p
        p = distribution.p
        q = distribution.q
        return n * p * q
