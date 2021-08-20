from .base import DiscreteDistribution, DiscreteDistributionAlgorithm, DistributionMean, DistributionVariance

from .cumulative import cumulative
from math import exp
from numpy import random

from functools import lru_cache


class Distribution(DiscreteDistribution):
    __μ: float

    def __init__(self, μ: float):
        assert μ > 0
        self.__μ = μ

    @property
    def µ(self) -> float:
        return self.__μ


class Algorithm(DiscreteDistributionAlgorithm[Distribution]):
    @staticmethod
    def default(distribution: Distribution, threshold: int = 88) -> 'Algorithm':
        if distribution.µ < threshold:
            return CumulativeAlgorithm()
        else:
            return NormalApproximationAlgorithm()


class CumulativeAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> int:
        µ = distribution.µ

        @lru_cache
        def p_of(x):
            if x == 0:
                return exp(-µ)
            else:
                return p_of(x - 1) * µ / x

        return cumulative(p_of)


class MultiplicationAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> int:
        µ = distribution.µ
        x = 0
        m = 1.0
        while True:
            m *= random.uniform()
            if m < exp(-µ):
                break
            else:
                x += 1
        return x


class NormalApproximationAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> int:
        µ = distribution.µ
        return round(random.normal(µ, µ))


class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        µ = distribution.µ
        return µ


class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        µ = distribution.µ
        return µ
