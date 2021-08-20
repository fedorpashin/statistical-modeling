from .base import DiscreteDistribution, DiscreteDistributionAlgorithm, DistributionMean, DistributionVariance
from dataclasses import dataclass

from .cumulative import cumulative
from math import exp
from numpy import random

from functools import lru_cache


@dataclass(frozen=True)
class Distribution(DiscreteDistribution):
    μ: float

    def __post_init__(self):
        assert self.μ > 0


class Algorithm(DiscreteDistributionAlgorithm[Distribution]):
    pass


class DefaultAlgorithm:
    def __new__(cls, distribution: Distribution, threshold: int = 88) -> Algorithm:
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
