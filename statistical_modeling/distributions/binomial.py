from .base import DiscreteDistribution, DiscreteDistributionAlgorithm, DistributionMean, DistributionVariance
from dataclasses import dataclass
from final_class import final

from .cumulative import cumulative
from math import sqrt
from numpy import random

from functools import cached_property, lru_cache


@final
@dataclass(frozen=True)
class Distribution(DiscreteDistribution):
    n: int
    p: float

    def __post__init__(self):
        assert self.n > 0
        assert 0 <= self.p <= 1

    @cached_property
    def q(self) -> float:
        return 1 - self.p


class Algorithm(DiscreteDistributionAlgorithm[Distribution]):
    pass


@final
class DefaultAlgorithm:
    def __new__(cls, distribution: Distribution, threshold: int = 100) -> Algorithm:
        if distribution.n < threshold:
            return CumulativeAlgorithm()
        else:
            return NormalApproximationAlgorithm()


@final
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


@final
class NormalApproximationAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> int:
        n = distribution.n
        p = distribution.p
        q = distribution.q
        return round(random.normal(n * p, sqrt(n * p * q)) + 0.5)


@final
class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        n = distribution.n
        p = distribution.p
        return n * p


@final
class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        n = distribution.n
        p = distribution.p
        q = distribution.q
        return n * p * q
