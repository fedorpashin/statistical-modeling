from .base import DiscreteDistribution, DiscreteDistributionAlgorithm, DistributionMean, DistributionVariance
from dataclasses import dataclass
from final_class import final
from overrides import overrides

from .cumulative import cumulative
from math import log
from numpy import random

from functools import cached_property, lru_cache


@final
@dataclass(frozen=True)
class Distribution(DiscreteDistribution):
    p: float

    def __post__init__(self):
        assert 0 <= self.p <= 1

    @cached_property
    def q(self) -> float:
        return 1 - self.p


class Algorithm(DiscreteDistributionAlgorithm[Distribution]):
    pass


@final
class DefaultAlgorithm:
    def __new__(cls, distribution: Distribution) -> Algorithm:
        return CumulativeAlgorithm()


@final
class CumulativeAlgorithm(Algorithm):
    @overrides
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


@final
class ForwardAlgorithm(Algorithm):
    @overrides
    def value(self, distribution: Distribution) -> int:
        p = distribution.p
        x = 0
        while random.uniform() > p:
            x += 1
        return x


@final
class ImprovedCumulativeAlgorithm(Algorithm):
    @overrides
    def value(self, distribution: Distribution) -> int:
        q = distribution.q
        M = random.uniform()
        return round(log(M) / log(q)) + 1


@final
class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        p = distribution.p
        return 1 / p


@final
class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        p = distribution.p
        q = distribution.q
        return q / p**2
