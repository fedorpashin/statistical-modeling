from .base import (
    DiscreteDistribution,
    DiscreteDistributionAlgorithm,
    DistributionMean,
    DistributionVariance,
)
from dataclasses import dataclass
from final_class import final
from overrides import overrides

from .cumulative import cumulative
from math import log

from functools import cached_property, lru_cache


@final
@dataclass(frozen=True)
class Distribution(DiscreteDistribution):
    q: float

    def __post__init__(self):
        assert 0 <= self.q <= 1

    @cached_property
    def p(self) -> float:
        return 1 - self.q

    @cached_property
    def α(self) -> float:
        return 1 / log(self.p)


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
        α = distribution.α
        q = distribution.q

        @lru_cache
        def p_of(x):
            if x == 1:
                return -α * q
            else:
                return q * (x - 1) / x

        return cumulative(p_of, x=1)


@final
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


@final
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
