from .base import (
    ContinuousDistribution,
    ContinuousDistributionAlgorithm,
    DistributionMean,
    DistributionVariance,
)
from dataclasses import dataclass
from final_class import final
from overrides import overrides

from math import sqrt, log, cos, pi
from numpy import random


@final
@dataclass(frozen=True, init=False)
class Distribution(ContinuousDistribution):
    μ: float = 0
    σ: float = 1


class Algorithm(ContinuousDistributionAlgorithm[Distribution]):
    pass


@final
class DefaultAlgorithm:
    def __new__(cls, distribution: Distribution) -> Algorithm:
        return BoxMullerAlgorithm()


@final
class BoxMullerAlgorithm(Algorithm):
    @overrides
    def value(self, distribution: Distribution) -> float:
        return sqrt(-2 * log(random.uniform())) * cos(2 * pi * random.uniform())


@final
class CentralLimitTheoremAlgorithm(Algorithm):
    @overrides
    def value(self, distribution: Distribution) -> float:
        return sum([random.uniform() for _ in range(12)]) - 6


@final
class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        µ = distribution.µ
        return µ


@final
class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        σ = distribution.σ
        return σ**2
