from .base import (
    ContinuousDistribution,
    ContinuousDistributionAlgorithm,
    DistributionMean,
    DistributionVariance,
)
from dataclasses import dataclass
from final_class import final
from overrides import overrides

from numpy import random


@final
@dataclass(frozen=True)
class Distribution(ContinuousDistribution):
    n: int

    def __post_init__(self):
        assert self.n > 0


class Algorithm(ContinuousDistributionAlgorithm[Distribution]):
    pass


@final
class DefaultAlgorithm:
    def __new__(cls, distribution: Distribution) -> Algorithm:
        return StandardAlgorithm()


@final
class StandardAlgorithm(Algorithm):
    @overrides
    def value(self, distribution: Distribution) -> float:
        n = distribution.n
        return sum([random.standard_normal()**2 for _ in range(n)])


@final
class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        n = distribution.n
        return n


@final
class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        n = distribution.n
        return 2 * n
