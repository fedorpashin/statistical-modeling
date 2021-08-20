from .base import ContinuousDistribution, ContinuousDistributionAlgorithm, DistributionMean, DistributionVariance
from dataclasses import dataclass
from final_class import final
from overrides import overrides

from numpy import random


@final
@dataclass(frozen=True)
class Distribution(ContinuousDistribution):
    a: int = 0
    b: int = 1

    def __post_init__(self):
        assert self.a < self.b


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
        a = distribution.a
        b = distribution.b
        return (b - a) * random.uniform() + a


@final
class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        a = distribution.a
        b = distribution.b
        return (a + b) / 2


@final
class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        a = distribution.a
        b = distribution.b
        return ((b - a)**2) / 12
