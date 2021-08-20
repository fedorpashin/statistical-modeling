from .base import ContinuousDistribution, ContinuousDistributionAlgorithm, DistributionMean, DistributionVariance
from dataclasses import dataclass

from numpy import random


@dataclass(frozen=True)
class Distribution(ContinuousDistribution):
    n: int

    def __post_init__(self):
        assert self.n > 0


class Algorithm(ContinuousDistributionAlgorithm[Distribution]):
    pass


class DefaultAlgorithm:
    def __new__(cls, distribution: Distribution) -> Algorithm:
        return StandardAlgorithm()


class StandardAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> float:
        n = distribution.n
        return sum([random.standard_normal()**2 for _ in range(n)])


class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        n = distribution.n
        return n


class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        n = distribution.n
        return 2 * n
