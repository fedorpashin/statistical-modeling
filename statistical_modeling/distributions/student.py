from .base import ContinuousDistribution, ContinuousDistributionAlgorithm, DistributionMean, DistributionVariance
from dataclasses import dataclass

from math import sqrt, inf
from numpy import random


@dataclass(frozen=True)
class Distribution(ContinuousDistribution):
    n: float

    def __post__init__(self):
        assert self.n > 0


class Algorithm(ContinuousDistributionAlgorithm[Distribution]):
    pass


class DefaultAlgorithm:
    def __new__(cls, distribution: Distribution) -> Algorithm:
        return StandardAlgorithm()


class StandardAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> float:
        n = distribution.n
        return random.uniform() / sqrt(random.chisquare(n) / n)


class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        return 0


class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        n = distribution.n
        if n > 2:
            return n / (n - 2)
        else:
            return inf
