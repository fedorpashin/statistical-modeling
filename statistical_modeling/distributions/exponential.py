from .base import ContinuousDistribution, ContinuousDistributionAlgorithm, DistributionMean, DistributionVariance
from dataclasses import dataclass

from math import log
from numpy import random


@dataclass(frozen=True)
class Distribution(ContinuousDistribution):
    β: float


class Algorithm(ContinuousDistributionAlgorithm[Distribution]):
    pass


class DefaultAlgorithm:
    def __new__(cls, distribution: Distribution) -> Algorithm:
        return StandardAlgorithm()


class StandardAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> float:
        β = distribution.β
        return β * log(random.uniform())


class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        β = distribution.β
        return β


class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        β = distribution.β
        return β**2
