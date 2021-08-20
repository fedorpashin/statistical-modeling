from .base import ContinuousDistribution, ContinuousDistributionAlgorithm, DistributionMean, DistributionVariance

from math import sqrt, inf
from numpy import random


class Distribution(ContinuousDistribution):
    __n: float

    def __init__(self, n: int):
        assert n > 0
        self.__n = n

    @property
    def n(self) -> float:
        return self.__n


class Algorithm(ContinuousDistributionAlgorithm[Distribution]):
    @staticmethod
    def default(distribution: Distribution) -> 'Algorithm':
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
