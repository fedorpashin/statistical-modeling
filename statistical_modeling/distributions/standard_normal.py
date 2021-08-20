from .base import ContinuousDistribution, ContinuousDistributionAlgorithm, DistributionMean, DistributionVariance

from math import sqrt, log, cos, pi
from numpy import random


class Distribution(ContinuousDistribution):
    __μ: float = 0
    __σ: float = 1

    @property
    def µ(self) -> float:
        return self.__μ

    @property
    def σ(self) -> float:
        return self.__σ


class Algorithm(ContinuousDistributionAlgorithm[Distribution]):
    @staticmethod
    def default(distribution: Distribution) -> 'Algorithm':
        return BoxMullerAlgorithm()


class BoxMullerAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> float:
        return sqrt(-2 * log(random.uniform())) * cos(2 * pi * random.uniform())


class CentralLimitTheoremAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> float:
        return sum([random.uniform() for _ in range(12)]) - 6


class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        µ = distribution.µ
        return µ


class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        σ = distribution.σ
        return σ**2
