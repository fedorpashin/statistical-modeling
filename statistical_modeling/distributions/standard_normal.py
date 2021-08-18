from .base import ContinuousDistribution, ContinuousDistributionAlgorithm
from abc import abstractmethod

from math import sqrt, log, cos, pi
from numpy import random

from functools import cached_property


class Distribution(ContinuousDistribution):
    __μ: float = 0
    __σ: float = 1

    @property
    def µ(self) -> float:
        return self.__μ

    @property
    def σ(self) -> float:
        return self.__σ


class Algorithm(ContinuousDistributionAlgorithm):
    @classmethod
    def default(cls, distribution: Distribution) -> 'Algorithm':
        return BoxMullerAlgorithm()

    @abstractmethod
    def value(self, distribution: Distribution) -> float:
        pass


class BoxMullerAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> float:
        return sqrt(-2 * log(random.uniform())) * cos(2 * pi * random.uniform())


class CentralLimitTheoremAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> float:
        return sum([random.uniform() for _ in range(12)]) - 6


class Mean:
    __distribution: Distribution

    @cached_property
    def __value(self) -> float:
        µ = self.__distribution.µ
        return µ


class Variance:
    __distribution: Distribution

    @cached_property
    def __value(self) -> float:
        σ = self.__distribution.σ
        return σ**2
