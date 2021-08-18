from .base import ContinuousDistribution, ContinuousDistributionAlgorithm, DistributionMean, DistributionVariance
from abc import abstractmethod

from math import log
from numpy import random


class Distribution(ContinuousDistribution):
    __β: float

    def __init__(self, β: float):
        self.__β = β

    @property
    def β(self) -> float:
        return self.__β


class Algorithm(ContinuousDistributionAlgorithm['Algorithm']):
    @abstractmethod
    def value(self, distribution: Distribution) -> float:
        pass

    @staticmethod
    def default(distribution: Distribution) -> 'Algorithm':
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
