from .base import (
    ContinuousDistribution,
    ContinuousDistributionAlgorithm,
    DistributionMean,
    DistributionVariance,
)
from dataclasses import dataclass
from final_class import final
from overrides import overrides

from math import log
from numpy import random


@final
@dataclass(frozen=True)
class Distribution(ContinuousDistribution):
    β: float


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
        β = distribution.β
        return β * log(random.uniform())


@final
class Mean(DistributionMean):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Mean':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        β = distribution.β
        return β


@final
class Variance(DistributionVariance):
    __distribution: Distribution

    def __new__(cls, distribution: Distribution) -> 'Variance':
        return super().__new__(cls, cls.__value(distribution))

    @staticmethod
    def __value(distribution: Distribution) -> float:
        β = distribution.β
        return β**2
