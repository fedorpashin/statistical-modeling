from .base import DiscreteDistribution, DiscreteDistributionAlgorithm
from abc import abstractmethod
from generic.multimethod import has_multimethods, multimethod

from .cumulative import cumulative
from math import exp
from numpy import random

from functools import cached_property, lru_cache


class Distribution(DiscreteDistribution):
    __μ: float

    def __init__(self, μ: float):
        assert μ > 0
        self.__μ = μ

    @property
    def µ(self) -> float:
        return self.__μ


@has_multimethods
class Algorithm(DiscreteDistributionAlgorithm['Algorithm']):
    @classmethod
    @multimethod()
    def default(cls, distribution: Distribution) -> 'Algorithm':
        return cls.default(88)

    @classmethod # noqa
    @default.register(Distribution, int)
    def default(cls, distribution: Distribution, threshold: int) -> 'Algorithm':
        µ = distribution.µ
        if µ < threshold:
            return CumulativeAlgorithm()
        else:
            return NormalApproximationAlgorithm()

    @abstractmethod
    def value(self, distribution: Distribution) -> int:
        pass


class CumulativeAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> int:
        µ = distribution.µ

        @lru_cache
        def p_of(x):
            if x == 0:
                return exp(-µ)
            else:
                return p_of(x - 1) * µ / x

        return cumulative(p_of)


class MultiplicationAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> int:
        µ = distribution.µ
        x = 0
        m = 1.0
        while True:
            m *= random.uniform()
            if m < exp(-µ):
                break
            else:
                x += 1
        return x


class NormalApproximationAlgorithm(Algorithm):
    def value(self, distribution: Distribution) -> int:
        µ = distribution.µ
        return round(random.normal(µ, µ))


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
        µ = self.__distribution.µ
        return µ
