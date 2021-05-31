from .base import Distribution, DistributionAlgorithm
from abc import abstractmethod
from typing import Callable

from numpy import random
import math

from functools import lru_cache, cached_property

__all__ = ['DiscreteUniformDistribution',
           'BinomialDistribution',
           'GeometricDistribution',
           'PoissonDistribution',
           'LogarithmicDistribution']


def _cumulative(p_of: Callable[[int], float], x: int = 0) -> int:
    m = random.uniform()
    while True:
        m -= p_of(x)
        if m < 0:
            break
        x += 1
    return x


class DiscreteDistribution(Distribution):
    class Algorithm(DistributionAlgorithm):
        pass

    @abstractmethod
    def value(self, algorithm: Algorithm) -> int:
        pass


class DiscreteUniformDistribution(DiscreteDistribution):
    class Algorithm(DistributionAlgorithm):
        standard = 1

    @property
    def default_algorithm(self) -> Algorithm:
        return self.Algorithm.standard

    _a: int
    _b: int

    def __init__(self, a: int, b: int):
        self._a = a
        self._b = b

    def value(self, algorithm: Algorithm = default_algorithm) -> int:  # type: ignore
        return math.floor(self.n * random.uniform() + self._a)

    @property
    def a(self) -> int:
        return self._a

    @property
    def b(self) -> int:
        return self._b

    @cached_property
    def n(self) -> int:
        return self._b - self._a + 1

    @cached_property
    def E(self) -> float:
        return (self._a + self._b) / 2

    @cached_property
    def D(self) -> float:
        return (self.n**2 - 1) / 12


class BinomialDistribution(DiscreteDistribution):
    class Algorithm(DistributionAlgorithm):
        cumulative = 1
        normal_approximation = 2

    @property
    def default_algorithm(self) -> Algorithm:
        return self.Algorithm.cumulative

    _n: int
    _p: float
    _threshold: int = 100

    def __init__(self, n: int, p: float):
        assert n > 0
        assert 0 <= p <= 1
        self._n = n
        self._p = p

    def value(self, algorithm: Algorithm = default_algorithm) -> int:  # type: ignore
        n = self._n
        p = self._p
        q = self._q

        if (algorithm == self.Algorithm.cumulative
                or algorithm is None and n < self._threshold):
            @lru_cache()
            def p_of(r):
                if r == 0:
                    return q**n
                else:
                    return p_of(r - 1) * p * (n - r + 1) / (r * q)

            return _cumulative(p_of)
        else:
            return round(random.normal(n * p, math.sqrt(n * p * q)) + 0.5)

    @property
    def n(self) -> int:
        return self._n

    @property
    def p(self) -> float:
        return self._p

    @property
    def threshold(self) -> int:
        return self._threshold

    @cached_property
    def _q(self) -> float:
        return 1 - self._p

    @cached_property
    def E(self) -> float:
        return self._n * self._p

    @cached_property
    def D(self) -> float:
        return self._n * self._p * self._q


class GeometricDistribution(DiscreteDistribution):
    class Algorithm(DistributionAlgorithm):
        cumulative = 1
        forward = 2
        improved_cumulative = 3

    @property
    def default_algorithm(self) -> Algorithm:
        return self.Algorithm.cumulative

    _p: float

    def __init__(self, p: float):
        assert 0 <= p <= 1
        self._p = p

    def value(self, algorithm: Algorithm = default_algorithm) -> int:  # type: ignore
        if algorithm == self.Algorithm.cumulative:
            @lru_cache
            def p_of(x):
                if x == 0:
                    return self._p
                else:
                    return p_of(x - 1) * self._q

            return _cumulative(p_of)
        elif algorithm == self.Algorithm.forward:
            x = 0
            while random.uniform() > self._p:
                x += 1
            return x
        else:
            M = random.uniform()
            return round(math.log(M) / math.log(self._q)) + 1

    @property
    def p(self) -> float:
        return self._p

    @cached_property
    def _q(self) -> float:
        return 1 - self._p

    @cached_property
    def E(self) -> float:
        return 1 / self._p

    @cached_property
    def D(self) -> float:
        return self._q / self._p**2


class PoissonDistribution(DiscreteDistribution):
    class Algorithm(DistributionAlgorithm):
        cumulative = 1
        multiplication = 2
        normal_approximation = 3

    @property
    def default_algorithm(self) -> Algorithm:
        return self.Algorithm.cumulative

    _μ: float
    _threshold: int = 88

    def __init__(self, μ: float):
        assert μ > 0
        self._μ = μ

    def value(self, algorithm: Algorithm = default_algorithm) -> int:  # type: ignore
        μ = self._μ
        if (algorithm == self.Algorithm.cumulative
                or algorithm is None and μ < self._threshold):
            @lru_cache
            def p_of(x):
                if x == 0:
                    return math.exp(-μ)
                else:
                    return p_of(x - 1) * μ / x

            return _cumulative(p_of)
        elif algorithm == self.Algorithm.multiplication:
            x = 0
            m = 1
            while True:
                m *= random.uniform()
                if m < math.exp(-μ):
                    break
                else:
                    x += 1
            return x
        else:
            return round(random.normal(μ, μ))

    @property
    def µ(self) -> float:
        return self._μ

    @property
    def threshold(self) -> int:
        return self._threshold

    @cached_property
    def E(self) -> float:
        return self._μ

    @cached_property
    def D(self) -> float:
        return self._μ


class LogarithmicDistribution(DiscreteDistribution):
    class Algorithm(DistributionAlgorithm):
        standard = 1

    @property
    def default_algorithm(self) -> Algorithm:
        return self.Algorithm.standard

    _q: float

    def __init__(self, q: float):
        assert 0 <= q <= 1
        self._q = q

    def value(self, algorithm: Algorithm = default_algorithm) -> int:  # type: ignore
        @lru_cache
        def p_of(x):
            if x == 1:
                return -self._α * self._q
            else:
                return self._q * (x - 1) / x

        return _cumulative(p_of, x=1)

    @property
    def q(self) -> float:
        return self._q

    @cached_property
    def _α(self) -> float:
        return 1 / math.log(self._p)

    @cached_property
    def _p(self) -> float:
        return 1 - self.q

    @cached_property
    def E(self) -> float:
        return -self._α * self.q / self._p

    @cached_property
    def D(self) -> float:
        return -self._α * self.q * (1 + self._α * self.q) / self._p**2
