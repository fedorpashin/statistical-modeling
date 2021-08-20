from ..distributions import (binomial,
                             chi_square,
                             continuous_uniform,
                             discrete_uniform,
                             exponential,
                             geometric,
                             logarithmic,
                             poisson,
                             standard_normal,
                             student)

from multimethod import multimeta


class DefaultAlgorithm(metaclass=multimeta):
    def __new__(cls, distribution: binomial.Distribution):
        return binomial.DefaultAlgorithm(distribution)

    def __new__(cls, distribution: chi_square.Distribution):
        return chi_square.DefaultAlgorithm(distribution)

    def __new__(cls, distribution: continuous_uniform.Distribution):
        return continuous_uniform.DefaultAlgorithm(distribution)

    def __new__(cls, distribution: discrete_uniform.Distribution):
        return discrete_uniform.DefaultAlgorithm(distribution)

    def __new__(cls, distribution: exponential.Distribution):
        return exponential.DefaultAlgorithm(distribution)

    def __new__(cls, distribution: geometric.Distribution):
        return geometric.DefaultAlgorithm(distribution)

    def __new__(cls, distribution: logarithmic.Distribution):
        return logarithmic.DefaultAlgorithm(distribution)

    def __new__(cls, distribution: poisson.Distribution):
        return poisson.DefaultAlgorithm(distribution)

    def __new__(cls, distribution: standard_normal.Distribution):
        return standard_normal.DefaultAlgorithm(distribution)

    def __new__(cls, distribution: student.Distribution):
        return student.DefaultAlgorithm(distribution)
