from ..sample import Sample, SampleVariance
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
from final_class import final


@final
class Variance(metaclass=multimeta):
    def __new__(cls, sample: Sample):
        return SampleVariance(sample)

    def __new__(cls, distribution: binomial.Distribution):
        return binomial.Variance(distribution)

    def __new__(cls, distribution: chi_square.Distribution):
        return chi_square.Variance(distribution)

    def __new__(cls, distribution: continuous_uniform.Distribution):
        return continuous_uniform.Variance(distribution)

    def __new__(cls, distribution: discrete_uniform.Distribution):
        return discrete_uniform.Variance(distribution)

    def __new__(cls, distribution: exponential.Distribution):
        return exponential.Variance(distribution)

    def __new__(cls, distribution: geometric.Distribution):
        return geometric.Variance(distribution)

    def __new__(cls, distribution: logarithmic.Distribution):
        return logarithmic.Variance(distribution)

    def __new__(cls, distribution: poisson.Distribution):
        return poisson.Variance(distribution)

    def __new__(cls, distribution: standard_normal.Distribution):
        return standard_normal.Variance(distribution)

    def __new__(cls, distribution: student.Distribution):
        return student.Variance(distribution)
