from ..sample import Sample, SampleMean
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
class Mean(metaclass=multimeta):
    def __new__(cls, sample: Sample):
        return SampleMean(sample)

    def __new__(cls, distribution: binomial.Distribution):
        return binomial.Mean(distribution)

    def __new__(cls, distribution: chi_square.Distribution):
        return chi_square.Mean(distribution)

    def __new__(cls, distribution: continuous_uniform.Distribution):
        return continuous_uniform.Mean(distribution)

    def __new__(cls, distribution: discrete_uniform.Distribution):
        return discrete_uniform.Mean(distribution)

    def __new__(cls, distribution: exponential.Distribution):
        return exponential.Mean(distribution)

    def __new__(cls, distribution: geometric.Distribution):
        return geometric.Mean(distribution)

    def __new__(cls, distribution: logarithmic.Distribution):
        return logarithmic.Mean(distribution)

    def __new__(cls, distribution: poisson.Distribution):
        return poisson.Mean(distribution)

    def __new__(cls, distribution: standard_normal.Distribution):
        return standard_normal.Mean(distribution)

    def __new__(cls, distribution: student.Distribution):
        return student.Mean(distribution)
