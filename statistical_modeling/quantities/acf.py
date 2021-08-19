from ..sample import Sample, SampleACF


class ACF:
    def __new__(cls, sample: Sample, f: int):
        return SampleACF(sample, f)
