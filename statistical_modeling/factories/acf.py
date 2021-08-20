from ..sample import Sample, SampleACF
from final_class import final


@final
class ACF:
    def __new__(cls, sample: Sample, f: int):
        return SampleACF(sample, f)
