from ..sample import AnySample, SampleACF
from final_class import final


@final
class ACF:
    def __new__(cls, sample: AnySample, f: int):
        return SampleACF(sample, f)
