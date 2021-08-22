from ..sample import AnySample, SampleACF
from final_class import final

__all__ = [
    'ACF',
]


@final
class ACF:
    def __new__(cls, sample: AnySample, f: int):
        return SampleACF(sample, f)
