import statistical_modeling as sm

from typing import Final

import unittest
from numpy.testing import assert_equal


class TestACF(unittest.TestCase):
    def test(self):
        s: Final = sm.Sample([1, 2, 3])
        f: Final = 3
        assert_equal(
            sm.ACF(s, f),
            sm.SampleACF(s, f)
        )
