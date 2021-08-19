from statistical_modeling.sample import Sample, SampleMean, SampleVariance, SampleACF

from typing import Final

import unittest
from numpy.testing import assert_equal, assert_allclose


class TestSample(unittest.TestCase):
    s: Final = Sample([458, -184, 59, 325, -530])

    def test_sample(self):
        assert_equal(self.s.x, [458, -184, 59, 325, -530])
        assert_equal(self.s.n, 5)

    def test_mean(self):
        assert_allclose(
            SampleMean(self.s),
            25.6
        )

    def test_variance(self):
        assert_allclose(
            SampleVariance(self.s),
            126069.84
        )

    def test_acf(self):
        assert_allclose(
            SampleACF(self.s, 3),
            0.39012,
            rtol=10e-4
        )
