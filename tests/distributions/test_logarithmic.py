from statistical_modeling.distributions.logarithmic import Distribution, Mean, Variance

from typing import Final

import unittest
from numpy.testing import assert_equal, assert_allclose


class TestLogarithmic(unittest.TestCase):
    d: Final = Distribution(0.5)

    def test_distribution(self):
        assert_equal(self.d.q, 0.5)
        assert_equal(self.d.p, 0.5)

    def test_mean(self):
        assert_allclose(
            Mean(self.d),
            1.44270,
            rtol=1e-4
        )

    def test_variance(self):
        assert_allclose(
            Variance(self.d),
            0.80402,
            rtol=1e-4
        )
