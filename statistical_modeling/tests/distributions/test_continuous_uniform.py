from statistical_modeling.distributions.continuous_uniform import Distribution, Mean, Variance

from typing import Final

import unittest
from numpy.testing import assert_equal, assert_allclose


class TestContinuousUniform(unittest.TestCase):
    d: Final = Distribution(0, 1)

    def test_distribution(self):
        assert_equal(self.d.a, 0)
        assert_equal(self.d.b, 1)

    def test_mean(self):
        assert_allclose(
            Mean(self.d),
            0.5
        )

    def test_variance(self):
        assert_allclose(
            Variance(self.d),
            0.08333,
            rtol=1e-04
        )
