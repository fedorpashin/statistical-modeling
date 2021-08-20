from statistical_modeling.distributions.discrete_uniform import Distribution, Mean, Variance

from typing import Final

import unittest
from numpy.testing import assert_equal, assert_allclose


class TestDiscreteUniform(unittest.TestCase):
    d: Final = Distribution(1, 100)

    def test_distribution(self):
        assert_equal(self.d.a, 1)
        assert_equal(self.d.b, 100)

    def test_mean(self):
        assert_allclose(
            Mean(self.d),
            50.5
        )

    def test_variance(self):
        assert_allclose(
            Variance(self.d),
            833.25
        )
