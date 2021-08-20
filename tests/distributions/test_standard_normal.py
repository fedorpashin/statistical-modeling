from statistical_modeling.distributions.standard_normal import Distribution, Mean, Variance

from typing import Final

import unittest
from numpy.testing import assert_equal, assert_allclose


class TestStandardNormal(unittest.TestCase):
    d: Final = Distribution()

    def test_distribution(self):
        assert_equal(self.d.µ, 0)
        assert_equal(self.d.σ, 1)

    def test_mean(self):
        assert_allclose(
            Mean(self.d),
            0
        )

    def test_variance(self):
        assert_allclose(
            Variance(self.d),
            1
        )
