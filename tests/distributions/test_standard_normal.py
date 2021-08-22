from statistical_modeling.distributions.standard_normal import (
    Distribution,
    Mean,
    Variance
)

from typing import Final

import unittest


class TestStandardNormal(unittest.TestCase):
    d: Final = Distribution()

    def test_distribution(self):
        self.assertEqual(self.d.µ, 0)
        self.assertEqual(self.d.σ, 1)

    def test_mean(self):
        self.assertAlmostEqual(
            Mean(self.d),
            0
        )

    def test_variance(self):
        self.assertAlmostEqual(
            Variance(self.d),
            1
        )
