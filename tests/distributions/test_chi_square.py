from statistical_modeling.distributions.chi_square import (
    Distribution,
    Mean,
    Variance
)

from typing import Final

import unittest


class TestChiSquare(unittest.TestCase):
    d: Final = Distribution(10)

    def test_distribution(self):
        self.assertEqual(self.d.n, 10)

    def test_mean(self):
        self.assertAlmostEqual(
            Mean(self.d),
            10
        )

    def test_variance(self):
        self.assertAlmostEqual(
            Variance(self.d),
            20
        )
