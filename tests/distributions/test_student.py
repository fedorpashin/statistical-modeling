from statistical_modeling.distributions.student import Distribution, Mean, Variance

from typing import Final

import unittest


class TestStudent(unittest.TestCase):
    d: Final = Distribution(10)

    def test_distribution(self):
        self.assertEqual(self.d.n, 10)

    def test_mean(self):
        self.assertAlmostEqual(
            Mean(self.d),
            0
        )

    def test_variance(self):
        self.assertAlmostEqual(
            Variance(self.d),
            1.25
        )
