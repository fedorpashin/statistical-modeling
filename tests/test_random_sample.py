from statistical_modeling.distributions import (
    geometric, logarithmic
)
from statistical_modeling import RandomSample

from typing import Final

import unittest


class TestRandomSample(unittest.TestCase):
    d: Final = geometric.Distribution(0.1)

    def test_n(self):
        self.assertEqual(
            RandomSample(3, self.d).n,
            3
        )

    def test_algorithm(self):
        a: Final = geometric.DefaultAlgorithm(self.d)
        self.assertEqual(
            RandomSample(3, self.d, a).algorithm,
            a
        )

    def test_default_algorithm(self):
        self.assertEqual(
            type(RandomSample(3, self.d).algorithm),
            type(geometric.DefaultAlgorithm(self.d))
        )

    def test_invalid_combination(self):
        with self.assertRaises(AttributeError):
            RandomSample(3, self.d, logarithmic.CumulativeAlgorithm()).x # noqa
