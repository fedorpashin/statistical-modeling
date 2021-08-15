from statistical_modeling.sample import Sample

import unittest
from numpy.testing import assert_allclose


class TestSample(unittest.TestCase):
    def test_quantities(self):
        s = Sample([458, -184, 59, 325, -530])
        with self.subTest():
            assert_allclose(s.Mean, 25.6)
        with self.subTest():
            assert_allclose(s.Variance, 126069.84)


if __name__ == '__main__':
    unittest.main()
