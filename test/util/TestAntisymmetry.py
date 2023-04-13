import unittest

from src.util.Antisymmetry import levi_civita_epsilon, determine_ordering


class TestAntisymmetriy(unittest.TestCase):
    """Tests the Antisymmetry Module functions"""

    def test_levi_civita_epsilon_0perm(self):
        testordering = [1, 2, 3, 4, 5]
        correct_res = 1
        self.assertEqual(levi_civita_epsilon(testordering), correct_res)

    def test_levi_civita_epsilon_1perm(self):
        testordering = [2, 1, 3, 4, 5, 6]
        correct_res = -1
        self.assertEqual(levi_civita_epsilon(testordering), correct_res)

    def test_levi_civita_epsilon_3perm(self):
        testordering = [2, 1, 5, 4, 3, 7, 6]
        correct_res = -1
        self.assertEqual(levi_civita_epsilon(testordering), correct_res)

    def test_levi_civita_epsilon_duplicate(self):
        testordering = [2, 2, 1, 3]
        correct_res = 0
        self.assertEqual(levi_civita_epsilon(testordering), correct_res)

    def test_determine_ordering_4entries(self):
        testdic = {'i': 2, 'j': 4, 'k': 7, 'l': 11}
        correct_res = [0, 1, 2, 3]
        self.assertEqual(determine_ordering(testdic), correct_res)

    def test_determine_ordering_4entries_unordered(self):
        testdic = {'i': 6, 'j': 7, 'k': 2, 'l': 3}
        correct_res = [2, 3, 0, 1]
        self.assertEqual(determine_ordering(testdic), correct_res)

    def test_determine_ordering_5entries_unordered(self):
        testdic = {'i': 6, 'j': 7, 'k': 2, 'l': 3, 'm': 5}
        correct_res = [3, 4, 0, 1, 2]
        self.assertEqual(determine_ordering(testdic), correct_res)
