"""Testing class for interaction group"""

import unittest

from src.util.InteractionGroup import *

from itertools import permutations

class TestInteractionGroup(unittest.TestCase):
    def test_determine_interaction_group_all_perms(self):
        testlist = [1,2,3,4]
        for perm in permutations(testlist):
            i,j,k,l = perm
            # filter those where i > j or l > k
            if i > j or l > k:
                continue
            # without loss of generality, we can assume that i has to be the smaller index
            if l < i:
                continue
            # print(perm)
            self.assertTrue(determine_interaction_group(i,j,k,l) in [InteractionGroup.FIRST, InteractionGroup.SECOND, InteractionGroup.THIRD])


    def test_determine_interaction_first_group(self):
        testlist = [3, 6, 11, 8]
        i,j,k,l = testlist
        self.assertEqual(determine_interaction_group(i,j,k,l), InteractionGroup.FIRST)


    def test_determine_interaction_second_group(self):
        testlist = [3, 7, 8, 5]
        i,j,k,l = testlist
        self.assertEqual(determine_interaction_group(i,j,k,l), InteractionGroup.SECOND)

    def test_determine_interaction_third_group(self):
        testlist = [3, 12, 6, 4]
        i,j,k,l = testlist
        self.assertEqual(determine_interaction_group(i,j,k,l), InteractionGroup.THIRD)