"""File containing the necessary tools for finding the correct permutation groups to apply in the sense
of the totally antisymmetric tensor. Used for the creation of PauliStrings for the double electron fermionic hamiltonian


"""

import numpy as np
from typing import Dict, List


def _determine_ordering(index_dict: Dict[str, int]):
    """Function which takes a dictionary with
    <key> ... name of the index
    <value> ... the value of the index
    and returns a List[int]
    where the ith number of the list signifies the
    order (starting with 1) of the ith index of the dict compared to all other indices.

    As an example, if list[3] = 2, it means that the 4th(zero-indexed) entry of the dict
    has the second-largest value.
    """
    sorted_dict = _sort_dict_by_values(index_dict)
    # Notice that this index_list [a, b, c, ... ] corresponds to the assignments alpha = a, beta = b, gamma = c, ...
    # within the paper
    ordering = []
    # since insertion order is preserved, can convert the keyset to list to do get position values:
    index_list = list(index_dict.keys())
    # iterate over indexes
    for elem in index_list:
        # find index in sorted list, add <position> to ordering
        position = list(sorted_dict.keys()).index(elem)
        ordering.append(position)

    return ordering


def _levi_civita_epsilon(ordering: List[int]):
    """For a list of integers representing an ordering, calculate the Levi-Civita epsilon
    https://en.wikipedia.org/wiki/Levi-Civita_symbol. Required to calculate the sign.
    Note that in the many-bodies paper it is simply denoted by epsilon.
    """
    # if duplicates are in the list, i.e. when the set representation has smaller cardinality, return 0
    if len(set(ordering)) != len(ordering):
        return 0
    elif _is_even_permutation(ordering):
        return 1
    else:
        return -1


def _is_even_permutation(ordering: List[int]):
    """Function which takes a list of integers and determines whether it is an even permuatation of
    the list containing the same elements sorted in ascending order"""
    # Taken from https://python-forum.io/thread-11603.html
    perm_count = 0
    for i, num in enumerate(ordering, start=1):
        # count how many are smaller
        perm_count += sum(num > num2 for num2 in ordering[i:])
    return not perm_count % 2


def _sort_dict_by_values(dict: Dict[str, int]):
    # Taken from https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value/613218#613218
    return {key: value for key, value in sorted(dict.items(), key=lambda item: item[1])}
