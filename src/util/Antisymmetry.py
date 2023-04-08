"""File containing the necessary tools for finding the correct permutation groups to apply in the sense
of the totally antisymmetric tensor. Used for the creation of PauliStrings for the double electron fermionic hamiltonian


"""

import numpy as np
from typing import Dict


def determine_ordering(index_dict: Dict[str, int]):
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
    size_list = []
    # since insertion order is preserved, can convert the keyset to list to do get position values:
    index_list = list(index_dict.keys())
    # iterate over indexes
    for elem in index_list:
        # find index in sorted list, add position to size_list
        position = list(sorted_dict.keys()).index(elem)
        size_list.append(position)

    return size_list


def _sort_dict_by_values(dict: Dict[str, int]):
    # Taken from https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value/613218#613218
    return {key: value for key, value in sorted(dict.items(), key=lambda item: item[1])}
