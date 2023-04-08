from enum import Enum


def determine_group(i: int, j: int, k: int, l: int):
    """Function to determine the group of interaction pairs from the order of indices"""
    if i < k < j < l:
        return IndexGroup.FIRST
    elif i < l < j < k:
        return IndexGroup.SECOND
    elif i < l < k < j:
        return IndexGroup.THIRD
    else:
        raise ValueError("Erroneous order of indices")


class IndexGroup(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3