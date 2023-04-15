from enum import Enum


def determine_interaction_group(i: int, j: int, k: int, l: int):
    """Function to determine the group of interaction pairs from the order of indices"""
    if i < j < l < k:
        return InteractionGroup.FIRST
    elif i < l < j < k:
        return InteractionGroup.SECOND
    elif i < l < k < j:
        return InteractionGroup.THIRD
    else:
        raise ValueError("Erroneous order of indices")


class InteractionGroup(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3