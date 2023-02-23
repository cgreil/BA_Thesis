from __future__ import annotations
from qiskit.quantum_info.operators import Pauli, Operator
from typing import Union


class PauliString:
    """Class to represent Pauli Strings, i.e. kronecker products of Pauli Operators with a weight multiplied to it"""

    weight: float
    # tuple of single qubit pauli operators
    pauli_tuple: tuple

    # Pauli List is a tuple containing either Pauli objects or LinearPauliCombination objects
    def __init__(self, pauli_tuple: tuple[Pauli | LinearPauliCombination], weight):
        self.weight = weight
        self.pauli_tuple = pauli_tuple

    def __len__(self):
        return len(self.pauli_tuple)


class LinearPauliCombination:
    """Datastructure to represent the linear combination of two Pauli operators within the PauliString datastructure"""
    pauli_op_1: Pauli
    pauli_op_2: Pauli

    def __init__(self, pauli1: Pauli, pauli2: Pauli):
        self.pauli_op_1 = pauli1
        self.pauli_op_2 = pauli2


class LinearPauliStringCombination:
    """Datastructure to allow a linear combination of two full Pauli Strings"""
    pauli_string_1: PauliString
    pauli_string_2: PauliString

    def __init__(self, pauli_string_1, pauli_string_2):
        self.pauli_string_1 = pauli_string_1
        self.pauli_string_2 = pauli_string_2
