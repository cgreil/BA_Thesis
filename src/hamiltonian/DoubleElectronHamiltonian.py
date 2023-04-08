"""Module which enables the generation of the double excitation fermionic hamiltonian in the second quantization using
Jordan-Wigner transformation.
"""

import numpy as np
from nptyping import NDArray, Shape, Float

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

from ..util.PauliStringCreation import pauli_string_from_dict


def generate_pauli_sum(num_qubits: int, weights: NDArray[Shape['4'], Float]):
    pass


def _generate_diagonal_paulis(num_qubits: int, weights: NDArray[Shape['4'], Float]):
    """Function which generates the Sum of Pauli strings which results off the mapping
    of the diagonal elements of the Two electron fermionic interaction Hamiltonian.
    Diagonal in this sense refers to the situation that for an index set ijkl, pairwaise indices
    are equal, i.e. i = k and j = l.

    The created linear combination of Strings will be returned as SparsePauliOp."""

    # initialize pauli and coeff lists
    pauli_list = []
    coeffs = []
    coeff_index = 0

    for j in range(num_qubits):
        for i in range(j):
            # retrieve correct coefficient
            coeffs[coeff_index] = (1 / 4) * weights[i, j, i, j]
            coeff_index = coeff_index + 1


def _identity_string_builder(num_qubits: int):
    # pass none to get identity string back
    return pauli_string_from_dict(num_qubits, None)


def _pauli_single_z_string_builder(num_qubits: int, i: int):
    return pauli_string_from_dict(num_qubits, {i: 'Z'})


def _pauli_double_z_string_builder(num_qubits: int, i: int, j: int):
    return pauli_string_from_dict(num_qubits, {i: 'Z', j: 'Z'})
