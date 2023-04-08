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

            pauli_identity_string = _identity_string_builder(num_qubits)
            pauli_Zi_string = _pauli_single_Z_string_builder(num_qubits, i)
            pauli_Zj_string = _pauli_single_Z_string_builder(num_qubits, j)
            pauli_double_Z_string = _pauli_double_Z_string_builder(num_qubits, i, j)

            # append all strings to the sum
            pauli_list.extend([pauli_identity_string, pauli_Zi_string, pauli_Zj_string, pauli_double_Z_string])

    # finally create the SparsePauliOp
    return SparsePauliOp(pauli_list, coeffs=np.array(coeffs))


def generate_offdiagonal_paulis(num_qubits: int, weights: NDArray[Shape['4'], Float]):
    """Function which creates the Sum of Pauli strings which results from the mapping of the offdiagonal elements
    of the two electron fermionic interaction Hamiltonian.

    Notice that there are three distinct groups of interaction types that one has to consider:
    I   ... i < j < l < k
    II  ... i < l < j < k
    III ... i < l < k < j
    """

    # j and k iterate over whole range, whereas i and l only up to j and k respectively
    for j in range(num_qubits):
        for i in range(j):
            if i == j:
                # is equal to a diagonal element
                continue
            for k in range(num_qubits):
                for l in range(k):
                    if l == k:
                        # also equal to a diagonal element (up to index swapping)
                        continue
                    # use method shown in the paper to determine the case
                    index_dict = {'i': i, 'j': j, 'l': l, 'k': k}

    # create empty lists for paulis and coeffs
    pass


def _identity_string_builder(num_qubits: int):
    # pass none to get identity string back
    return pauli_string_from_dict(num_qubits, None)


def _pauli_single_Z_string_builder(num_qubits: int, i: int):
    return pauli_string_from_dict(num_qubits, {i: 'Z'})


def _pauli_double_Z_string_builder(num_qubits: int, i: int, j: int):
    return pauli_string_from_dict(num_qubits, {i: 'Z', j: 'Z'})
