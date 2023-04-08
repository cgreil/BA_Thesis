"""Module that enables the generaton of the single electron interaction hamiltonian in the second
quantization using the Jordan-Wigner transformation.

The single particle interaction Hamiltonian H1 can be described as
$$H_1 = \sum_i h_{ii} a_i^{\dagger} a_i + \sum_{i < j} h_{ij} (a_i^{\dagger} a_j + a_j^{\dagger} a_i)$$
where $a, a^\dagger$ denote the annihilation and creation operators, respectively.

The top level generation function is generate_pauli_sum, which returns a SparsePauliOp
(see https://qiskit.org/documentation/stubs/qiskit.opflow.primitive_ops.PauliSumOp.html#qiskit.opflow.primitive_ops.PauliSumOp)
which lets one combine weights with sparse Pauli Operators, where Pauli Operators can be Tensor products of
Pauli Gates.
"""

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp


def generate_pauli_sum(num_qubits: int, weights: np.ndarray[float, float]):
    """Function which returns the full PauliSumOp for the whole single electron fermionic hamiltonian."""
    diagonal_sparse_paulis = _generate_diagonal_paulis(num_qubits, weights)
    offdiagonal_sparse_paulis = _generate_offdiagonal_paulis(num_qubits, weights)

    complete_sparse_paulis = diagonal_sparse_paulis.compose(offdiagonal_sparse_paulis)

    return PauliSumOp(complete_sparse_paulis)

def _generate_diagonal_paulis(num_qubits: int, weights: np.ndarray[float, float]):
    """Generates the sparse pauli operator resulting from the diagonal elements of the hamiltonian"""

    # initialize empty pauli list and coeff array
    pauli_list = []
    coeffs = []

    for i in range(num_qubits):
        # store coeff
        coeffs[i] = 1 / 2 * weights[i, i]

        # identity pauli string
        pauli_I_list = ['I' for _ in range(num_qubits)]
        pauli_list.append(pauli_I_list)

        # Z pauli string
        pauli_Z_list = ['I' for _ in range((i - 1))]
        pauli_Z_list.append('Z')
        pauli_Z_list.extend(['I' for _ in range(i + 1, num_qubits)])
        pauli_list.append(pauli_Z_list)

    # finally create the Sparse Pauli Operator
    return SparsePauliOp(pauli_list, coeffs=np.array(coeffs))


def _generate_offdiagonal_paulis(num_qubits: int, weights: np.ndarray[float, float]):
    # initialize the pauli list and coeff list
    pauli_list = []
    coeffs = []

    coeff_index = 0

    # iterate over combinations where i < j
    for j in range(num_qubits):
        for i in range(j):
            # add coefficient
            coeffs[coeff_index] = weights[i, j]
            coeff_index = coeff_index + 1

            pauli_X_string = _pauli_X_string_builder(i, j, num_qubits)
            pauli_Y_string = _pauli_Y_string_builder(i, j, num_qubits)
            # add strings to the list of all
            pauli_list.append(pauli_X_string)
            pauli_list.append(pauli_Y_string)

    # create the Sparse Pauli Operator
    return SparsePauliOp(pauli_list, coeffs=np.array(coeffs))


def _pauli_Y_string_builder(i, j, num_qubits):
    """Creates a string corresponding to a transform on the Hilbert space for num_qubits qubits, with transformation
        Y iff qubit index k is i or j
        Z iff qubit index i < k < j
        I otherwise
    """

    # all qubits below I have identity transformation
    pauli_Y_list = ['I' for _ in range(i - 1)]
    # apply X on qubit i
    pauli_Y_list.append('Y')
    # between i and j, apply Z transformations
    pauli_Y_list.extend(['Y' for _ in range(i, j)])
    # apply another X on qubit j
    pauli_Y_list.append('Y')
    # fill with identities
    pauli_Y_list.extend(['I' for _ in range(j + 1, num_qubits)])

    return pauli_Y_list


def _pauli_X_string_builder(i, j, num_qubits):
    """Creates a string corresponding to a transform on the Hilbert space for num_qubits qubits, with transformation
        X iff qubit index k is i or j
        Z iff qubit index i < k < j
        I otherwise
    """

    # all qubits below I have identity transformation
    pauli_X_list = ['I' for _ in range(i - 1)]
    # apply X on qubit i
    pauli_X_list.append('X')
    # between i and j, apply Z transformations
    pauli_X_list.extend(['Z' for _ in range(i, j)])
    # apply another X on qubit j
    pauli_X_list.append('X')
    # fill with identities
    pauli_X_list.extend(['I' for _ in range(j + 1, num_qubits)])

    return pauli_X_list
