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
from nptyping import Shape, NDArray, Float

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

from ..util.PauliStringCreation import pauli_string_from_dict


def generate_1e_hamiltonian(num_qubits: int, weights: NDArray[Shape['2'], Float]):
    """Function which returns the full PauliSumOp for the whole single electron fermionic hamiltonian."""
    diagonal_sparse_paulis = _generate_diagonal_paulis(num_qubits, weights)
    offdiagonal_sparse_paulis = _generate_offdiagonal_paulis(num_qubits, weights)

    complete_sparse_paulis = SparsePauliOp.sum([diagonal_sparse_paulis, offdiagonal_sparse_paulis])

    return PauliSumOp(complete_sparse_paulis)


def _generate_diagonal_paulis(num_qubits: int, weights: NDArray[Shape['2'], Float]):
    """Generates the sparse pauli operator resulting from the diagonal elements of the hamiltonian"""

    # initialize empty pauli list and coeff array
    pauli_list = []
    coeffs = []

    for i in range(num_qubits):
        # store coeff
        coeffs[i] = 1 / 2 * weights[i, i]

        # identity pauli string
        pauli_I_list = pauli_string_from_dict(num_qubits, None)
        pauli_list.append(pauli_I_list)

        # Z pauli string
        pauli_Z_list = pauli_string_from_dict(num_qubits, {i: 'Z'})
        pauli_list.append(pauli_Z_list)

    # finally create the Sparse Pauli Operator
    return SparsePauliOp(pauli_list, coeffs=np.array(coeffs))


def _generate_offdiagonal_paulis(num_qubits: int, weights: NDArray[Shape['2'], Float]):
    # initialize the pauli list and coeff list
    pauli_list = []
    coeffs = []

    coeff_index = 0

    # iterate over combinations where i < j
    for j in range(num_qubits):
        for i in range(j):
            if i == j:
                # equal to a diagonal element
                continue
            # add coefficient
            coeffs[coeff_index] = -(1 / 2) * weights[i, j]
            coeff_index = coeff_index + 1

            pauli_X_string = _pauli_X_string_builder(i, j, num_qubits)
            pauli_Y_string = _pauli_Y_string_builder(i, j, num_qubits)
            # add strings to the list of all
            pauli_list.append(pauli_X_string)
            pauli_list.append(pauli_Y_string)

    # create the Sparse Pauli Operator
    return SparsePauliOp(pauli_list, coeffs=np.array(coeffs))


def _pauli_Y_string_builder(i: int, j: int, num_qubits: int):
    """Creates a string corresponding to a transform on the Hilbert space for num_qubits qubits, with transformation
        Y iff qubit index k is i or j
        Z iff qubit index i < k < j
        I otherwise
    """

    pauli_dict = {i: 'Y'}
    for k in range(i + 1, j):
        pauli_dict[k] = 'Z'
    pauli_dict[j] = 'Y'

    pauli_string = pauli_string_from_dict(num_qubits, pauli_dict)
    return pauli_string


def _pauli_X_string_builder(i: int, j: int, num_qubits: int):
    """Creates a string corresponding to a transform on the Hilbert space for num_qubits qubits, with transformation
        X iff qubit index k is i or j
        Z iff qubit index i < k < j
        I otherwise
    """

    pauli_dict = {i: 'X'}
    for k in range(i + 1, j):
        pauli_dict[k] = 'Z'
    pauli_dict[j] = 'X'

    pauli_string = pauli_string_from_dict(num_qubits, pauli_dict)
    return pauli_string
