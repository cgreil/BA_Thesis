import numpy as np
from nptyping import NDArray, Shape, Float
from qiskit.quantum_info import SparsePauliOp

from src.util.PauliStringCreation import pauli_string_from_dict


def generate_diagonal_paulis(num_qubits: int, interaction_integrals: NDArray[Shape['2'], Float]):
    """Generates the sparse pauli operator resulting from the diagonal elements of the hamiltonian"""

    # initialize empty pauli list and coeff array
    pauli_list = []
    coeffs = []


    for i in range(num_qubits):
        # store coeff
        coeff = (1/2) * interaction_integrals[i, i]
        # append coeff twice, since for every diagonal element there will be two pauli strings added
        coeffs.extend([coeff, coeff])

        # I pauli string
        pauli_I_list = pauli_string_from_dict(num_qubits, None)
        pauli_list.append(pauli_I_list)

        # Z pauli string    # finally create the Sparse Pauli Operator

        pauli_Z_list = pauli_string_from_dict(num_qubits, {i: 'Z'})
        pauli_list.append(pauli_Z_list)

    # finally create the Sparse Pauli Operator
    return SparsePauliOp(pauli_list, coeffs=np.array(coeffs))


def generate_offdiagonal_paulis(num_qubits: int, interaction_integrals: NDArray[Shape['2'], Float]):
    # initialize the pauli list and coeff list
    pauli_list = []
    coeffs = []

    coeff_index = 0

    # iterate over combinations where i < j_builder
    for j in range(num_qubits):
        for i in range(j):
            if i == j:
                # equal to a diagonal element
                continue
            # add coefficient
            coeffs[coeff_index] = -(1 / 2) * interaction_integrals[i, j]
            coeff_index = coeff_index + 1

            pauli_X_string = _pauli_X_string_builder(num_qubits, i, j)
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


def _pauli_X_string_builder(num_qubits: int, i: int, j: int):
    """Creates a string corresponding to a transform on the Hilbert space for num_qubits qubits, with transformation
        X iff qubit index k is i or j
        Z iff qubit index i < k < j
        I otherwise
    """
    if j < i:
        raise ValueError("first index value has to be lower than the second")

    if i == j:
        raise ValueError("Position of X Paulis cannot be equal")

    pauli_dict = {i: 'X'}
    for k in range(i + 1, j):
        pauli_dict[k] = 'Z'
    pauli_dict[j] = 'X'

    pauli_string = pauli_string_from_dict(num_qubits, pauli_dict)
    return pauli_string
