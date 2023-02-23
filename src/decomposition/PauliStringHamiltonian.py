import array
import numpy as np
from qiskit.quantum_info.operators import Operator, Pauli
from PauliString import PauliString, LinearPauliCombination


def get_single_electron_hamiltonian(coeffs, num_qubits):
    """For a given number of qubits N and a corresponding NxN matrix containing the one electron interaction integrals,
     calculate a representation of the single electron interaction hamiltonian in a sequence of Pauli strings according
     to the Jordan Wigner transformation.
     The methodology is described in https://arxiv.org/abs/0705.1928"""

    # dimension of coeffs has to be NxN
    assert (len(coeffs.shape()) == 2)
    assert (coeffs.shape()[0] == num_qubits)
    assert (coeffs.shape()[1] == num_qubits)
    # get pauli operators
    I = Pauli(label=0)
    X = Pauli(label=1)
    Y = Pauli(label=2)
    Z = Pauli(label=3)
    # store the corresponding Pauli-strings
    pauli_string_list = []
    # diagonal elements (consist of linear combinations of pauli operators)
    for i in range(num_qubits):
        first_part_diag = [I for _ in range(0, i)]
        second_part_diag = [I for _ in range(i + 1, num_qubits)]
        center_diag = [LinearPauliCombination(I, Z)]

        # concat lists and transform to tuple
        pauli_tuple = tuple(
            [element for partial_list in [first_part_diag, center_diag, second_part_diag] for element in
             partial_list])

        # retrieve weight (factor 1/2 is part of the JW transformation)
        weight = (1 / 2) * coeffs[i, i]

        # create the full Pauli String object and add to list
        pauli_string_list.append(PauliString(pauli_tuple, weight))

        # off-diagonal elements
        for j in range(i, num_qubits):
            first_part_off = [I for _ in range(0, i)]
            second_part_off = [I for _ in range(j, num_qubits)]

            # for the center (linear combination of two Pauli strings), first create the individual strings:
            # cnot staircase corresponds to the kronecker product of Z-operators
            cnot_staircase = [Z for _ in range((i + 1), j)]

            # generate first pauli string operator list
            pauli_string_list_1 = np.zeros(num_qubits)
            pauli_string_list_1[:i] = first_part_off
            pauli_string_list_1[i] = X
            pauli_string_list_1[(i + 1):j] = cnot_staircase
            pauli_string_list_1[j] = X
            pauli_string_list_1[j:] = second_part_off

            # generate second pauli string operator list
            pauli_string_list_2 = np.zeros(num_qubits)
            pauli_string_list_2[:i] = first_part_off
            pauli_string_list_2[i] = Y
            pauli_string_list_2[(i + 1):j] = cnot_staircase
            pauli_string_list_2[j] = Y
            pauli_string_list_2[j:] = second_part_off

            # generate the linear combinations of Pauli gates, put them in tuple and generate the final string from it
            pauli_string_list.append(PauliString(tuple(
                [LinearPauliCombination(pauli_string_list_1[k], pauli_string_list_2[k]) for k in
                 range(0, num_qubits)]), (1 / 2) * coeffs[i, j]))

    # return final pauli strings
    return tuple(pauli_string_list)


def get_double_electron_hamiltonian():
    return


class PauliStringHamiltonian:
    """Represents the Pauli String version of the Hamiltonian"""
    num_qubits = 0
    H1 = None  # 1 - interaction part
    H2 = None  # 2 - interaction part
    # tuple for pauli
    pauli_strings: tuple

    def __init__(self, num_qubits, coeffs: array):
        """For a tuple of qubits, and an array of coefficients, create representations of the 1- and 2- electron
        interaction hamiltonian onto these qubits according to the Jordan-Wigner transformation
        """
        self.num_qubits = num_qubits
        get_single_electron_hamiltonian(coeffs, num_qubits)
