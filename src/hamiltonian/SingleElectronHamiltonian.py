"""Module that enables the creation of the single electron interaction hamiltonian in the second
quantization using the Jordan-Wigner transformation.

The single particle interaction Hamiltonian H1 can be described as
$$H_1 = \sum_i h_{ii} a_i^{\dagger} a_i + \sum_{i < j} h_{ij} (a_i^{\dagger} a_j + a_j^{\dagger} a_i)$$
where $a, a^\dagger$ denote the annihilation and creation operators, respectively.

This module returns a PauliSumOp
(see https://qiskit.org/documentation/stubs/qiskit.opflow.primitive_ops.PauliSumOp.html#qiskit.opflow.primitive_ops.PauliSumOp)
which lets one combine weights with sparse Pauli Operators, where Pauli Operators can be Tensor products of
Pauli Gates.
"""

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import PauliList
from typing import List
def generate_pauli_sum(num_qubits: int, weights:List[List[float]]):
    """Function which returns the PauliSumOp for the whole single electron fermionic hamiltonian."""


def generate_diagonal_paulis(num_qubits: int, i: int, weights:List[List[float]]):
    """Generates the sparse pauli operator resulting from the diagonal elements of the hamiltonian"""

    # initialize empty pauli list and coeff array
    PauliList = []
    coeffs = []

    for i in range(num_qubits):
        coeff = weights[i][i]
        # identity pauli string
        pauli_I_list = ['I' for _ in range(num_qubits)]
        PauliList.append(pauli_I_list)
        # already multiplies with 1/2
        coeffs[i] = 1/2 * weights[i]

        #Z pauli string
        pauli_Z_list = ['I' for _ in range((i-1))]
        pauli_Z_list.append('Z')
        pauli_Z_list.extend(['I' for _ in range(i+1, num_qubits)])
        PauliList.append(pauli_Z_list)

    # finally create the Sparse Pauli Operator
    SparsePauliOp(PauliList, coeffs=np.array(coeffs))



def generate_offdiagonal_paulis():

    return







