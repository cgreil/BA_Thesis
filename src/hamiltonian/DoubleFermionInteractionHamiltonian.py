"""Module which enables the generation of the double excitation fermionic hamiltonian in the second quantization using
Jordan-Wigner transformation.
"""

from nptyping import NDArray, Shape, Float

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

from ..fermion_interaction.DoubleFermionicInteraction import generate_diagonal_paulis, generate_offdiagonal_paulis


def generate_2e_hamiltonian(num_qubits: int, interaction_integrals: NDArray[Shape['4'], Float]):
    diagonal_pauli_op = generate_diagonal_paulis(num_qubits, interaction_integrals)
    offiagonal_pauli_op = generate_offdiagonal_paulis(num_qubits, interaction_integrals)

    # combine the pauli operators and return the final PauliSumOperator
    complete_pauli_op = SparsePauliOp.sum([diagonal_pauli_op, offiagonal_pauli_op])

    return PauliSumOp(complete_pauli_op)


