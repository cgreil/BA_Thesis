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

from nptyping import Shape, NDArray, Float

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

from ..fermion_interaction.SingleFermionicInteraction import generate_diagonal_paulis, generate_offdiagonal_paulis


def generate_1e_hamiltonian(num_qubits: int, weights: NDArray[Shape['2'], Float]):
    """Function which returns the full PauliSumOp for the whole single electron fermionic hamiltonian."""
    diagonal_sparse_paulis = generate_diagonal_paulis(num_qubits, weights)
    offdiagonal_sparse_paulis = generate_offdiagonal_paulis(num_qubits, weights)

    complete_sparse_paulis = SparsePauliOp.sum([diagonal_sparse_paulis, offdiagonal_sparse_paulis])

    return PauliSumOp(complete_sparse_paulis)


