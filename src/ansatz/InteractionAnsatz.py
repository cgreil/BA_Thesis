"""Helper class to create the """


from nptyping import Shape, NDArray, Float

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

from ..fermion_interaction import SingleFermionicInteraction as sf
from ..fermion_interaction import DoubleFermionicInteraction as df

def generate_1e_ansatz_part(num_qubits: int, interaction_integrals: NDArray[Shape['2'], Float]):
    """Function which returns the full PauliSumOp for the whole single interaction fermionic hamiltonian."""
    diagonal_sparse_paulis = sf.generate_diagonal_paulis(num_qubits, interaction_integrals)
    offdiagonal_sparse_paulis = sf.generate_offdiagonal_paulis(num_qubits, interaction_integrals)

    complete_sparse_paulis = SparsePauliOp.sum([diagonal_sparse_paulis, offdiagonal_sparse_paulis])

    return PauliSumOp(complete_sparse_paulis)

def generate_2e_ansatz_part(num_qubits: int, interaction_integrals: NDArray[Shape['4'], Float]):
    """Function which returns teh full PauliSumOp for the double interaction fermionic hamiltonian."""
    diagonal_sparse_paulis = df.generate_diagonal_paulis(num_qubits, interaction_integrals)
    offdiagonal_sparse_paulis = df.generate_offdiagonal_paulis(num_qubits, interaction_integrals)

    complete_sparse_paulis = SparsePauliOp.sum([diagonal_sparse_paulis, offdiagonal_sparse_paulis])

    return PauliSumOp(complete_sparse_paulis)

