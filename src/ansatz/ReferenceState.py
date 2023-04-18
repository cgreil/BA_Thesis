from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

# pauli string builder
from src.util.PauliStringCreation import pauli_string_from_dict


class ReferenceState:
    """Class providicng functionality for creation of a reference state in the
     orbital occupation theory."""
    num_qubits: int = 0
    num_occupied: int = 0
    # operator for creating the occupied state from a |000...> state.
    reference_operator: PauliSumOp = None

    def __init__(self, num_qubits: int, num_occupied: int):
        self.num_qubits = num_qubits
        self.num_occupied = num_occupied
        # call generate method
        self._generate_ref_op()

    def _generate_ref_op(self):
        first_occupied = self.num_qubits - self.num_occupied
        op_dic = {k: 'X' for k in range(first_occupied, self.num_qubits)}
        string = pauli_string_from_dict(self.num_qubits, op_dic)
        sparse_op = SparsePauliOp(string)
        self.reference_operator = PauliSumOp(sparse_op)

    def get_ref_op(self):
        if self.reference_operator is None:
            raise ValueError("Reference Ansatz is None Object")
        return self.reference_operator

    def get_matrix(self):
        if self.reference_operator is None:
            raise ValueError("Reference Ansatz is None Object")
        return self.reference_operator.to_matrix()