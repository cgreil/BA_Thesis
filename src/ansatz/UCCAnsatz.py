"""Provides auxiliary functions needed for the preparation of the operator form of the kUCC-Ansatz"""

from nptyping import NDArray

from qiskit.opflow import PauliSumOp, OperatorBase
from qiskit.quantum_info import SparsePauliOp

from RandomWeightGenerator import generate_random_2dim, generate_random_4dim
from InteractionAnsatz import generate_1e_ansatz_part, generate_2e_ansatz_part
class UCCAnsatz:

    num_qubits: int = 0
    # UCC Object stored as PauliSumOp
    ansatz_operator: PauliSumOp = None


    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self._generate_ansatz_operator()


    def _generate_ansatz_operator(self):
        """Creates the necessary random initialization weights, invokes the operator creation."""
        initial_2d_weights = generate_random_2dim(self.num_qubits)
        initial_4d_weights = generate_random_4dim(self.num_qubits)

        single_interaction_part = generate_1e_ansatz_part(self.num_qubits, initial_2d_weights)
        double_interaction_part = generate_2e_ansatz_part(self.num_qubits, initial_4d_weights)

        self.ansatz_operator = single_interaction_part.add(double_interaction_part)

    def get_ansatz(self) -> PauliSumOp:
        if self.ansatz_operator is None:
            raise ValueError("Ansatz is NoneObject")
        else:
            return self.ansatz_operator

    def get_matrix(self) -> NDArray:
        if self.ansatz_operator is None:
            raise ValueError("Ansatz is NoneObject")
        else:
            return self.ansatz_operator.to_matrix()

    def exponential(self) -> OperatorBase:
        if self.ansatz_operator is None:
            raise ValueError("Ansatz is NoneObject")
        else:
            return self.ansatz_operator.exp_i()
