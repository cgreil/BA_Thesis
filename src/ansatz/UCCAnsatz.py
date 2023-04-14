"""Provides auxiliary functions needed for the preparation of the operator form of the kUCC-Ansatz"""

from nptyping import NDArray

from qiskit.opflow import PauliSumOp, OperatorBase

from src.ansatz.InteractionAnsatz import generate_1e_ansatz_part, generate_2e_ansatz_part


class UCCAnsatz:
    num_qubits: int = 0
    # UCC Object stored as PauliSumOp
    ansatz_operator: PauliSumOp = None
    single_interaction_weights: NDArray = None
    double_interaction_weights: NDArray = None

    def __init__(self, num_qubits: int, eri1_weights: NDArray, eri2_weights: NDArray):
        self.num_qubits = num_qubits
        self._generate_ansatz_operator()
        self.single_interaction_weights = eri1_weights
        self.double_interaction_weights = eri2_weights

    def _generate_ansatz_operator(self):
        single_interaction_part = generate_1e_ansatz_part(self.num_qubits, self.single_interaction_weights)
        double_interaction_part = generate_2e_ansatz_part(self.num_qubits, self.double_interaction_weights)

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
