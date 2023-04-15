"""VQE Object that will be built by VQE Builder and run by VQE Runner. stores the
operators needed and provides the methods necessary for measurements, gradients, expectation values
and the likes."""

from qiskit.opflow import OperatorBase

from src.ansatz.RandomWeightGenerator import generate_random_2dim, generate_random_4dim
from src.VQE.VQEBuilder import *


class VQEObject:
    num_qubits: int
    num_occupied: int
    reference_operator: OperatorBase = None
    ansatz_operator: OperatorBase = None
    hamiltonian_operator: OperatorBase = None

    # TODO: Wrap num_qubits, num_occupied, and all things specific to the molecule into a dedicated object
    def __init__(self, num_qubits: int, num_occupied: int):
        self.num_qubits = num_qubits
        self.num_occupied = num_occupied

        # generate reference op
        self.reference_operator = VQEBuilder.build_reference_state_operator(self.num_qubits, self.num_occupied)

        # generate ansatz op with random values
        eri1_weights = generate_random_2dim(self.num_qubits)
        eri2_weights = generate_random_4dim(self.num_qubits)
        self.ansatz_operator = VQEBuilder.build_kUCC_ansatz_operator(self.num_qubits, eri1_weights, eri2_weights)

        # generate hamiltonian op
        self.hamiltonian_operator = VQEBuilder.build_hamiltonian_operator(self.num_qubits)
