"""Class for building the circuit corresponding to the VQE algorithm.
Builds all necessary operators, transforms it to circuits and appends it together

"""

from qiskit.opflow import CircuitOp
from qiskit import QuantumCircuit

from ..hamiltonian.FermionicHamiltonian import FermionicHamiltonian


class VQECircuit:
    num_qubits: int
    num_classical_bits: int
    hamiltonian_operator = None
    ansatz_operator = None
    measurement_operator = None

    def get_hamiltonian_circuit(self):
        pass

    def get_ansatz_circuit(self):
        pass

    def get_measurement_circuit(self):
        pass

    def get_optimizer(self):
        pass
