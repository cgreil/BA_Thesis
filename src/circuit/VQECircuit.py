"""Class for building the circuit corresponding to the VQE algorithm.
Builds all necessary operators, transforms it to circuits and appends it together

"""

from qiskit.opflow import CircuitOp
from qiskit import QuantumCircuit

from ..hamiltonian.FermionicHamiltonian import FermionicHamiltonian
from ..ansatz.UCCAnsatz import  UCCAnsatz

class VQECircuit:
    num_qubits: int
    num_classical_bits: int
    hamiltonian_operator = None
    ansatz_operator = None
    measurement_operator = None

    def get_hamiltonian_circuit(self):
        """Creates the hamiltonian class for num_qubits, stores the operator"""
        FermionicHamiltonian(self.num_qubits)
        self.hamiltonian_operator = FermionicHamiltonian.fermionic_hamiltonian(None)


    def get_ansatz_circuit(self):
        """Creates the Ansatz object for num_qubits, stores the operator"""
        UCCAnsatz(self.num_qubits)
        self.ansatz_operator = UCCAnsatz.get_ansatz()

    def get_measurement_circuit(self):
        pass

    def get_optimizer(self):
        pass
