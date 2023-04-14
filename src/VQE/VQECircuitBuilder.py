"""Class for building the VQE corresponding to the VQE algorithm.
Builds all necessary operators, transforms it to circuits and appends it together

"""

from qiskit.opflow import CircuitOp
from qiskit import QuantumCircuit

from ..hamiltonian.FermionicHamiltonian import FermionicHamiltonian
from ..ansatz.UCCAnsatz import UCCAnsatz

class VQECircuit:

    """Class which includes static methods that allow to build the quantum VQE for the """


    @staticmethod
    def build_hamiltonian_circuit(self, num_qubits):
        """Creates the hamiltonian class for num_qubits, stores the operator"""
        FermionicHamiltonian(num_qubits)
        hamiltonian_operator = FermionicHamiltonian.get_hamiltonian(num_qubits)
        return hamiltonian_operator.exp_i().to_circuit_op()

    @staticmethod
    def build_reference_state_circuit(num_qubits: int, num_occupated: int):
        """For a VQE with num_qubits qubits (corresponding to the number of orbitals)
        and an integer num_populated, build a VQE that will apply X gates to the num_occupated
        qubits with the lowest bit significance.
        This is corresponding to providing a reference stat in the occupation number theory
        representation of quantum orbitals
        """
        # note that qiskit is uses little endian notation, i.e. for example the lowest orbitals are
        # |000111> where the rightmost qubit would have index 0
        pass



    @staticmethod
    def build_kUCC_ansatz_circuit(self):
        """Creates the Ansatz object for num_qubits, stores the operator"""
        UCCAnsatz(self.num_qubits)
        ucc_ansatz = UCCAnsatz.get_ansatz()
        
