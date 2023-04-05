from qiskit import QuantumCircuit
from qiskit.opflow import Operato

from src.evolution.BasicGates import *

def get_circuit(N: int):
    '''returns a circuit with N qubit registers'''
    return QuantumCircuit(N)


def electronic_hamiltonian(circ:QuantumCircuit):
    """Function taking a circuit and appending unitary gates representing the electronic Hamiltonian.
    Afterwards returns the edited circuit again.

    The Hamiltonian will be split into single electron interaction and double electron interaction."""
    #TODO
    return circ


def append_single_interaction_hamiltonian(N:int, i:int, j:int):
    """Function which creates a circuit representing the single interaction hamiltonian between qubits at position i and
     qubit at position j onto the given circuit"""

    #Diagonal elements

    #Off-Diagonal elements
    return







#Testing function
if __name__ == '__main__':
    #circ = get_circuit(5)
    #add_creation_op(circ, 4)

    opX = Operator(Pauli('X'))
    opY = Operator(Pauli('Y'))
    partial_op = 0.5 * (opX - 1j * opY)

    #print(circ)