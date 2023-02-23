"""Module implementing the one- and two qubit fermionic excitations under correct handling of antisymmetry
by the second quantization. """

import math

from qiskit.circuit.library import PauliGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import *


def SingleQuFermExcitation(circuit: QuantumCircuit, i: int, j: int, theta: float):
    """Function that creates a circuit which corresponds for a single qubit excitation on a qubit in
    orbital with index i to orbital with index j. It is assumed that i < j."""

    N = circuit.num_qubits

    # make sure that indices fit inside the circuit
    assert (N > i)
    assert (N > j)
    # assert order of indices
    assert (i <= j)



    # --------------------------------------------------------------






if __name__ == '__main__':
    circ = QuantumCircuit(4)
    SingleQuFermExcitation(circ, 0, 3, math.pi)
    # circ.qasm(filename="test")
    circ.draw(output='mpl', filename='test')
