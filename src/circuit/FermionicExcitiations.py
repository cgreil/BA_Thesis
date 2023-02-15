"""Module implementing the one- and two qubit fermionic excitations under correct handling of antisymmetry
by the second quantization. """

import math

from qiskit.circuit.library import PauliGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import *


def SingleQuFermExcitation(circuit:QuantumCircuit, i:int, j:int, theta:float):
    """Function that creates a circuit which corresponds for a single qubit excitation on a qubit in
    orbital with index i to orbital with index j. It is assumed that i < j."""

    N = circuit.num_qubits

    #make sure that indices fit inside the circuit
    assert(N > i)
    assert(N > j)
    #assert order of indices
    assert(i <= j)

    circuit.rx((math.pi / 2), j)
    circuit.h(i)

    #cnot staircase
    #iterate top down
    for t in range(j, (i+1)):
        #t is control and t-1 is target qubit
        circuit.cnot(t, t-1)

    #apply Rz rotation with theta
    circuit.rz(theta, i)

    #cnot staircase upwards
    for t in range(i, j):
        #t+1 is target and t is control
        circuit.cnot(t+1, t)

    circuit.rx((-math.pi / 2), j)
    circuit.h(i)

    circuit.h(j)
    circuit.rx((math.pi / 2), i)

    #cnot staircase
    #iterate top down
    for t in range(j, (i+1)):
        #t is control and t-1 is target qubit
        circuit.cnot(t, t-1)

    circuit.rz(-theta, i)

    #cnot staircase upwards
    for t in range(i, j):
        #t+1 is target and t is control
        circuit.cnot(t+1, t)

    circuit.h(j)
    circuit.rx((-math.pi / 2), i)


if __name__ == '__main__':
    circ = QuantumCircuit(4)
    SingleQuFermExcitation(circ, 0, 3, math.pi)

    circ.draw()


