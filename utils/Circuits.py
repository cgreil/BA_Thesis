import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator, Pauli
from typing import *


def get_circuit(N: int):
    '''returns a circuit with N qubit registers'''
    return QuantumCircuit(N)


def add_creation_op(circ: QuantumCircuit, i: int):
    '''For a given circuit and index i, appends gates to the circuit that
    correspond to a creation operator at qubit with index i.
    Preserves the antisymmetry conditions as required in second
    quantized form'''

    # prepare operators
    opX = Operator(Pauli('X'))
    opY = Operator(Pauli('Y'))
    #TODO: Not unitary yet
    partial_op = 0.5 * (opX + 1j * opY)

    print(partial_op)
    circ.append(partial_op, i)
    for k in range(i-1):
        circ.z(k)

    return circ


def add_annihilation_op(circ:QuantumCircuit, i:int):
    '''For a given circuit and index i, appends gates to the circuit that
    correspond to a annihilation operator at qubit with index i. Preserves the antisymmetry
    conditions as required in the second quantized form'''

    #prepare operators
    opX = Operator(Pauli('X'))
    opY = Operator(Pauli('Y'))
    #TODO: Not unitary et
    partial_op = 0.5 * (opX - 1j * opY)

    circ.append(partial_op, i)
    for k in range(i-1):
        circ.z(k)

    return circ






#Testing function
if __name__ == '__main__':
    #circ = get_circuit(5)
    #add_creation_op(circ, 4)

    opX = Operator(Pauli('X'))
    opY = Operator(Pauli('Y'))
    partial_op = 0.5 * (opX - 1j * opY)

    #print(circ)