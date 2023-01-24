import qiskit
from qiskit.quantum_info  import Operator, Pauli
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import I, Z, X, Y
import numpy as np


def evol_x():
    matrix = -(np.pi/4) * X
    return PauliEvolutionGate(matrix)

def evol_y():
    matrix = -(np.pi/4) * Y
    return PauliEvolutionGate(matrix)


def evol_z():
    matrix = -(np.pi / 4) * Z
    return PauliEvolutionGate(matrix)