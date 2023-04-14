"""Creates the random weights needed for the k-UCC Ansatz state"""

from nptyping import NDArray, Shape, Float
import numpy as np


def generate_random_2dim(num_qubits: int) -> NDArray[Shape['2'], Float]:
    """Generates a random numpy ndarray with dimension num_qubits x num_qubits as initial values
    for the k-UCC Ansatz state
    """
    return np.random.rand(num_qubits, num_qubits)


def generate_random_4dim(num_qubits: int) -> NDArray[Shape['4'], Float]:
    """Generates a random numpy ndarray with dimension num_qubits x num_qubits x num_qubits x num_qbits
     as initial values for the k-UCC Ansatz state
    """
    return np.random.rand(num_qubits, num_qubits, num_qubits, num_qubits)
