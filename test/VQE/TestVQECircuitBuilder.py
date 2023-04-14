import unittest

from src.VQE.VQECircuitBuilder import VQECircuitBuilder
from src.ansatz.RandomWeightGenerator import *



"""Testclass for VQE CircuitBuilder"""


class TestVQECircuitBuilder(unittest.TestCase):
    # testvariables
    num_qubits = 14
    eri1 = generate_random_2dim(num_qubits)
    eri2 = generate_random_4dim(num_qubits)

    def test_sanitycheck(self):
        hamiltonian = VQECircuitBuilder.build_hamiltonian_circuit(num_qubits=self.num_qubits)
        print(hamiltonian)
        self.assertTrue(True)

