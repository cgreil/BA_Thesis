"""Testclass for the VQEBuilder class"""

import unittest
import numpy as np

from src.VQE.VQEBuilder import VQEBuilder


class TestVQEBuilder(unittest.TestCase):

    def test_build_kUCC_Ansatz_op_sanity(self):
        num_qubits = 4
        eri1_random = np.random.random((num_qubits, num_qubits))
        eri2_random = np.random.random((num_qubits, num_qubits, num_qubits, num_qubits))
        ansatz = VQEBuilder.build_kUCC_ansatz_operator(num_qubits, eri1_random, eri2_random)
        print(ansatz)
        self.assertTrue(True)
