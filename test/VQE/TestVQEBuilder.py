"""Testclass for the VQEBuilder class"""

import unittest
import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

from src.VQE.VQEBuilder import VQEBuilder
from src.molecule.BeH2 import BeH2


class TestVQEBuilder(unittest.TestCase):

    def test_build_kUCC_Ansatz_op_sanity(self):
        num_qubits = 4
        eri1_random = np.random.random((num_qubits, num_qubits))
        eri2_random = np.random.random((num_qubits, num_qubits, num_qubits, num_qubits))
        ansatz = VQEBuilder.build_kUCC_ansatz_operator(num_qubits, eri1_random, eri2_random)
        print(ansatz)
        self.assertTrue(True)

    def test_build_kUCC_Ansatz_op_wrong_num_qubits(self):
        num_qubits = 4
        eri1_random = np.random.random((num_qubits, num_qubits))
        eri2_random = np.random.random((num_qubits, num_qubits, num_qubits, num_qubits))
        with self.assertRaises(AssertionError):
            ansatz = VQEBuilder.build_kUCC_ansatz_operator(num_qubits + 1, eri1_random, eri2_random)

    def test_build_reference_state(self):
        num_qubits = 10
        num_occupied = 7
        reference_op = VQEBuilder.build_reference_state_operator(num_qubits, num_occupied)
        correct_op_string = 'IIIXXXXXXX'
        correct_op = PauliSumOp(SparsePauliOp(correct_op_string))
        self.assertEqual(reference_op, correct_op)

    def test_build_reference_state_wrong_args(self):
        num_qubits = 5
        num_occupied = 8
        with self.assertRaises(ValueError):
            VQEBuilder.build_reference_state_operator(num_qubits, num_occupied)

    def test_build_hamiltonian_op_sanity(self):
        num_qubits = 14
        beh2 = BeH2("Test", 14, 6)
        print(VQEBuilder.build_hamiltonian_operator(num_qubits, beh2))
        self.assertTrue(True)
