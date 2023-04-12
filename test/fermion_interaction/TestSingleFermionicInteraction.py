import unittest
import numpy as np

from qiskit.quantum_info import SparsePauliOp

from src.fermion_interaction.SingleFermionicInteraction import _pauli_X_string_builder as build
from src.fermion_interaction.SingleFermionicInteraction import generate_diagonal_paulis as diag


class TestSingleFermionicInteraction(unittest.TestCase):

    def test_pauli_x_string_builder_odd_qubits(self):
        correctstring = 'IIXZZZXII'
        built_string = build(9, 2, 6)
        self.assertEquals(built_string, correctstring)

    def test_pauli_x_string_builder_even_qubits(self):
        correctstring = 'IIXZZZXIII'
        built_string = build(10, 2, 6)
        self.assertEquals(built_string, correctstring)

    def test_pauli_x_string_builder_x_at_bounds(self):
        correctstring = 'XZZZX'
        built_string = build(5, 0, 4)
        self.assertEquals(built_string, correctstring)

    def test_pauli_x_string_builder_adjacent_x(self):
        correctstring = 'IIXXII'
        built_string = build(6, 2, 3)
        self.assertEquals(built_string, correctstring)

    def test_pauli_x_string_builder_only_x(self):
        correctstring = 'XX'
        built_string = build(2, 0, 1)
        self.assertEquals(built_string, correctstring)

    def test_pauli_x_string_builder_equal_indices(self):
        with self.assertRaises(ValueError):
            build(3, 1, 1)

    def test_pauli_x_string_builder_wrong_index_order(self):
        with self.assertRaises(ValueError):
            build(3, 2, 1)

    def test_diagonal_pauli_generator_2qubit(self):
        num_qubits = 2
        coeffs_sample = np.array([[0.3, 1], [3.4, 8.2]])
        # np.repeat(x, n) creates a new list, where each x is repeated n times
        correct_coeffs = np.repeat(np.array([coeffs_sample[k, k] * 1 / 2 for k in range(num_qubits)]), 2)
        correct_op = SparsePauliOp(['II', 'ZI', 'II', 'IZ'], coeffs=correct_coeffs)
        result = diag(2, coeffs_sample)
        self.assertEqual(result, correct_op)

    def test_diagonal_pauli_generator_3qubit(self):
        num_qubits = 3
        coeffs_sample = np.array([[2.2, 1.0, 3.2], [8.9, 4.1, 0.98], [0.63, 2.2, 0.82]])
        correct_coeffs = np.repeat(np.array([coeffs_sample[k, k] * 1 / 2 for k in range(num_qubits)]), 2)
        correct_op = SparsePauliOp(['III', 'ZII', 'III', 'IZI', 'III', 'IIZ'], coeffs=correct_coeffs)

        result = diag(3, coeffs_sample)
        self.assertEqual(result, correct_op)

    def test_diagonal_pauli_generator_2qubit_wrong_order(self):
        num_qubits = 2
        coeffs_sample = np.array([[0.3, 1], [3.4, 8.2]])
        # np.repeat(x, n) creates a new list, where each x is repeated n times
        correct_coeffs = np.repeat(np.array([coeffs_sample[k, k] * 1 / 2 for k in range(num_qubits)]), 2)
        # note that IZ and II are swapped to what they should be
        correct_op = SparsePauliOp(['II', 'ZI', 'IZ', 'II'], coeffs=correct_coeffs)
        result = diag(2, coeffs_sample)
        self.assertNotEqual(result, correct_op)

    def test_diagonal_pauli_generator_wrong_coeff_matrix_size(self):
        num_qubits = 2
        coeffs_sample = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with self.assertRaises(AssertionError):
            diag(num_qubits, coeffs_sample)

    def test_diagonal_pauli_generator_wrong_coeff_matrix_shape(self):
        num_qubits = 2
        coeffs_sample = np.array([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            diag(num_qubits, coeffs_sample)


