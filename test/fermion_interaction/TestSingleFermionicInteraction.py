import unittest

from src.fermion_interaction.SingleFermionicInteraction import _pauli_X_string_builder as build


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


