import unittest

from src.fermion_interaction.SingleFermionicInteraction import _pauli_X_string_builder

class TestSingleFermionicInteraction(unittest.TestCase):

    def test_pauli_x_string_builder_even_qubits(self):
        correctstring = 'IIXZZZXII'
        built_string = _pauli_X_string_builder(9, 2, 6)
        self.assertEquals(built_string, correctstring)

    def test_pauli_x_string_builder_odd_qubits(self):
        correctstring = 'IIXZZZXII'
        built_string = _pauli_X_string_builder(9, 2, 6)
        self.assertEquals(built_string, correctstring)

