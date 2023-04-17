"""Testclass for the VQE object"""

import unittest

from src.VQE.VQEObject import VQEObject
from src.molecule.BeH2 import BeH2

from qiskit.opflow import operator_base, expectations


class TestVQEObject(unittest.TestCase):

    def test_full_op_sanity(self):
        num_qubits = 14
        num_occupied = 6
        beh2 = BeH2("Test", num_qubits, num_occupied)

        test_obj = VQEObject(beh2)
        print(test_obj.full_op())

    def test_exp_val_sanity(self):
        num_qubits = 14
        num_occupied = 6
        beh2 = BeH2("Test", num_qubits, num_occupied)

        test_obj = VQEObject(beh2)
        print(test_obj.expectation_value())
