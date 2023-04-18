import unittest

from src.util.PauliStringCreation import pauli_string_from_dict as create


class TestPauliStringCreation(unittest.TestCase):

    def test_create_identity(self):
        correctstring = 'IIIIII'
        built_string = create(6, None)
        self.assertEqual(correctstring, built_string)

    def test_create_only_X(self):
        correctstring = 'XXX'
        builder_dic = {0: 'X', 1: 'X', 2: 'X'}
        built_string = create(3, builder_dic)
        self.assertEqual(correctstring, built_string)

    def test_create_sample(self):
        correctstring = 'YXIZXY'
        builder_dic = {0: 'Y', 1: 'X', 3: 'Z', 4: 'X', 5: 'Y'}
        built_string = create(6, builder_dic)
        self.assertEqual(correctstring, built_string)

    def test_create_invalid_identifier(self):
        with self.assertRaises(ValueError):
            create(3, {0: 'AB'})

    def test_create_dic_too_long(self):
        with self.assertRaises(ValueError):
            create(2, {0: 'X', 1: 'X', 2: 'X'})
