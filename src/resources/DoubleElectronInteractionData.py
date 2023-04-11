from typing import List

from ..util.InteractionGroup import InteractionGroup


class DoubleElectronInteractionData:
    """static Class for storing the speficic configuration of pauli string terms for the 24 term
    sum in the double electron hamiltonian"""

    _number_of_terms = 24
    _pauli_list = ['XXXX', 'XXYY', 'XYXY', 'XYYX', 'YXXY', 'YXYX', 'YYXX', 'YYYY']
    _sign_lookup_matrix = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1], [1, 1, -1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]

    def get_number_of_terms(self):
        return self._number_of_terms

    def get_pauli_list(self):
        return self._pauli_list

    def get_sign(self, interaction_group: InteractionGroup, term_number: int):
        """Returns the sign of a term that is specified by the interaction group and the
        term number. Specifically, the sign is in the lookup matrix at entry i,j where row i
        corresponds to the term number and interaction group j corresponds to the column"""
        # Notice that the enum interaction group has integer correspondence
        return self._sign_lookup_matrix[term_number][interaction_group.value]