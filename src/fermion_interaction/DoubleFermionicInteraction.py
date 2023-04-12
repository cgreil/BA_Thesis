"""Module which enables the creation of electron interaction operators as pauli strings"""
from typing import List

import numpy as np
from nptyping import NDArray, Shape, Float
from qiskit.quantum_info import SparsePauliOp

from src.resources.DoubleElectronInteractionData import DoubleElectronInteractionData
from src.util.Antisymmetry import determine_ordering, levi_civita_epsilon
from src.util.InteractionGroup import InteractionGroup, determine_interaction_group
from src.util.PauliStringCreation import pauli_string_from_dict


def _determine_positions(interaction_group: InteractionGroup, pauli_indices: List[int]):
    # s_alpha will hold the position of the lowest, s_beta of the second lowest and so on
    # For 2 interaction pairs, only 4 indices
    assert (len(pauli_indices) == 4)

    if interaction_group is InteractionGroup.FIRST:
        s_alpha, s_beta, s_gamma, s_delta = pauli_indices
        positions = [s_alpha, s_beta, s_gamma, s_delta]
    elif interaction_group is InteractionGroup.SECOND:
        s_alpha, s_gamma, s_beta, s_delta = pauli_indices
        positions = [s_alpha, s_beta, s_gamma, s_delta]
    elif interaction_group is InteractionGroup.THIRD:
        s_alpha, s_delta, s_beta, s_gamma = pauli_indices
        positions = [s_alpha, s_beta, s_gamma, s_delta]
    else:
        raise ValueError("This ordering of indices is not allowed!")

    assert (positions is not None)
    return positions


def _identity_string_builder(num_qubits: int):
    # pass none to get identity string for dimension num_qubits
    return pauli_string_from_dict(num_qubits, None)


def _pauli_single_Z_string_builder(num_qubits: int, i: int):
    return pauli_string_from_dict(num_qubits, {i: 'Z'})


def _pauli_double_Z_string_builder(num_qubits: int, i: int, j: int):
    return pauli_string_from_dict(num_qubits, {i: 'Z', j: 'Z'})


def _pauli_quadra_term_builder(num_qubits: int, positions: List[int], paulis='IIII'):
    """Utilizies pauli_string from dict to build a single sum  term for the 24 term interaction
    hamiltonian.
    paulis is a string of length 4 where each entry is either X or Y, the corresponding Pauli matrix
    will then be inserted at the respective index given from the positions list"""
    assert (len(positions) == 4)
    assert (len(paulis) == 4)

    pauli_dict = {}
    # add the four 'main' pauli matrices
    for position, pauli in zip(positions, paulis):
        pauli_dict[position] = pauli

    # add the asymmetry Z strings between s_alpha and s_beta aswell as between s_gamma and s_delta
    s_alpha, s_beta, s_gamma, s_delta = positions
    for i in range(s_alpha + 1, s_beta):
        pauli_dict[i] = 'Z'

    for i in range(s_gamma + 1, s_delta):
        pauli_dict[i] = 'Z'

    return pauli_string_from_dict(num_qubits, pauli_dict)


def _generate_diagonal_paulis(num_qubits: int, interaction_integrals: NDArray[Shape['4'], Float]):
    """Function which generates the Sum of Pauli strings which results off the mapping
    of the diagonal elements of the Two electron fermionic interaction Hamiltonian.
    Diagonal in this sense refers to the situation that for an index set ijkl, pairwaise indices
    are equal, i.e. i = k and j = l.

    The created linear combination of Strings will be returned as SparsePauliOp."""

    # initialize pauli and coeff lists
    pauli_list = []
    coeffs = []
    coeff_index = 0

    for j in range(num_qubits):
        for i in range(j):
            # retrieve correct coefficient
            coeffs[coeff_index] = (1 / 4) * interaction_integrals[i, j, i, j]
            coeff_index = coeff_index + 1

            pauli_identity_string = _identity_string_builder(num_qubits)
            pauli_Zi_string = _pauli_single_Z_string_builder(num_qubits, i)
            pauli_Zj_string = _pauli_single_Z_string_builder(num_qubits, j)
            pauli_double_Z_string = _pauli_double_Z_string_builder(num_qubits, i, j)

            # append all strings to the sum
            pauli_list.extend([pauli_identity_string, pauli_Zi_string, pauli_Zj_string, pauli_double_Z_string])

    # finally create the SparsePauliOp
    return SparsePauliOp(pauli_list, coeffs=np.array(coeffs))


def _generate_offdiagonal_paulis(num_qubits: int, interaction_integrals: NDArray[Shape['4'], Float]):
    """Function which creates the Sum of Pauli strings which results from the mapping of the offdiagonal elements
    of the two electron fermionic interaction Hamiltonian.

    Notice that there are three distinct groups of interaction types that one has to consider:
    I   ... i < j < l < k
    II  ... i < l < j < k
    III ... i < l < k < j
    """

    # initialize pauli string
    interaction_strings = []


    # j and k iterate over whole range, whereas i and l only up to j and k respectively
    for j in range(num_qubits):
        for i in range(j):
            if i == j:
                # is equal to a diagonal element
                continue
            for k in range(num_qubits):
                for l in range(k):
                    if l == k:
                        # also equal to a diagonal element (up to index swapping)
                        continue
                    # use method shown in the paper to determine the case
                    index_dict = {'i': i, 'j': j, 'l': l, 'k': k}
                    # determine the group of interaction pairs
                    interaction_group = determine_interaction_group(i, j, k, l)
                    # Determine the ordering and from it, determine the result of the levi-citavi epsilon
                    ordering = determine_ordering(index_dict)
                    coeff = (1 / 8) * levi_civita_epsilon(ordering) * interaction_integrals[i, j, k, l]

                    # Calculate the full pauli string for a single interaction
                    pauli_string = _build_24_term_string(num_qubits, interaction_group, [i, j, k, l],
                                                         coeff=coeff)
                    # construct the string for the full excitation
                    interaction_strings.append(pauli_string)

    excitation_string = SparsePauliOp.sum(interaction_strings)
    return excitation_string


def _build_24_term_string(num_qubits: int, interaction_group: InteractionGroup, pauli_indices: List[int], coeff=1):
    """ Build the 24-term Pauli string corresponding to a single summation term.
        In particular, each of the terms will have the form
        II...IAZZ...ZBCZZ...ZDII...I
        where A,B,C,D are from {X, Y} respectively.
    """
    # pauli indices is the list [i, j, k, l]
    positions = _determine_positions(interaction_group, pauli_indices)
    substrings = []

    for term_index in range(DoubleElectronInteractionData.get_number_of_terms()):
        sign = DoubleElectronInteractionData.get_sign(interaction_group, term_index)
        term = _pauli_quadra_term_builder(num_qubits, positions,
                                          DoubleElectronInteractionData.get_pauli_list()[term_index])
        substring = SparsePauliOp(data=term, coeffs=sign * coeff)
        # add the operator to the whole thing
        substrings.append(substring)

    interaction_string = SparsePauliOp.sum(substrings)
    return interaction_string
