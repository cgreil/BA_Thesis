"""File containing helper functions to create Pauli strings"""

from typing import Dict


def pauli_string_from_dict(num_qubits: int, non_identity: Dict[int, str] | None):
    """Creates a string representing a tensor product of Pauli matrices where all elements are I elements except the
    elements within the dict, where for a pair of (key, value), the pauli gate at position <key> has the identifier
     <value>."""
    # Standard is identity
    pauli_string = ['I' for _ in range(num_qubits)]

    if non_identity is None:
        return pauli_string

    else:
        for key in non_identity:
            identifier = non_identity[key]
            # check whether identifier is allowed
            if len(identifier) > 1:
                raise ValueError("Only gates with Identifiers of length 1 are allowed")
            else:
                pauli_string[key] = identifier

    return ''.join(pauli_string)
