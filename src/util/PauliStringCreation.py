"""File containing helper functions to create Pauli strings"""

from typing import Dict


def pauli_string_from_dict(num_qubits: int, non_identity: Dict[int, str] | None):
    """Creates a string representing a tensor product of Pauli matrices where all elements are I elements except the
    elements within the dict, where for a pair of (key, value), the pauli gate at position <key> has the identifier
     <value>."""
    # Standard is identity
    pauli_list = ['I' for _ in range(num_qubits)]

    if non_identity is None:
        return ''.join(pauli_list)

    if len(non_identity) > num_qubits:
        raise ValueError("Non-identity dic length cannot be larger than the number of qubits")

    else:
        for key in non_identity:
            identifier = non_identity[key]
            # check whether identifier is allowed
            if identifier not in ['X', 'Y', 'Z']:
                raise ValueError("Only Pauli gates are allowed as identifiers")
            else:
                pauli_list[key] = identifier

    # create string from list
    return ''.join(pauli_list)
