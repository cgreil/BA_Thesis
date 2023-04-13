"""File constructing the fermionic interaction hamiltonian in second quantization.
In this current approximation, single-electron-interaction hamiltonian and double-electron-interaction
Hamiltonian parts are considered.
 """

from nptyping import NDArray, Float, Shape, ndarray
from qiskit.opflow import PauliSumOp, OperatorBase

from SingleFermionInteractionHamiltonian import generate_1e_hamiltonian
from DoubleFermionInteractionHamiltonian import generate_2e_hamiltonian


class FermionicHamiltonian:

    num_qubits: int = 0
    # Hamiltonian object which will be stored as PauliSumOp
    hamiltonian_operator: PauliSumOp = None

    # constructor
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        # invoke hamiltonian generation
        self._generate_hamiltonian_operator(self.num_qubits, None, None)

    def fermionic_hamiltonian(self, molecule):
        """Creates the Fermionic excitation Hamiltonian for the given molecule.
        Will retrieve the necessary basis elements from basis set exchange,
        calculate the interaction integrals, compute the full excitation hamiltonian
        and return it.
        TODO: Parse Molecule and invoke Integration
        """
        pass


    # TODO: Set private later
    def _generate_hamiltonian_operator(self, num_qubits: int, weights_1e: NDArray[Shape['2'], Float],
                                       weights_2e: NDArray[Shape['4'], Float]):
        """From the number of qubits and the weight matrices for the respective interactions, form the hamiltonian
        and return it as qiskit PauliSumOp data structure."""
        single_electron_hamiltonian = generate_1e_hamiltonian(num_qubits, weights_1e)
        double_electron_hamiltonian = generate_2e_hamiltonian(num_qubits, weights_2e)

        self.hamiltonian_operator = single_electron_hamiltonian.add(double_electron_hamiltonian)

    def get_hamiltonian(self) -> PauliSumOp:
        """Returns the PauliSumOp object"""
        if self.hamiltonian_operator is None:
            raise ValueError("Hamiltonian is NoneObject")
        else:
            return self.hamiltonian_operator

    def get_matrix(self) -> ndarray:
        """Returns the matrix representation of the hamiltonian"""
        if self.hamiltonian_operator is None:
            raise ValueError("Hamiltonian is NoneObject")
        else:
            return self.hamiltonian_operator.to_matrix()

    def exponential(self) -> OperatorBase:
        """Returns a CircuitOperator equivalent to the exponentiation of the Hamiltonian"""
        if self.hamiltonian_operator is None:
            raise ValueError("Hamiltonian is NoneObject")
        else:
            return self.hamiltonian_operator.exp_i()
