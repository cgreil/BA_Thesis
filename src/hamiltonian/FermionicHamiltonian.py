"""File constructing the fermionic interaction hamiltonian in second quantization.
In this current approximation, single-electron-interaction hamiltonian and double-electron-interaction
Hamiltonian parts are considered.
 """

from pyscf.gto import Mole

from nptyping import NDArray, Float, Shape
from qiskit.opflow import PauliSumOp, OperatorBase

from SingleFermionInteractionHamiltonian import generate_1e_hamiltonian
from DoubleFermionInteractionHamiltonian import generate_2e_hamiltonian
from ..electron_integrals.BeH2Integrals import BeH2Integrals


class FermionicHamiltonian:

    num_qubits: int = 0
    # Hamiltonian object which will be stored as PauliSumOp
    hamiltonian_operator: PauliSumOp = None
    single_interaction_weights: NDArray[Shape['4'], Float]
    double_interaction_weights: NDArray[Shape['2'], Float]

    # constructor
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        # invoke hamiltonian generation
        self.fermionic_hamiltonian()

    def fermionic_hamiltonian(self):
        """Creates the Fermionic excitation Hamiltonian for the given molecule.
        Will retrieve the necessary basis elements from basis set exchange,
        calculate the interaction integrals, compute the full excitation hamiltonian
        and return it.

        Invoked at __init__ time, but is idempotent and can be reinvoked arbitrarily
        """
        # calculate integrals using dedicated class, unpack 2-tuple
        eri1_matrix, eri2_matrix = BeH2Integrals.get_integral_matrices()
        self.single_interaction_weights = eri1_matrix
        self.double_interaction_weights = eri2_matrix

        # invoke generation of hamiltonian
        self._generate_hamiltonian_operator()

    def get_hamiltonian(self) -> PauliSumOp:
        """Returns the PauliSumOp object"""
        if self.hamiltonian_operator is None:
            raise ValueError("Hamiltonian is NoneObject")
        else:
            return self.hamiltonian_operator

    def get_matrix(self) -> NDArray:
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

    def _generate_hamiltonian_operator(self):
        """From the number of qubits and the weight matrices for the respective interactions, form the hamiltonian
        and return it as qiskit PauliSumOp data structure."""

        single_electron_hamiltonian = generate_1e_hamiltonian(self.num_qubits, self.single_interaction_weights)
        double_electron_hamiltonian = generate_2e_hamiltonian(self.num_qubits, self.double_interaction_weights)

        self.hamiltonian_operator = single_electron_hamiltonian.add(double_electron_hamiltonian)
