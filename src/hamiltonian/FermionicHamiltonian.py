"""File constructing the fermionic interaction hamiltonian in second quantization.
In this current approximation, single-electron-interaction hamiltonian and double-electron-interaction
 Hamiltonian parts are considered. """

from nptyping import NDArray, Float, Shape
from qiskit.opflow import PauliSumOp

from SingleElectronHamiltonian import generate_1e_hamiltonian
from DoubleElectronHamiltonian import generate_2e_hamiltonian


class FermionicHamiltonian:

    hamiltonian = None

    def fermionic_hamiltonian(self, molecule):
        """Creates the Fermionic excitation Hamiltonian for the given molecule.
        Will retrieve the necessary basis elements from basis set exchange,
        calculate the interaction integrals, compute the full excitation hamiltonian
        and return it.
        TODO: Parse Molecule and invoke Integration
        """
        pass


    # TODO: Probably set private later
    def compose_hamiltonian(self, num_qubits: int, weights_1e: NDArray[Shape['2'], Float], weights_2e: NDArray[Shape['4'], Float]):
        """From the number of qubits and the weight matrices for the respective interactions, form the hamiltonian
        and return it as qiskit PauliSumOp data structure."""
        single_electron_hamiltonian = generate_1e_hamiltonian(num_qubits, weights_1e)
        double_electron_hamiltonian = generate_2e_hamiltonian(num_qubits, weights_2e)

        self.hamiltonian = single_electron_hamiltonian.add(double_electron_hamiltonian)



