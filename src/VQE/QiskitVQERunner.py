from src.molecule.AbstractMolecule import AbstractMolecule
from src.VQE.VQEBuilder import VQEBuilder

from nptyping import Float, NDArray
from qiskit.algorithms import VQE
from qiskit.opflow import OperatorBase
from qiskit_nature.circuit.library.ansatzes import UCC
# import ucc ansatz from qiskit-nature
# import fitting gradient
from qiskit.algorithms.optimizers import COBYLA


class QiskitVQERunner:
    """The Runner class that configures the VQE algorithm specifics, such as the Ansatz,
    the molecule and the specifics of the algorithm evaluation.
    Specifically, QiskitVQERunner will utilize the Qiskit specific VQE calss to perform
    the optimization"""

    num_shots: int = 0
    final_energy: Float = 0.0
    molecule: AbstractMolecule = None
    solver: VQE = None
    hamiltonian: OperatorBase

    def __init__(self, num_shots: int, molecule: AbstractMolecule):
        """Init function creating the runner, specifically constructing the VQE Solver needed for
        Eigentsate creation"""
        self.num_shots = num_shots
        self.molecule = molecule

        # get Ansatz, Hamiltonian, Reference state
        reference = VQEBuilder.build_reference_state_operator(molecule.num_orbitals, molecule.num_electrons)
        eri1_weights, eri2_weights = molecule.get_integral_matrices()
        ansatz = VQEBuilder.build_kUCC_ansatz_operator(molecule.num_orbitals, eri1_weights, eri2_weights)

        # TODO: Change so that only molecule is provided as parameter
        self.hamiltonian = VQEBuilder.build_hamiltonian_operator(molecule.num_orbitals, molecule)

        # Create Qiskit VQE Object
        # supply ansatz, optimizer, gradient,
        self.solver = VQE()
        # supply number of max_shots
        self.solver.ansatz = UCC()

    def run(self):
        # perform the computation
        return self.solver.compute_minimum_eigenvalue(self.hamiltonian)
