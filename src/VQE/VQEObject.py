"""VQE Object that will be built by VQE Builder and run by VQE Runner. stores the
operators needed and provides the methods necessary for measurements, gradients, expectation values
and the likes."""

from qiskit.circuit import ParameterVector, Parameter

from qiskit.opflow import OperatorBase, PauliSumOp
from qiskit.opflow.gradients import *
from qiskit.opflow.expectations import *
from qiskit.opflow.state_fns import CircuitStateFn

from src.ansatz.RandomWeightGenerator import generate_random_2dim, generate_random_4dim
from src.VQE.VQEBuilder import *


class VQEObject:
    num_qubits: int
    num_occupied: int
    reference_operator: OperatorBase = None
    ansatz_operator: PauliSumOp = None
    hamiltonian_operator: PauliSumOp = None

    exp_ansatz_op: OperatorBase = None
    exp_hamiltonian_op: OperatorBase = None

    # TODO: Wrap num_qubits, num_occupied, and all things specific to the molecule into a dedicated object
    def __init__(self, num_qubits: int, num_occupied: int):
        self.num_qubits = num_qubits
        self.num_occupied = num_occupied

        # generate reference op
        self.reference_operator = VQEBuilder.build_reference_state_operator(self.num_qubits, self.num_occupied)

        # generate ansatz op with random values
        eri1_weights = generate_random_2dim(self.num_qubits)
        eri2_weights = generate_random_4dim(self.num_qubits)
        self.ansatz_operator = VQEBuilder.build_kUCC_ansatz_operator(self.num_qubits, eri1_weights, eri2_weights)
        self.exp_ansatz_op = self.ansatz_operator.exp_i()

        # generate hamiltonian op
        self.hamiltonian_operator = VQEBuilder.build_hamiltonian_operator(self.num_qubits)
        self.exp_hamiltonian_op = self.hamiltonian_operator.exp_i()

    def measure(self):
        """Method to perform a measurement onto the operators"""
        pass

    def full_op(self):
        return self.exp_ansatz_op.adjoint().compose(self.exp_hamiltonian_op)

    def expectation_value(self):
        """Method to retrieve the expectation value """
        # create |000...> state to measure
        psi0 = CircuitStateFn(QuantumCircuit(self.num_qubits))
        # do measurement <psi|U_ref^dag e^H U_ref|psi>
        return psi0.adjoint().compose(self.exp_ansatz_op.adjoint()).compose(self.exp_hamiltonian_op) \
            .compose(self.exp_ansatz_op).compose(psi0)

    def gradient(self):
        """Method to retrieve the first order gradient for the respective hamiltonian"""
        # TODO: Change
        # a = Parameter('a')
        # theta_vec = [a, b, c]
        gradient = Gradient().convert(self.full_op(), params=[])
        return gradient

    def hamiltonian_matrix(self):
        """Method to create the matrix representation of the hamiltonian operator and return it"""
        return self.hamiltonian_operator.to_matrix()
