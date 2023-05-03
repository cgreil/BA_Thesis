from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, ControlledGate
from qiskit.opflow import PauliSumOp


class QPE:
    """
    Wrapper class for the QPE algorithm
    """

    QPECircuit: QuantumCircuit = None
    unitaryOp: PauliSumOp = None
    ansatzCircuit: QuantumCircuit = None
    num_readout_qubits: int
    num_state_qubits: int

    def __init__(self, unitaryOp: PauliSumOp, ansatzCirc: QuantumCircuit, num_readout: int, num_state: int):
        """init method which enables the object creation for the case that the ansatz is given as circuit form"""
        self.unitaryCircuit = unitaryOp
        self.ansatzCircuit = ansatzCirc
        self.num_readout_qubits = num_readout
        self.num_state_qubits = num_state

        self._build_from_circuits()

    @classmethod
    def from_ops(cls, unitaryOp: PauliSumOp, ansatzOp: PauliSumOp, num_readout: int, num_state: int):
        """Classmethod that takes the Ansatz and the Unitary in PauliSumOp form and enables QPE construction from there"""
        ansatzCirc = ansatzOp.to_circuit()

        return cls(unitaryOp, ansatzCirc, num_readout, num_state)

    def _build_from_circuits(self):
        """
        Creates the full circuit for the QPE calculation as auxiliary method for __init__()
        """

        # Make sure the state register has the correct size
        assert self.ansatzCircuit.num_qubits == self.num_state_qubits
        assert self.unitaryCircuit.num_qubits == self.num_state_qubits


        # Start building full circuit
        state_reg = QuantumCircuit(self.num_state_qubits)
        readout_reg = QuantumCircuit(self.num_readout_qubits)

        # Create equal superposition in readout reg by applying hadamard on all readout qubits
        readout_reg.h(range(0, self.num_readout_qubits))
        # Apply the Ansatz circuit onto the state register
        state_reg.compose(self.ansatzCircuit)

        # Create Controlled Unitary from the unitary operator
        #gate = Gate(self.unitaryOp, self.num_state_qubits)
        #controlled_unitary = ControlledGate(,  )