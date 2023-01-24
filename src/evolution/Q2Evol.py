import qiskit
from qiskit.circuit.library import PauliGate
from qiskit.quantum_info  import Operator, Pauli
from typing import *

class Q2Evol:
    """class for intermediate representation of evolution operator on 2 qubits, that
    are in general nondajacent"""
    q1id = None
    q2id = None
    operationQ1 = None
    operationQ2 = None
    #constant factor applied to both gates
    factor = None

    def getCircuit(self):
        """Method to create a circuit implementing the 2qubit evolution, that can later be composed
        together with other circuits to create the whole thing"""

        assert(self.q1id != None)
        assert(self.q2id != None)

        id1 = self.q1id
        id2 = self.q2id

        #calculate size of subcircuit
        N = None
        if (id1 < id2):
            N = id2 - id1
        else:
            N = id1 - id2

        circ = qiskit.QuantumCircuit(N)

        #TODO: actual implementation, just test for now
        for i in range(N):
            circ.h(i)

        return circ


