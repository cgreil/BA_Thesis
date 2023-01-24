"""Module intended for full creation of the VQE circuit by putting all required building blocks together."""
from src.evolution.Q2Evol import *


def compileVQECircuit():
    evol = Q2Evol()
    evol.q1id = 0
    evol.q2id = 2
    return evol.getCircuit()



if __name__ == '__main__':
    print("Circuit:")

    circ = compileVQECircuit()
    circ.to_gate()
    print(circ.qasm())
    #circ.draw()


