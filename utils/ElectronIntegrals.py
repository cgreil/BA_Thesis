import math
import numpy as np
from scipy.integrate import quad
from sympy import lambdify
from sympy.abc import r

def integrand(r):
    return 0.136444023226142 * math.exp(-1.31483311 * r ** 2) + 0.177986726950098 * math.exp(
        -0.3055389383 * r ** 2) + 0.0494416395477091 * math.exp(-0.0993707456 * r ** 2)


"""Module for calculating one- and two- electron interaction integrals """


if __name__ == '__main__':
    #numeric integration with scipy
    print(quad(integrand, 0, 1))
    print("Hi")


def int_1e(basis:list):
    """Takes a list of MO basis functions of size N and returns a NxN matrix describing the
    one-electron interaction between the electrons in the respective MOs"""
    #prepare NxN matrix
    N = len(basis)
    integrals = np.zeros((N,N))

    #outer iteratation
    for p in range(N):
        # turn sympy value to usual python expression
        numeric_MO1 = lambdify(basis[p], r)
        #inner iteration:
        for q in range(N):
            numeric_MO2 = lambdify(basis[q], r)
            #define r1 and r2


    return integrals


def int_2e(basis:list):
    """Takes a list of MO basis functions of size N and returns a NxNxNxN matrix describing the two-electron interaction
    between the electrons in the respective MOs"""
    #prepare NxNxNxN matrix
    N = len(basis)
    integrals = np.zeros((N,N,N,N))

    #first iteration
    for p in range(N):
        #second iteration
        for q in range(N):
            #third iteration
            for r in range(N):
                #fourth iteration
                for s in range(N):
                    integrals[p,q,r,s] = 1


    return integrals



def lambdify_sympy(expression, symbol):
    """Auxilliary function taking a sympy expression and a symbol
     and using lambdify to create a python expression from it"""
    return lambdify(expression, symbol)








