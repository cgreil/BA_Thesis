import math

import basis_set_exchange as bse
import json

import sympy
from sympy import *
from typing import *
from scipy.integrate import quad


def get_sto3g_json(elemnumber: int):
    """Invokes an API call to Basis Set Exchange and retrieves the coefficients for the element
    with the periodic number corresponding to elemnumber. Returns the retrieved information in
    json format"""

    return bse.get_basis('STO-3G', elements=[elemnumber], fmt='json')


def get_sto_3g_coefficients(elemnumber: int, level: int):
    """For a given element number and shell, returns the coeffs as list. The sequence of shell
    numbers follows the Aufbau principle"""
    reply = get_sto3g_json(elemnumber)
    jsonobject = json.loads(reply)

    element = jsonobject['elements'][str(elemnumber)]
    shell = element['electron_shells'][level]
    return shell['coefficients']


def get_sto_3g_exponents(elemnumber: int, level: int):
    """For a given element number and shell, returns the exponents as list. The sequence of shell
    numbers follows the Aufbau principle"""
    reply = get_sto3g_json(elemnumber)
    jsonobject = json.loads(reply)

    element = jsonobject['elements'][str(elemnumber)]
    shell = element['electron_shells'][level]
    return shell['exponents']


def print_AOs():
    # hydrogen
    h_coeffs = get_sto_3g_coefficients(1, 0)
    h_exponents = get_sto_3g_exponents(1, 0)
    print("Hydrogen coefficients: ")
    print(h_coeffs[0])
    print("Hydrogen exponents: ")
    print(h_exponents)
    print("\n")

    print("Be 1s orbital:")
    be_coeffs = get_sto_3g_coefficients(4, 0)
    be_exponents = get_sto_3g_exponents(4, 0)
    print("Beryllium coeffs: ")
    print(be_coeffs)
    print("Beryllium exponents: ")
    print(be_exponents)
    print("\n")

    print("Be 2s orbital:")
    be_coeffs = get_sto_3g_coefficients(4, 1)
    be_exponents = get_sto_3g_exponents(4, 1)
    print("Beryllium coeffs: ")
    print(be_coeffs[0])
    print("Beryllium exponents: ")
    print(be_exponents)

    print("Be 2p orbital:")
    be_coeffs = get_sto_3g_coefficients(4, 1)
    be_exponents = get_sto_3g_exponents(4, 1)
    print("Beryllium coeffs: ")
    print(be_coeffs[1])
    print("Beryllium exponents: ")
    print(be_exponents)


def get_gaussian_primitive_sym(alpha: float):
    """Function returning the Gaussian primitive symbolically in r"""
    r = Symbol('r')
    primitive = pow((2 * alpha) / math.pi, (3 / 4)) * sympy.exp(-alpha * r ** 2)
    return primitive


def get_STO(coefficients: list, exponents: list):
    """Get the STO form as linear combination of the individual primitives"""

    N = len(coefficients)
    assert (N == len(exponents))

    # convert str list to int list
    if (type(coefficients[0]) == str):
        coeffs = [float(elem) for elem in coefficients]
    else:
        coeffs = coefficients

    if (type(exponents[0]) == str):
        exps = [float(elem) for elem in exponents]
    else:
        exps = exponents

    STO = 0
    for i in range(N):
        STO = STO + coeffs[i] * get_gaussian_primitive_sym(exps[i])
    return STO


# test
if __name__ == '__main__':
    # hydrogen
    h_exponents = get_sto_3g_exponents(1, 0)
    h_coeffs = get_sto_3g_coefficients(1, 0)[0]

    h_orbital_1s = get_STO(h_coeffs, h_exponents)
    print("Hydrogen 1s MO:")
    print(h_orbital_1s)
    print("\n")

    # beryllium
    be_1s_exponents = get_sto_3g_exponents(4, 0)
    be_1s_coeffs = get_sto_3g_coefficients(4, 0)[0]
    be_orbital_1s = get_STO(be_1s_coeffs, be_1s_exponents)
    print("Beryllium 1s MO:")
    print(be_orbital_1s)
    print("\n")

    be_2s_exponents = get_sto_3g_exponents(4, 1)
    be_2s_coeffs = get_sto_3g_coefficients(4, 1)[0]
    be_orbital_2s = get_STO(be_2s_coeffs, be_2s_exponents)
    print("Beryllium 2s MO:")
    print(be_orbital_2s)
    print("\n")

    be_2p_exponents = get_sto_3g_exponents(4, 1)
    be_2p_coeffs = get_sto_3g_coefficients(4, 1)[1]
    be_orbital_2p = get_STO(be_2p_coeffs, be_2p_exponents)
    print("Beryllium 2p MO:")
    print(be_orbital_2p)
    #Symbolic integration with sympy
    print(integrate(be_orbital_2p, (Symbol('r'),0,100)))
