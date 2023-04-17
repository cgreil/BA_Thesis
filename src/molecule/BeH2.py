import pyscf
from pyscf import gto
from pyscf.gto.basis import parse_gaussian
import numpy as np

from nptyping import NDArray, Shape, Complex
from typing import Tuple

from AbstractMolecule import AbstractMolecule


class BeH2Integrals(AbstractMolecule):
    """Class providing staticmethod needed for creation of integrals"""

    def __init__(self, name: str, num_orbitals: int, num_electrons: int):
        super.name = name
        super.num_orbitals = num_orbitals
        super.num_electrons = num_electrons

    def get_integral_matrices(self) -> Tuple[NDArray, NDArray]:
        """Overwrites superclass method from AbstractMolecule"""
        molecule = BeH2Integrals._pyscf_create_beh2()
        integral_tuple = (BeH2Integrals._pyscf_1e_integrals(molecule), BeH2Integrals._pyscf_2e_integrals(molecule))
        return integral_tuple

    @staticmethod
    def _pyscf_1e_integrals(molecule: gto.Mole):
        """Function to calculate the 1-electron integrals in PySCF and returning the respective values as NxN matrix
        for a given molecule"""
        h1e = molecule.intor("int1e_kin", aosym='s1') + molecule.intor("int1e_nuc", aosym='s1')
        return h1e

    @staticmethod
    def _pyscf_2e_integrals(molecule: gto.Mole):
        """Function to calculate the 2-electron integrals in PySCF and returning the respective values as NxN matrix
        for a given molecule"""
        h2e = molecule.intor("int2e", aosym='s1')
        return h2e

    @staticmethod
    def _pyscf_create_beh2():
        """Function to create the beh2 molecule as pyscf object
         using the MO basis defined in the gbs file"""
        molecule = gto.Mole()
        # geometry of the molecule
        molecule.atom = '''
            Be 0. 0. 0. 
            H  0. 1. 0. 
            H  0. 0. 1. 
            '''
        # B for Bohr referring to Bohr radius
        molecule.unit = 'B'
        # custom sto-3g basis is being parsed here
        molecule.basis = {
            # absolute file paths used
            'Be': parse_gaussian.load(
                '/home/christoph/PycharmProjects/Ba_Thesis_Documents/src/resources/BeH2_basis_new.gbs', 'Be'),
            'H': parse_gaussian.load(
                '/home/christoph/PycharmProjects/Ba_Thesis_Documents/src/resources/BeH2_basis_new.gbs', 'H')
        }

        molecule.spin = 0
        molecule.build()
        # molecule.symmetry = True
        # molecule.symmetry_subgroup = 'DooH'
        return molecule

    @staticmethod
    def _dim2_from_dim4(array: NDArray[Shape['4'], Complex]) -> NDArray[Shape['2'], Complex]:
        """Util function taking a NxNxNxN array and returning a N^2xN^2 array in the following scheme:
        Element [p,q,r,s] in the 4 dim array will be found at position [N*p + r, N*q + s] in the 2 dimensional array.
        For example, element at [1,2,4,8] for N = 14 will be at [14*1 + 4, 14*2+8] i.e. [18,36].
        Useful for printing to files/tables.
        """
        N = array.shape[0]
        matrix = np.zeros((N ** 2, N ** 2))

        for p in range(N):
            for q in range(N):
                for r in range(N):
                    for s in range(N):
                        matrix[N * p + r][N * q + s] = array[p][q][r][s]
        return matrix
