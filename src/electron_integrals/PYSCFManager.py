import pyscf
from pyscf import gto
from pyscf.gto.basis import parse_gaussian
import numpy as np


class PYSCFManager:

    """Class providing staticmethod needed for creation of integrals"""

    @staticmethod
    def create_beh2():
        """Function to create the beh2 molecule as pyscf object
         using the MO basis defined in the gbs file"""
        mol = gto.Mole()
        mol.atom = '''
            Be 0. 0. 0. 
            H  0. 1. 0. 
            H  0. 0. 1. 
            '''
        # B for Bohr referring to Bohr radius
        mol.unit = 'B'
        mol.basis = {
            # file paths from content root
            'Be': parse_gaussian.load(
                '/home/christoph/PycharmProjects/Ba_Thesis_Documents/src/resources/BeH2_basis_new.gbs', 'Be'),
            'H': parse_gaussian.load(
                '/home/christoph/PycharmProjects/Ba_Thesis_Documents/src/resources/BeH2_basis_new.gbs',
                'H')
        }

        mol.spin = 0
        mol.build()
        # mol.symmetry = True
        # mol.symmetry_subgroup = 'DooH'
        return mol

    @staticmethod
    def print_beh2_HF_energy():
        """Reference method calculating the Hartree Fock Energy of BeH2 using PySCF"""
        print("Reference Energy:")
        beh2 = PYSCFManager.create_beh2()
        mf = beh2.UHF()
        mf.init_guess = '1e'
        mf.kernel()

    @staticmethod
    def get_1e_integrals(molecule):
        """Function to calculate the 1-electron integrals in PySCF and returning the respective values as NxN matrix
        for a given molecule"""
        h1e = molecule.intor("int1e_kin", aosym='s1') + molecule.intor("int1e_nuc", aosym='s1')
        return h1e

    @staticmethod
    def get_2e_integrals(molecule):
        """Function to calculate the 2-electron integrals in PySCF and returning the respective values as NxN matrix
        for a given molecule"""
        h2e = molecule.intor("int2e", aosym='s1')
        return h2e

    @staticmethod
    def dim2_from_dim4(array):
        """Util function taking a NxNxNxN array and returning a N^2xN^2 array in the following scheme:
        Element [p,q,r,s] in the 4 dim array will be found at position [N*p + r, N*q + s] in the 2 dimensional array.
        For example, element at [1,2,4,8] for N = 14 will be at [14*1 + 4, 14*2+8] i.e. [18,36].

        Useful for printing to files/tables."""
        N = array.shape[0]
        matrix = np.zeros((N ** 2, N ** 2))

        for p in range(N):
            for q in range(N):
                for r in range(N):
                    for s in range(N):
                        matrix[N * p + r][N * q + s] = array[p][q][r][s]
        return matrix


if __name__ == '__main__':
    beh2 = PYSCFManager.create_beh2()

    # print reference unrestricted hartree fock energy
    # print_beh2_HF_energy()
    print("1-Electron interaction integrals (kinetic + nuclear)")
    eri1 = np.array(PYSCFManager.get_1e_integrals(beh2))
    print(eri1.shape)
    print("2-Electron interaction integrals:")
    eri2 = np.array(PYSCFManager.get_2e_integrals(beh2))
    print(eri2.shape)

    # reshape eri2 so it can be written to file
    N = len(eri2)
    np.savetxt('eri1', eri1, fmt='%4.6f', delimiter=' ')
    # np.savetxt('eri2', eri2.reshape((N*N,N*N)), fmt='%4.6f', delimiter=' ')
    # Use custom reshape function to have well defined order
    np.savetxt('eri2', PYSCFManager.dim2_from_dim4(eri2), fmt='%4.6f', delimiter=' ')
