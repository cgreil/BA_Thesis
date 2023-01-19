import pyscf
from pyscf import gto
from pyscf.gto.basis import parse_gaussian
import numpy as np


def create_beh2():
    """Function to create the beh2 molecule as pyscf object
     using the MO basis defined in the gbs file"""
    mol = gto.Mole()
    mol.atom = '''
        Be 0. 0. 0. 
        H  0. 1. 0. 
        H  0. 0. 1. 
        '''
    mol.unit='B' #B for Bohr
    mol.basis={
        #For WSL interpreter use linux like paths
        'Be': parse_gaussian.load('/mnt/c/Users/cmose/PycharmProjects/BA/Calculation/resources/BeH2_basis_new.gbs', 'Be'),
        'H': parse_gaussian.load('/mnt/c/Users/cmose/PycharmProjects/BA/Calculation/resources/BeH2_basis_new.gbs', 'H')
        #'Be': parse_gaussian.load('C:\\Users\\cmose\\PycharmProjects\\BA\\Calculation\\resources\\BeH2_basis.gbs', 'Be'),
        #'H': parse_gaussian.load('/C:\\Users\\cmose\\PycharmProjects\\BA\\Calculation\\resources\\BeH2_basis.gbs', 'H')

    }

    mol.spin = 0
    mol.build()
    #mol.symmetry = True
    #mol.symmetry_subgroup = 'DooH'
    return mol



def print_beh2_HF_energy():
    """Reference method calculating the Hartree Fock Energy of BeH2 using PySCF"""
    print("Reference Energy:")
    beh2 = create_beh2()
    mf = beh2.UHF()
    mf.init_guess = '1e'
    mf.kernel()



def get_1e_integrals(molecule):
    """Function to calculate the 1-electron integrals in PySCF and returning the respective values as NxN matrix
    for a given molecule"""
    h1e = molecule.intor("int1e_kin", aosym='s1') + molecule.intor("int1e_nuc", aosym='s1')
    return h1e


def get_2e_integrals(molecule):
    """Function to calculate the 2-electron integrals in PySCF and returning the respective values as NxN matrix
    for a given molecule"""
    h2e = molecule.intor("int2e", aosym='s1')
    return h2e


if __name__ == '__main__':

    beh2 = create_beh2()

    #print reference
    #print_beh2_HF_energy()
    print("1-Electron interaction integrals (kinetic + nuclear)")
    eri1 = np.array(get_1e_integrals(beh2))
    print(eri1.shape)
    print("2-Electron interaction integrals:")
    eri2 = np.array(get_2e_integrals(beh2))
    print(eri2.shape)

    #reshape eri2 so it can be written to file
    N = len(eri2)
    np.savetxt('eri1', eri1, fmt='%4.6f', delimiter=' ')
    np.savetxt('eri2', eri2.reshape((N*N,N*N)), fmt='%4.6f', delimiter=' ')


