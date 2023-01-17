from pyscf import gto
from pyscf.gto.basis import parse_gaussian


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
        'Be': parse_gaussian.load('/mnt/c/Users/cmose/PycharmProjects/BA/Calculation/resources/BeH2_basis.gbs', 'Be'),
        'H': parse_gaussian.load('/mnt/c/Users/cmose/PycharmProjects/BA/Calculation/resources/BeH2_basis.gbs', 'H')
        #'Be': parse_gaussian.load('C:\\Users\\cmose\\PycharmProjects\\BA\\Calculation\\resources\\BeH2_basis.gbs', 'Be'),
        #'H': parse_gaussian.load('/C:\\Users\\cmose\\PycharmProjects\\BA\\Calculation\\resources\\BeH2_basis.gbs', 'H')

    }

    mol.spin = 0
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



if __name__ == '__main__':
    #print reference
    print_beh2_HF_energy()



