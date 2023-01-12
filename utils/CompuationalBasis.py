import numpy as np



def get_basis(N):
    basis = []
    for i in range(N):
        elem = np.zeros(N)
        elem[i] = 1
        basis.append(elem)
    return basis


def kronecker_delta(i, j):
    if i == j:
        return i
    else:
        return 0


#def get_creation_decomposition()