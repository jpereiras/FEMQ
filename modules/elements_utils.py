import numpy as np
from modules.shape_functions import shape_function_Q4, shape_function_derivatives_Q4

def compute_B_matrix(dN_dxi, J_inv):
    """
    Calcula la matriz B para un elemento Q4 dado el Jacobiano inverso.
    """
    dN_dx = J_inv @ dN_dxi  # (2x4)
    B = np.zeros((3, 8))
    for i in range(4):
        B[0, 2*i]     = dN_dx[0, i]
        B[1, 2*i+1]   = dN_dx[1, i]
        B[2, 2*i]     = dN_dx[1, i]
        B[2, 2*i+1]   = dN_dx[0, i]
    return B

def compute_element_stiffness(coords, D):
    """
    Calcula la matriz de rigidez de un elemento Q4 usando 2x2 Gauss.
    """
    ke = np.zeros((8, 8))
    gauss_pts = [(-1/np.sqrt(3), -1/np.sqrt(3)),
                 ( 1/np.sqrt(3), -1/np.sqrt(3)),
                 ( 1/np.sqrt(3),  1/np.sqrt(3)),
                 (-1/np.sqrt(3),  1/np.sqrt(3))]

    for xi, eta in gauss_pts:
        N, dN_dxi = shape_function_Q4(xi, eta)
        J = dN_dxi @ coords  # Jacobiano 2x2
        detJ = np.linalg.det(J)
        J_inv = np.linalg.inv(J)
        B = compute_B_matrix(dN_dxi, J_inv)
        ke += B.T @ D @ B * detJ
    return ke
