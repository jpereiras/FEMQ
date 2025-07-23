import numpy as np
from modules.elements_utils import compute_B_matrix, compute_element_stiffness

def constitutive_matrix(E, nu, element_type):
    """
    Devuelve la matriz constitutiva D para el tipo de análisis indicado.
    """
    if element_type == "CPS4":
        # Plane stress
        coef = E / (1 - nu**2)
        D = coef * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])
    elif element_type == "CPE4":
        # Plane strain
        coef = E / ((1 + nu) * (1 - 2 * nu))
        D = coef * np.array([
            [1 - nu,     nu,         0],
            [nu,         1 - nu,     0],
            [0,          0,      (1 - 2 * nu) / 2]
        ])
    elif element_type == "CAX4":
        # Axisymmetric - simplificada (solo sirve para referencias)
        coef = E / ((1 + nu) * (1 - 2 * nu))
        D = coef * np.array([
            [1 - nu, nu, nu, 0],
            [nu, 1 - nu, nu, 0],
            [nu, nu, 1 - nu, 0],
            [0, 0, 0, (1 - 2 * nu) / 2]
        ])
    else:
        raise ValueError("Tipo de elemento no reconocido")
    return D

def assemble_stiffness_matrix(nodes, elements, D):
    """
    Ensambla la matriz global de rigidez.
    """
    n_nodes = nodes.shape[0]
    K = np.zeros((2 * n_nodes, 2 * n_nodes))

    for elem in elements:
        ke = compute_element_stiffness(nodes[elem], D)
        dof = []
        for n in elem:
            dof.extend([2*n, 2*n+1])
        for i in range(8):
            for j in range(8):
                K[dof[i], dof[j]] += ke[i, j]
    return K, None  # Placeholder para DOF map si se expande más adelante
