import numpy as np

def shape_function_Q4(xi, eta):
    """
    Devuelve las funciones de forma y sus derivadas respecto a xi y eta.
    """
    N = 0.25 * np.array([
        (1 - xi)*(1 - eta),
        (1 + xi)*(1 - eta),
        (1 + xi)*(1 + eta),
        (1 - xi)*(1 + eta)
    ])

    dN_dxi = np.array([
        [-(1 - eta), -(1 - xi)],
        [ (1 - eta), -(1 + xi)],
        [ (1 + eta),  (1 + xi)],
        [-(1 + eta),  (1 - xi)]
    ]) * 0.25

    return N, dN_dxi.T  # dN_dxi: 2x4


def shape_function_derivatives_Q4(xi, eta):
    """
    Derivadas de las funciones de forma Q4 respecto a xi y eta.
    Devuelve una matriz 2x4 con [dNi/dxi; dNi/deta].
    """
    dN_dxi = np.array([
        [-(1 - eta), -(1 - xi)],
        [ (1 - eta), -(1 + xi)],
        [ (1 + eta),  (1 + xi)],
        [-(1 + eta),  (1 - xi)]
    ]) * 0.25
    return dN_dxi.T
