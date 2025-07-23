import numpy as np
import json
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def load_mesh_data(problem_name):
    # Leer nodos y descartar ID y coordenada Z
    nodes_raw = np.loadtxt(f"Model_nodos_{problem_name}.sal")
    coords = nodes_raw[:, 1:3]  # x, y
    # Leer elementos y descartar ID, convertir a 0-based
    elems_raw = np.loadtxt(f"Model_elem_{problem_name}.sal", dtype=int)
    connectivity = elems_raw[:, 1:5] - 1
    return coords, connectivity

def apply_boundary_conditions(problem_name, n_nodes):
    bc = np.zeros(2 * n_nodes, dtype=bool)
    bc_data = np.loadtxt(f"Model_bc_{problem_name}.sal", dtype=int)
    for row in bc_data:
        node = int(row[0]) - 1
        dof = int(row[1])
        bc[2 * node + dof] = True
    return bc

def apply_loads(problem_name, n_nodes):
    F = np.zeros(2 * n_nodes)
    loads = np.loadtxt(f"Model_load_{problem_name}.sal")
    for row in loads:
        node = int(row[0]) - 1
        fx = float(row[1])
        fy = float(row[2])
        F[2 * node] += fx
        F[2 * node + 1] += fy
    return F

def solve_system(K, F, bc):
    K = lil_matrix(K)
    free_dofs = ~bc
    K_ff = K[free_dofs, :][:, free_dofs]
    F_f = F[free_dofs]
    U = np.zeros_like(F)
    U[free_dofs] = spsolve(K_ff.tocsr(), F_f)
    return U

def save_results(U, problem_name):
    u_dict = {f"U[{i}]": float(val) for i, val in enumerate(U)}
    with open("output_data.json", "w") as f:
        json.dump(u_dict, f, indent=4)
