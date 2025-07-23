import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from modules.elements_utils import compute_B_matrix
from modules.shape_functions import shape_function_Q4

def plot_mesh(nodes, elements, show_ids=False):
    plt.figure()
    for e, elem in enumerate(elements):
        x = nodes[elem, 0]
        y = nodes[elem, 1]
        plt.plot(np.append(x, x[0]), np.append(y, y[0]), 'k-')
        if show_ids:
            cx = np.mean(x)
            cy = np.mean(y)
            plt.text(cx, cy, f"E{e}", color='blue', fontsize=8)
            for i, nid in enumerate(elem):
                plt.text(nodes[nid, 0], nodes[nid, 1], f"N{nid}", color='red', fontsize=7)
    plt.axis('equal')
    plt.title("Mesh")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def plot_displacements(nodes, elements, U, scale=1.0):
    displaced_nodes = nodes + scale * U.reshape(-1,2)
    plt.figure()
    for elem in elements:
        # Construir vector de desplazamientos ue [u1,v1,...,u4,v4]
        ue = np.zeros(8)
        for i, n in enumerate(elem):
            ue[2*i] = U[2*n]
            ue[2*i+1] = U[2*n+1]

        orig = nodes[elem]
        deformed = displaced_nodes[elem]
        plt.plot(np.append(orig[:,0], orig[0,0]), np.append(orig[:,1], orig[0,1]), 'k--')
        plt.plot(np.append(deformed[:,0], deformed[0,0]), np.append(deformed[:,1], deformed[0,1]), 'r-')
    plt.axis('equal')
    plt.title("Deformed shape")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def compute_stress_at_nodes(nodes, elements, U, D):
    n_nodes = nodes.shape[0]
    stress_nodal = np.zeros((n_nodes, 3))
    count = np.zeros(n_nodes)

    for elem in elements:
        # Construir vector de desplazamientos ue [u1,v1,...,u4,v4]
        ue = np.zeros(8)
        for i, n in enumerate(elem):
            ue[2*i] = U[2*n]
            ue[2*i+1] = U[2*n+1]

        coords = nodes[elem]
        ue = ue.reshape(8,)
        for xi, eta in [(-1/np.sqrt(3), -1/np.sqrt(3)),
                        (1/np.sqrt(3), -1/np.sqrt(3)),
                        (1/np.sqrt(3), 1/np.sqrt(3)),
                        (-1/np.sqrt(3), 1/np.sqrt(3))]:
            _, dN_dxi = shape_function_Q4(xi, eta)
            J = dN_dxi @ coords
            J_inv = np.linalg.inv(J)
            B = compute_B_matrix(dN_dxi, J_inv)
            stress = D @ (B @ ue)
            for i, nid in enumerate(elem):
                stress_nodal[nid] += stress
                count[nid] += 1
    stress_nodal /= count[:, None]
    return stress_nodal

def plot_stress_component(nodes, elements, U, E, nu, element_type, component='sxx'):
    comp_idx = {'sxx': 0, 'syy': 1, 'sxy': 2}[component]
    from modules.elements import constitutive_matrix
    D = constitutive_matrix(E, nu, element_type)
    stress_nodal = compute_stress_at_nodes(nodes, elements, U, D)
    x, y, z = nodes[:,0], nodes[:,1], stress_nodal[:, comp_idx]

    triangles = []
    for elem in elements:
        # Construir vector de desplazamientos ue [u1,v1,...,u4,v4]
        ue = np.zeros(8)
        for i, n in enumerate(elem):
            ue[2*i] = U[2*n]
            ue[2*i+1] = U[2*n+1]

        triangles.append([elem[0], elem[1], elem[2]])
        triangles.append([elem[0], elem[2], elem[3]])
    triang = tri.Triangulation(x, y, triangles)
    plt.figure()
    tpc = plt.tricontourf(triang, z, levels=30, cmap='viridis')
    plt.colorbar(tpc, label=component)
    plt.title(f"{component.upper()} Stress Field")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plot_von_mises(nodes, elements, U, E, nu, element_type):
    from modules.elements import constitutive_matrix
    D = constitutive_matrix(E, nu, element_type)
    stress_nodal = compute_stress_at_nodes(nodes, elements, U, D)
    sxx = stress_nodal[:, 0]
    syy = stress_nodal[:, 1]
    sxy = stress_nodal[:, 2]
    von_mises = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)

    x, y = nodes[:, 0], nodes[:, 1]
    triangles = []
    for elem in elements:
        # Construir vector de desplazamientos ue [u1,v1,...,u4,v4]
        ue = np.zeros(8)
        for i, n in enumerate(elem):
            ue[2*i] = U[2*n]
            ue[2*i+1] = U[2*n+1]

        triangles.append([elem[0], elem[1], elem[2]])
        triangles.append([elem[0], elem[2], elem[3]])
    triang = tri.Triangulation(x, y, triangles)
    plt.figure()
    tpc = plt.tricontourf(triang, von_mises, levels=30, cmap='plasma')
    plt.colorbar(tpc, label='von Mises')
    plt.title("Von Mises Stress Field")
    plt.axis('equal')
    plt.grid(True)
    plt.show()
