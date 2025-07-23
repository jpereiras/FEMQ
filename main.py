import json
from modules.elements import assemble_stiffness_matrix, constitutive_matrix
from modules.postprocessing import plot_mesh, plot_displacements, plot_stress_component, plot_von_mises
from modules.io_utils import load_mesh_data, apply_boundary_conditions, apply_loads, solve_system, save_results

def run_fem_analysis(input_data_path):
    # Leer archivo JSON
    with open(input_data_path, "r") as f:
        input_data = json.load(f)

    E = input_data["E"]
    nu = input_data["nu"]
    eltype = input_data["type"]
    problem_name = input_data["problemName"]

    # Cargar malla y condiciones
    nodes, elements = load_mesh_data(problem_name)
    K, dof_map = assemble_stiffness_matrix(nodes, elements, constitutive_matrix(E, nu, eltype))
    F = apply_loads(problem_name, len(nodes))
    bc = apply_boundary_conditions(problem_name, len(nodes))

    # Resolver sistema
    U = solve_system(K, F, bc)

    # Guardar resultados
    save_results(U, problem_name)

    # Plots
    plot_mesh(nodes, elements, show_ids=True)
    plot_displacements(nodes, elements, U, scale=10)
    plot_stress_component(nodes, elements, U, E, nu, eltype, component='sxx')
    plot_stress_component(nodes, elements, U, E, nu, eltype, component='syy')
    plot_stress_component(nodes, elements, U, E, nu, eltype, component='sxy')
    plot_von_mises(nodes, elements, U, E, nu, eltype)

if __name__ == "__main__":
    run_fem_analysis("input_data.json")
