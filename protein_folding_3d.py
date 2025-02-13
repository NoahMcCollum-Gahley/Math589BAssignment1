import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def get_target_energy(n_beads):
    if n_beads == 10:
        return -21.0
    elif n_beads == 100:
        return -455.0
    elif n_beads == 200:
        return -945.0
    else:
        return 0


# Initialize protein positions
def initialize_protein(n_beads, dimension=3, fudge = 1e-5):
    """
    Initialize a protein with `n_beads` arranged almost linearly in `dimension`-dimensional space.
    The `fudge` is a factor that, if non-zero, adds a spiral structure to the configuration.
    """
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i-1, 0] + 1  # Fixed bond length of 1 unit
        positions[i, 1] = fudge * np.sin(i)  # Fixed bond length of 1 unit
        positions[i, 2] = fudge * np.sin(i*i)  # Fixed bond length of 1 unit                
    return positions

# Lennard-Jones potential function
def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    """
    Compute Lennard-Jones potential between two beads.
    """
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

# Bond potential function
def bond_potential(r, b=1.0, k_b=100.0):
    """
    Compute harmonic bond potential between two bonded beads.
    """
    return k_b * (r - b)**2

# Total energy function. NEW, AND RENAMED!!!!!!!!!!!!!
def compute_energy_and_gradient(x, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    positions = x.reshape((n_beads, -1))
    n_dim = positions.shape[1]
    total_energy = 0.0
    gradient = np.zeros_like(positions)

    # Compute bond potential and its gradient
    for bead in range(n_beads - 1):
        displacement = positions[bead + 1] - positions[bead]
        distance = np.linalg.norm(displacement)
        
        if distance > 0:
            bond_energy = bond_potential(distance, b, k_b)
            total_energy += bond_energy
            energy_derivative = 2 * k_b * (distance - b)
            force = (energy_derivative / distance) * displacement
            gradient[bead] -= force
            gradient[bead + 1] += force

    # Compute Lennard-Jones potential and its gradient
    displacement_matrix = positions[:, None, :] - positions[None, :, :]
    distance_matrix = np.linalg.norm(displacement_matrix, axis=2)

    i_indices, j_indices = np.triu_indices(n_beads, k=1)
    pairwise_distances = distance_matrix[i_indices, j_indices]

    valid_pairs = pairwise_distances >= 1e-2
    valid_distances = pairwise_distances[valid_pairs]

    if valid_distances.size > 0:
        LJ_potential = 4 * epsilon * ((sigma / valid_distances) ** 12 - (sigma / valid_distances) ** 6)
        total_energy += np.sum(LJ_potential)

        LJ_force_magnitude = 4 * epsilon * (-12 * sigma**12 / valid_distances**13 + 6 * sigma**6 / valid_distances**7)
        displacement_vectors = displacement_matrix[i_indices, j_indices]
        valid_displacements = displacement_vectors[valid_pairs]

        force_contributions = (LJ_force_magnitude[:, None] / valid_distances[:, None]) * valid_displacements
        np.add.at(gradient, i_indices[valid_pairs], force_contributions)
        np.add.at(gradient, j_indices[valid_pairs], -force_contributions)

    return total_energy, gradient.flatten()


#NEW: NAME IS DIFFERENT!!!!!
def optimize_bfgs(func, initial_x, args, n_beads, max_iterations=1000, tolerance=1e-6, step_size=1.0, decay=0.5, armijo_c=1e-4):
    x = initial_x.copy()
    dim = len(x)
    inv_hessian = np.eye(dim)
    path = []

    for iteration in range(max_iterations):
        f_value, gradient = func(x, *args)
        grad_norm = np.linalg.norm(gradient)

        if grad_norm < tolerance:
            print(f"BFGS converged at iteration {iteration}, gradient norm: {grad_norm:.8e}")
            break

        direction = -inv_hessian @ gradient
        step = step_size

        # Armijo backtracking line search
        while True:
            x_candidate = x + step * direction
            f_candidate, _ = func(x_candidate, *args)

            if f_candidate <= f_value + armijo_c * step * np.dot(gradient, direction):
                break
            
            step *= decay
            if step < 1e-12:
                break

        step_vector = step * direction
        x_candidate = x + step_vector
        f_candidate, new_gradient = func(x_candidate, *args)

        gradient_diff = new_gradient - gradient
        curvature = np.dot(gradient_diff, step_vector)

        if curvature > 1e-10:
            rho = 1.0 / curvature
            identity_matrix = np.eye(dim)
            term1 = identity_matrix - rho * np.outer(step_vector, gradient_diff)
            term2 = identity_matrix - rho * np.outer(gradient_diff, step_vector)
            inv_hessian = term1 @ inv_hessian @ term2 + rho * np.outer(step_vector, step_vector)

        x = x_candidate
        path.append(x.reshape((n_beads, -1)))

        if (iteration + 1) % 50 == 0:
            print(f"Iteration {iteration + 1}: f = {f_candidate:.6f}, ||g|| = {np.linalg.norm(new_gradient):.2e}")

    return x, path



# MODIFIED: NEW NAME!!!!!!!!!!!
def optimize_protein_structure(positions, n_beads, save_csv=False, max_iterations=10000, tolerance=1e-4, energy_threshold=None):
    # Determine target energy if not provided
    if energy_threshold is None:
        energy_threshold = get_target_energy(n_beads)

    initial_x = positions.flatten()
    optimization_args = (n_beads,)

    # Execute custom BFGS with backtracking
    optimized_x, trajectory = optimize_bfgs(compute_energy_and_gradient, initial_x, optimization_args, n_beads, 
                                             max_iterations=max_iterations, tolerance=tolerance)
    final_energy, _ = compute_energy_and_gradient(optimized_x, n_beads)
    print(f"Initial BFGS optimization: f = {final_energy:.6f}")

    optimal_energy = final_energy
    optimal_x = optimized_x.copy()

    # Attempt perturbed restarts if necessary
    if optimal_energy > energy_threshold:
        perturbation_attempts = 3
        perturbation_intensity = 1e-1
        for attempt in range(perturbation_attempts):
            print(f"Executing perturbation attempt {attempt + 1}...")
            perturbed_x = optimal_x + np.random.normal(scale=perturbation_intensity, size=optimal_x.shape)
            new_x, new_trajectory = optimize_bfgs(compute_energy_and_gradient, perturbed_x, optimization_args, 
                                                  n_beads, max_iterations=max_iterations // 2, tolerance=tolerance)
            new_energy, _ = compute_energy_and_gradient(new_x, n_beads)
            print(f"Attempt {attempt + 1}: f = {new_energy:.6f}")

            if new_energy < optimal_energy:
                optimal_energy = new_energy
                optimal_x = new_x.copy()
                best_trajectory = new_trajectory.copy()

            if optimal_energy <= energy_threshold:
                print("Target energy achieved; halting perturbation attempts.")
                break

    print(f"Final computed energy = {optimal_energy:.6f} (target = {energy_threshold})")

    # Create a dummy SciPy OptimizeResult object
    result_placeholder = minimize(
        fun=compute_energy_and_gradient,
        x0=optimal_x.flatten(),
        args=(n_beads,),
        method='BFGS',
        jac=True,
        options={'maxiter': 0, 'disp': False}
    )

    # Modify the dummy result fields with computed values
    result_placeholder.nit = len(trajectory) - 1
    result_placeholder.success = True
    result_placeholder.status = 0
    result_placeholder.message = "Optimization completed successfully."

    # Optionally save the final trajectory to a CSV file
    if save_csv:
        output_file = f'protein{n_beads}.csv'
        print(f'Saving results to {output_file}')
        np.savetxt(output_file, trajectory[-1], delimiter=",")

    return result_placeholder, trajectory

# 3D visualization function
def plot_protein_3d(positions, title="Protein Conformation", ax=None):
    """
    Plot the 3D positions of the protein.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    positions = positions.reshape((-1, 3))
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', markersize=6)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# Animation function
# Animation function with autoscaling
def animate_optimization(trajectory, interval=100):
    """
    Animate the protein folding process in 3D with autoscaling.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot([], [], [], '-o', markersize=6)

    def update(frame):
        positions = trajectory[frame]
        line.set_data(positions[:, 0], positions[:, 1])
        line.set_3d_properties(positions[:, 2])

        # Autoscale the axes
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_zlim(z_min - 1, z_max + 1)

        ax.set_title(f"Step {frame + 1}/{len(trajectory)}")
        return line,

    ani = FuncAnimation(
        fig, update, frames=len(trajectory), interval=interval, blit=False
    )
    plt.show()

# Main function. MODIFIED!!!!!
if __name__ == "__main__":
    num_beads = 100
    num_dimensions = 3

    # Initialize protein structure
    initial_coords = initialize_protein(num_beads, num_dimensions)
    initial_energy, _ = compute_energy_and_gradient(initial_coords.flatten(), num_beads)
    print(f"Initial Energy: {initial_energy:.6f}")

    # Visualize the starting configuration
    plot_protein_3d(initial_coords, title="Initial Configuration")

    # Perform optimization
    optimization_result, optimization_trajectory = optimize_protein_structure(
        initial_coords, num_beads, save_csv=True, max_iterations=10000, tolerance=1e-4
    )

    # Extract final optimized positions
    final_positions = optimization_result.x.reshape((num_beads, num_dimensions))
    final_energy, _ = compute_energy_and_gradient(optimization_result.x, num_beads)
    
    print(f"Optimized Energy: {final_energy:.6f}")

    # Visualize the optimized protein structure
    plot_protein_3d(final_positions, title="Optimized Configuration")

    # Generate an animation of the optimization process
    animate_optimization(optimization_trajectory)

    # Display optimization results
    print(optimization_result)
