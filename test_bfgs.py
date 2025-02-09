import numpy as np

# Import functions from protein_folding_3d.py
from protein_folding_3d import optimize_protein, initialize_protein, total_energy

# Define test parameters
n_beads = 10
dimension = 3

# Initialize bead positions
initial_positions = initialize_protein(n_beads, dimension)

# Print initial energy
print("Initial Energy:", total_energy(initial_positions.flatten(), n_beads))

print(optimize_protein.__code__.co_names)

# Run optimization
optimized_positions, trajectory = optimize_protein(initial_positions, n_beads, write_csv=True)

# Print optimized energy
print("Optimized Energy:", total_energy(optimized_positions.flatten(), n_beads))
