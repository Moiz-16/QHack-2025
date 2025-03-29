import networkx as nx
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.applications import Maxcut
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.visualization import plot_histogram

# Step 1: Define the Graph
square_graph = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0)])

# Step 2: Convert Graph to QUBO using Maxcut class
maxcut = Maxcut(square_graph)
qubo = maxcut.to_quadratic_program()

# Step 3: Convert QUBO to Ising Hamiltonian
ising, ising_offset = qubo.to_ising()

# Step 4: Create QAOA Circuit
qaoa_reps = 2
qaoa_ansatz = QAOAAnsatz(cost_operator=ising, reps=qaoa_reps, name='qaoa')
qaoa_ansatz.measure_active()

# Step 5: Assign Parameters
betas = np.random.uniform(0, np.pi, qaoa_reps)
gammas = np.random.uniform(0, 2 * np.pi, qaoa_reps)
parameter_values = [*betas, *gammas]
qaoa_with_parameters = qaoa_ansatz.assign_parameters(dict(zip(qaoa_ansatz.parameters,
                                                              parameter_values)))

# Step 6: Execute on Simulator
aer_simulator = AerSimulator()
compiled_qaoa = transpile(qaoa_with_parameters, aer_simulator)
result = aer_simulator.run(compiled_qaoa, shots=10 ** 5).result()
counts = result.get_counts()


# Step 7: Discard 0000 and 1111 as these are Not Cuts -> [][0, 1, 2, 3] and [1, 2, 3, 4][]
def is_trivial_solution(binary_string):
    """
    Check if the solution is trivial (either all 0's or all 1's).
    """
    return binary_string == '0' * len(binary_string) or binary_string == '1' * len(binary_string)


# Step 8: Filter out trivial solutions (0000, 1111)
filtered_counts = {binary_string: count for binary_string, count in counts.items() if not
is_trivial_solution(binary_string)}


# Step 9: Calculate the Max-Cut and Max Number of Cuts
def calculate_max_cut(graph):
    """
    Calculate the maximum number of cuts for a given graph.
    """
    # Define all possible partitions
    max_cuts = 0
    best_partition = None

    # Loop over all possible partitions of the graph into two sets
    num_nodes = len(graph.nodes)
    for i in range(1, 1 << num_nodes):
        set_a = [node for node in range(num_nodes) if (i & (1 << node)) > 0]
        set_b = [node for node in range(num_nodes) if (i & (1 << node)) == 0]

        # Calculate cut edges between set A and set B
        cut_edges = []
        for edge in graph.edges:
            if (edge[0] in set_a and edge[1] in set_b) or (edge[0] in set_b and edge[1] in
                                                           set_a):
                cut_edges.append(edge)

            # Check if this partition has more cuts
        if len(cut_edges) > max_cuts:
            max_cuts = len(cut_edges)
            best_partition = (set_a, set_b, cut_edges)

    return max_cuts, best_partition


# Step 10: Calculate the maximum cut for the graph
max_cuts, best_partition = calculate_max_cut(square_graph)

# Step 11: Output the results
print(f"Maximum␣number␣of␣cuts:␣{max_cuts}")
print(f"Best␣partition:␣Set␣A:␣{best_partition[0]},␣Set␣B:␣{best_partition[1]}")
print(f"Cut␣edges:␣{best_partition[2]}")