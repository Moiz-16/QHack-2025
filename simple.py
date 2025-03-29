import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

# Create a simple square graph with weights
graph = nx.DiGraph()

# Add nodes and edges
graph.add_edge(0, 1, weight=4.0)  # Top edge
graph.add_edge(1, 0, weight=4.0)
graph.add_edge(1, 2, weight=2.0)  # Right edge
graph.add_edge(2, 1, weight=2.0)
graph.add_edge(2, 3, weight=5.0)  # Bottom edge
graph.add_edge(3, 2, weight=5.0)
graph.add_edge(3, 0, weight=2.0)  # Left edge
graph.add_edge(0, 3, weight=2.0)


# Define start and end nodes
start_node = 0
end_node = 2

# Create QuadraticProgram for shortest path
qp = QuadraticProgram()

# Add binary variables for each edge
edge_vars = {}
for i, (u, v, data) in enumerate(graph.edges(data=True)):
    var_name = f'x_{u}_{v}'
    edge_vars[(u, v)] = var_name
    qp.binary_var(var_name)

# Objective: minimize path weight
linear = {}
for (u, v), var_name in edge_vars.items():
    linear[var_name] = graph[u][v]['weight']

qp.minimize(linear=linear)

# Add flow constraints
for node in graph.nodes():
    incoming = [edge_vars.get((u, node)) for u in graph.predecessors(node)
                if (u, node) in edge_vars]
    outgoing = [edge_vars.get((node, v)) for v in graph.successors(node)
                if (node, v) in edge_vars]

    # Flow constraint dictionary
    flow_dict = {}

    if node == start_node:
        # Start node: outflow - inflow = 1
        for var in outgoing:
            flow_dict[var] = 1
        for var in incoming:
            flow_dict[var] = -1
        qp.linear_constraint(linear=flow_dict, sense='==', rhs=1)

    elif node == end_node:
        # End node: inflow - outflow = 1
        for var in incoming:
            flow_dict[var] = 1
        for var in outgoing:
            flow_dict[var] = -1
        qp.linear_constraint(linear=flow_dict, sense='==', rhs=1)

    else:
        # Transit nodes: inflow = outflow
        for var in incoming:
            flow_dict[var] = 1
        for var in outgoing:
            flow_dict[var] = -1
        qp.linear_constraint(linear=flow_dict, sense='==', rhs=0)

# Convert to QUBO
qubo_converter = QuadraticProgramToQubo()
qubo = qubo_converter.convert(qp)

# Get the Ising Hamiltonian
ising, offset = qubo.to_ising()

# Create QAOA circuit using QAOAAnsatz
qaoa_reps = 1  # Using p=1 for simplicity
qaoa_ansatz = QAOAAnsatz(cost_operator=ising, reps=qaoa_reps)
qaoa_ansatz.measure_all()

# Set QAOA parameters
betas = [0.5]  # Mixer Hamiltonian angles
gammas = [0.1]  # Problem Hamiltonian angles
parameter_values = betas + gammas

# Assign parameters to the circuit
parameter_dict = dict(zip(qaoa_ansatz.parameters, parameter_values))
qaoa_with_parameters = qaoa_ansatz.assign_parameters(parameter_dict)

# Run on simulator
simulator = AerSimulator()
compiled_circuit = transpile(qaoa_with_parameters, simulator)
job = simulator.run(compiled_circuit, shots=1024)
result = job.result()
counts = result.get_counts()

# Get the most frequent bitstring
most_frequent = max(counts, key=counts.get)
print(f"Most frequent measurement: {most_frequent}")

# We need a mapping from bit positions to variable names
variable_names = list(edge_vars.values())
variable_names.sort()  # Ensure consistent ordering

# Map result back to edge selection
selected_edges = []
for (u, v), var_name in edge_vars.items():
    # Find the position of this variable in the list
    var_idx = variable_names.index(var_name)
    # Check if this edge is selected in the solution
    if most_frequent[len(most_frequent) - 1 - var_idx] == '1':
        selected_edges.append((u, v))
        print(f"Selected edge: {u} -> {v}")

# Construct the path
path_graph = nx.DiGraph()
path_graph.add_edges_from(selected_edges)

# Check if we have a valid path
if nx.has_path(path_graph, start_node, end_node):
    path = nx.shortest_path(path_graph, start_node, end_node)
    total_weight = sum(graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
    print(f"QAOA solution: Path = {path}, Weight = {total_weight}")
else:
    print("No valid path found from selected edges")

# Compare with classical solution
classical_path = nx.shortest_path(graph, start_node, end_node, weight='weight')
classical_weight = sum(graph[classical_path[i]][classical_path[i + 1]]['weight']
                       for i in range(len(classical_path) - 1))
print(f"Classical solution: Path = {classical_path}, Weight = {classical_weight}")

# Visualize the graph
plt.figure(figsize=(8, 6))
pos = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (0, 0)}  # Square layout

# Draw nodes and edges
nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=500)
nx.draw_networkx_labels(graph, pos)
nx.draw_networkx_edges(graph, pos, alpha=0.3)
edge_labels = {(u, v): f"{d['weight']}" for u, v, d in graph.edges(data=True)}
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

# Highlight selected edges
nx.draw_networkx_edges(graph, pos, edgelist=selected_edges, width=3, edge_color='r')

# Highlight the path if found
if 'path' in locals():
    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, width=3, edge_color='g')

plt.title("Square Graph with Shortest Path")
plt.axis('off')

# Save the figure instead of showing it interactively
plt.savefig('shortest_path_qaoa.png')
print("Graph visualization saved as 'shortest_path_qaoa.png'")