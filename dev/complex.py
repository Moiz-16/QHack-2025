import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo


# 1. Generate a smaller random graph
def generate_random_graph(num_nodes=8, edge_probability=0.25, min_weight=1, max_weight=10):
    """Generate a random directed graph with weighted edges"""
    G = nx.DiGraph()

    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)

    # Add random edges with weights (limiting total edges)
    edge_count = 0
    max_edges = 25  # Limit total edges to stay under qubit limit

    for i in range(num_nodes):
        for j in range(num_nodes):
            if edge_count >= max_edges:
                break
            if i != j and random.random() < edge_probability:
                weight = random.uniform(min_weight, max_weight)
                G.add_edge(i, j, weight=weight)
                edge_count += 1

    # Ensure there's at least one path from start to end
    start, end = 0, num_nodes - 1
    if not nx.has_path(G, start, end):
        # Find a path by connecting nodes if needed
        path = []
        current = start
        while current != end:
            next_node = min(current + 1, end)  # Ensure we don't exceed end
            G.add_edge(current, next_node, weight=random.uniform(min_weight, max_weight))
            path.append((current, next_node))
            current = next_node
            edge_count += 1
        print(f"Added path to ensure connectivity: {path}")

    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


# 2. Formulate and solve shortest path using QAOA
def solve_shortest_path_qaoa(graph, start_node, end_node):
    """Find shortest path using QAOA"""
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

        # Skip if no connections
        if not incoming and not outgoing:
            continue

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

    print(f"Created QUBO with {len(qp.variables)} variables and {len(qp.linear_constraints)} constraints")

    # Convert to QUBO
    qubo_converter = QuadraticProgramToQubo()
    qubo = qubo_converter.convert(qp)

    # Get the Ising Hamiltonian
    ising, offset = qubo.to_ising()

    # Create QAOA circuit using QAOAAnsatz
    qaoa_reps = 2  # Using p=1 for simplicity and to reduce circuit depth
    qaoa_ansatz = QAOAAnsatz(cost_operator=ising, reps=qaoa_reps)
    qaoa_ansatz.measure_all()

    print(f"QAOA circuit has {qaoa_ansatz.num_qubits} qubits")

    # Set up simulator with unlimited qubits
    simulator = AerSimulator(method='statevector')

    # Create appropriate number of parameters based on reps
    betas = np.random.uniform(0, np.pi, qaoa_reps)  # Need qaoa_reps beta values
    gammas = np.random.uniform(0, 2 * np.pi, qaoa_reps)  # Need qaoa_reps gamma values
    parameter_values = list(betas) + list(gammas)

    # Assign parameters to the circuit
    parameter_dict = dict(zip(qaoa_ansatz.parameters, parameter_values))
    qaoa_with_parameters = qaoa_ansatz.assign_parameters(parameter_dict)

    # Run on simulator
    compiled_circuit = transpile(qaoa_with_parameters, simulator)
    job = simulator.run(compiled_circuit, shots=1024)
    result = job.result()
    counts = result.get_counts()

    # Get the most frequent bitstring
    most_frequent = max(counts, key=counts.get)
    print(f"Most frequent measurement: {most_frequent} (frequency: {counts[most_frequent] / 1024:.2%})")

    # Create mapping from bit positions to variable names
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
        print(f"QAOA solution: Path = {path}")
        print(f"Path weight = {total_weight:.2f}")
        return path, total_weight, selected_edges
    else:
        print("No valid path found from selected edges")
        return None, None, selected_edges


# 3. Visualize graph and results
def visualize_complex_graph(graph, qaoa_path=None, selected_edges=None, classical_path=None):
    """Visualize a complex graph with paths"""
    plt.figure(figsize=(12, 10))

    # Use spring layout for complex graphs
    pos = nx.spring_layout(graph, seed=42)  # Fixed seed for consistency

    # Draw nodes and all edges
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=300)
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edges(graph, pos, alpha=0.2)

    # Draw edge labels (weights)
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

    # Highlight QAOA selected edges
    if selected_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=selected_edges, width=1.5, edge_color='red',
                               style='dashed', alpha=0.7)

    # Highlight QAOA path
    if qaoa_path and len(qaoa_path) > 1:
        qaoa_path_edges = [(qaoa_path[i], qaoa_path[i + 1]) for i in range(len(qaoa_path) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=qaoa_path_edges, width=3, edge_color='green')

        # Draw the QAOA path nodes with different color
        nx.draw_networkx_nodes(graph, pos, nodelist=qaoa_path, node_color='lightgreen',
                               node_size=400)

    # Highlight classical path for comparison if different from QAOA
    if classical_path and len(classical_path) > 1 and classical_path != qaoa_path:
        classical_path_edges = [(classical_path[i], classical_path[i + 1]) for i in range(len(classical_path) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=classical_path_edges, width=2,
                               edge_color='blue', style='dotted')

    plt.title("Random Graph with Shortest Path")
    plt.axis('off')
    return plt


# Main execution
def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # 1. Generate random graph (smaller)
    num_nodes = 8  # Reduced number of nodes
    graph = generate_random_graph(num_nodes=num_nodes)

    # Define start and end nodes
    start_node = 0
    end_node = 4

    print(f"Finding shortest path from node {start_node} to node {end_node}")

    # 2. Find path using QAOA
    qaoa_path, qaoa_weight, selected_edges = solve_shortest_path_qaoa(graph, start_node, end_node)

    # 3. Find path using classical algorithm for comparison
    if nx.has_path(graph, start_node, end_node):
        classical_path = nx.shortest_path(graph, start_node, end_node, weight='weight')
        classical_weight = sum(graph[classical_path[i]][classical_path[i + 1]]['weight']
                               for i in range(len(classical_path) - 1))
        print(f"Classical solution: Path = {classical_path}")
        print(f"Classical path weight = {classical_weight:.2f}")
    else:
        print("No classical path exists (should not happen due to our graph generation)")
        classical_path = None
        classical_weight = None

    # 4. Compare results
    if qaoa_path and classical_path:
        print("\nComparison:")
        if qaoa_weight == classical_weight:
            print("✓ QAOA found the optimal solution!")
        else:
            print(f"QAOA weight: {qaoa_weight:.2f}, Classical weight: {classical_weight:.2f}")
            print(f"Difference: {abs(qaoa_weight - classical_weight):.2f}")
            if qaoa_weight < classical_weight:
                print("⚠️ QAOA found a better solution than the classical algorithm (unlikely, check for errors)")
            else:
                approx_ratio = classical_weight / qaoa_weight if qaoa_weight > 0 else 0
                print(f"QAOA approximation ratio: {approx_ratio:.2f}")

    # 5. Visualize
    plt = visualize_complex_graph(graph, qaoa_path, selected_edges, classical_path)

    # Save the figure
    plt.savefig('random_graph_shortest_path.png', dpi=300, bbox_inches='tight')
    print("Graph visualization saved as 'random_graph_shortest_path.png'")


if __name__ == "__main__":
    main()