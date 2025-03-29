import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import time
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo


# 1. Generate a controlled random graph
def generate_controlled_random_graph(num_nodes=10, max_edges=18, min_weight=1, max_weight=10):
    """Generate a random directed graph with controlled number of edges"""
    G = nx.DiGraph()

    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)

    # First create a path through all nodes to ensure connectivity
    for i in range(num_nodes - 1):
        weight = random.uniform(min_weight, max_weight)
        G.add_edge(i, i + 1, weight=weight)

    # Add remaining random edges (staying under max_edges)
    remaining_edges = max_edges - (num_nodes - 1)
    edge_count = num_nodes - 1

    # Try to add random edges, up to the limit
    attempts = 0
    while edge_count < max_edges and attempts < 100:
        i = random.randint(0, num_nodes - 1)
        j = random.randint(0, num_nodes - 1)
        if i != j and not G.has_edge(i, j):
            weight = random.uniform(min_weight, max_weight)
            G.add_edge(i, j, weight=weight)
            edge_count += 1
        attempts += 1

    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


# 2. Generate random source and destination points
def generate_random_points(graph, min_distance=2):
    """Generate random source and destination points with minimum distance between them"""
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        raise ValueError("Graph must have at least 2 nodes")

    # Make sure we only try valid pairs of nodes
    attempts = 0
    max_attempts = 100

    while attempts < max_attempts:
        source = random.choice(nodes)
        dest = random.choice(nodes)

        # Ensure source and dest are different
        if source == dest:
            attempts += 1
            continue

        # Check if there's a path between them
        if not nx.has_path(graph, source, dest):
            attempts += 1
            continue

        # Check minimum distance
        path = nx.shortest_path(graph, source, dest)
        if len(path) >= min_distance:
            print(f"Selected source: {source}, destination: {dest}")
            return source, dest

        attempts += 1

    # If we couldn't find suitable random points, return first and last node
    print("Couldn't find suitable random points, using first and last node")
    return nodes[0], nodes[-1]


# 3. Formulate and solve shortest path using QAOA
def solve_shortest_path_qaoa(graph, start_node, end_node):
    """Find shortest path using QAOA"""
    # Make sure the nodes exist in the graph
    if start_node not in graph or end_node not in graph:
        raise ValueError(f"Nodes {start_node} and {end_node} must exist in the graph")

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

    if len(qp.variables) > 25:
        print("Warning: Large number of variables may cause issues with the simulator")

    # Convert to QUBO
    qubo_converter = QuadraticProgramToQubo()
    qubo = qubo_converter.convert(qp)

    # Get the Ising Hamiltonian
    ising, offset = qubo.to_ising()

    # Create QAOA circuit using QAOAAnsatz
    qaoa_reps = 2  # Using p=2 for better approximation
    qaoa_ansatz = QAOAAnsatz(cost_operator=ising, reps=qaoa_reps)
    qaoa_ansatz.measure_all()

    print(f"QAOA circuit has {qaoa_ansatz.num_qubits} qubits")

    # Set up simulator with unlimited qubits
    simulator = AerSimulator(method='statevector')

    # Create appropriate number of parameters based on reps
    betas = np.random.uniform(0, np.pi, qaoa_reps)  # Need qaoa_reps beta values
    gammas = np.random.uniform(0, 2 * np.pi, qaoa_reps)  # Need qaoa_reps gamma values
    parameter_values = list(betas) + list(gammas)

    # Check if parameter count matches circuit parameters
    if len(qaoa_ansatz.parameters) != len(parameter_values):
        print(
            f"Warning: Circuit has {len(qaoa_ansatz.parameters)} parameters but we're providing {len(parameter_values)}")
        # Adjust parameter list to match
        if len(qaoa_ansatz.parameters) > len(parameter_values):
            # Add more parameters if needed
            additional_params = [random.uniform(0, np.pi) for _ in
                                 range(len(qaoa_ansatz.parameters) - len(parameter_values))]
            parameter_values.extend(additional_params)
        else:
            # Truncate if we have too many
            parameter_values = parameter_values[:len(qaoa_ansatz.parameters)]

    # Assign parameters to the circuit
    parameter_dict = dict(zip(qaoa_ansatz.parameters, parameter_values))
    qaoa_with_parameters = qaoa_ansatz.assign_parameters(parameter_dict)

    # Run on simulator
    try:
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
            bit_pos = len(most_frequent) - 1 - var_idx
            if bit_pos >= 0 and bit_pos < len(most_frequent) and most_frequent[bit_pos] == '1':
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

    except Exception as e:
        print(f"Error during QAOA execution: {e}")
        return None, None, []


# 4. Visualize graph and results with legend
def visualize_complex_graph_with_legend(graph, start_node, end_node, qaoa_path=None, selected_edges=None, classical_path=None):
    """Visualize a complex graph with paths and a legend"""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Use spring layout for complex graphs
    pos = nx.spring_layout(graph, seed=42)  # Fixed seed for reproducibility

    # Draw nodes and all edges
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=300, ax=ax)
    nx.draw_networkx_labels(graph, pos, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.2, ax=ax)

    # Draw edge labels (weights)
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    # Highlight source and destination
    nx.draw_networkx_nodes(graph, pos, nodelist=[start_node],
                          node_color='yellow', node_size=500, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=[end_node],
                          node_color='orange', node_size=500, ax=ax)

    # Create legend handles
    legend_elements = [
        mpatches.Patch(color='lightblue', label='Regular Node'),
        mpatches.Patch(color='yellow', label='Start Node'),
        mpatches.Patch(color='orange', label='End Node')
    ]

    # Highlight QAOA selected edges
    if selected_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=selected_edges, width=1.5, edge_color='red',
                               style='dashed', alpha=0.7, ax=ax)
        legend_elements.append(mpatches.Patch(color='red', alpha=0.7, label='QAOA Selected Edges'))

    # Highlight QAOA path
    if qaoa_path and len(qaoa_path) > 1:
        qaoa_path_edges = [(qaoa_path[i], qaoa_path[i + 1]) for i in range(len(qaoa_path) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=qaoa_path_edges, width=3, edge_color='green', ax=ax)
        legend_elements.append(mpatches.Patch(color='green', label='QAOA Optimal Path'))

        # Draw the QAOA path nodes with different color
        intermediate_nodes = qaoa_path[1:-1] if len(qaoa_path) > 2 else []
        if intermediate_nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=intermediate_nodes,
                                  node_color='lightgreen', node_size=400, ax=ax)
            legend_elements.append(mpatches.Patch(color='lightgreen', label='QAOA Path Intermediate Nodes'))

    # Highlight classical path for comparison if different from QAOA
    if classical_path and len(classical_path) > 1 and classical_path != qaoa_path:
        classical_path_edges = [(classical_path[i], classical_path[i + 1]) for i in range(len(classical_path) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=classical_path_edges, width=2,
                               edge_color='blue', style='dotted', ax=ax)
        legend_elements.append(mpatches.Patch(color='blue', label='Classical Optimal Path'))

    # Add the legend to the plot with a title
    ax.legend(handles=legend_elements, loc='upper right', title='Graph Elements')

    plt.title(f"Random Graph with Shortest Path from Node {start_node} to Node {end_node}")
    plt.axis('off')
    return plt


# Main execution
def main():
    # Initialize random with current time for true randomness
    current_time = int(time.time())
    random.seed(current_time)
    np.random.seed(current_time)

    print(f"Using random seed: {current_time}")

    try:
        # 1. Generate controlled random graph
        num_nodes = 10
        max_edges = 18
        graph = generate_controlled_random_graph(num_nodes=num_nodes, max_edges=max_edges)

        # 2. Generate random source and destination points
        try:
            start_node, end_node = generate_random_points(graph)
        except Exception as e:
            print(f"Error generating random points: {e}")
            # Fallback to first and last node
            start_node, end_node = 0, num_nodes - 1

        print(f"Finding shortest path from node {start_node} to node {end_node}")

        # 3. Find path using QAOA
        qaoa_path, qaoa_weight, selected_edges = solve_shortest_path_qaoa(graph, start_node, end_node)

        # 4. Find path using classical algorithm for comparison
        classical_path = None
        classical_weight = None

        if nx.has_path(graph, start_node, end_node):
            classical_path = nx.shortest_path(graph, start_node, end_node, weight='weight')
            classical_weight = sum(graph[classical_path[i]][classical_path[i + 1]]['weight']
                                   for i in range(len(classical_path) - 1))
            print(f"Classical solution: Path = {classical_path}")
            print(f"Classical path weight = {classical_weight:.2f}")
        else:
            print(f"No classical path exists from {start_node} to {end_node}")

        # 5. Compare results
        if qaoa_path and classical_path:
            print("\nComparison:")
            if abs(qaoa_weight - classical_weight) < 0.01:  # Allow for floating point differences
                print("✓ QAOA found the optimal solution!")
            else:
                print(f"QAOA weight: {qaoa_weight:.2f}, Classical weight: {classical_weight:.2f}")
                print(f"Difference: {abs(qaoa_weight - classical_weight):.2f}")
                if qaoa_weight < classical_weight:
                    print("⚠️ QAOA found a better solution than the classical algorithm (unlikely, check for errors)")
                else:
                    approx_ratio = classical_weight / qaoa_weight if qaoa_weight > 0 else 0
                    print(f"QAOA approximation ratio: {approx_ratio:.2f}")

        # 6. Generate unique filename based on time
        filename = f'random_graph_shortest_path_{current_time}.png'

        # 7. Visualize with legend
        plt = visualize_complex_graph_with_legend(graph, start_node, end_node, qaoa_path, selected_edges, classical_path)

        # Save the figure
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved as '{filename}'")

    except Exception as e:
        print(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()