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


# 1. Generate a city graph with a bank and safehouse
def generate_city_graph(num_nodes=10, max_edges=18, min_weight=1, max_weight=10):
    """Generate a random directed graph representing a city with varying difficulties of travel"""
    G = nx.DiGraph()

    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)

    # First create a path through all nodes to ensure connectivity
    for i in range(num_nodes - 1):
        # The weight represents difficulty (police presence, traffic, cameras, etc.)
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

    print(f"Created city graph with {G.number_of_nodes()} locations and {G.number_of_edges()} routes")
    return G


# 2. Designate bank and safehouse locations
def designate_bank_and_safehouse(graph, min_distance=3):
    """Designate which nodes represent the bank and the safehouse"""
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        raise ValueError("Graph must have at least 2 nodes")

    # Make sure we only try valid pairs of nodes
    attempts = 0
    max_attempts = 100

    while attempts < max_attempts:
        bank = random.choice(nodes)
        safehouse = random.choice(nodes)

        # Ensure bank and safehouse are different
        if bank == safehouse:
            attempts += 1
            continue

        # Check if there's a path between them
        if not nx.has_path(graph, bank, safehouse):
            attempts += 1
            continue

        # Check minimum distance to make it interesting
        path = nx.shortest_path(graph, bank, safehouse)
        if len(path) >= min_distance:
            print(f"Bank is at location {bank}, safehouse is at location {safehouse}")
            return bank, safehouse

        attempts += 1

    # If we couldn't find suitable locations, return first and last node
    print("Couldn't find suitable bank and safehouse locations, using first and last node")
    return nodes[0], nodes[-1]


# 3. Find getaway path using QAOA with improved parameters
def solve_getaway_path_qaoa(graph, bank_node, safehouse_node):
    """Find optimal getaway path using QAOA with optimized parameters"""
    print(
        f"Planning optimal getaway route from bank (location {bank_node}) to safehouse (location {safehouse_node})...")

    # Improved parameters: More repetitions for better results
    qaoa_reps = 3  # Increased from 2
    return solve_shortest_path_qaoa(graph, bank_node, safehouse_node, qaoa_reps=qaoa_reps)


# Core function to solve shortest path using QAOA
def solve_shortest_path_qaoa(graph, start_node, end_node, qaoa_reps=3):
    """Find shortest path using QAOA with optimized parameters"""
    from scipy.optimize import minimize

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

    # Objective: minimize path difficulty (weight)
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
    qaoa_ansatz = QAOAAnsatz(cost_operator=ising, reps=qaoa_reps)

    # Set up simulator with unlimited qubits and higher optimization level
    simulator = AerSimulator(method='statevector')

    # Create function to evaluate the QAOA cost for a given set of parameters
    def objective_function(parameters):
        # Assign parameters to the circuit
        parameter_dict = dict(zip(qaoa_ansatz.parameters, parameters))
        qaoa_with_parameters = qaoa_ansatz.assign_parameters(parameter_dict)

        # Add measurement to all qubits
        qaoa_circuit = qaoa_with_parameters.copy()
        qaoa_circuit.measure_all()

        # Transpile with optimization
        compiled_circuit = transpile(qaoa_circuit, simulator, optimization_level=3)

        # Run on simulator
        job = simulator.run(compiled_circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()

        # Calculate expected energy for each measured state
        energy = 0
        for bitstring, count in counts.items():
            # Convert bitstring to a dictionary of qubit values
            qubit_values = {}
            variable_names = list(edge_vars.values())
            variable_names.sort()  # Ensure consistent ordering

            for i, var_name in enumerate(variable_names):
                # Map from bitstring positions to variable values
                bit_pos = len(bitstring) - 1 - i
                if bit_pos >= 0 and bit_pos < len(bitstring):
                    qubit_values[var_name] = int(bitstring[bit_pos])
                else:
                    qubit_values[var_name] = 0  # Default if out of range

            # Calculate the cost of this solution
            solution_cost = sum(graph[u][v]['weight'] * qubit_values[var_name]
                                for (u, v), var_name in edge_vars.items()
                                if var_name in qubit_values)

            # Weight by the probability of this outcome
            probability = count / 1024
            energy += solution_cost * probability

        return energy

    # Better initial parameters for QAOA
    initial_betas = np.linspace(0.01, np.pi / 2, qaoa_reps)  # Gradually increasing beta values
    initial_gammas = np.linspace(0.01, np.pi, qaoa_reps)  # Gradually increasing gamma values
    initial_params = np.concatenate([initial_betas, initial_gammas])

    # Number of optimization iterations
    max_iter = 50  # Adjust based on your needs

    print(f"Optimizing getaway route parameters with {max_iter} iterations...")

    # Run optimization - COBYLA is often good for noisy functions like quantum circuits
    result = minimize(
        objective_function,
        initial_params,
        method='COBYLA',
        options={'maxiter': max_iter, 'disp': True}
    )

    optimized_params = result.x
    print(f"Optimization complete. Final parameters: {optimized_params}")
    print(f"Final objective value: {result.fun}")

    # Use the optimized parameters for the final circuit
    parameter_dict = dict(zip(qaoa_ansatz.parameters, optimized_params))
    final_qaoa = qaoa_ansatz.assign_parameters(parameter_dict)
    final_qaoa.measure_all()

    # Run final circuit with optimized parameters
    try:
        compiled_circuit = transpile(final_qaoa, simulator, optimization_level=3)
        job = simulator.run(compiled_circuit, shots=2048)  # More shots for final run
        result = job.result()
        counts = result.get_counts()

        # Get the most frequent bitstring
        most_frequent = max(counts, key=counts.get)
        print(f"Most frequent measurement: {most_frequent} (frequency: {counts[most_frequent] / 2048:.2%})")

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
                print(f"Selected route: {u} -> {v}")

        # Handle the case where no edges are selected (all zeros)
        if not selected_edges:
            print("Warning: No edges selected in solution (all zeros measurement)")
            # Fall back to classical solution in this case
            path = nx.shortest_path(graph, start_node, end_node, weight='weight')
            total_weight = sum(graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            return path, total_weight, {'classical_fallback': True, 'path_edges': path_edges, 'reason': 'all_zeros'}

        # Construct the path
        path_graph = nx.DiGraph()
        path_graph.add_edges_from(selected_edges)

        # Check if we have a valid path
        if nx.has_path(path_graph, start_node, end_node):
            path = nx.shortest_path(path_graph, start_node, end_node)
            total_weight = sum(graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
            print(f"QAOA solution: Getaway route = {path}")
            print(f"Route difficulty = {total_weight:.2f}")
            return path, total_weight, selected_edges
        else:
            print("No valid getaway route found from selected edges")
            # If no valid path is found, try post-processing repair
            return repair_invalid_solution(graph, selected_edges, start_node, end_node)

    except Exception as e:
        print(f"Error during QAOA execution: {e}")
        # Fall back to classical solution
        try:
            path = nx.shortest_path(graph, start_node, end_node, weight='weight')
            total_weight = sum(graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            return path, total_weight, {'classical_fallback': True, 'path_edges': path_edges, 'reason': 'error'}
        except:
            return None, None, []


# Improved repair mechanism
def repair_invalid_solution(graph, selected_edges, start_node, end_node):
    """Attempt to repair an invalid solution by adding necessary edges"""
    print("Attempting to repair invalid getaway route...")

    # Create a graph from selected edges
    path_graph = nx.DiGraph()
    path_graph.add_edges_from(selected_edges)

    # Make sure start and end nodes are in the graph
    for node in [start_node, end_node]:
        if node not in path_graph:
            path_graph.add_node(node)
            print(f"Added missing node {node} to path graph")

    # If there's no path from start to end, we need to add edges
    if not nx.has_path(path_graph, start_node, end_node):
        # Find all nodes reachable from start
        try:
            reachable_from_start = set(nx.descendants(path_graph, start_node))
            reachable_from_start.add(start_node)
        except nx.NetworkXError:
            reachable_from_start = {start_node}  # Just the start node if it's isolated

        # Find all nodes that can reach end
        try:
            # If end_node has no incoming edges, this might fail
            can_reach_end = set()
            reversed_graph = path_graph.reverse()
            if end_node in reversed_graph:
                can_reach_end = set(nx.descendants(reversed_graph, end_node))
            can_reach_end.add(end_node)
        except nx.NetworkXError:
            can_reach_end = {end_node}  # Just the end node if it's isolated

        # Find connecting edges from the original graph
        connecting_edges = []
        for u in reachable_from_start:
            for v in can_reach_end:
                if u != v and graph.has_edge(u, v) and not path_graph.has_edge(u, v):
                    connecting_edges.append((u, v, graph[u][v]['weight']))

        # Sort by weight and add the lightest connecting edge
        if connecting_edges:
            connecting_edges.sort(key=lambda x: x[2])
            u, v, _ = connecting_edges[0]
            path_graph.add_edge(u, v, weight=graph[u][v]['weight'])
            selected_edges.append((u, v))
            print(f"Added connecting route: {u} -> {v}")
        else:
            # If no connecting edge found, add direct edges from start or to end
            # First try to add an edge from start to any node
            start_connections = []
            for v in graph.successors(start_node):
                if graph.has_edge(start_node, v):
                    start_connections.append((start_node, v, graph[start_node][v]['weight']))

            if start_connections:
                start_connections.sort(key=lambda x: x[2])
                _, v, _ = start_connections[0]
                path_graph.add_edge(start_node, v, weight=graph[start_node][v]['weight'])
                selected_edges.append((start_node, v))
                print(f"Added edge from start: {start_node} -> {v}")

            # Then try to add an edge to end from any node
            end_connections = []
            for u in graph.predecessors(end_node):
                if graph.has_edge(u, end_node):
                    end_connections.append((u, end_node, graph[u][end_node]['weight']))

            if end_connections:
                end_connections.sort(key=lambda x: x[2])
                u, _, _ = end_connections[0]
                path_graph.add_edge(u, end_node, weight=graph[u][end_node]['weight'])
                selected_edges.append((u, end_node))
                print(f"Added edge to end: {u} -> {end_node}")

    # Try to find a path again
    if nx.has_path(path_graph, start_node, end_node):
        path = nx.shortest_path(path_graph, start_node, end_node)
        total_weight = sum(graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
        print(f"Repaired getaway route: Path = {path}")
        print(f"Route difficulty = {total_weight:.2f}")
        return path, total_weight, {'repaired': True, 'path_edges': selected_edges}
    else:
        print("Could not repair solution, falling back to classical algorithm")
        path = nx.shortest_path(graph, start_node, end_node, weight='weight')
        total_weight = sum(graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
        # Create the edges from the path
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        return path, total_weight, {'classical_fallback': True, 'path_edges': path_edges, 'reason': 'repair_failed'}


# Visualize the city and getaway paths with improved robustness
def visualize_heist_scenario(graph, bank_node, safehouse_node, qaoa_path=None, selected_edges=None,
                             classical_path=None):
    """Visualize the city with bank, safehouse, and getaway paths"""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Use spring layout for city layout with a fixed seed for reproducibility
    pos = nx.spring_layout(graph, seed=42)

    # Draw all locations and routes
    nx.draw_networkx_nodes(graph, pos, node_color='lightgray', node_size=300, ax=ax)
    nx.draw_networkx_labels(graph, pos, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.2, ax=ax)

    # Draw edge labels (difficulty values)
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    # Highlight bank and safehouse
    nx.draw_networkx_nodes(graph, pos, nodelist=[bank_node],
                           node_color='red', node_size=500, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=[safehouse_node],
                           node_color='green', node_size=500, ax=ax)

    # Create legend handles
    legend_elements = [
        mpatches.Patch(color='lightgray', label='City Location'),
        mpatches.Patch(color='red', label='Bank (Start)'),
        mpatches.Patch(color='green', label='Safehouse (End)')
    ]

    # Determine if we used a classical fallback
    used_classical_fallback = isinstance(selected_edges, dict) and selected_edges.get('classical_fallback', False)

    # Highlight selected edges
    if selected_edges and not used_classical_fallback:
        # Handle dictionary format for repaired solutions
        if isinstance(selected_edges, dict) and 'path_edges' in selected_edges:
            edges_to_draw = selected_edges['path_edges']
            if isinstance(edges_to_draw, list):
                nx.draw_networkx_edges(graph, pos, edgelist=edges_to_draw, width=1.5, edge_color='orange',
                                       style='dashed', alpha=0.7, ax=ax)
                legend_elements.append(mpatches.Patch(color='orange', alpha=0.7, label='Repaired QAOA Routes'))
        else:
            # Regular QAOA selected edges
            nx.draw_networkx_edges(graph, pos, edgelist=selected_edges, width=1.5, edge_color='orange',
                                   style='dashed', alpha=0.7, ax=ax)
            legend_elements.append(mpatches.Patch(color='orange', alpha=0.7, label='QAOA Considered Routes'))

    # Highlight the optimal getaway path
    if qaoa_path and len(qaoa_path) > 1:
        qaoa_path_edges = [(qaoa_path[i], qaoa_path[i + 1]) for i in range(len(qaoa_path) - 1)]

        # If we used classical fallback, draw the path in blue as "Classical Optimal Path"
        if used_classical_fallback:
            nx.draw_networkx_edges(graph, pos, edgelist=qaoa_path_edges, width=3, edge_color='blue', ax=ax)
            legend_elements.append(mpatches.Patch(color='blue', label='Classical Optimal Path'))
        else:
            # Otherwise draw as QAOA path in purple
            nx.draw_networkx_edges(graph, pos, edgelist=qaoa_path_edges, width=3, edge_color='purple', ax=ax)
            legend_elements.append(mpatches.Patch(color='purple', label='Optimal Getaway Path'))

        # Draw the path intermediate locations
        intermediate_nodes = qaoa_path[1:-1] if len(qaoa_path) > 2 else []
        if intermediate_nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=intermediate_nodes,
                                   node_color='plum', node_size=400, ax=ax)
            legend_elements.append(mpatches.Patch(color='plum', label='Getaway Route Stops'))

    # Only highlight classical path for comparison if:
    # 1. It exists and has more than 1 node
    # 2. It's different from the qaoa_path
    # 3. We didn't use classical fallback (which would make qaoa_path = classical_path)
    if classical_path and len(classical_path) > 1 and classical_path != qaoa_path and not used_classical_fallback:
        classical_path_edges = [(classical_path[i], classical_path[i + 1]) for i in range(len(classical_path) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=classical_path_edges, width=2,
                               edge_color='blue', style='dotted', ax=ax)
        legend_elements.append(mpatches.Patch(color='blue', label='Classical Optimal Path'))

    # Add the legend to the plot with a title
    ax.legend(handles=legend_elements, loc='upper right', title='Heist Scenario Elements')

    plt.title(f"Bank Heist Scenario: Getaway from Bank (Node {bank_node}) to Safehouse (Node {safehouse_node})")
    plt.axis('off')
    return plt


# Main execution with better error handling
def main():
    # Initialize random with current time for true randomness
    current_time = int(time.time())
    random.seed(current_time)
    np.random.seed(current_time)

    print(f"Starting bank heist simulation with random seed: {current_time}")

    try:
        # 1. Generate city graph
        num_nodes = 8  # Reduced for better reliability
        max_edges = 15  # Adjusted for better reliability
        city_graph = generate_city_graph(num_nodes=num_nodes, max_edges=max_edges)

        # 2. Designate bank and safehouse locations
        try:
            bank_node, safehouse_node = designate_bank_and_safehouse(city_graph)
        except Exception as e:
            print(f"Error designating bank and safehouse: {e}")
            # Fallback to first and last node
            bank_node, safehouse_node = 0, num_nodes - 1

        print(
            f"Bank heist scenario: Escape from bank at location {bank_node} to safehouse at location {safehouse_node}")

        # 3. Find getaway path using QAOA
        qaoa_start_time = time.time()
        qaoa_path, qaoa_difficulty, selected_edges = solve_getaway_path_qaoa(city_graph, bank_node, safehouse_node)
        qaoa_time = time.time() - qaoa_start_time
        print(f"QAOA solution time: {qaoa_time:.2f} seconds")

        # 4. Find path using classical algorithm for comparison
        classical_path = None
        classical_difficulty = None

        if nx.has_path(city_graph, bank_node, safehouse_node):
            classical_start_time = time.time()
            classical_path = nx.shortest_path(city_graph, bank_node, safehouse_node, weight='weight')
            classical_time = time.time() - classical_start_time

            classical_difficulty = sum(city_graph[classical_path[i]][classical_path[i + 1]]['weight']
                                       for i in range(len(classical_path) - 1))
            print(f"Classical solution: Getaway route = {classical_path}")
            print(f"Classical route difficulty = {classical_difficulty:.2f}")
            print(f"Classical solution time: {classical_time:.6f} seconds")

            # Speed comparison
            speedup = qaoa_time / classical_time if classical_time > 0 else 0
            print(f"QAOA was {speedup:.1f}x slower than classical algorithm")
        else:
            print(f"No classical getaway route exists from bank to safehouse")

        # 5. Compare results
        if qaoa_path and classical_path:
            print("\nComparing getaway methods:")

            # Check if we had to fall back to classical solution
            used_classical_fallback = False
            if isinstance(selected_edges, dict) and selected_edges.get('classical_fallback', False):
                used_classical_fallback = True
                print(f"Note: Used classical fallback because: {selected_edges.get('reason', 'unknown')}")

            if not used_classical_fallback:
                if abs(qaoa_difficulty - classical_difficulty) < 0.01:  # Allow for floating point differences
                    print("✓ QAOA found the optimal getaway route!")
                else:
                    print(f"QAOA difficulty: {qaoa_difficulty:.2f}, Classical difficulty: {classical_difficulty:.2f}")
                    print(f"Difference: {abs(qaoa_difficulty - classical_difficulty):.2f}")
                    if qaoa_difficulty < classical_difficulty:
                        print("⚠️ QAOA found a better route than the classical algorithm (unlikely, check for errors)")
                    else:
                        approx_ratio = classical_difficulty / qaoa_difficulty if qaoa_difficulty > 0 else 0
                        print(f"QAOA approximation ratio: {approx_ratio:.2f}")
                        if approx_ratio < 0.8:
                            print("QAOA solution is significantly worse than classical - police might catch you!")
                        else:
                            print("QAOA solution is close to optimal - should be a clean getaway!")
            else:
                print("⚠️ QAOA failed to find a valid solution, using classical solution instead.")

        # 6. Generate unique filename based on time
        filename = f'bank_heist_simulation_{current_time}.png'

        # 7. Visualize with legend
        plt = visualize_heist_scenario(city_graph, bank_node, safehouse_node,
                                       qaoa_path, selected_edges, classical_path)

        # Save the figure
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Bank heist visualization saved as '{filename}'")

        # Check if we used a fallback solution
        used_classical_fallback = isinstance(selected_edges, dict) and selected_edges.get('classical_fallback', False)
        used_repair = isinstance(selected_edges, dict) and selected_edges.get('repaired', False)

        # Success message
        if qaoa_path:
            if used_classical_fallback:
                print("\nQAOA failed to find a valid getaway route. The classical algorithm was used instead.")
            elif used_repair:
                print("\nQAOA solution needed repair but a valid getaway route was found.")
            elif classical_path and qaoa_path == classical_path:
                print("\nThe quantum getaway was successful! You found the optimal escape route!")
            elif classical_path:
                if qaoa_difficulty <= classical_difficulty * 1.2:  # Within 20% of optimal
                    print("\nThe quantum getaway was successful, though not perfectly optimal!")
                else:
                    print("\nThe quantum getaway route was suboptimal. Police are suspicious!")
            else:
                print("\nThe quantum getaway was completed, but we couldn't verify if it was optimal.")
        else:
            print("\nThe quantum getaway planning failed. Back to the drawing board!")

    except Exception as e:
        print(f"Error in bank heist simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()