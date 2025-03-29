import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
import seaborn as sns
from tqdm import tqdm
import json
import os
from datetime import datetime

# Import all functions from the original script
# Assume all the original functions are available in a file named 'bank_heist_qaoa.py'
from advanced_bank_heist_simulation import (
    generate_city_graph,
    designate_bank_and_safehouse,
    solve_getaway_path_qaoa,
    visualize_heist_scenario
)

# Create results directory if it doesn't exist
results_dir = "heist_results"
os.makedirs(results_dir, exist_ok=True)


def run_heist_experiment(num_runs=30, min_nodes=6, max_nodes=12, save_samples=5):
    """
    Run the bank heist simulation multiple times and collect performance data

    Parameters:
    -----------
    num_runs : int
        Number of simulation runs to perform
    min_nodes, max_nodes : int
        Range of city graph sizes to test
    save_samples : int
        Number of sample visualizations to save

    Returns:
    --------
    DataFrame with collected statistics
    """
    # Prepare data collection
    results = []

    # For saving sample visualizations
    saved_samples = 0

    # Run multiple experiments
    for run in tqdm(range(num_runs), desc="Running heist simulations"):
        # Generate a random seed for this run
        run_seed = int(time.time()) + run
        random.seed(run_seed)
        np.random.seed(run_seed)

        # Choose number of nodes for this run
        num_nodes = random.randint(min_nodes, max_nodes)
        max_edges = min(num_nodes * 2, num_nodes * (num_nodes - 1) // 2)  # Ensure reasonable edge density

        # Store run metrics
        run_data = {
            "run_id": run,
            "seed": run_seed,
            "num_nodes": num_nodes,
            "num_edges": 0,
            "qaoa_success": False,
            "qaoa_fallback": False,
            "qaoa_repair": False,
            "classical_success": False,
            "qaoa_difficulty": None,
            "classical_difficulty": None,
            "approx_ratio": None,
            "qaoa_time": None,
            "classical_time": None,
            "speedup_factor": None,
            "path_match": False
        }

        try:
            # 1. Generate city graph
            city_graph = generate_city_graph(num_nodes=num_nodes, max_edges=max_edges)
            run_data["num_edges"] = city_graph.number_of_edges()

            # 2. Designate bank and safehouse locations
            try:
                bank_node, safehouse_node = designate_bank_and_safehouse(city_graph)
            except Exception as e:
                print(f"Error designating bank and safehouse (Run {run}): {e}")
                # Fallback to first and last node
                bank_node, safehouse_node = 0, num_nodes - 1

            # 3. Find getaway path using QAOA
            qaoa_start_time = time.time()
            qaoa_path, qaoa_difficulty, selected_edges = solve_getaway_path_qaoa(city_graph, bank_node, safehouse_node)
            qaoa_time = time.time() - qaoa_start_time

            run_data["qaoa_time"] = qaoa_time

            # Check if QAOA was successful
            if qaoa_path is not None:
                run_data["qaoa_success"] = True
                run_data["qaoa_difficulty"] = qaoa_difficulty

                # Check if we used fallback or repair
                if isinstance(selected_edges, dict):
                    if selected_edges.get('classical_fallback', False):
                        run_data["qaoa_fallback"] = True
                    if selected_edges.get('repaired', False):
                        run_data["qaoa_repair"] = True

            # 4. Find path using classical algorithm for comparison
            if nx.has_path(city_graph, bank_node, safehouse_node):
                classical_start_time = time.time()
                classical_path = nx.shortest_path(city_graph, bank_node, safehouse_node, weight='weight')
                classical_time = time.time() - classical_start_time

                classical_difficulty = sum(city_graph[classical_path[i]][classical_path[i + 1]]['weight']
                                           for i in range(len(classical_path) - 1))

                run_data["classical_success"] = True
                run_data["classical_difficulty"] = classical_difficulty
                run_data["classical_time"] = classical_time

                # Calculate speedup factor (higher means classical is faster)
                if classical_time > 0:
                    run_data["speedup_factor"] = qaoa_time / classical_time

                # Compare paths if both methods succeeded
                if run_data["qaoa_success"] and run_data["classical_success"]:
                    # Calculate approximation ratio (classical_difficulty / qaoa_difficulty)
                    # Lower is better for difficulties, so ratio closer to 1 is better
                    if qaoa_difficulty > 0:
                        run_data["approx_ratio"] = classical_difficulty / qaoa_difficulty

                    # Check if paths match
                    if qaoa_path == classical_path:
                        run_data["path_match"] = True

            # Save sample visualizations
            if saved_samples < save_samples and run_data["qaoa_success"] and run_data["classical_success"]:
                try:
                    # Create visualization
                    plt_fig = visualize_heist_scenario(
                        city_graph, bank_node, safehouse_node,
                        qaoa_path, selected_edges, classical_path
                    )

                    # Save with detailed filename
                    sample_filename = f"{results_dir}/sample_run_{run}_nodes{num_nodes}_ratio{run_data['approx_ratio']:.2f}.png"
                    plt_fig.savefig(sample_filename, dpi=300, bbox_inches='tight')
                    plt_fig.close()

                    saved_samples += 1
                    print(f"Saved visualization sample {saved_samples}/{save_samples}")
                except Exception as viz_err:
                    print(f"Error saving visualization for run {run}: {viz_err}")

        except Exception as e:
            print(f"Error in run {run}: {e}")
            import traceback
            traceback.print_exc()

        # Add run data to results
        results.append(run_data)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"{results_dir}/heist_results_{timestamp}.csv", index=False)

    return results_df


def analyze_results(results_df):
    """
    Analyze and visualize the experimental results

    Parameters:
    -----------
    results_df : DataFrame
        Results from the experiments
    """
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Print summary statistics
    print("\n===== EXPERIMENT SUMMARY =====")
    print(f"Total runs: {len(results_df)}")
    print(f"QAOA success rate: {results_df['qaoa_success'].mean():.2%}")
    print(f"QAOA required fallback: {results_df['qaoa_fallback'].mean():.2%}")
    print(f"QAOA required repair: {results_df['qaoa_repair'].mean():.2%}")
    print(f"Path matched classical solution: {results_df['path_match'].mean():.2%}")

    # Calculate average approximation ratio (excluding failed runs)
    valid_ratios = results_df[results_df['approx_ratio'].notna()]
    if len(valid_ratios) > 0:
        print(f"Average approximation ratio: {valid_ratios['approx_ratio'].mean():.4f}")
        print(f"Median approximation ratio: {valid_ratios['approx_ratio'].median():.4f}")

    # Calculate average speedup factor
    valid_speedups = results_df[results_df['speedup_factor'].notna()]
    if len(valid_speedups) > 0:
        print(f"Classical algorithm was {valid_speedups['speedup_factor'].median():.1f}x faster (median)")

    # Save summary to text file
    with open(f"{results_dir}/summary_{timestamp}.txt", 'w') as f:
        f.write("===== BANK HEIST QAOA EXPERIMENT SUMMARY =====\n")
        f.write(f"Total runs: {len(results_df)}\n")
        f.write(f"QAOA success rate: {results_df['qaoa_success'].mean():.2%}\n")
        f.write(f"QAOA required fallback: {results_df['qaoa_fallback'].mean():.2%}\n")
        f.write(f"QAOA required repair: {results_df['qaoa_repair'].mean():.2%}\n")
        f.write(f"Path matched classical solution: {results_df['path_match'].mean():.2%}\n")

        if len(valid_ratios) > 0:
            f.write(f"Average approximation ratio: {valid_ratios['approx_ratio'].mean():.4f}\n")
            f.write(f"Median approximation ratio: {valid_ratios['approx_ratio'].median():.4f}\n")

        if len(valid_speedups) > 0:
            f.write(f"Median speedup factor: {valid_speedups['speedup_factor'].median():.4f}\n")
            f.write(f"(Higher speedup factor means classical is faster)\n")

    # Generate visualizations
    # 1. Success rates pie chart
    plt.figure(figsize=(10, 6))
    success_data = [
        results_df['qaoa_success'].mean(),
        results_df['qaoa_fallback'].mean(),
        results_df['qaoa_repair'].mean(),
        1 - results_df['qaoa_success'].mean()
    ]
    labels = [
        'QAOA Success (No Issues)',
        'QAOA with Classical Fallback',
        'QAOA with Repair',
        'QAOA Failed'
    ]
    # Adjust for overlapping categories
    success_data[0] = success_data[0] - success_data[1] - success_data[2]

    colors = ['#2ecc71', '#f39c12', '#3498db', '#e74c3c']
    plt.pie(success_data, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('QAOA Success Rate Breakdown')
    plt.savefig(f"{results_dir}/success_rate_pie_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Approximation ratio distribution
    if len(valid_ratios) > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(valid_ratios['approx_ratio'], kde=True, bins=20)
        plt.axvline(x=1.0, color='red', linestyle='--', label='Optimal Ratio (1.0)')
        plt.title('Distribution of Approximation Ratios')
        plt.xlabel('Approximation Ratio (Higher is Better)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f"{results_dir}/approx_ratio_dist_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Execution time comparison
    if len(valid_speedups) > 0:
        plt.figure(figsize=(12, 6))
        time_data = results_df[['qaoa_time', 'classical_time']].copy()
        # Convert to milliseconds for better visualization
        time_data['qaoa_time'] = time_data['qaoa_time'] * 1000
        time_data['classical_time'] = time_data['classical_time'] * 1000

        # Plot only valid data
        valid_times = time_data.dropna()
        sns.boxplot(data=valid_times)
        plt.title('Execution Time Comparison')
        plt.ylabel('Time (milliseconds)')
        plt.yscale('log')  # Log scale due to large difference
        plt.savefig(f"{results_dir}/time_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Performance by graph size
    plt.figure(figsize=(12, 6))
    success_by_size = results_df.groupby('num_nodes').agg({
        'qaoa_success': 'mean',
        'qaoa_fallback': 'mean',
        'qaoa_repair': 'mean'
    })
    success_by_size = success_by_size.reset_index()

    # Calculate pure success rate (no fallback or repair)
    success_by_size['pure_success'] = success_by_size['qaoa_success'] - success_by_size['qaoa_fallback'] - \
                                      success_by_size['qaoa_repair']

    # Plot
    success_by_size.plot(x='num_nodes', y=['pure_success', 'qaoa_fallback', 'qaoa_repair'],
                         kind='bar', stacked=True, figsize=(12, 6),
                         color=['#2ecc71', '#f39c12', '#3498db'])
    plt.title('QAOA Performance by Graph Size')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Success Rate')
    plt.legend(['Clean Success', 'Classical Fallback', 'Required Repair'])
    plt.savefig(f"{results_dir}/performance_by_size_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Approximation ratio by graph size
    if len(valid_ratios) > 0:
        plt.figure(figsize=(12, 6))
        ratio_by_size = results_df.groupby('num_nodes')['approx_ratio'].mean().reset_index()
        ratio_by_size.plot(x='num_nodes', y='approx_ratio', kind='bar', figsize=(12, 6), color='#3498db')
        plt.axhline(y=1.0, color='red', linestyle='--', label='Optimal Ratio (1.0)')
        plt.title('Average Approximation Ratio by Graph Size')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Approximation Ratio (Higher is Better)')
        plt.legend()
        plt.savefig(f"{results_dir}/ratio_by_size_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Return the timestamp for reference
    return timestamp


def main():
    """Run the main experiment and analysis workflow"""
    print("Starting Bank Heist QAOA Experiment Suite")

    # Get parameters from user or use defaults
    try:
        num_runs = int(input("Enter number of runs to perform [default 30]: ") or "30")
        min_nodes = int(input("Enter minimum number of nodes [default 6]: ") or "6")
        max_nodes = int(input("Enter maximum number of nodes [default 12]: ") or "12")
        save_samples = int(input("Enter number of sample visualizations to save [default 5]: ") or "5")
    except ValueError:
        print("Invalid input, using default values")
        num_runs = 30
        min_nodes = 6
        max_nodes = 12
        save_samples = 5

    print(f"\nRunning {num_runs} simulations with graphs of {min_nodes}-{max_nodes} nodes")
    print(f"Saving {save_samples} sample visualizations")

    # Run experiments
    start_time = time.time()
    results_df = run_heist_experiment(num_runs, min_nodes, max_nodes, save_samples)

    # Analyze results
    timestamp = analyze_results(results_df)

    # Report execution time
    total_time = time.time() - start_time
    print(f"\nExperiment completed in {total_time:.2f} seconds")
    print(f"Results saved in '{results_dir}' with timestamp {timestamp}")


if __name__ == "__main__":
    main()