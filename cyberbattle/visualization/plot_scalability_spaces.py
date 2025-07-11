# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    plot_scalability_spaces.py
    Script to plot the scalability of the architectures (obs/action spaces).
"""

import os
import re
from datetime import datetime
import math
from tensorboard.backend.event_processing import event_accumulator
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
script_dir = os.path.dirname(os.path.abspath(__file__))

# Dictionary to map goals to specific metrics to use for the plotting
goal_metrics = {
    "control": "train/Relative owned nodes percentage",
    "discovery": "train/Relative discovered amount percentage",
    "disruption": "train/Relative disrupted nodes percentage"
}

# List of directories to search for logs to plot
dirs = ['../agents/logs/']

# Initialize results dictionary to store AUC values for each approach
results = {
    'global': {},
    'local': {},
    'continuous': {}
}

# Regular expressions to parse folder names
folder_scenario_pattern = re.compile(r'(?P<approach>TRPO_\w+)_graphs_nodes=(?P<nodes>\d+)_vulns=(?P<vulns>\d+)_(?P<scenario>\d+)_'
                                     r'(?P<goal>\w+)_(?P<model>\w+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}')
folder_pattern = re.compile(r'(?P<approach>TRPO_\w+)_graphs_nodes=(?P<nodes>\d+)_vulns=(?P<vulns>\d+)_'
                            r'(?P<goal>\w+)_(?P<model>\w+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}')

# Dictionary to store failure points for each approach, in order to plot some 'X' points in the 3D plot
failure_points = {
    'global': [],
    'local': [],
    'continuous': []
}

# Function to read the AUC of the specified metric from the TensorBoard log
def extract_auc_from_tensorboard(tensorboard_path, metric_name, normalizer=None):
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    if metric_name in ea.Tags()['scalars']:
        events = ea.Scalars(metric_name)
        values = [event.value for event in events]
        if normalizer:
            values = [((value * normalizer) - 1) / (normalizer - 1) for value in values]
            values = [value if value >= 0 else 0 for value in values]
        auc = np.trapz(values, dx=1)  # Normalized AUC
        perfect_values = [1 for _ in values]
        ideal_auc = np.trapz(perfect_values, dx=1)
        auc /= ideal_auc
        return auc
    else:
        assert False, f"Metric {metric_name} not found in TensorBoard log {tensorboard_path}. Available metrics: {ea.Tags()['scalars']}"

# Function to compute the relative improvement between two values, in order to compute overall improvements between approaches
def compute_relative_improvement(f2, f1, epsilon=1e-6):
    """
    Compute the relative improvement between two values f1 and f2.
    """
    if f1 != 0:
        return (f2 - f1) / abs(f1)
    elif f1 == 0 and f2 > 0:
        return f2 / epsilon  # Large improvement if f1 is zero and f2 is positive
    elif f1 == 0 and f2 == 0:
        return 0.0  # No improvement if both are zero
    else:
        return (f2 - f1) / (abs(f1) + epsilon)  # Handle edge cases for negative values

# Function to compute bootstrap confidence intervals a set of values
def bootstrap_ci(data, n_iterations=1000, ci_percentile=95):
    means = []
    n = len(data)
    for _ in range(n_iterations):
        resampled_data = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(resampled_data))
    lower = np.percentile(means, (100 - ci_percentile) / 2)
    upper = np.percentile(means, 100 - (100 - ci_percentile) / 2)
    return lower, upper

# Function to plot the 3D AUC vs Nodes and Vulnerabilities with smoothing
def plot_3d_auc_vs_nodes_vulns_smoothed(results, all_combinations, logs_folder):
    all_x_vals = []
    all_y_vals = []
    all_z_means = []
    approach_labels = []

    # Define fixed colors for surfaces
    approach_colors = {
        "continuous": "steelblue",
        "local": "darkorange",
        "global": "forestgreen",
    }

    # Initialize a dictionary to hold surface objects for each approach
    surface_objects = {
        "continuous": [],
        "local": [],
        "global": [],
    }

    scores_dict = {}

    # Iterate over all approaches
    for approach, data in results.items():
        if len(data) == 0:
            continue

        if approach == "compressed":
            approach = "continuous" # change label for plotting

        x_vals = []
        y_vals = []
        z_means = []

        surface_color = approach_colors[approach]

        for combination in all_combinations:
            nodes, vulns = combination
            if combination in data:
                combined_data = data[combination]
                mean_score = np.mean(combined_data)
                x_vals.append(nodes)
                y_vals.append(vulns)
                z_means.append(mean_score)
                if approach not in scores_dict:
                    scores_dict[approach] = {}

                if (nodes, vulns) not in scores_dict[approach]:
                    scores_dict[approach][(nodes, vulns)] = mean_score

        all_x_vals.extend(x_vals)
        all_y_vals.extend(y_vals)
        all_z_means.extend(z_means)
        approach_labels.extend([approach] * len(x_vals))

        # Create a grid for smoothing
        xi = np.linspace(min(x_vals), max(x_vals), 100)
        yi = np.linspace(min(y_vals), max(y_vals), 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate to smooth the surface, checking if there's data for each approach
        if len(x_vals) > 0:
            zi = griddata((x_vals, y_vals), z_means, (xi, yi), method='cubic')
            zi = gaussian_filter(zi, sigma=5)

            surface_objects[approach].append((xi, yi, zi, surface_color, x_vals, y_vals, z_means))

    # Refill all values from 5 to 250 in order to have a complete grid
    for approach, data in scores_dict.items():
        for nodes in [5, 10, 15, 20, 35, 50, 100, 150, 200, 250]:
            for vulns in [5, 10, 15, 20, 35, 50, 100, 150, 200, 250]:
                if (nodes, vulns) not in data:
                    scores_dict[approach][(nodes, vulns)] = math.nan


    # Compute relative improvements between approaches
    print("Calculating relative improvements...")
    for approach, data in scores_dict.items():
        total_improvement = 0.0
        improvement_count = 0
        print(f"Processing approach: {approach}")
        for (x_vals, y_vals), z_means in data.items():
            if approach != 'continuous':  # Define your base approach to compare against
                for (x_vals_2, y_vals_2), z_means_2 in scores_dict['continuous'].items():
                    if x_vals == x_vals_2 and y_vals == y_vals_2 and not math.isnan(z_means) and not math.isnan(z_means_2):
                        improvement = compute_relative_improvement(z_means_2, z_means)
                        total_improvement += improvement
                        improvement_count += 1

        # Calculate the average relative improvement
        average_relative_improvement = total_improvement / improvement_count if improvement_count > 0 else 0
        print(f"Average Relative Improvement: {average_relative_improvement}")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Iterate over the stored surfaces and plot them
    for approach, surfaces in surface_objects.items():
        surface_color = approach_colors[approach]
        for xi, yi, zi, surface_color, x_vals, y_vals, z_means in surfaces:
            # Plot the smoothed surface with a fixed color
            ax.plot_surface(xi, yi, zi, color=surface_color, edgecolor='none', alpha=0.6)
        ax.scatter([], [], color=surface_color, label=f"{approach.capitalize()}", s=50)

        for item in failure_points[approach]:
            # Calculate opacity based on the number of failures (max 3 failures)
            alpha = item['count'] / 3  # Normalized to [0, 1]
            ax.scatter(item['nodes'], item['vulns'], 0, color=approach_colors[approach], marker='x', s=100, alpha=alpha)


    ax.scatter([], [], color='black', marker='x', s=100, label='Failure')

    ax.set_zlim(0, 0.8)
    ax.set_xlabel("Nodes", fontsize=25, labelpad=25)  # Adjust labelpad as needed
    ax.set_ylabel("Vulnerabilities", fontsize=25, labelpad=25)  # Adjust labelpad as needed
    ax.set_zlabel("Mean Scores' AUC", fontsize=25, labelpad=205)  # Adjust labelpad as needed
    ax.set_title("Scores' AUC Surfaces", fontsize=30)
    ax.legend(fontsize=23)

    unique_nodes = [5, 30, 50, 100, 150, 200, 250]
    unique_vulns = [5, 30, 50, 100, 150, 200, 250]
    ax.set_xticks(unique_nodes)
    ax.set_yticks(unique_vulns)

    ax.set_xlim(left=0, right=260)
    ax.set_ylim(bottom=0, top=260)

    ax.set_xticklabels(unique_nodes, rotation=45, fontsize=18)
    ax.set_yticklabels(unique_vulns, rotation=45, fontsize=18)

    # Save the 3D plot with different orientations
    ax.set_zticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], fontsize=18)
    orientations = [15, 30, 45, 60, 90, 120, 135, 150, 180]  # Elevation, Azimuth pairs
    for elev in orientations:
        for azim in orientations:
            print(f"Saving 3D plot with elevation: {elev}, azimuth: {azim}")
            ax.view_init(elev=elev, azim=azim)
            ax.dist = 50
            ax.set_zlabel("Mean Scores' AUC", fontsize=25, labelpad=5)  # Adjust labelpad as needed
            plt.tight_layout()
            plt.savefig(os.path.join(logs_folder,f"3d_plot_elev_{elev}_azim_{azim}.pdf"),  bbox_inches='tight',  pad_inches=0.3)

    fig_zoom = plt.figure(figsize=(7, 7))
    ax_zoom = fig_zoom.add_subplot(111, projection='3d')

    x_lim, y_lim = 50, 50

    x_vals_zoom = np.array(all_x_vals)
    y_vals_zoom = np.array(all_y_vals)

    for approach, surfaces in surface_objects.items():
        surface_color = approach_colors[approach]
        for xi, yi, zi, surface_color, x_vals, y_vals, z_means in surfaces:
            # Filter points for zoomed-in view
            x_vals = np.array(x_vals)
            y_vals = np.array(y_vals)
            z_means = np.array(z_means)

            mask = (x_vals <= x_lim) & (y_vals <= y_lim)
            x_vals_zoom = x_vals[mask]
            y_vals_zoom = y_vals[mask]
            z_means_zoom = z_means[mask]

            xi_zoom = np.linspace(min(x_vals_zoom), max(x_vals_zoom), 100)
            yi_zoom = np.linspace(min(y_vals_zoom), max(y_vals_zoom), 100)
            xi_zoom, yi_zoom = np.meshgrid(xi_zoom, yi_zoom)
            zi_zoom = griddata((x_vals_zoom, y_vals_zoom), z_means_zoom, (xi_zoom, yi_zoom), method='cubic')
            zi_zoom = gaussian_filter(zi_zoom, sigma=5)

            # Plot the smoothed surface
            ax_zoom.plot_surface(xi_zoom, yi_zoom, zi_zoom, color=surface_color, edgecolor='none', alpha=0.6)

        ax_zoom.scatter([], [], color=surface_color, label=f"{approach.capitalize()}", s=50)

    # Set zoomed-in view's limits and labels
    ax_zoom.set_xlim(0, 55)
    ax_zoom.set_ylim(0, 55)
    ax_zoom.set_zlim(0, 0.8)
    ax.set_zlim(0, 0.8)
    # Set labels and title
    ax_zoom.set_xlabel("Nodes", fontsize=25, labelpad=15)  # Adjust labelpad as needed
    ax_zoom.set_ylabel("Vulnerabilities", fontsize=25, labelpad=15)  # Adjust labelpad as needed
    ax_zoom.set_zlabel("Mean Scores' AUC", fontsize=25, labelpad=15)  # Adjust labelpad as needed
    ax_zoom.set_title("Learning Score Surface Plot", fontsize=25)
    ax_zoom.legend(fontsize=20)

    unique_nodes = sorted(set(x_vals_zoom))
    unique_vulns = sorted(set(y_vals_zoom))

    ax_zoom.set_xticks(unique_nodes)
    ax_zoom.set_yticks(unique_vulns)
    ax_zoom.set_xticklabels(unique_nodes, rotation=45, fontsize=15)
    ax_zoom.set_yticklabels(unique_vulns, rotation=45, fontsize=15)
    ax_zoom.set_zticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], fontsize=15)

    # Save the zoomed-in plot
    for elev in orientations:
        for azim in orientations:
            print(f"Saving zoomed-in view with elevation: {elev}, azimuth: {azim}")
            ax_zoom.view_init(elev=elev, azim=azim)
            plt.savefig(os.path.join(logs_folder,f"3d_plot_zoomed_elev_{elev}_azim_{azim}.pdf"))

# Script used to generate the smoothed 3D plots
if __name__ == "__main__":
    print("Starting to process directories and extract AUC values...")
    # Main script to process the directories and extract AUC values
    for base_dir in dirs:
        for folder_name in os.listdir(base_dir):
            if 'graphs_nodes' not in folder_name:
                continue
            match = folder_pattern.match(folder_name)
            match2 = folder_scenario_pattern.match(folder_name)
            if match or match2:
                if match2:
                    approach, nodes, vulns, scenario, goal, model = match2.groups()
                else:  # if match2:
                    approach, nodes, vulns, goal, model = match.groups()
                approach = approach.split("_")[1]
                print("Processing folder:", folder_name)
                print("Approach:", approach, "Nodes:", nodes, "Vulns:", vulns, "Scenario:", scenario, "Goal:", goal, "Model:", model)
                if approach == "compressed":
                    approach = "continuous"
                nodes, vulns = int(nodes), int(vulns)
                subfolder = os.listdir(os.path.join(base_dir, folder_name))[0]  # Get the first subfolder
                for subfolder in os.listdir(os.path.join(base_dir, folder_name)):
                    if subfolder.startswith('TRPO_'):
                        break
                tensorboard_dir = os.path.join(base_dir, folder_name, subfolder, 'TRPO_1')
                tensorboard_files = [f for f in os.listdir(tensorboard_dir) if f.startswith('events.out.tfevents')]
                if nodes == 2 or vulns == 2:
                    continue
                if tensorboard_files:
                    tensorboard_path = os.path.join(tensorboard_dir, tensorboard_files[0])
                    metric_name = goal_metrics.get(goal)
                    # Extract and normalize the AUC
                    auc_value = extract_auc_from_tensorboard(tensorboard_path, metric_name, normalizer=None)
                    if math.isnan(auc_value):
                        # store like (nodes, vulns, num_failures)
                        found = False
                        for item in failure_points[approach]:
                            if item['nodes'] == nodes and item['vulns'] == vulns:
                                item['count'] += 1
                                found = True
                                break
                        if not found:
                            failure_points[approach].append({'nodes': nodes, 'vulns': vulns, 'count': 1})
                    else:
                        normalized_auc = auc_value / 1.0
                        if (nodes, vulns) not in results[approach]:
                            results[approach][(nodes, vulns)] = []
                        results[approach][(nodes, vulns)].append(normalized_auc)
                        print(
                            f"Approach: {approach}, Goal: {goal}, Nodes: {nodes}, Vulns: {vulns}, AUC: {normalized_auc}")

    all_combinations = []
    for nodes in [5, 10, 15, 20, 35, 50, 100, 150, 200, 250]:
        for vulns in [5, 10, 15, 20, 35, 50, 100, 150, 200, 250]:
            all_combinations.append((nodes, vulns))
    logs_folder = os.path.join(script_dir, 'logs', "scalability_" + datetime.now().strftime('%Y-%m-%d_%H'))
    os.makedirs(logs_folder, exist_ok=True)
    plot_3d_auc_vs_nodes_vulns_smoothed(results, all_combinations, logs_folder)
