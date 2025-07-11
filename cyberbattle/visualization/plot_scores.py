# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    plot_scores.py
    Script to plot the scores of the RL agents for goals/NLP extractors
"""
# TODO: The code may need readaptation to work in all cases

import shutil
import glob
from datetime import datetime
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.visualization.rliable.rliable_metrics import plot_indicators, plot_score_hist, plot_performance_profile, plot_probability_improvement, plot_ranks, plot_alternative_metrics # noqa: E402
script_dir = os.path.dirname(__file__)

def calculate_bootstrap_ci(data, confidence=0.99, n_iterations=10000):
    rng = np.random.default_rng()
    bootstrap_means = np.array([
        np.mean(rng.choice(data, replace=True, size=len(data)))
        for _ in range(n_iterations)
    ])
    lower_bound = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
    original_mean = np.mean(data)
    return original_mean, lower_bound, upper_bound


def plot_difference_scores(results):
    # Extracting data
    goals = ['control', 'discovery', 'disruption']
    sets = ['train', 'val', 'test']
    data = {goal: {s: None for s in sets} for goal in goals}

    for entry in results:
        goal = entry['Goal']
        set_type = entry['Set']
        mean = entry['Mean Performance']
        ci_error = (entry['Mean Performance'] - entry['CI Lower'], entry['CI Upper'] - entry['Mean Performance'])
        data[goal][set_type] = [mean, ci_error]

    # Plotting
    x = np.arange(len(goals))  # X-axis positions for goals
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_title('TRPO Scores per Goal and Split', fontsize=42)
    colors = ["blue", "orange", "green"]
    set_rewrite = {
        "train": "Training",
        "val": "Validation",
        "test": "Test"
    }
    for i, set_type in enumerate(sets):
        means = [data[goal][set_type][0] for goal in goals]  # type: ignore
        ci_errors = [data[goal][set_type][1] for goal in goals]  # type: ignore
        ax.bar(
            x + i * bar_width, means, bar_width,
            label=set_rewrite[set_type].capitalize(),
            yerr=np.array(ci_errors).T,  # Error bars
            capsize=5, edgecolor='black', alpha=0.7,
            color=colors[i],
        )

    # Formatting
    ax.set_ylabel('Mean Score & BCI', fontsize=38)
    ax.set_xlabel('Goal', fontsize=40, labelpad=42)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(goals, fontsize=40)
    ax.legend(
        fontsize=28,
        loc='upper left', bbox_to_anchor=(0.61, 1), handlelength=2
    )
    ax.tick_params(axis='y', labelsize=34)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Removing extra x-ticks
    ax.set_xticks([])
    for i, goal in enumerate(goals):
        ax.text(
            x[i] + 1 * bar_width, -0.07, goal.capitalize(),
            fontsize=34, ha='center', va='bottom', color='black'
        )

    plt.tight_layout()
    plt.savefig(os.path.join(logs_folder, 'difference_scores.pdf'))
    plt.show()


def process_results(yaml_data, dimension, indicators):
    df = pd.DataFrame(yaml_data)

    # Ensure the dimension column is valid
    if dimension not in df.columns:
        raise ValueError(f"Dimension '{dimension}' is not in the YAML data columns")

    # Get unique values for the specified dimension
    unique_values = df[dimension].unique()

    results = {}
    order = ["control", "discovery", "disruption"]

    for value in unique_values:
        subset = df[df[dimension] == value]
        if indicators:
            scores = subset['Score']
            if value == "random500":
                value = "mixed"
            if value == "distilbert":
                value = "DistilBERT"
            elif value == "gpt2":
                value = "GPT-2"
            elif value == "roberta":
                value = "RoBERTa"
            elif value == "bert":
                value = "BERT"
            results[f"{value}"] = scores
        elif dimension != "RL":
            subset_control = subset[subset["Goal"] == "control"]
            subset_discovery = subset[subset["Goal"] == "discovery"]
            subset_disruption = subset[subset["Goal"] == "disruption"]
            scores_control = subset_control['Score']
            scores_discovery = subset_discovery['Score']
            scores_disruption = subset_disruption['Score']
            if value == "random500":
                value = "mixed"
            if value == "distilbert":
                value = "DistilBERT"
            elif value == "gpt2":
                value = "GPT-2"
            elif value == "roberta":
                value = "RoBERTa"
            elif value == "bert":
                value = "BERT"
            results[f"{value}"] = np.array([scores_control, scores_discovery, scores_disruption])
        else:
            subset_control = subset[subset["Goal"] == "control"]
            subset_discovery = subset[subset["Goal"] == "discovery"]
            subset_disruption = subset[subset["Goal"] == "disruption"]

            scores_control = subset_control['Score']
            scores_discovery = subset_discovery['Score']
            scores_disruption = subset_disruption['Score']

            if value == "random":
                value = "Random"
            results[value] = np.array([scores_control, scores_discovery, scores_disruption])

    return results, order


def process_files(directory, difference=False):
    file_pattern = os.path.join(directory, '*.csv')
    file_list = glob.glob(file_pattern)
    scores = []

    for file_path in file_list:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        algo, goal, llm, test_set, K, additional, experiment = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6].split(".")[0]
        df = pd.read_csv(file_path)
        if algo == 'random':
            df_agent = df
        else:
            df_agent = df[df['agent'] == 'agent'] # it can potentially contain random performances as well

        for row in df_agent.iterrows():
            if goal in ['discovery_node', 'control_node', 'disruption_node']:
                # Use episode won as the score
                score_column = 'won'

            else:
                # Use appropriate column based on the goal
                if goal == 'discovery':
                    score_column = 'discovered_amount_percentage'
                elif goal == 'control':
                    score_column = 'owned_nodes_percentage'
                elif goal == 'disruption':
                    score_column = 'disrupted_nodes_percentage'
                else:
                    raise ValueError(f"Unknown goal: {goal}")
            if difference:
                score = {
                    'RL': algo,
                    'NLP': llm,
                    'Goal': goal,
                    'Test Set': test_set,
                    'Score': row[1][score_column],
                    'K': int(K),
                    'Set': additional,
                    'Experiment': experiment
                }
            else:
                score = {
                    'RL': algo,
                    'NLP': llm,
                    'Goal': goal,
                    'Test Set': test_set,
                    'Score': row[1][score_column],
                    'K': int(K),
                    'Defender Parameter': additional,
                    'Experiment': experiment
                }
            scores.append(score)

    return scores

def get_most_recent_file(path, file_extension='csv'):
    """Get the most recent file with the given extension in the directory."""
    files = glob.glob(os.path.join(path, f"*.{file_extension}"))
    if not files:
        return None
    # filter files that should have average in name
    files = [file for file in files if 'average' in file or 'random' in file]
    latest_file = max(files, key=os.path.getctime)
    return latest_file


def find_next_run_number(dest_dir, base_filename):
    """Find the next available run number for a given base filename in the destination directory."""
    run_number = 1
    while os.path.exists(os.path.join(dest_dir, f"{base_filename}_{run_number}.csv")):
        run_number += 1
    return run_number

def copy_and_rename_files(src_root, dest_dir, variation=False, difference=False):
    for root, dirs, files in os.walk(src_root):
        # Check if the current directory is a target directory
        if 'test' in dirs:
            # Parse the current path to get algorithm, goal, and LLM information
            path_parts = root.split(os.sep)[-1].split("_")
            if len(path_parts) == 1 and path_parts[0] == 'test':
                continue
            print("path parts:", path_parts)
            try:
                algo = path_parts[0]
                goal = path_parts[2]
                llm = path_parts[3]
            except ValueError:
                continue
            test_dir = os.path.join(root, 'test')
            for test_set_dir in os.listdir(test_dir):
                if algo == 'random':
                    test_set_path = os.path.join(test_dir, test_set_dir, 'random')
                else:
                    test_set_path = os.path.join(test_dir, test_set_dir, 'validation', '1')
                if not os.path.isdir(test_set_path):
                    test_set_path = os.path.join(test_dir, test_set_dir, 'train', '1')
                if os.path.isdir(test_set_path):
                    if variation:
                        all_csv_files = glob.glob(os.path.join(test_set_path, '*.csv'))
                        for csv_file in all_csv_files:
                            csv_filename = os.path.basename(csv_file)
                            test_set_name = ''.join(test_set_dir.split('_'))
                            test_set_name = ''.join(test_set_name.split('-'))
                            if len(csv_filename.split("_")) > 7:
                                defender_parameter = csv_filename.split("_")[6]
                                if '-' in defender_parameter:  # reimage case
                                    defender_parameter = defender_parameter.split('-')[0]
                                else:
                                    defender_parameter = defender_parameter
                            else:
                                defender_parameter = '0.0'
                            K = csv_filename.split("_")[4]
                            # events case it is already good like this
                            base_filename = f"{algo}_{goal}_{llm}_{test_set_name}_{K}_{defender_parameter}"
                            run_number = find_next_run_number(dest_dir, base_filename)
                            new_filename = f"{algo}_{goal}_{llm}_{test_set_name}_{K}_{defender_parameter}_{run_number}.csv"
                            dest_path = os.path.join(dest_dir, new_filename)
                            shutil.copy2(csv_file, dest_path)
                    elif difference:
                        all_csv_files = glob.glob(os.path.join(test_set_path, '*.csv'))
                        for csv_file in all_csv_files:
                            csv_filename = os.path.basename(csv_file)
                            test_set_name = ''.join(test_set_dir.split('_'))
                            test_set_name = ''.join(test_set_name.split('-'))
                            if len(csv_filename.split("_")) == 9:
                                set = csv_filename.split("_")[5]
                                if set == "trainingset":
                                    set = "train"
                                elif set == "validationset":
                                    set = "val"
                            else:
                                set = "test"
                            K = csv_filename.split("_")[4]
                            # events case it is already good like this
                            base_filename = f"{algo}_{goal}_{llm}_{test_set_name}_{K}_{set}"
                            run_number = find_next_run_number(dest_dir, base_filename)
                            new_filename = f"{algo}_{goal}_{llm}_{test_set_name}_{K}_{set}_{run_number}.csv"
                            dest_path = os.path.join(dest_dir, new_filename)
                            shutil.copy2(csv_file, dest_path)
                    else:
                        most_recent_csv = get_most_recent_file(test_set_path)
                        if most_recent_csv:
                            csv_filename = os.path.basename(most_recent_csv)
                            test_set_name = ''.join(test_set_dir.split('_'))
                            test_set_name = ''.join(test_set_name.split('-'))
                            if len(csv_filename.split("_")) > 6:
                                defender_parameter = csv_filename.split("_")[6]
                                if len(defender_parameter) <= 2: # otherwise date, no defender parameter
                                    if '-' in defender_parameter:  # reimage case
                                        defender_parameter = defender_parameter.split('-')[0]
                                else:
                                    defender_parameter = ""
                            else:
                                defender_parameter = ""

                            K = csv_filename.split("_")[4]
                            # events case it is already good like this
                            base_filename = f"{algo}_{goal}_{llm}_{test_set_name}_{K}_{defender_parameter}"
                            run_number = find_next_run_number(dest_dir, base_filename)
                            new_filename = f"{algo}_{goal}_{llm}_{test_set_name}_{K}_{defender_parameter}_{run_number}.csv"
                            dest_path = os.path.join(dest_dir, new_filename)
                            shutil.copy2(most_recent_csv, dest_path)


def group_defender_probability(scores):
    df = pd.DataFrame(scores)
    df['Defender Action Probability'] = df['Defender Parameter'].str.extract(r'(\d+\.\d+)').astype(float)/100

    df.loc[df['Test Set'] == 'base', 'Defender Action Probability'] = 0

    results = []

    for goal in df['Goal'].unique():
        subset = df[df['Goal'] == goal]
        for value in subset['Defender Action Probability'].unique():
            # skip value Nan
            if pd.isna(value):
                continue
            subset_nodes = subset[subset['Defender Action Probability'] == value]
            scores_list = subset_nodes['Score'].values

            mean_score, ci_low, ci_high = calculate_bootstrap_ci(scores_list)
            if value == 0.075 or value == 0.025:
                continue
            results.append({
                'Defender Action Probability': value,
                'Goal': goal,
                'Mean Performance': mean_score,
                'CI Lower': ci_low,
                'CI Upper': ci_high
            })

    return results

def group_sets(scores):
    df = pd.DataFrame(scores)
    results = []

    for goal in df['Goal'].unique():
        subset = df[df['Goal'] == goal]
        for value in subset['Set'].unique():
            # skip value Nan
            if pd.isna(value):
                continue
            subset_nodes = subset[subset['Set'] == value]
            scores_list = subset_nodes['Score'].values

            mean_score, ci_low, ci_high = calculate_bootstrap_ci(scores_list)
            results.append({
                'Set': value,
                'Goal': goal,
                'Mean Performance': mean_score,
                'CI Lower': ci_low,
                'CI Upper': ci_high
            })

    # Compute relative test-to-training ratio
    for goal in df['Goal'].unique():
        subset = df[df['Goal'] == goal]
        train_scores = subset[subset['Set'] == 'train']['Score'].values
        test_scores = subset[subset['Set'] == 'test']['Score'].values
        val_scores = subset[subset['Set'] == 'val']['Score'].values
        mean_train, _, _ = calculate_bootstrap_ci(train_scores)
        mean_test, _, _ = calculate_bootstrap_ci(test_scores)
        mean_val, _, _ = calculate_bootstrap_ci(val_scores)


    train_scores = df[df['Set'] == 'train']['Score'].values
    test_scores = df[df['Set'] == 'test']['Score'].values
    val_scores = df[df['Set'] == 'val']['Score'].values
    mean_train, _, _ = calculate_bootstrap_ci(train_scores)
    mean_test, _, _ = calculate_bootstrap_ci(test_scores)
    mean_val, _, _ = calculate_bootstrap_ci(val_scores)
    return results


def group_steps(scores):
    df = pd.DataFrame(scores)
    df['Steps per Node'] = df['K']

    results = []

    for goal in df['Goal'].unique():
        subset = df[df['Goal'] == goal]
        for value in subset['Steps per Node'].unique():
            # skip value Nan
            if pd.isna(value):
                continue
            subset_nodes = subset[subset['Steps per Node'] == value]
            scores_list = subset_nodes['Score'].values

            mean_score, ci_low, ci_high = calculate_bootstrap_ci(scores_list)

            results.append({
                'Steps per Node': value,
                'Goal': goal,
                'Mean Performance': mean_score,
                'CI Lower': ci_low,
                'CI Upper': ci_high
            })

    return results

def calculate_average_slope(subset):
    # Calculate the average slope based on y-value changes relative to x-value changes
    y_values = subset['Mean Performance'].values
    slopes = [(y_values[i + 1] - y_values[i])  for i in range(len(y_values) - 1)]
    return sum(slopes)


def plot_performance(data, logs_folder, metric='Number of Nodes',):
    df = pd.DataFrame(data)

    plt.figure(figsize=(13, 8))
    sns.set(style="white", font_scale=1.2)  # Set style and font scale for better readability
    goals = df['Goal'].unique()
    palette = sns.color_palette("tab10", len(goals))  # Choose a color palette

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Initialize legend labels list
    legend_labels = []

    # Plot regular results
    for i, goal in enumerate(goals):
        subset = df[(df['Goal'] == goal)]
        subset = subset.sort_values(metric)

        if not subset.empty:
            # Calculate average slope for the legend
            average_slope = calculate_average_slope(subset)
            legend_labels.append(f'{goal.capitalize()} (slope: {average_slope:.2f})')
            # Plot shaded area for confidence intervals
            plt.fill_between(subset[metric],
                             subset['CI Lower'],
                             subset['CI Upper'],
                             color=palette[i],
                             alpha=0.3,  # Adjust transparency of shaded area
                             label=None)

            # Plot mean performance with error bars
            plt.errorbar(subset[metric],
                         subset['Mean Performance'],
                         yerr=[subset['Mean Performance'] - subset['CI Lower'],
                               subset['CI Upper'] - subset['Mean Performance']],
                         fmt='-o',
                         color=palette[i],
                         label=f'{goal} (regular)',
                         capsize=5,
                         elinewidth=2,
                         markersize=8,
                         capthick=2)

    # X-axis ticks based on metric
    if metric == 'Defender Action Probability':
        ticks = [0.01, 0.1, 0.25, 0.5]
        labels = ['{:.0f}%'.format(ticks[0]*100), '{:.0f}%'.format(ticks[1]*100), '{:.0f}%'.format(ticks[2]*100),
                  '{:.0f}%'.format(ticks[3]*100)] #, '{:.0f}%'.format(ticks[4]*100)] #, '{:.0f}%'.format(ticks[5]*100),
                  #'{:.0f}%'.format(ticks[6]*100), '{:.0f}%'.format(ticks[7]*100)]
        plt.xticks(ticks=ticks, labels=labels, fontsize=36)
        loc = 'lower right'
        bbox_to_anchor = None
    else: #if metric == 'Steps per Node':
        ticks = [1, 5, 10, 20, 30, 40, 50]
        plt.xticks(ticks=ticks, fontsize=42)
        loc = 'lower right'
        bbox_to_anchor = None

    xlabels = {
        'Defender Action Probability': r'Node-Intervention Probability ($\mathit{p_{n}}$)',
        'Steps per Node': r'Steps per Node ($\mathit{K}$)'
    }

    plt.xlabel(xlabels[metric], fontsize=42)

    plt.yticks(fontsize=46)
    plt.legend(labels=legend_labels, fontsize=34, loc=loc, bbox_to_anchor=bbox_to_anchor)
    plt.ylabel('Mean Score & BCI', fontsize=42)
    plt.title("TRPO Goal Scores vs Defender", fontsize=38)
    plt.ylim(0, 0.6)
    plt.grid(False)  # Disable background grid
    plt.tight_layout()
    plt.savefig(
        os.path.join(logs_folder, f"performance_plot_{'_'.join(metric.lower().split(' '))}.pdf"))  # Save the figure as a PDF for high quality
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy and rename files from source to destination directory.')
    parser.add_argument('-f', '--folders', type=str, nargs='+', required=True, help='Path to the directories to involve in the experiments.')
    parser.add_argument('-d', '--dimension', type=str, required=True, choices=['RL', 'NLP'],
                        help='Dimension to aggregate and calculate statistics (e.g., Goal).')
    parser.add_argument('-o', '--option', type=str, default='interval_estimate',
                        help='Type of analysis to be performed.',
                        choices=['rank', 'histogram', 'performance_profile',
                                 'probability_improvement', 'advanced', 'indicators', 'variation', 'train_val_test'])
    args = parser.parse_args()

    logs_folder = os.path.join(script_dir, "logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(logs_folder, exist_ok=True)

    for folder in args.folders:
        source_dir = os.path.join(script_dir, "..", "agents", "logs", folder)
        copy_and_rename_files(source_dir, logs_folder, args.option=="variation", args.option=="train_val_test")

    output = os.path.join(logs_folder, 'results.yaml')
    results = process_files(logs_folder, args.option=="train_val_test")
    with open(output, 'w') as file:
        yaml.dump(results, file, default_flow_style=False)
    if args.option != "variation" and args.option != "train_val_test":
        scores_dict, tasks = process_results(results, args.dimension, indicators=args.option == "indicators")
        dimensions = list(scores_dict.keys())

        if args.option == 'indicators':
            plot_indicators(scores_dict, label_name=args.dimension, logs_folder=logs_folder)
        elif args.option == 'histogram':
            for dimension_value in dimensions:
                plot_score_hist(scores_dict[dimension_value], bins=10, N=min(3, len(tasks)), figsize=(26, 11), names=tasks,
                                algorithm_name=dimension_value, logs_folder=logs_folder)
        elif args.option == 'performance_profile':
            plot_performance_profile(scores_dict, logs_folder=logs_folder)
        elif args.option == 'probability_improvement':
            plot_probability_improvement(scores_dict, dimensions, logs_folder=logs_folder)
        elif args.option == 'rank':
            plot_ranks(scores_dict, dimensions, tasks, logs_folder=logs_folder)
        elif args.option == 'advanced':
            plot_alternative_metrics(scores_dict, logs_folder=logs_folder)
    if args.option == 'train_val_test':
        results = group_sets(results)
        plot_difference_scores(results)
    elif args.option == 'variation':
        if args.dimension == 'defender':
            results = group_defender_probability(results)
            plot_performance(results, metric='Defender Action Probability', logs_folder=logs_folder)
        elif args.dimension == 'steps':
            results = group_steps(results)
            plot_performance(results, metric='Steps per Node', logs_folder=logs_folder)
