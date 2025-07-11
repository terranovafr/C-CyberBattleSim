# coding=utf-8
# Copyright 2021 The Rliable Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import library as rly, metrics, plot_utils
from .plot_utils import decorate_axis, plot_interval_estimates_shifted
from .metrics_utils import get_rank_matrix
import numpy as np
import os
import warnings
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib import rcParams
from matplotlib import rc
from itertools import combinations

warnings.filterwarnings('default')
RAND_STATE = np.random.RandomState(42)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
sns.set_style("white")
# Matplotlib params
rcParams['legend.loc'] = 'best'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rc('text', usetex=False)

color_dict = {
    # RL algorithms with blue and green shades
    'TRPO': '#ff4c4c',  # Bright red
    'PPO': '#1f77b4',  # Medium blue
    'SAC': '#2ca02c',  # Dark green
    'TD3': '#17becf',  # Teal
    'TQC': '#ff7f0e',  # Orange
    'RPPO': '#9467bd',  # Medium purple
    'A2C': '#e377c2',  # Light pink
    'Random': '#8c564b',  # Brown

    # LM algorithms with yellow and red shades
    'BERT': (0.929, 0.694, 0.125),  # Deep yellow
    'DistilBERT': (0.980, 0.745, 0.235),  # Light yellow
    'RoBERTa': (0.929, 0.125, 0.125),  # Deep red
    'GPT-2': (0.980, 0.235, 0.235),  # Light red
    'CySecBERT': (0.929, 0.125, 0.313),  # Deep reddish-pink
    'SecureBERT': (0.980, 0.235, 0.451),  # Light reddish-pink
    'SecBERT': (0.929, 0.313, 0.125),  # Deep orange-red
    'SecRoBERTa': (0.980, 0.451, 0.235),  # Light orange-red
}

# Define custom metric functions
def custom_median(data):
    return np.median(data)

def custom_mean(data):
    return np.mean(data)

def custom_iqm(data):
    # Interquartile mean
    q1, q3 = np.percentile(data, [25, 75])
    iqm = np.mean([x for x in data if q1 <= x <= q3])
    return iqm

def custom_difficulty_progress(data, percentile=0.25):
    threshold = np.percentile(data, percentile * 100)
    progress = np.mean([x for x in data if x <= threshold])
    return progress

def plot_score_hist(score_matrix, bins=20, figsize=(28, 14),
                    fontsize='xx-large', N=6, extra_row=1,
                    names=None, algorithm_name="unknown", logs_folder="./logs"):
    num_tasks = score_matrix.shape[0]
    N1 = (num_tasks // N) + extra_row

    fig, ax = plt.subplots(nrows=N1, ncols=N, figsize=figsize)

    # Ensure ax is always 2D
    if num_tasks <= N:
        ax = np.expand_dims(ax, axis=0) if N1 == 1 else np.array(ax).reshape(1, N)
    elif N1 == 1:
        ax = np.array(ax).reshape(1, N)
    else:
        ax = np.array(ax).reshape(N1, N)

    for i in range(N):
        for j in range(N1):
            idx = j * N + i
            if idx < num_tasks:
                ax[j, i].set_title(names[idx], fontsize=fontsize)
                print(score_matrix[idx, :])
                sns.histplot(score_matrix[idx, :], bins=bins, ax=ax[j, i], kde=True)
            else:
                ax[j, i].axis('off')

            decorate_axis(ax[j, i], wrect=5, hrect=5, labelsize='xx-large')
            ax[j, i].xaxis.set_major_locator(plt.MaxNLocator(4))
            if idx % N == 0:
                ax[j, i].set_ylabel('Count', size=fontsize)
            else:
                ax[j, i].yaxis.label.set_visible(False)
            ax[j, i].grid(axis='y', alpha=0.1)

    fig.subplots_adjust(hspace=0.85, wspace=0.17)
    plt.savefig(os.path.join(logs_folder, f"histogram_{algorithm_name}.pdf"))
    plt.show()


def bootstrap_metric_ci(data, metric_func, num_resamples=10000, ci=95):
    data = np.array(data)
    resamples = np.random.choice(data, (num_resamples, len(data)), replace=True)
    metric_values = np.apply_along_axis(metric_func, 1, resamples)
    lower = np.percentile(metric_values, (100 - ci) / 2)
    upper = np.percentile(metric_values, 100 - (100 - ci) / 2)
    return lower, upper


def plot_indicators(scores_dict, label_name, reps=10000, logs_folder="./logs"):
    plt.rcParams.update({'font.size': 9})

    # Define aggregate function using the metrics
    metric_functions = {
        'Median': custom_median,
        'IQM': custom_iqm,
        'Mean': custom_mean,
        'Difficulty Progress (25%)': lambda x: custom_difficulty_progress(x, 0.25)
    }

    metric_names = list(metric_functions.keys())
    metric_colors = sns.color_palette('deep', n_colors=4)  # Use a color palette

    results = {algorithm: {metric: None for metric in metric_names} for algorithm in scores_dict}
    ci_results = {algorithm: {metric: {'lower': None, 'upper': None} for metric in metric_names} for algorithm in
                  scores_dict}

    for algorithm, scores in scores_dict.items():
        for metric_name, metric_func in metric_functions.items():
            metric_value = metric_func(scores)
            results[algorithm][metric_name] = metric_value
            ci_lower, ci_upper = bootstrap_metric_ci(scores, metric_func, num_resamples=reps, ci=95)
            ci_results[algorithm][metric_name]['lower'] = ci_lower
            ci_results[algorithm][metric_name]['upper'] = ci_upper

    # Sort algorithms by the Mean metric
    sorted_algorithms = sorted(scores_dict.keys(), key=lambda algo: results[algo]['Mean'], reverse=False)

    # Prepare data for plotting using sorted order
    algorithms = sorted_algorithms
    num_algorithms = len(algorithms)
    bar_width = 0.20
    index = np.arange(num_algorithms)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot bars with error bars for confidence intervals
    for i, metric_name in enumerate(metric_names):
        metric_means = [results[algorithm][metric_name] for algorithm in algorithms]
        metric_cis_low = [ci_results[algorithm][metric_name]['lower'] for algorithm in algorithms]
        metric_cis_high = [ci_results[algorithm][metric_name]['upper'] for algorithm in algorithms]

        # Ensure error bars are non-negative
        error_low = np.clip(np.array(metric_means) - np.array(metric_cis_low), 0, None)
        error_high = np.clip(np.array(metric_cis_high) - np.array(metric_means), 0, None)

        # Position of bars
        positions = index + i * bar_width - (len(metric_names) / 2) * bar_width + bar_width / 2

        # Plot bars with error bars for confidence intervals
        ax.bar(positions, metric_means, bar_width,
               yerr=[error_low, error_high],
               capsize=5, color=metric_colors[i], label=metric_name)

    # Set x-ticks and labels
    ax.set_xlabel(label_name, fontsize=24)
    ax.set_ylabel('Score Indicator and CI', fontsize=24)
    ax.set_title('Comparison of Performances Across LLMs', fontsize=24)
    ax.set_xticks(index)
    ax.set_xticklabels(algorithms, rotation=45, fontsize=24)
    yticks = np.arange(0, 0.7, 0.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{tick:.1f}' for tick in yticks], fontsize=24)

    ax.spines['bottom'].set_visible(False)  # Hide x-axis line
    ax.spines['top'].set_visible(False)  # Hide top line
    ax.spines['right'].set_visible(False)  # Hide right line
    ax.spines['left'].set_visible(True)  # Show y-axis line

    ax.legend(fontsize=24)

    # Save and show plot
    plt.tight_layout()
    plt.savefig(os.path.join(logs_folder, 'indicators.pdf'))
    plt.show()

def calculate_indicators(scores_dict, reps=10000, logs_folder="./logs"):
    plt.rcParams.update({'font.size': 9})

    # Define aggregate function using the metrics
    metric_functions = {
        'Median': custom_median,
        'IQM': custom_iqm,
        'Mean': custom_mean,
        'Difficulty Progress (25%)': lambda x: custom_difficulty_progress(x, 0.25)
    }

    metric_names = list(metric_functions.keys())

    results = {algorithm: {metric: None for metric in metric_names} for algorithm in scores_dict}
    ci_results = {algorithm: {metric: {'lower': None, 'upper': None} for metric in metric_names} for algorithm in
                  scores_dict}

    for algorithm, scores in scores_dict.items():
        for metric_name, metric_func in metric_functions.items():
            metric_value = metric_func(scores)
            results[algorithm][metric_name] = metric_value
            ci_lower, ci_upper = bootstrap_metric_ci(scores, metric_func, num_resamples=reps, ci=95)
            ci_results[algorithm][metric_name]['lower'] = ci_lower
            ci_results[algorithm][metric_name]['upper'] = ci_upper

def plot_performance_profile(scores_dict, reps=10000, logs_folder="./logs"):
    plt.rcParams.update({'font.size': 16})
    algorithms = list(scores_dict.keys())
    score_dict = {key: scores_dict[key] for key in algorithms}
    taus = np.linspace(0.0, 0.75, 101)

    score_distributions, score_distributions_cis = rly.create_performance_profile(
        score_dict, taus, reps=reps)

    fig, axes = plt.subplots(ncols=1, figsize=(12, 10))
    xticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    plot_utils.plot_performance_profiles(
        score_distributions, taus,
        performance_profile_cis=score_distributions_cis,
        colors=color_dict,
        xlabel=r'Normalized Score $(\tau)$',
        labelsize='x-large',
        xticks=xticks,
        ax=axes)

    desired_order = ['TRPO', 'PPO', 'RPPO', 'SAC', 'TD3', 'TQC', 'A2C', 'Random', 'SecureBERT', 'CySecBERT',
                     'SecRoBERTa', 'SecBERT', 'RoBERTa', 'GPT-2', 'DistilBERT', 'BERT']
    desired_order = [algo for algo in desired_order if algo in algorithms]

    fake_patches = [mpatches.Patch(color=color_dict[alg],
                                   alpha=0.75) for alg in desired_order]
    fig.legend(fake_patches, desired_order, loc='upper center',
                        fancybox=True, ncol=len(algorithms)/2,
                        fontsize='large')
    fig.subplots_adjust(top=0.9, wspace=0.1, hspace=0.05)
    plt.savefig(os.path.join(logs_folder,"performance_profile.pdf"))
    plt.show()


def plot_probability_improvement(scores_dict, algorithms, reps=10000, logs_folder="./logs"):
    # probability of improvement, one algorithm with respect to another
    plt.rcParams.update({'font.size': 16})
    possible_pairs = list(combinations(algorithms, 2))

    # just focus on a subset of pairs
    required_pairs = [("PPO", "TRPO"),("TRPO", "PPO"),("QRDQN", "DQN"),("DQN", "QRDQN"),("QRDQN","RecurrentPPO"),("RecurrentPPO","QRDQN"), ("A2C", "DQN"), ("DQN", "A2C"),
                      ("TRPO", "RecurrentPPO"), ("RecurrentPPO", "TRPO"), ("QRDQN", "A2C"), ("A2C", "QRDQN")]

    all_pairs = {}
    for pair in possible_pairs:
        if pair in required_pairs:
            pair_name = f'{pair[0]}_{pair[1]}'
            all_pairs[pair_name] = (
                scores_dict[pair[0]], scores_dict[pair[1]])

    probabilities, probability_cis = rly.get_interval_estimates(
        all_pairs, metrics.probability_of_improvement, reps=reps)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
    h = 0.6

    ax2 = ax.twinx()

    for i, (pair, p) in enumerate(probabilities.items()):
        (l, u), p = probability_cis[pair], p

        ax.barh(y=i, width=u - l, height=h,
                left=l, color="purple",
                alpha=0.75, label=pair[0])
        ax2.barh(y=i, width=u - l, height=h,
                 left=l, color="purple",
                 alpha=0.0, label=pair[1])
        ax.vlines(x=p, ymin=i - 7.5 * h / 16, ymax=i + (6 * h / 16),
                  color='k', alpha=0.85)

    ax.set_yticks(list(range(len(all_pairs))))
    ax2.set_yticks(range(len(all_pairs)))
    pairs = [x.split('_') for x in probabilities.keys()]
    ax2.set_yticklabels([pair[1] for pair in pairs], fontsize='large')
    ax.set_yticklabels([pair[0] for pair in pairs], fontsize='large')
    ax2.set_ylabel('Algorithm Y', fontweight='bold', rotation='horizontal', fontsize='x-large')
    ax.set_ylabel('Algorithm X', fontweight='bold', rotation='horizontal', fontsize='x-large')
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.yaxis.set_label_coords(0, 1.0)
    ax2.yaxis.set_label_coords(1.0, 1.0375)
    decorate_axis(ax, wrect=5)
    decorate_axis(ax2, wrect=5)

    ax.tick_params(axis='both', which='major', labelsize='x-large')
    ax2.tick_params(axis='both', which='major', labelsize='x-large')
    ax.set_xlabel('P(X > Y)', fontsize='xx-large')
    ax.grid(axis='x', alpha=0.2)
    plt.subplots_adjust(wspace=0.05)
    ax.spines['left'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(logs_folder,"probability_improvement.pdf"))
    plt.show()


def plot_ranks(scores_dict, algorithms, tasks, logs_folder="./logs"):
    plt.rcParams.update({'font.size': 16})

    all_ranks = get_rank_matrix(scores_dict, 10000, algorithms=algorithms)
    #print("All ranks: ", all_ranks)

    mean_ranks = np.mean(all_ranks, axis=0)

    index = 0
    keys_list = list(scores_dict.keys())

    for mean_rank in mean_ranks:
        print("Mean rank: ", mean_rank, "for algorithm: ", keys_list[index])
        index += 1

    keys = algorithms
    labels = list(range(1, len(keys) + 1))
    width = 1.0  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
    bottom = np.zeros_like(mean_ranks[0])

    for i, key in enumerate(keys):
        ranks = mean_ranks

        bars = ax.bar(labels, ranks[i], width, color=color_dict[key],
                      bottom=bottom, alpha=0.9)
        bottom += mean_ranks[i]

        # Add text annotations for each bar
        for bar in bars:
            height = bar.get_height()
            text_position = bar.get_y() + height / 2

            # Only show text if height is greater than 0.1
            if height > 0.1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    text_position,
                    f'{height:.2f}',  # Format the text to 2 decimal places
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=28,
                    #fontweight='bold'  # Make text bold
                )

        if i == 0:
            ax.set_ylabel('Probability Distribution', size=40)

    ax.set_xlabel('Position', size=40)
    ax.set_title('Mean Ranking across Experiments', size=40)
    ax.set_xticks(labels)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(labels, size=38)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', bottom=False, top=False,
                   left=False, right=False, labeltop=False,
                   labelbottom=True, labelleft=False, labelright=False)

    desired_order = ['TRPO', 'PPO', 'RPPO', 'SAC', 'TD3', 'TQC', 'A2C', 'Random','SecureBERT', 'CySecBERT', 'SecRoBERTa', 'SecBERT', 'RoBERTa', 'GPT-2', 'DistilBERT', 'BERT']
    desired_order = [algo for algo in desired_order if algo in algorithms]
    fake_patches = [mpatches.Patch(color=color_dict[m], alpha=0.75)
                    for m in desired_order]
    fig.legend(fake_patches, desired_order, loc='upper center',
               fancybox=True, ncols=4, fontsize=26)
    fig.subplots_adjust(top=0.8, wspace=0.1, hspace=0.05)
    plt.savefig(os.path.join(logs_folder, "ranks.pdf"))
    plt.show()


def difficulty_progress(scores, proportion=0.25):
    sorted_scores = sorted(scores.flatten())
    max_index = int(proportion * len(sorted_scores))
    return np.mean(sorted_scores[:max_index])

def plot_alternative_metrics(scores_dict, logs_folder="./logs"):
    # Compute Difficulty Progress/ SuperHuman Probability
    DP_25 = lambda scores: difficulty_progress(scores, 0.25)
    DP_50 = lambda scores: difficulty_progress(scores, 0.5)
    improvement_prob = lambda scores: np.mean(scores > 1)

    aggregate_func = lambda x: np.array([DP_25(x), DP_50(x), improvement_prob(x)])
    aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
        scores_dict, aggregate_func, reps=10000)

    algorithms = list(scores_dict.keys())
    metric_names = ['Difficulty Progress (25%)', 'Difficulty Progress (50%)',
                    'Superhuman Probability']
    plot_interval_estimates_shifted(
        aggregate_scores,
        aggregate_interval_estimates,
        algorithms=algorithms,
        metric_names=metric_names,
        xlabel='Human Normalized Score')
    plt.savefig(os.path.join(logs_folder,"alternative.pdf"))
    plt.show()
