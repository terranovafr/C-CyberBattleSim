# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    math_utils.py
    This file contains the utilities related to math modules and functions.
"""

import torch
import numpy as np
import random
from typing import Callable

# Linear schedule for decay of a metric (e.g. learning rate) from an initial value to a final value
def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return initial_value + (final_value - initial_value) * (1.0 - progress_remaining)
    return func

# Calculate the area under the curve using the trapezoidal rule
def calculate_auc(times, values):
    auc = np.trapz(values, times)
    return auc

# Calculate the area under the curve using the trapezoidal rule normalizing the metric values by subtracting the starting value -> net improvement
def calculate_normalized_auc(times, values):
    times = np.array(times)
    values = np.array(values)
    starting_value = values[0]
    normalized_values = values - starting_value
    area_normalized = np.trapz(normalized_values, times)
    net_improvement = values[-1] - starting_value
    return area_normalized, net_improvement

# Generate a number of convex combinations using Dirichlet distribution satisfying a minimum value for each element
def generate_convex_combinations(num_samples, num_percentages, min_value=0.05):
    list_samples = []
    while len(list_samples) < num_samples:
        sample = np.random.dirichlet(np.ones(num_percentages))
        # Check if all elements have at least minimum while generating random convex combinations
        if all(p >= min_value for p in sample):
            sample_list = [ float(sample[i]) for i in range(len(sample))]
            list_samples.append(sample_list)
    return list_samples

# Calculate bootstrap confidence intervals based on confidence level and number of iterations
def bootstrap_ci(data, confidence=0.95, n_iterations=10000):
    rng = np.random.default_rng()
    bootstrap_means = np.array([np.mean(rng.choice(data, replace=True, size=len(data))) for _ in range(n_iterations)])
    lower_bound = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
    original_mean = np.mean(data)
    return original_mean, lower_bound, upper_bound

# Dynamically scaling coefficients for the loss function: scaling factor as the inverse of the standard deviation
def compute_dynamic_scale(data):
    scale = 1 / (1.5*data.std() + 1e-6)  # Adding a small epsilon to avoid division by zero
    return scale

# Diversity loss ensures that the model does not collapse to giving the same point to all nodes
# Application: GAE loss into the encoder ensures as soft constraint that the embeddings are diverse and not collapsed to a single point
def diversity_loss(features, scale=1.0):
    # Calculate pairwise distances
    pairwise_distance = torch.cdist(features, features, p=2)  # p=2 for Euclidean distance
    # Apply an exponential function to emphasize larger distances
    exp_distances = torch.exp(-pairwise_distance / scale)
    # Sum the elements of the exponential distances matrix
    loss = torch.mean(exp_distances)
    return loss

# Set seeds for reproducibility
def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
