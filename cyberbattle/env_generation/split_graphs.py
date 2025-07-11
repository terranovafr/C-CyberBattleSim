# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    split_graphs.py
    Script to split environments based on complexity criteria
"""

from typing import List, Dict, Tuple
import argparse
import os
from pathlib import Path
import sys
import csv
from math import floor
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.utils.file_utils import load_yaml, save_yaml # noqa: E402
from cyberbattle.utils.log_utils import setup_logging # noqa: E402
script_dir = Path(__file__).parent

# Normalizing features in [0,1] range ensures that all features are treated equally when calculating complexity
def normalize_features(data, features: List[str]) -> List[Dict]:
    min_max = {feature: {'min': float('inf'), 'max': float('-inf')} for feature in features}
    index = 0
    for env in data:
        index += 1
        for feature in features:
            if feature in env:
                value = float(env[feature])
                if value < min_max[feature]['min']:
                    min_max[feature]['min'] = value
                if value > min_max[feature]['max']:
                    min_max[feature]['max'] = value

    for env in data:
        for feature in features:
            if feature in env:
                min_val = min_max[feature]['min']
                max_val = min_max[feature]['max']
                value = float(env[feature])
                normalized_value = (value - min_val) / (max_val - min_val) if max_val != min_val else 0
                env[f'normalized_{feature}'] = normalized_value

    return data

# complexity score computed as sum of normalized features summed or subtracted based on if ascending or descending (weights can be introduced)
def calculate_complexity(env: Dict, criteria: Dict) -> float:
    complexity_score = 0.0
    for feature, direction in criteria.items():
        if feature in env:
            value = env[f'normalized_{feature}']
            if direction == 'maximize':
                complexity_score += value
            elif direction == 'minimize':
                complexity_score += 1 - value
    return complexity_score

def split_envs(data, train_pct: float, val_pct: float, score: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    sorted_data = sorted(data, key=lambda x: x[score])
    total = len(sorted_data)
    train_size = floor(total * train_pct)
    val_size = floor(total * val_pct)
    if total % 2 == 1:
        # If total number of environments is odd, assign one more to the training set
        train_size += 1

    train_data = [{"id": int(env['environment_ID']), "score": env[score]} for env in sorted_data[:train_size]]
    val_data = [{"id": int(env['environment_ID']), "score": env[score]} for env in sorted_data[train_size:train_size + val_size]]
    test_data = [{"id": int(env['environment_ID']), "score": env[score]} for env in sorted_data[train_size + val_size:]]

    return train_data, val_data, test_data

# Split the environments based on complexity criteria (most complex in test set, most complex among the remaining in validation set, rest in training set)
def split_environments(stats_file: str, criteria_file: str, train_pct: float, val_pct: float):
    with open(stats_file, mode='r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    criteria = load_yaml(criteria_file)
    features = criteria.keys()
    normalized_data = normalize_features(data, features)

    for env in normalized_data:
        env['complexity_score'] = calculate_complexity(env, criteria)
    train_data, val_data, test_data = split_envs(normalized_data, train_pct, val_pct, score='complexity_score')

    output_data = {
        'training_set': train_data,
        'validation_set': val_data,
        'test_set': test_data,
        'params': {
            'train_pct': train_pct,
            'val_pct': val_pct,
            'test_pct': 1 - train_pct - val_pct
        }
    }
    save_yaml(output_data, os.path.dirname(stats_file), 'split.yaml')

# Eventually possible to use it as external script if split was not performed during generation or in case of re-split
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split environments based on complexity criteria')
    parser.add_argument('-f', '--folder', type=str, help='Path to the folder with the CSV file containing environment data')
    parser.add_argument('--complexity_criteria_file', type=str,
                        default=os.path.join("config", "complexity_criteria.yaml"),
                        help='Path to the YAML file containing complexity criteria')
    parser.add_argument('-t', '--split_train', type=float, default=0.6, help='Percentage of data to use for training')
    parser.add_argument('-v', '--split_val', type=float, default=0.2, help='Percentage of data to use for validation')
    parser.add_argument('--no_save_log_file', action='store_false', dest='save_log_file',
                        default=True, help='Disable logging to file; log only to terminal')

    # test size will be implicit
    args = parser.parse_args()
    if args.split_train + args.split_val > 1:
        raise ValueError("The sum of training and validation percentages must be less than 1.")

    logger = setup_logging(os.path.join(script_dir, "..", "data", "env_samples", args.folder, "split_log"), log_to_file=args.save_log_file)

    logger.info("Splitting environments of folder %s based on complexity criteria %s: train %.2f, val %.2f, test %.2f", args.folder, args.complexity_criteria_file, args.split_train, args.split_val, 1 - args.split_train - args.split_val)
    stats_file = os.path.join(script_dir, "..", "data", "env_samples", args.folder, "graphs_stats.csv")
    split_environments(stats_file, os.path.join(script_dir, args.complexity_criteria_file), args.split_train, args.split_val)
