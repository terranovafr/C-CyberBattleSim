# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    train_utils.py
    This file contains the supporting functions used during training.
"""

import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, TD3, SAC, DDPG, DQN
from stable_baselines3.a2c import A2C
from sb3_contrib import RecurrentPPO, TRPO, TQC

# Mapping algorithm type to model class and additional parameters
algorithm_models = {
    'ppo': PPO,
    'a2c': A2C,
    'rppo': RecurrentPPO,
    'trpo': TRPO,
    'ddpg': DDPG,
    'sac': SAC,
    'td3': TD3,
    'tqc': TQC,
    'dqn': DQN
}

reccurrent_algorithms = ["rppo"]

# Function to replace the strings with the actual classes
def replace_with_classes(policy_kwargs): # done after saving these elements in the config file as strings
    activation_functions = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid,
        "ELU": nn.ELU
    }
    if 'activation_fn' in policy_kwargs:
        policy_kwargs['activation_fn'] = activation_functions[policy_kwargs['activation_fn']]
    # Handle optimizers
    optimizers = {
        "Adam": optim.Adam,
        "SGD": optim.SGD
    }
    if 'optimizer_class' in policy_kwargs:
        policy_kwargs['optimizer_class'] = optimizers[policy_kwargs['optimizer_class']]

    return policy_kwargs

# Check arguments and returns error and why in case args are not properly set
def check_args(args):
    if args.load_envs and args.load_processed_envs:
        return True, "You cannot specify both the path to the environments to load and the path to the processed environments. Please choose one."
    if args.save_embeddings_csv_file and not args.save_csv_file:
        return True, "You must set the saving to CSV file with interval to save the embeddings to if you want to save the embeddings."
    return False, "OK"

# Function to clean the config file before saving it
def clean_config_save(config):
    if not config['static_defender_agent'] == "reimage":
        config.pop('detect_probability_min', None)
        config.pop('detect_probability_max', None)
        config.pop('scan_capacity_min', None)
        config.pop('scan_capacity_max', None)
        config.pop('scan_frequency_min', None)
        config.pop('scan_frequency_max', None)
    if not config['static_defender_agent'] == "events":
        config.pop('random_event_probability', None)
        config.pop('random_event_probability_min', None)
        config.pop('random_event_probability_max', None)
    config.pop('graph_encoder_models_path', None)
    config.pop('train_config', None)
    config.pop('rewards_config', None)
    config.pop('algo_config', None)
    return config
