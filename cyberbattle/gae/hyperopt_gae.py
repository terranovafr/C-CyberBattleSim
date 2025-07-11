# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    hyperopt_gae.py
    This file contains the module used to hyper-optimize the Graph Autoencoder (GAE) model.
    The file relies on the train_gae.py script to train the model with the hyperparameters sampled by Optuna.
"""

import argparse
import copy
from datetime import datetime
import os
import numpy as np
import random
import optuna
import sys
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.gae.train_gae import main as train_main # noqa: E402
from cyberbattle.utils.file_utils import load_yaml # noqa: E402
from cyberbattle.utils.log_utils import setup_logging # noqa: E402
script_dir = Path(__file__).parent

map_name_to_sampler = {
    'grid': optuna.samplers.GridSampler,
    'random': optuna.samplers.RandomSampler,
    'tpe': optuna.samplers.TPESampler,
    'cmaes': optuna.samplers.CmaEsSampler,
    'GPS': optuna.samplers.GPSampler,
    'partial_fixed': optuna.samplers.PartialFixedSampler,
    'nsga2': optuna.samplers.NSGAIISampler,
    'nsga3': optuna.samplers.NSGAIIISampler,
    'qmc': optuna.samplers.QMCSampler,
    'bruteforce': optuna.samplers.BruteForceSampler
}

# Sample the characteristics of a layer
def sample_layer(trial, layer_template, is_first_layer=False, is_last_layer=False):
    layer = {}
    for k, v in layer_template.items():
        if isinstance(v, dict):
            if v['type'] == 'categorical':
                if k == 'type' and is_first_layer:
                    layer[k] = 'NNConv'  # Force the first layer to be NNConv in order to include edge feature vectors
                elif k == 'activation' and is_last_layer: # Force the last layer to have a linear activation in order to ensure the range is normal
                    layer[k] = 'null'
                else:
                    layer[k] = trial.suggest_categorical(k, v['values'])
            elif v['type'] == 'int':
                layer[k] = trial.suggest_int(k, v['min'], v['max'])
        else:
            layer[k] = v  # For non-dict values, directly assign
    return layer

def sample_hyperparameters(trial, trial_config, hyperparams_ranges):
    # Sample hyperparameters
    for k, v in hyperparams_ranges.items():
        if k == 'model_config':
            layers = []
            num_layers = trial.suggest_int("num_layers", hyperparams_ranges.get('num_layers', {}).get('min', 1),
                                           hyperparams_ranges.get('num_layers', {}).get('max', 4))
            for i in range(num_layers):
                if i == 0:
                    layer_template = v['layer_template'][
                        0]  # First layer must be NNConv otherwise the edge feature vectors will not be included
                else:
                    layer_template = random.choice(v['layer_template'][1:])  # Choose from flexible layer templates

                is_first_layer = (i == 0)
                is_last_layer = (i == num_layers - 1)
                layers.append(
                    sample_layer(trial, layer_template, is_first_layer=is_first_layer, is_last_layer=is_last_layer))

            trial_config['model_config']['layers'] = layers
        else:
            if v['type'] == 'categorical':
                trial_config[k] = trial.suggest_categorical(k, v['values'])
            elif v['type'] == 'uniform':
                trial_config[k] = trial.suggest_uniform(k, v['low'], v['high'])
            elif v['type'] == 'int':
                trial_config[k] = trial.suggest_int(k, v['min'], v['max'])
    return trial_config

# Objective function for hyperparameter optimization
def objective(trial, logs_folder, envs_folder, config, hyperparams_ranges, logger):
    trial_config = copy.deepcopy(config)
    trial_config = sample_hyperparameters(trial, trial_config, hyperparams_ranges)
    if config['verbose']:
        logger.info("Trial configuration:")
        indices_keys = ['continuous_indices', 'binary_indices', 'multi_class_info', 'node_feature_vector_size',
                        'edge_feature_vector_size', 'train_config', 'hyperparams_ranges_file']
        trial_config_copy = {key: trial_config[key] for key in trial_config if key not in indices_keys}
        logger.info(trial_config_copy)
    # Creating unique run folder
    trial_folder = os.path.join(logs_folder, f"trial_{str(trial.number + 1)}")
    if config['verbose']:
        logger.info(f"Trial folder: {os.path.basename(trial_folder.rstrip('/'))}")
    os.makedirs(trial_folder, exist_ok=True)
    # Save the modified configuration for reproducibility
    return train_main(config, trial_folder, envs_folder, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a GNN Autoencoder')
    parser.add_argument('--name', default=False, help='Name of the logs folder related to the run')
    parser.add_argument('--static_seeds', action='store_true', default=False, help='Use a static seed for training')
    parser.add_argument('--load_seeds', default="config",
                        help='Path of the folder where the seeds.yaml should be loaded from (e.g. previous experiment)')
    parser.add_argument('--random_seeds', action='store_true', default=False, help='Use random seeds for training')
    parser.add_argument('--yaml', default=False,
                        help='Read configuration file from YAML file of a previous training')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs to perform')
    parser.add_argument('--holdout', default=False, action="store_true", help='Switch between graphs')
    parser.add_argument('--load_envs', default=False, type=str, help='Path to the .pkl file containing the graph')
    parser.add_argument('--no_save_log_file', action='store_false', dest='save_log_file',
                        default=True, help='Disable logging to file; log only to terminal')
    parser.add_argument('-v', '--verbose', type=int, default=1, help='Verbose level: 0, 1, 2, 3')
    parser.add_argument('-nlp', '--nlp_extractors', default=['distilbert', 'roberta', 'CySecBERT', 'SecBERT', 'SecRoBERTa', 'SecureBERT', 'gpt2', 'bert'], nargs='+', type=str,
                        help='Name of the NLP extractor(s) to be used for generating embeddings to compress. If multiple are specified, they will be used separately and results will be merged.')
    parser.add_argument('-pca', '--pca_components', default=None, type=int, help='Invoke with the use of PCA for the feature vectors')
    # Configuration files
    parser.add_argument('--train_config', type=str, default=os.path.join('config', 'train_config.yaml'),
                        help='Path to the configuration YAML file')
    # Hyper-parameters optimization file
    parser.add_argument('--num_trials', type=int, default=25, help='Number of runs to perform')
    parser.add_argument('--hyperparams_ranges_file', type=str,
                        default=os.path.join('config', 'hyperparams_ranges.yaml'),
                        help='Path to the hyperopt configuration YAML file')
    parser.add_argument('--optimization_type', type=str,
                        choices=['grid', 'random', 'tpe', 'cmaes', 'GPS', 'partial_fixed', 'nsga2', 'nsga3', 'qmc',
                                 'bruteforce'], default='tpe',
                        help='Type of hyperparameter optimization to use')
    args = parser.parse_args()

    general_config = load_yaml(os.path.join(script_dir, "..", "..", "config.yaml"))
    if not args.load_envs:
        args.load_envs = general_config['default_environments_path']

    args.train_config = os.path.join(script_dir, args.train_config)
    args.hyperparams_ranges_file = os.path.join(script_dir, args.hyperparams_ranges_file)

    # Creating logs folder
    if args.name:
        logs_folder = os.path.join(script_dir, 'logs/',
                                   "hyperopt_" + args.name + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        logs_folder = os.path.join(script_dir, 'logs/', "hyperopt_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(logs_folder, exist_ok=True)  # Ensure logs folder exists
    envs_folder = os.path.join(logs_folder, 'envs')

    logger = setup_logging(logs_folder, log_to_file=args.save_log_file)

    # Potential seed setting before the generation of environment
    if args.static_seeds:
        if args.verbose:
            logger.info("Setting static seeds for each run of training")
        seeds_runs = [42 for _ in range(args.num_runs)]
    elif args.random_seeds:
        if args.verbose:
            logger.info("Setting random seeds for training")
        seeds_runs = [np.random.randint(1000) for _ in range(args.num_runs)]
    else: #  args.load_seeds:
        if args.verbose:
            logger.info("Loading seeds from seeds file %s", args.load_seeds)
        seeds_loaded = load_yaml(os.path.join(script_dir, args.load_seeds, 'seeds.yaml'))
        seeds_runs = seeds_loaded['seeds'][:args.num_runs]

    if args.yaml:
        logger.info("Reading YAML file from a previous training for reproducibility...")
        # Read eventually YAML configuration file of a previous training
        config = load_yaml(os.path.join(script_dir, "logs", args.yaml))
        config.update({"yaml": args.yaml})
        config.update({"verbose": args.verbose})
        config.update({"save_log_file": args.save_log_file})
    else:
        # Read default YAML configuration files and merge
        config = load_yaml(args.train_config)
        # not updated if loaded configuration file from previous experiment
        config.update({"seeds": seeds_runs})
        config.update(vars(args))
        if not args.pca_components:
            config['pca_components'] = config['default_vulnerability_embeddings_size']
        else:
            config['pca_components'] = args.pca_components

    # Load hyperopt ranges
    hyperparams_ranges = load_yaml(args.hyperparams_ranges_file)

    if args.optimization_type in map_name_to_sampler:
        sampler = map_name_to_sampler[args.optimization_type]()
    else:
        raise ValueError("Unsupported optimization type specified:", args.optimization_type)

    study = optuna.create_study(direction='minimize', sampler=sampler, study_name='gae_hyperopt', storage=f"sqlite:///{os.path.abspath(logs_folder)}/gae_hyperopt.db", load_if_exists=True)  # type: ignore
    study.optimize(
        lambda trial: objective(trial, logs_folder=logs_folder, envs_folder=envs_folder,
                                hyperparams_ranges=hyperparams_ranges, config=config, logger=logger),
        n_trials=config['num_trials']
    )
    if config['verbose']:
        logger.info(f"Best hyperparameters: {study.best_params}")
