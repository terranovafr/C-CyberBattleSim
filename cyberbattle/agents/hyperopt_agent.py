# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    hyperopt_agent.py
    Script to hyper-optimize a RL algorithm on a C-CyberBattleSim Environment for all goals, and sampling nlp extractors.
"""

import argparse
import sys
import os
import yaml
from datetime import datetime
import numpy as np
import optuna
import random
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.agents.train_agent import setup_train_via_args, train_rl_algorithm # noqa: E402
from cyberbattle.utils.train_utils import check_args # noqa: E402
from cyberbattle.utils.log_utils import setup_logging # noqa: E402
from cyberbattle.utils.file_utils import save_yaml # noqa: E402
script_dir = os.path.dirname(__file__)

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

goal_metrics = {
    "validation" : {
        "control": "validation/Relative owned percentage (mean)",
        "discovery": "validation/Relative discovered amount (mean)",
        "disruption": "validation/Relative disrupted percentage (mean)"
    },
    "train": {
        "control": "train/Relative owned nodes percentage",
        "discovery": "train/Relative discovered amount percentage",
        "disruption": "train/Relative disrupted nodes percentage"
    }
}

# suggest hyperparams based on type of range
def suggest_hyperparameters(trial, hyperparam_ranges):
    suggested_params = {}
    for param_name, param_config in hyperparam_ranges.items():
        if param_config['type'] == 'categorical':
            suggested_params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
        elif param_config['type'] == 'float':
            suggested_params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'],
                                                               log=param_config.get('log', False))
        elif param_config['type'] == 'int':
            suggested_params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
    return suggested_params

# objective function for optuna trials
def objective(trial, hyperparam_ranges, args, original_logs_folder, logger):
    # sample a set of hyperparams
    suggested_params = suggest_hyperparameters(trial, hyperparam_ranges)

    logger.info(f"Training {args.algorithm} with suggested params:")
    logger.info(suggested_params)

    overall_metric = 0
    # Loop all the goals and merge the results
    for goal in args.goals:
        # randomly extract the LLM to avoid too many iterations to consider them all for every goal
        random_nlp_extractor = random.choice(args.nlp_extractors)
        envs_folder = os.path.join(original_logs_folder, "envs", random_nlp_extractor)
        logs_folder = os.path.join(original_logs_folder,  args.algorithm.upper() + "_trial_" + str(trial.number) + "_" + goal + "_" + random_nlp_extractor)
        os.makedirs(logs_folder, exist_ok=True)
        save_yaml(suggested_params, logs_folder, "hyperparams.yaml")
        args.goal = goal
        args.nlp_extractor = random_nlp_extractor
        logger, logs_folder, _, config, train_ids, val_ids = setup_train_via_args(args, logs_folder, envs_folder=envs_folder)

        config['name'] = f"{config['algorithm']}_{config['goal']}_{config['nlp_extractor']}"
        for param_name, value in suggested_params.items():
            config['algorithm_hyperparams'][param_name] = value
        config['algorithm_hyperparams']['learning_rate_type'] = "constant"
        if args.holdout:
            config['metric_name'] = goal_metrics['validation'][goal]
        else:
            config['metric_name'] = goal_metrics['train'][goal]

        # invoke the training function
        runs_metric = train_rl_algorithm(config=config, envs_folder=envs_folder, logs_folder=logs_folder, logger=logger, metric_name=config['metric_name'], train_ids=train_ids, val_ids=val_ids)
        avg_metric = np.mean(runs_metric)

        # average AUC of the metric across all runs for the trial
        logger.info("Metric: ", config['metric_name'], ":", avg_metric)
        overall_metric += avg_metric
    logger.info("Overall metric", config['metric_name'], "averaged across goals:", overall_metric / 3)
    return overall_metric / 3 # average across goals

# Function hyperopt_rl to perform hyperparameter optimization using Optuna
def hyperopt_rl(hyperparam_ranges, args, logs_folder, logger):
    if args.optimization_type in map_name_to_sampler:
        sampler = map_name_to_sampler[args.optimization_type]()
    else:
        raise ValueError("Unsupported optimization type specified:", args.optimization_type)

    # Create study or overwrite if it exists
    study = optuna.create_study(direction=args.direction, sampler=sampler, storage = f"sqlite:///{os.path.join(logs_folder, args.name + '.db')}", study_name=args.name, load_if_exists=True)  # type: ignore
    study.optimize(lambda trial: objective(trial, hyperparam_ranges, args, logs_folder, logger), n_trials=args.num_trials)

    logger.info(f"Best parameters found: {study.best_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-optimize RL Agent for C-CyberBattleSim environments")
    # same code of train_agent but with hyperopt logic
    parser.add_argument('--algorithm', type=str,
                        choices=['ppo', 'a2c', 'rppo', 'trpo', 'ddpg', 'sac', 'td3', 'tqc'], default='trpo',
                        help='RL algorithm to train')
    parser.add_argument('-nlp', '--nlp_extractors', type=str,
                        choices=["bert", "distilbert", "roberta", "gpt2", "CySecBERT", "SecureBERT", "SecBERT",
                                 "SecRoBERTa"],
                        default=["bert", "distilbert", "roberta", "gpt2", "CySecBERT", "SecureBERT", "SecBERT",
                                 "SecRoBERTa"], nargs='+',
                        help='NLP extractors to be used for extracting vulnerability embeddings')
    parser.add_argument('-g', '--goals', type=str, choices=['control', 'discovery', 'disruption'],
                        default=['control', 'discovery', 'disruption'], nargs='+',
                        help='Goals to be used for sampling the agent')
    parser.add_argument('--environment_type', type=str, choices=['continuous', 'local', 'global'], default='continuous',
                        help='Type of environment to be used for training')  # to be extended in the future to LOCAL or DISCRETE or others
    parser.add_argument('--static_seeds', action='store_true', default=False, help='Use a static seed for training')
    parser.add_argument('--load_seeds', default="config",
                        help='Path of the folder where the seeds.yaml should be loaded from (e.g. previous experiment)')
    parser.add_argument('--random_seeds', action='store_true', default=False, help='Use random seeds for training')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--holdout', action='store_true', default=False,
                        help='Use validation and test sets and switch among environments periodically on the training and validation sets')
    parser.add_argument('--finetune_model', type=str,
                        help='Path to the model to eventually finetune (relative to the logs folder)')
    parser.add_argument('--early_stopping', type=int, default=0,
                        help='Early stopping on the validation environments setting the number of patience runs')
    parser.add_argument('--name', default=False, help='Name of the logs folder related to the run')
    parser.add_argument('--load_envs', default=False,
                        help='Path of the run folder where the networks should be loaded from')
    parser.add_argument('--load_processed_envs', default=False,
                        help='Path of the run folder where the envs already processed should be loaded from')
    parser.add_argument('--static_defender_agent', default=None, choices=['reimage', 'events', None],
                        help='Defender agent to use')
    parser.add_argument('-pca', '--pca_components', default=None, type=int,
                        help='Invoke with the use of PCA for the feature vectors specifying the number of components')
    parser.add_argument('-v', '--verbose', type=int, default=2, help='Verbose level: 0 - no output, 1 - training/validation information, 2 - episode level information, 3 - iteration level information')
    parser.add_argument('--no_save_log_file', action='store_false', dest='save_log_file',
                        default=True, help='Disable logging to file; log only to terminal')
    parser.add_argument('--save_csv_file', default=0,
                        help='Flag to decide whether trajectories should be saved periodically to a csv file during training (the value determines the interval in episodes)')
    parser.add_argument('--save_embeddings_csv_file', action='store_true', default=False,
                        help='Flag to decide whether also embeddings should be saved periodically to a csv file during training')
    # hyperopt specific arguments
    parser.add_argument('--optimization_type', type=str, choices=['grid', 'random', 'tpe', 'cmaes', 'GPS', 'partial_fixed', 'nsga2', 'nsga3', 'qmc', 'bruteforce'], default='tpe',
                        help='Type of hyperparameter optimization to use')
    parser.add_argument('--hyperparams_ranges_file', type=str,
                        default="hyperparams_ranges.yaml",
                        help='Path to YAML file specifying hyperparameter ranges')
    parser.add_argument('--metric_name', type=str, default="validation/Episode reward (mean)",
                        help='Name of the metric to hyper-optimize')
    parser.add_argument('--direction', type=str, default="maximize", choices=['maximize', 'minimize'],
                        help='Direction of the optimization (maximize or minimize)')
    parser.add_argument('--num_trials', type=int, default=25, help='Number of trials for hyperparameter optimization')
    # Configuration files
    parser.add_argument('--train_config', type=str, default='config/train_config.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--rewards_config', type=str, default='config/rewards_config.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--algo_config', type=str, default='config/algo_config.yaml',
                        help='Path to the configuration YAML file')
    args = parser.parse_args()

    check_args(args)

    # read hyperparams ranges
    with open(os.path.join(script_dir, "config", args.hyperparams_ranges_file), 'r') as file:
        hyperparams_ranges = yaml.safe_load(file)

    # read only those of the target algorithms
    hyperparams_ranges = hyperparams_ranges.get(args.algorithm, {})

    if args.name:
        logs_folder = os.path.join(script_dir, 'logs',
                                   "hyperopt_" + args.name + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        logs_folder = os.path.join(script_dir, 'logs', "hyperopt_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    logger = setup_logging(logs_folder, log_to_file=args.save_log_file)

    if not args.name:
        args.name = "hyperopt_" + args.algorithm

    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    save_yaml(hyperparams_ranges,  logs_folder, "hyperparams_ranges.yaml")
    hyperopt_rl(hyperparams_ranges, args, logs_folder, logger)
