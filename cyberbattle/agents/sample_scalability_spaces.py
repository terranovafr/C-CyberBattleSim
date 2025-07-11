# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.


"""
    sample_scalability_spaces.py
    Script to sample agents for each obs/action space type and assess scalability.
"""

import argparse
import sys
import os
from datetime import datetime
import shutil
import re
import random
import numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.agents.train_agent import setup_train_via_args, train_rl_algorithm  # noqa: E402
script_dir = os.path.dirname(__file__)

# goals and nlp extractors to consider for the sampling
goal_metrics = {  # metrics to use for each goal
    "control": "train/Relative owned nodes percentage",
    "discovery": "train/Relative discovered amount percentage",
    "disruption": "train/Relative disrupted nodes percentage"
}

algorithm_hyperparameters = {
    # all found with hyper-parameter tuning for each space
    "trpo": {
        "local": {
            "batch_size": {"type": "categorical", "values": [64]},
            "gamma": {"type": "categorical", "values": [0.9]},
            "n_steps": {"type": "categorical", "values": [2048]},
            "target_kl": {"type": "categorical", "values": [0.05]},
            "gae_lambda": {"type": "categorical", "values": [0.95]},
            "learning_rate": {"type": "categorical", "values": [0.01]},
        },
        "global": {
            "batch_size": {"type": "categorical", "values": [64]},
            "gamma": {"type": "categorical", "values": [0.9]},
            "n_steps": {"type": "categorical", "values": [2048]},
            "target_kl": {"type": "categorical", "values": [0.05]},
            "gae_lambda": {"type": "categorical", "values": [0.95]},
            "learning_rate": {"type": "categorical", "values": [0.001]},
        },
        "continuous": { # default in files
        }
    }
}

def suggest_hyperparameters(algorithm, environment_type, hyperparam_ranges):
    suggested_params = {}
    hyperparam_ranges = hyperparam_ranges[algorithm][environment_type]
    for hyperparam in hyperparam_ranges:
        if hyperparam_ranges[hyperparam]["type"] == "categorical":
            suggested_params[hyperparam] = random.choice(hyperparam_ranges[hyperparam]["values"])
        elif hyperparam_ranges[hyperparam]["type"] == "uniform":
            suggested_params[hyperparam] = np.random.uniform(hyperparam_ranges[hyperparam]["low"], hyperparam_ranges[hyperparam]["high"])
        elif hyperparam_ranges[hyperparam]["type"] == "loguniform":
            suggested_params[hyperparam] = np.exp(np.random.uniform(np.log(hyperparam_ranges[hyperparam]["low"]), np.log(hyperparam_ranges[hyperparam]["high"])))
    return suggested_params

# sample all combinations of (nlp_extractor, goals) for the algorithm
def sample_games_rl(args):
    trial_id = 0
    topology_set = []
    for topology in os.listdir(os.path.join(script_dir, "..", "data", "env_samples", "scalability_study")):
        if topology.startswith("graphs_nodes="):
            already_done = False
            if not os.path.exists(os.path.join(script_dir, "logs")):
                os.makedirs(os.path.join(script_dir, "logs"))
            for folder in os.listdir(os.path.join(script_dir, "logs")):
                if folder.startswith("TRPO_"+args.environment_type+"_graphs_nodes="+topology.split('_nodes=')[1].split("_")[0]+"_vulns="+topology.split('_vulns=')[1].split("_")[0]):
                    already_done = True
                    break
            if not already_done:
                topology_set.append(topology)

    # Function to extract the number of nodes and vulnerabilities
    def extract_complexity(topology):
        parts = topology.split('_')
        num_nodes = int(parts[1].split('=')[1])
        num_vulnerabilities = int(parts[2].split('=')[1].split("_")[0])
        return num_nodes + num_vulnerabilities

    # Sort the topologies by complexity
    topology_set.sort(key=extract_complexity)

    NUM_EXPERIMENTS = 1
    i = 0
    original_load_envs = args.load_envs
    for topology in topology_set:
        args.load_envs = os.path.join(original_load_envs, topology)
        for _ in range(NUM_EXPERIMENTS):
            args.algorithm = random.choice(list(algorithm_hyperparameters.keys())) # support also the possibility for many RL algorithms
            trial_id += 1
            args.goal = args.goals[i % 3]
            args.nlp_extractor = random.choice(args.nlp_extractors) # randomize NLP extractor each time
            args.name = args.algorithm.upper()+ "_" + args.environment_type + "_" + topology + "_" + args.goal + "_" + args.nlp_extractor
            logs_folder = os.path.join(script_dir, 'logs', args.name + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            if not os.path.exists(logs_folder):
                os.makedirs(logs_folder)
            # Finetuning the right model from a folder of sampling experiment (loading the analogous experiment)
            if args.finetune_folder:
                args.finetune_folder = os.path.join(script_dir, 'logs', args.finetune_folder, args.algorithm.upper() + "_" + str(trial_id) + "_" + args.goal + "_" + args.nlp_extractor)
                checkpoint_files = [file for file in os.listdir(os.path.join(args.finetune_folder, "validation", str(1))) if file.startswith("checkpoint_")]
                episode_pattern = re.compile(r'checkpoint_(-?\d+)')
                checkpoint_files.sort(key=lambda x: int(episode_pattern.search(x).group(1)))
                args.finetune_model = os.path.join(args.finetune_folder, "validation", str(1), checkpoint_files[-1])
            else:
                args.finetune_model = None
            metric_name = goal_metrics[args.goal]
            logs_folder = os.path.join(logs_folder, args.algorithm.upper() + "_" + str(trial_id) + "_" + args.goal + "_" + args.nlp_extractor)
            os.makedirs(logs_folder, exist_ok=True)
            logger, logs_folder, envs_folder, config, train_ids, val_ids = setup_train_via_args(args, logs_folder)
            suggested_params = suggest_hyperparameters(args.algorithm, args.environment_type, algorithm_hyperparameters)
            for param_name, value in suggested_params.items():
                config['algorithm_hyperparams'][param_name] = value
            if args.verbose:
                logger.info(f"Performing sample iteration {trial_id} with goal {config['goal']} and nlp_extractor {config['nlp_extractor']}")
            train_rl_algorithm(logs_folder, envs_folder, config, train_ids, val_ids, logger=logger, metric_name=metric_name, verbose=args.verbose)
            shutil.rmtree(envs_folder)
        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample different trials with NLP extractor and goals for an RL Agent in C-CyberBattleSim environments")
    # same code of train_agent but without nlp extractor and goal
    parser.add_argument('--environment_type', type=str, choices=['continuous', 'local', 'global'], default='continuous',
                        help='Type of environment to be used for training')  # to be extended in the future to LOCAL or DISCRETE or others
    parser.add_argument('--static_seeds', action='store_true', default=False, help='Use a static seed for training')
    parser.add_argument('--load_seeds', default="config",
                        help='Path of the folder where the seeds.yaml should be loaded from (e.g. previous experiment)')
    parser.add_argument('--random_seeds', action='store_true', default=False, help='Use random seeds for training')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--holdout', action='store_true', default=False,
                        help='Use validation and test sets and switch among environments periodically on the training and validation sets')
    parser.add_argument('-g', '--goals', type=str, default=['control', 'discovery', 'disruption'], nargs='+',
                        help='Goals to be used for the training (default: all three goals)')
    parser.add_argument('-nlp', '--nlp_extractors', type=str, default=['bert', 'distilbert', 'roberta', 'gpt2', 'CySecBERT', 'SecureBERT', 'SecBERT', 'SecRoBERTa'], nargs='+',
                        help='NLP extractors to be used for the sampling (default: all available)')
    parser.add_argument('--finetune_folder', type=str,
                        help='Path to the folder with the models to eventually finetune of a previous sampling experiment (relative to the logs folder)')
    parser.add_argument('--early_stopping', type=int, default=0,
                        help='Early stopping on the validation environments setting the number of patience runs')
    parser.add_argument('--load_envs', type=str, default='scalability_study',
                        help='Path to the folder with the environments to be loaded (relative to the logs folder)')
    parser.add_argument('--name', default=False, help='Name of the logs folder related to the run')
    parser.add_argument('--static_defender_agent', default=None, choices=['reimage', 'events', None],
                        help='Defender agent to use')
    parser.add_argument('-pca', '--pca_components', default=None, type=int,
                        help='Invoke with the use of PCA for the feature vectors specifying the number of components')
    parser.add_argument('-v', '--verbose', type=int, default=2, help='Verbose level: 0 - no output, 1 - training/validation information, 2 - episode level information, 3 - iteration level information')
    parser.add_argument('--save_log_file', action='store_true', default=False,
                        help='Log to file instead of terminal output')
    parser.add_argument('--save_csv_file', default=0,
                        help='Flag to decide whether trajectories should be saved periodically to a csv file during training (the value determines the interval in episodes)')
    parser.add_argument('--save_embeddings_csv_file', action='store_true', default=False,
                        help='Flag to decide whether also embeddings should be saved periodically to a csv file during training')
    parser.add_argument('--train_config', type=str, default='config/train_config.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--rewards_config', type=str, default='config/rewards_config.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--algo_config', type=str, default='config/algo_config.yaml',
                        help='Path to the configuration YAML file')
    args = parser.parse_args()

    # Environments are enforced based on names of the logs
    args.load_processed_envs = False
    sample_games_rl(args)
