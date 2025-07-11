# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    sample_agent.py
    Script to train a RL algorithm on a C-CyberBattleSim Environment for all goals and nlp extractors.
"""

import argparse
import sys
import os
from datetime import datetime
import shutil
import re
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.agents.train_agent import setup_train_via_args, train_rl_algorithm # noqa: E402
from cyberbattle.utils.train_utils import check_args # noqa: E402
from cyberbattle.utils.file_utils import load_yaml # noqa: E402
script_dir = os.path.dirname(__file__)

# sample all combinations of (nlp_extractor, goals) for the algorithm
def sample_games_rl(args, logs_folder):
    trial_id = 0
    nlp_extractor_list = args.nlp_extractors if isinstance(args.nlp_extractors, list) else [args.nlp_extractors]
    goal_list = args.goals if isinstance(args.goals, list) else [args.goals]
    for nlp_extractor in nlp_extractor_list:
        for goal in goal_list:
            trial_id += 1
            args.goal = goal
            args.nlp_extractor = nlp_extractor
            # Finetuning the right model from a folder of sampling experiment (loading the analogous experiment)
            if args.finetune_folder:
                args.finetune_folder = os.path.join(args.finetune_folder, args.algorithm.upper() + "_" + str(trial_id) + "_" + args.goal + "_" + args.nlp_extractor)
                checkpoint_files = [file for file in os.listdir(os.path.join(args.finetune_folder, "validation", str(1))) if file.startswith("checkpoint_")]
                episode_pattern = re.compile(r'checkpoint_(-?\d+)')
                checkpoint_files.sort(key=lambda x: int(episode_pattern.search(x).group(1)))
                args.finetune_model = os.path.join(args.finetune_folder, "validation", str(1), checkpoint_files[-1])
            else:
                args.finetune_model = None
            run_logs_folder = os.path.join(logs_folder, args.algorithm.upper() + "_" + str(trial_id) + "_" + args.goal + "_" + args.nlp_extractor)
            os.makedirs(run_logs_folder, exist_ok=True)
            logger, run_logs_folder, envs_folder, config, train_ids, val_ids = setup_train_via_args(args, run_logs_folder)
            if args.verbose:
                logger.info(f"Performing sample iteration {trial_id} with goal {config['goal']} and nlp_extractor {config['nlp_extractor']}")
            train_rl_algorithm(run_logs_folder, envs_folder, config, train_ids, val_ids, logger=logger, verbose=args.verbose)
            # delete directory envs folder at the end otherwise it will be too big
            if not args.load_processed_envs: # otherwise it removes the folder from the specific run containing the processed envs
                if args.verbose:
                    logger.info(f"Removing environment folder {envs_folder} after sampling")
                shutil.rmtree(envs_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample different trials with NLP extractor and goals for an RL Agent in C-CyberBattleSim environments")
    # same code of train_agent but without nlp extractor and goal
    parser.add_argument('--algorithm', type=str,
                        choices=['ppo', 'a2c', 'rppo', 'trpo', 'ddpg', 'sac', 'td3', 'tqc'], default='ppo',
                        help='RL algorithm to train')
    parser.add_argument('-nlp', '--nlp_extractors', type=str,
                        choices=["bert", "distilbert", "roberta", "gpt2", "CySecBERT", "SecureBERT", "SecBERT",
                                 "SecRoBERTa"], default=["bert", "distilbert", "roberta", "gpt2", "CySecBERT", "SecureBERT", "SecBERT",
                                 "SecRoBERTa"], nargs='+',
                        help='NLP extractors to be used for extracting vulnerability embeddings')
    parser.add_argument('-g', '--goals', type=str, choices=['control', 'discovery', 'disruption'], default=['control', 'discovery', 'disruption'], nargs='+',
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
    parser.add_argument('--finetune_folder', type=str,
                        help='Path to the folder with the models to eventually finetune of a previous sampling experiment (relative to the logs folder)')
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

    error, message = check_args(args)
    if error:
        raise ValueError(message)

    # Creating logs folder
    if args.name:
        logs_folder = os.path.join('logs', args.name + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        logs_folder = os.path.join('logs', "sample_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    # Load general configuration files
    general_config = load_yaml(os.path.join(script_dir, "..", "..", "config.yaml"))
    if not args.load_envs and not args.load_processed_envs:
        args.load_envs = general_config['default_environments_path']

    sample_games_rl(args, logs_folder)
