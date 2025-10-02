# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    visualize_agent.py
    Script to visualize the a trained model tested a C-CyberBattleSim Environment.
"""
import copy
import argparse
import re
import numpy as np
import yaml
import sys
import os
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.utils.train_utils import algorithm_models # noqa: E402
from cyberbattle.utils.math_utils import set_seeds # noqa: E402
from cyberbattle.utils.file_utils import remove_folder_and_files  # noqa: E402
from cyberbattle.utils.log_utils import setup_logging  # noqa: E402
from cyberbattle.agents.test_agent import load_test_envs  # noqa: E402
from cyberbattle.utils.visualization_utils import create_interactive_agent_visualization  # noqa: E402
script_dir = Path(__file__).parent

def visualize_agent(config, envs, logger, verbose):
    # Determine where the checkpoint to load should be a training or a validation one
    if config['val_checkpoints']:
        if verbose:
            logger.info("Focusing on validation checkpoints...")
        checkpoints_folder = "validation"
    else:
        if verbose:
            logger.info("Focusing on training checkpoints...")
        checkpoints_folder = "checkpoints"

    for internal_run in os.listdir(os.path.join(config['run_folder'], checkpoints_folder)):

        if config['val_checkpoints']:
            test_folder = os.path.join("test", config['test_folder'], "validation")
        else:
            test_folder = os.path.join("test", config['test_folder'], "train")

        config['run_id'] = internal_run

        if not os.path.exists(os.path.join(config['run_folder'], test_folder, str(config['run_id']))):
            os.makedirs(os.path.join(config['run_folder'], test_folder, str(config['run_id'])))

        # Load the proper checkpoint file(s)
        checkpoint_files = [file for file in os.listdir(
            os.path.join(config['run_folder'], checkpoints_folder, str(config['run_id']))) if
                            file.startswith("checkpoint_")]

        episode_pattern = re.compile(r'checkpoint_(-?\d+)')

        checkpoint_files.sort(key=lambda x: int(episode_pattern.search(x).group(1)))
        checkpoints = []

        # Case of a single checkpoint: the last one
        if config['val_checkpoints']:
            config['last_checkpoint'] = True # enforced due to merging reasons, otherwise we cannot merge checkpoints without knowing the timestep (which is available in the training case)

        if config['last_checkpoint']:
            # last checkpoint only
            if len(checkpoint_files) > 0:
                if verbose:
                    logger.info("Focusing on the last checkpoint only...")
                checkpoints.append(os.path.join(config['run_folder'], checkpoints_folder, str(config['run_id']),
                                                checkpoint_files[-1]))
        else:
            # use all checkpoints ordered
            if verbose:
                logger.info("Focusing on all checkpoints...")
            for checkpoint_file in checkpoint_files:
                checkpoints.append(
                    os.path.join(config['run_folder'], checkpoints_folder, str(config['run_id']), checkpoint_file))

        if len(checkpoints) == 0:
            if verbose:
                logger.error("No checkpoint to load from the folder")

        for checkpoint_index in range(len(checkpoints)):
            checkpoint_path = checkpoints[checkpoint_index]
            model = algorithm_models[config['algorithm']].load(checkpoint_path)
            if verbose:
                logger.info("Focusing on the checkpoint: %s", checkpoint_path)
            # Options of visualization supported (TO EXTEND)
            if config['option'] == 'interactive_path':
                create_interactive_agent_visualization(model, envs, config['num_episodes_per_checkpoint'])

def main():
    parser = argparse.ArgumentParser(description='Test the trained RL agent on C-CyberBattleSim environments.')
    parser.add_argument('-f', '--logs_folder', required=True, help='Path to the specific logs folder with runs')
    parser.add_argument('--run_set', default="all", type=str, help='Run folder name to gather the correct run metrics (or "all" for all periodically)')
    parser.add_argument('--algorithm', choices=['ppo', 'a2c', 'rppo', 'trpo', 'ddpg', 'sac', 'td3', 'tqc'], default='trpo', help='Algorithm to use ')
    parser.add_argument('--environment_type', type=str, choices=['continuous'], default='continuous',
                        help='Type of environment to be used for training (only continuous supported)')  # to be extended in the future to LOCAL or DISCRETE or others
    parser.add_argument('--load_default_test_envs', default=False, action="store_true", help='Load test environments using default location')
    parser.add_argument('--load_custom_envs', required=False,
                        help='Path to the test folder customized (different from default test folder)')
    parser.add_argument('--load_custom_test_envs', required=False, help='Path to the test folder customized (different from default test folder), focusing only in its test set')
    parser.add_argument('--load_custom_val_envs', required=False,
                        help='Path to the test folder customized (different from default test folder), focusing only in its test set')
    parser.add_argument('--load_custom_train_envs', required=False,
                        help='Path to the test folder customized (different from default test folder), focusing only in its test set')
    parser.add_argument('--static_seed', action='store_true', default=False, help='Use a static seed for training')
    parser.add_argument('--load_seed', default="config",
                        help='Path of the folder where the seeds.yaml should be loaded from (e.g. previous experiment)')
    parser.add_argument('--random_seed', action='store_true', default=False, help='Use random seeds for training')
    parser.add_argument('--last_checkpoint', default=False, action="store_true", help='Load the last checkpoint only (best for validation or last for training)')
    parser.add_argument('--val_checkpoints', default=False, action="store_true",
                        help='Use validation checkpoints instead of training checkpoints')
    parser.add_argument('--option', default='interactive_path', choices=['interactive_path'], help='Decide which visualization approach')
    parser.add_argument('--static_defender_agent', default=None, choices=['reimage', 'events', None],
                        help='Static defender agent to use')
    parser.add_argument('--test_config', type=str, default='config/test_config.yaml', help='Path to the test configuration YAML file')
    parser.add_argument('-v', '--verbose', default=2, type=int, help='Verbose level: 0 - no output, 1 - training/validation information, 2 - episode level information, 3 - iteration level information')
    parser.add_argument('--no_save_log_file', action='store_false', dest='save_log_file',
                        default=True, help='Disable logging to file; log only to terminal')
    args = parser.parse_args()

    if not args.load_default_test_envs and not args.load_custom_test_envs and not args.load_custom_envs and not args.load_custom_val_envs and not args.load_custom_train_envs:
        raise ValueError("ERROR: Need to specify either default or custom test environments...")
    if not args.last_checkpoint and args.val_checkpoints:
        raise ValueError("ERROR: Can only use last checkpoint in case of validation checkpoints due to merging reason...")
    args.logs_folder = os.path.join(script_dir, 'logs', args.logs_folder)

    logger = setup_logging(args.logs_folder, args.save_log_file)

    # Consider all runs
    if args.run_set == "all":
        args.run_set = [folder for folder in os.listdir(args.logs_folder) if os.path.isdir(os.path.join(args.logs_folder, folder)) and folder != "test"]
        if args.verbose:
            logger.info(f"Using all {len(args.run_set)} runs in the logs folder...")
    else:
        if args.verbose:
            logger.info(f"Using the single run {args.run_set} of the logs folder...")

    # Read YAML configuration files
    with open(os.path.join(script_dir, args.test_config), 'r') as config_file:
        test_config = yaml.safe_load(config_file)
    for key, value in vars(args).items():
        test_config[key] = value

    if args.load_custom_test_envs:
        test_config['test_folder'] = copy.deepcopy(args.load_custom_test_envs).split("/")[-1]
    elif args.load_custom_val_envs:
        test_config['test_folder'] = copy.deepcopy(args.load_custom_val_envs).split("/")[-1]
    elif args.load_custom_train_envs:
        test_config['test_folder'] = copy.deepcopy(args.load_custom_train_envs).split("/")[-1]
    elif args.load_custom_envs:
        test_config['test_folder'] = copy.deepcopy(args.load_custom_envs).split("/")[-1]
    else:
        test_config['test_folder'] = "default"

    # Eventual seeds
    if args.static_seed:
        seed_test = 42
    elif args.random_seed:
        seed_test = np.random.randint(1000)
    else: # args.load_seed, first seed
        if args.verbose:
            logger.info(f"Reading seeds from folder {args.load_seed}")
        with open(os.path.join(args.load_seed, 'seeds.yaml'), 'r') as seeds_file:
            seeds_loaded = yaml.safe_load(seeds_file)
        seed_test = seeds_loaded['seeds'][0] # use the first one only by default
    test_config.update({"seeds_test": seed_test})
    set_seeds(seed_test)

    # random agent only case: independent from checkpoints
    if args.verbose:
        logger.info("Option selected: {}".format(test_config['option']))
    if isinstance(test_config['run_set'], list):
        runs_outcomes = []
        for run in test_config['run_set']:
            if run == "test":
                continue
            if args.verbose:
                logger.info("Run ID: %s", run)
            run_folder = os.path.join(args.logs_folder, run)
            train_config_file = os.path.join(run_folder, 'train_config.yaml')
            with open(train_config_file, 'r') as train_config_file:
                train_config = yaml.safe_load(train_config_file)
            if not args.algorithm:
                args.algorithm = train_config['algorithm']
                test_config['algorithm'] = train_config['algorithm']
            test_run_config = copy.deepcopy(test_config)
            test_run_config['run_name'] = run
            test_run_config['run_folder'] = run_folder
            test_run_config['goal'] = run.split("_")[2]

            test_envs, envs_folder = load_test_envs(run_folder, args, train_config, test_config, logger=logger)
            visualize_agent(test_run_config, test_envs, logger, args.verbose)
            if args.load_custom_test_envs or args.load_custom_envs or args.load_custom_val_envs or args.load_custom_train_envs:
                remove_folder_and_files(envs_folder)
    else:
        if args.verbose:
            logger.info("Run ID: %s", test_config['run_set'])
        run_folder = os.path.join(args.logs_folder, test_config['run_set'])
        train_config_file = os.path.join(run_folder, 'train_config.yaml')
        with open(train_config_file, 'r') as train_config_file:
            train_config = yaml.safe_load(train_config_file)
        if not args.algorithm:
            args.algorithm = train_config['algorithm']
            test_config['algorithm'] = train_config['algorithm']
        test_config['run_name'] = test_config['run_set']
        test_config['run_folder'] = run_folder
        test_config['goal'] = run_folder.split("_")[2]
        test_envs, envs_folder = load_test_envs(run_folder, args, train_config, test_config, logger=logger)
        visualize_agent(test_config, test_envs, logger, args.verbose)
        if args.load_custom_test_envs or args.load_custom_envs or args.load_custom_val_envs or args.load_custom_train_envs:
            remove_folder_and_files(envs_folder)

if __name__ == "__main__":
    main()
