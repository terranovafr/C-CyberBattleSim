# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    train_agent.py
    Script to train a RL algorithm on scenarios of the C-CyberBattleSim Environment for a certain goal, nlp extractor, and parameters
    Several options are available to customize the training process
"""

import argparse
import copy
import sys
import os
import yaml
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import random
import pickle
from tqdm import tqdm
import shutil
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import re
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.utils.train_utils import replace_with_classes, check_args, clean_config_save, algorithm_models, reccurrent_algorithms # noqa: E402
from cyberbattle.utils.math_utils import linear_schedule, calculate_auc, set_seeds # noqa: E402
from cyberbattle.utils.file_utils import extract_metric_data, load_yaml # noqa: E402
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv # noqa: E402
from cyberbattle.agents.callbacks import TrainingCallback as TrainingCallbackGraph, ValidationCallback as ValidationCallbackGraph # noqa: E402
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs, wrap_graphs_to_global_envs, wrap_graphs_to_local_envs # noqa: E402
from cyberbattle.utils.file_utils import save_yaml # noqa: E402
from cyberbattle._env.static_defender import ScanAndReimageCompromisedMachines, ExternalRandomEvents # noqa: E402
from cyberbattle.gae.model import GAEEncoder # noqa: E402
from cyberbattle.utils.log_utils import setup_logging # noqa: E402

torch.set_default_dtype(torch.float32)
script_dir = os.path.dirname(__file__)

# Handles the training for different runs of the algorithm and extraction of average metric values
def train_rl_algorithm(logs_folder, envs_folder, config, train_ids, val_ids, metric_name=None, logger=None, verbose=1):
    metric_values = []
    if verbose:
        logger.info(f"Training algorithm {config['algorithm']} on {config['num_environments']} environments with {config['num_runs']} runs.")
    for run_id in range(config['num_runs']):

        # Varying the seed (if needed) at each run
        seed = config['seeds_runs'][run_id]
        set_seeds(seed)

        if verbose:
           logger.info(f"Run {run_id + 1}/{config['num_runs']} with seed {seed} started.")

        # Wrap training and validation environments using switchers

        train_envs = RandomSwitchEnv(train_ids, config['switch_interval'], envs_folder=envs_folder, csv_folder=logs_folder, save_to_csv=config['save_csv_file'], save_to_csv_interval=config['save_csv_file'], save_embeddings=config['save_embeddings_csv_file'], verbose=verbose)

        if config['holdout']:
            val_envs = RandomSwitchEnv(val_ids, config['val_switch_interval'],  envs_folder=envs_folder, csv_folder=os.path.join(logs_folder, "validation"), save_to_csv=config['save_csv_file'], save_to_csv_interval=config['save_csv_file'], save_embeddings=config['save_embeddings_csv_file'], verbose=verbose)
        else:
            val_envs = None

        check_env(train_envs)

        train_model(train_envs, logs_folder, config, run_id+1, val_envs=val_envs, logger=logger, verbose=verbose)

        # extract metric data (used by hyperopt to determine best hyperparameters based on their score)
        if metric_name:
            if config['algorithm'] != "rppo": # handling name difference in case of recurrent PPO
                tensorboard_dir = os.path.join(logs_folder, config['algorithm'].upper() + "_" + str(run_id + 1))
            else:
                tensorboard_dir = os.path.join(logs_folder, "RecurrentPPO_" + str(run_id + 1))
            # determine the evolution of curves
            times, values = extract_metric_data(tensorboard_dir, metric_name)
            auc = calculate_auc(times, values) # AUC normalized to starting value in order to normalize to initial random performances
            if verbose:
                logger.info(f"The AUC of metric {metric_name} for the run {run_id + 1} is {auc}")
            metric_values.append(auc)

    if verbose:
        logger.info("Training finished.")
    return metric_values

# Function used to train the model based on the configuration provided
def train_model(train_envs, logs_folder, config, run_id, val_envs=None, logger=None, verbose=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        logger.info(f"Training on device: {device}")

    # Learning rate scheduling (potential)
    algorithm_config = copy.deepcopy(config['algorithm_hyperparams'])
    if algorithm_config['learning_rate_type'] == "linear":
        learning_rate = linear_schedule(algorithm_config['learning_rate'], algorithm_config['learning_rate_final'])
    else: # constant learning rate
        learning_rate = algorithm_config['learning_rate']

    algorithm_config.pop('learning_rate_type', None)
    algorithm_config.pop('learning_rate', None)
    algorithm_config.pop('learning_rate_final', None)

    # Create the model
    model_class = algorithm_models[config['algorithm']]

    # Use DummyVecEnv to wrap the environments
    train_envs = DummyVecEnv([lambda: Monitor(train_envs)])

    # potential normalization if specified
    train_envs = VecNormalize(train_envs, norm_obs=config['norm_obs'], norm_reward=config['norm_reward'])
    if val_envs:
        val_envs = DummyVecEnv([lambda: Monitor(val_envs)])
        val_envs = VecNormalize(val_envs, norm_obs=config['norm_obs'], norm_reward=False)

    if verbose:
        logger.info(f"Normalization of observations: {config['norm_obs']}")
        logger.info(f"Normalization of rewards: {config['norm_reward']}")

    # different policy according to whether the algorithm is using history of transitions
    if config['algorithm'] in reccurrent_algorithms:
        if config['finetune_model'] and os.path.exists(config['finetune_model']):
            model = model_class.load(config['finetune_model'], env=train_envs, device=device)
            if verbose:
                logger.info(f"Loaded model to finetune from {config['finetune_model']}")
        else:
            if verbose:
                logger.info("Initialized new model from scratch")
            model = model_class("MultiInputLstmPolicy", train_envs, policy_kwargs=config['policy_kwargs'],
                                learning_rate=learning_rate,
                                tensorboard_log=logs_folder, **algorithm_config, verbose=verbose, device=device)
    else:
        lstm_keys = ['lstm_hidden_size', 'n_lstm_layers']
        for key in lstm_keys:
            if key in config['policy_kwargs']:
                del config['policy_kwargs'][key]
        if 'finetune_model' in config and config['finetune_model'] and os.path.exists(config['finetune_model']):
            model = model_class.load(config['finetune_model'], tensorboard_log=logs_folder, env=train_envs, verbose=1, learning_rate=learning_rate, device=device)
            if verbose:
                logger.info(f"Loaded model to finetune from {config['finetune_model']}")
        else:
            if verbose:
                logger.info("Initialized new model from scratch")
            model = model_class("MultiInputPolicy", train_envs, policy_kwargs=config['policy_kwargs'],
                                    learning_rate=learning_rate,
                                    tensorboard_log=logs_folder, **algorithm_config, verbose=verbose, device=device)

    # Checkpoint periodic saving
    checkpoint_callback = CheckpointCallback(save_freq=config['checkpoints_save_freq'],
                                             save_path=os.path.join(logs_folder, "checkpoints", str(run_id)),
                                             name_prefix='checkpoint')
    # Logging additional training metrics
    train_callback = TrainingCallbackGraph(env=train_envs, verbose=verbose)

    callbacks = [checkpoint_callback, train_callback]

    if val_envs:
        # Logging validation metrics, saves validation checkpoints, and use early stopping if requested
        val_callback = ValidationCallbackGraph(
                val_env=val_envs,
                n_val_episodes=config['n_val_episodes'],
                val_freq=config['val_freq'],
                validation_folder_path=os.path.join(logs_folder, "validation", str(run_id)),
                early_stopping=config['early_stopping'],
                patience=config['early_stopping'],
                output_logger=logger,
                verbose=verbose
        )
        callbacks.append(val_callback)
    if verbose:
        logger.info(f"Training started for run {run_id} lasting {config['train_iterations']} iterations with episode length {config['episode_iterations']}")
    # Final call to learn method with all callbacks to handle training and validation
    try:
        model.learn(total_timesteps=config['train_iterations'], callback=callbacks)
    except Exception as e:
        logger.error(f"Training failed due to errors: {e}, skipping this run")
        return

# Setup the training via arguments provided
def setup_train_via_args(args, logs_folder=None, envs_folder=None):
    args.yaml = args.yaml if 'yaml' in args else False # only train_agent for now due to complexity
    if args.yaml: # done before otherwise logs folder already set, supported only for train_agent.py for now
        with open(os.path.join(script_dir, "logs", str(args.yaml), "train_config.yaml"), 'r') as config_file:
            config = yaml.safe_load(config_file)
        # Save information that yaml was loaded
        args.goal = config['goal']
        args.nlp_extractor = config['nlp_extractor']
        args.algorithm = config['algorithm']

    if not logs_folder:
        # Creating logs folder
        if args.name:
            logs_folder = os.path.join(script_dir, 'logs', args.name + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            logs_folder = os.path.join(script_dir, 'logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        logs_folder = os.path.join(logs_folder,
                                   args.algorithm.upper() + "_x_" + args.goal + "_" + args.nlp_extractor)  # just for uniformity
        os.makedirs(logs_folder, exist_ok=True)

    # Setting up the logger (terminal output and or file based)
    logger = setup_logging(logs_folder, log_to_file=args.save_log_file)

    # Read YAML configuration file of a previous training
    if 'yaml' in args and args.yaml:
        # Save information that yaml was loaded
        config['yaml'] = args.yaml
    else:
        # Read otherwise provided (or default) YAML configuration file
        if args.verbose:
            logger.info("Reading default configuration YAML files")
        with open(os.path.join(script_dir, args.train_config), 'r') as config_file:
            config = yaml.safe_load(config_file)
        with open(os.path.join(script_dir,args.rewards_config), 'r') as config_file:
            rewards_config = yaml.safe_load(config_file)
        with open(os.path.join(script_dir,args.algo_config), 'r') as config_file:
            algorithm_config = yaml.safe_load(config_file)
        args.algorithm_hyperparams = algorithm_config[args.algorithm]
        args.rewards_dict = rewards_config['rewards_dict'][args.goal]
        args.penalties_dict = rewards_config['penalties_dict'][args.goal]
        args.policy_kwargs = algorithm_config['policy_kwargs']
        config.update(vars(args)) # put all arguments in the configuration dict only if yaml not loaded, otherwise overwrite

    # Finetune an already existing model
    if 'finetune_model' in args and args.finetune_model:
        if '.zip' not in args.finetune_model:
            raise ValueError("The path provided should be the zip file related to the model to finetune!")
        config['finetune_model'] = os.path.join(script_dir, "logs", config['finetune_model'])

    # Seeds handling
    if config['static_seeds']:
        if config['verbose']:
            logger.info("Setting a static seed for all runs of training")
        seeds_runs = [42 for _ in range(config['num_runs'])]
    elif config['random_seeds']:
        if config['verbose']:
            logger.info("Setting random seeds for training")
        seeds_runs = [np.random.randint(1000) for _ in range(config['num_runs'])]
    else: # load_seeds case
        if config['verbose']:
            logger.info("Loading seeds from seeds file %s", config['load_seeds'])
        seeds_loaded = load_yaml(os.path.join(script_dir, args.load_seeds, 'seeds.yaml'))
        seeds_runs = seeds_loaded['seeds'][0:config['num_runs']]

    # Saving seeds for reproducibility
    save_yaml({"seeds": seeds_runs}, logs_folder, "seeds.yaml")
    config.update({"seeds_runs": seeds_runs})
    set_seeds(seeds_runs[0]) # do it before initial setting steps

    def get_base_path(base_dir):
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        run_dirs = sorted([d for d in subdirs if re.match(r'run_\d+', d)])
        if run_dirs:
            # Assuming you want the latest run_X folder
            return os.path.join(base_dir, run_dirs[-1])  # or use run_dirs[0] for the first
        else:
            return base_dir

    # Load the graph encoder which changes according to the LM and eventual dimensionality reduction used
    if config['pca_components']:
        if config['verbose']:
            logger.info(
                "Loading graph encoder model related to NLP extractor %s with dimensionality reduction components %d",
                config['nlp_extractor'], config['pca_components'])
        if config['yaml']:  # already pre-computed
            graph_encoder_path = config['graph_encoder_path']
            graph_encoder_config_path = config['graph_encoder_config_path']
            graph_encoder_spec_path = config['graph_encoder_spec_path']
        else:
            general_config = load_yaml(os.path.join(script_dir, "..", "..", "config.yaml"))
            base_dir = os.path.join(script_dir, "..", "gae", "logs",
                                    general_config['gae_dimensionality_reduction_path'][config['pca_components']],
                                    f"num_components={config['pca_components']}", config['nlp_extractor'])
            base_dir = get_base_path(base_dir)
            graph_encoder_path = os.path.join(base_dir, "encoder.pth")
            graph_encoder_config_path = os.path.join(base_dir, "train_config_encoder.yaml")
            graph_encoder_spec_path = os.path.join(base_dir, "model_spec.yaml")
    else:
        if config['verbose']:
            logger.info("Loading graph encoder model related to NLP extractor %s", config['nlp_extractor'])
        if config['yaml']:  # already pre-computed
            graph_encoder_path = config['graph_encoder_path']
            graph_encoder_config_path = config['graph_encoder_config_path']
            graph_encoder_spec_path = config['graph_encoder_spec_path']
        else:
            general_config = load_yaml(os.path.join(script_dir, "..", "..", "config.yaml"))
            base_dir = os.path.join(script_dir, "..", "gae", "logs",
                                    general_config['gae_path'], config['nlp_extractor'])
            base_dir = get_base_path(base_dir)
            graph_encoder_path = os.path.join(base_dir, "encoder.pth")
            graph_encoder_spec_path = os.path.join(base_dir, "model_spec.yaml")
            graph_encoder_config_path = os.path.join(base_dir, "train_config_encoder.yaml")

    # Eventual dimensionality reduction also needed in the environment if it was done on the LM embedding space (e.g. in order to modify action space size)
    if not config['pca_components']:
        config.update({"pca_components": config['default_vulnerability_embeddings_size']})  # default output layer size of NN of most LLMs

    # Load the graph encoder configuration file to extract the embeddings dimensions
    with open(graph_encoder_config_path, 'r') as config_file:
        config_encoder = yaml.safe_load(config_file)
    with open(graph_encoder_spec_path, 'r') as config_file:
        spec_encoder = yaml.safe_load(config_file)
    config_encoder.update(spec_encoder)
    config['node_embeddings_dimensions'] = config_encoder['model_config']['layers'][-1]['out_channels']
    config['edge_feature_aggregations'] = config_encoder['edge_feature_aggregations']

    # Load the graph encoder model
    graph_encoder = GAEEncoder(config_encoder['node_feature_vector_size'], config_encoder['model_config']['layers'],
                               config_encoder['edge_feature_vector_size'])
    graph_encoder.load_state_dict(torch.load(str(graph_encoder_path)))
    graph_encoder.eval()

    config.update({"graph_encoder_path": graph_encoder_path})
    config.update({"graph_encoder_config_path": graph_encoder_config_path})
    config.update({"graph_encoder_spec_path": graph_encoder_spec_path})

    # Removing potential configuration keys that are not needed to be saved
    config = clean_config_save(config)
    # Saving configurations to ensure reproducibility
    save_yaml(config, logs_folder)

    config.update({"graph_encoder_path": os.path.join(script_dir, graph_encoder_path)})
    config.update({"graph_encoder_config_path": os.path.join(script_dir, graph_encoder_config_path)})
    config.update({"graph_encoder_spec_path": os.path.join(script_dir, graph_encoder_spec_path)})

    if not config['holdout']:
        if config['verbose']:
            logger.info("Holdout not set, hence processing only one environment from the environment folder")
        config['num_environments'] = 1

    if config['load_processed_envs']: # environments already processed from previous run
        if '/' not in config['load_processed_envs']:
            raise ValueError("The path of the run folder inside the logs folder should be provided!")
        if config['verbose']:
            logger.info("Loading processed environments from the run folder %s", config['load_processed_envs'])
        envs_folder = os.path.join(script_dir, "logs", config['load_processed_envs'], "envs")
        config['num_environments'] = 0
        for file in os.listdir(envs_folder):
            if file.endswith(".pkl"):
                config['num_environments'] += 1
        if config['verbose']:
            logger.info("Loaded %d environments from the folder", config['num_environments'])
    elif config['load_envs']: # environments to be processed from a folder in env_samples
        if config['verbose']:
            logger.info("Loading environments from the folder %s", config['load_envs'])

        original_envs_folder = os.path.join(script_dir, '..', 'data', 'env_samples', config['load_envs'])

        if not envs_folder: # destination folder where to have the processed environments
            envs_folder = os.path.join(logs_folder, "envs")
        if not os.path.exists(os.path.join(envs_folder)):
            os.makedirs(os.path.join(envs_folder))

        config['num_environments'] = 0
        for element in tqdm(os.listdir(original_envs_folder), desc="Loading environments from the folder"):
            if os.path.isfile(os.path.join(original_envs_folder, element)):
                with open(os.path.join(envs_folder, element), 'wb') as f: # saving into the destination folder
                    shutil.copyfile(os.path.join(original_envs_folder, element), os.path.join(envs_folder, element))
            if os.path.isdir(os.path.join(original_envs_folder,element)) and element.isdigit():
                config['num_environments'] += 1

                # loading the correct environments based on nlp extractor and eventual dimensionality reduction
                if config['pca_components'] == 768:
                    network_folder = os.path.join(original_envs_folder, element, f"network_{config['nlp_extractor']}.pkl")
                else:
                    network_folder = os.path.join(original_envs_folder, element, "pca/num_components="+str(config['pca_components']) ,f"network_{config['nlp_extractor']}.pkl")
                with open(network_folder, 'rb') as f:
                    network = pickle.load(f)

                # map simple values to classes after saving the configuration file (to avoid saving the object)
                if config['static_defender_agent']:
                    if config['static_defender_agent']=='reimage':
                        config['static_defender_agent'] = ScanAndReimageCompromisedMachines(random.uniform(config['detect_probability_min'], config['detect_probability_max']),
                                                                                           random.randint(config['scan_capacity_min'], config['scan_capacity_max']),
                                                                                           random.randint(config['scan_frequency_min'], config['scan_frequency_max']), logger=logger, verbose=config['verbose'])
                    elif config['static_defender_agent']=='events':
                        config['static_defender_agent'] = ExternalRandomEvents(random.uniform(config['random_event_probability_min'], config['random_event_probability_max']), logger=logger, verbose=config['verbose'])

                # Use the network graph environment and map it to a C-CyberBattleSim environment
                if args.environment_type == "global":
                    env = wrap_graphs_to_global_envs(network, logger, **config)
                elif args.environment_type == "local":
                    env = wrap_graphs_to_local_envs(network, logger, **config)
                else: #if args.environment_type == "continuous":
                    env = wrap_graphs_to_compressed_envs(network, logger, **config)
                    env.set_graph_encoder(graph_encoder)

                env.set_pca_components(config['pca_components'])

                with open(os.path.join(envs_folder, f"{element}.pkl"), 'wb') as f: # saving into the destination folder
                    pickle.dump(env, f)

        if config['verbose']:
            logger.info("Loaded %d environments from the folder", config['num_environments'])

    # Splitting environment in case of holdout method
    if config['holdout']:
        yaml_split_path = os.path.join(envs_folder, "split.yaml")
        with open(yaml_split_path, 'r') as file:
            yaml_info = yaml.safe_load(file)
        train_ids = []
        for elem in yaml_info['training_set']:
            train_ids.append(elem['id'])
        val_ids = []
        for elem in yaml_info['validation_set']:
            val_ids.append(elem['id'])
    else:
        train_ids = [i + 1 for i in range(config['num_environments'])]
        val_ids = None
        yaml_info = {"training_set": [{"id": i} for i in train_ids], "validation_set": []}

    # save split information in logs
    with open(os.path.join(logs_folder, "split.yaml"), 'w') as f:
        yaml.dump(yaml_info, f)

    # map simple values to classes after saving the configuration file (to avoid saving the object)
    config['policy_kwargs'] = replace_with_classes(config['policy_kwargs'])

    return logger, logs_folder, envs_folder, config, train_ids, val_ids



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL algorithm on a C-CyberBattleSim Environment!")
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'a2c', 'rppo', 'trpo', 'ddpg', 'sac', 'td3', 'tqc'], default='trpo', help='RL algorithm to train')
    parser.add_argument('--environment_type', type=str, choices=['continuous', 'global', 'local'], default='continuous', help='Type of environment to be used for training') # to be extended in the future to LOCAL or DISCRETE or others
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('-nlp', '--nlp_extractor', type=str, choices=["bert", "distilbert", "roberta", "gpt2", "CySecBERT", "SecureBERT", "SecBERT", "SecRoBERTa"], default="bert", help='NLP extractor to be used for extracting vulnerability embeddings')
    parser.add_argument('--holdout', action='store_true', default=False, help='Use holdout strategy and periodically evaluate on validation sets')
    parser.add_argument('--finetune_model', type=str, help='Path to the model to eventually finetune (relative to the logs folder)')
    parser.add_argument('--early_stopping', type=int, default=0, help='Early stopping on the validation environments setting the number of patience runs')
    parser.add_argument('-g','--goal', type=str,  default="control", choices=["control", "discovery", "disruption", "control_node", "discovery_node", "disruption_node"], help='Goal (threat model) of the agent')
    parser.add_argument('--yaml', default=False, help='Read configuration file from YAML file of a previous training')
    parser.add_argument('--name', default=False, help='Name of the logs folder related to the run')
    parser.add_argument('--load_envs', default=False, help='Path of the envs folder where the networks should be processed and loaded from')
    parser.add_argument('--load_processed_envs', default=False, help='Path of the run folder where the envs already processed should be loaded from')
    parser.add_argument('--static_seeds', action='store_true', default=False, help='Use a static seed for training')
    parser.add_argument('--load_seeds', default="config", help='Path of the folder where the seeds.yaml should be loaded from (e.g. previous experiment)')
    parser.add_argument('--random_seeds', action='store_true', default=False, help='Use random seeds for training')
    parser.add_argument('--static_defender_agent', default=None, choices=['reimage', 'events', None], help='Static defender agent to use')
    parser.add_argument('-pca', '--pca_components', default=None, type=int,
                        help='Invoke with the use of PCA for the feature vectors specifying the number of components')
    parser.add_argument('-v', '--verbose', default=2, type=int, help='Verbose level: 0 - no output, 1 - training/validation information, 2 - episode level information, 3 - iteration level information')
    parser.add_argument('--no_save_log_file', action='store_false', dest='save_log_file',
                        default=True, help='Disable logging to file; log only to terminal')
    parser.add_argument('--save_csv_file', action='store_true', default=False, help='Flag to decide whether trajectories should be saved periodically to a csv file during training (the value determines the interval in episodes)')
    parser.add_argument('--save_embeddings_csv_file', action='store_true', default=False, help='Flag to decide whether also embeddings should be saved periodically to a csv file during training')
    parser.add_argument('--train_config', type=str, default='config/train_config.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--rewards_config', type=str, default='config/rewards_config.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--algo_config', type=str, default='config/algo_config.yaml',
                        help='Path to the configuration YAML file')
    args = parser.parse_args()

    # Check consistency of arguments
    error, message = check_args(args)
    if error:
        raise ValueError(message)

    # Load general configuration
    general_config = load_yaml(os.path.join(script_dir, "..", "..", "config.yaml"))
    if not args.load_envs and not args.load_processed_envs:
        args.load_envs = general_config['default_environments_path']

    # Setup configuration and environments
    logger, logs_folder, envs_folder, config, train_ids, val_ids = setup_train_via_args(args, None)
    # Train the RL algorithm with the provided configuration
    train_rl_algorithm(logs_folder, envs_folder, config, train_ids, val_ids, logger=logger, verbose=args.verbose)
