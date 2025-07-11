# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    test_heuristic.py
    Script to test an heuristic on a C-CyberBattleSim Environment.
    Several options are present to assess the heuristic's performance.
"""

import copy
import torch
import argparse
import sys
import os
import numpy as np
import yaml
import pickle
import datetime
from tqdm import tqdm
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv # noqa: E402
from cyberbattle.gae.model import GAEEncoder # noqa: E402
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs # noqa: E402
from cyberbattle.utils.test_utils import calculate_average_performances, print_save_performance_metrics # noqa: E402
from cyberbattle.utils.math_utils import set_seeds # noqa: E402
from cyberbattle.utils.file_utils import remove_folder_and_files # noqa: E402
from cyberbattle._env.static_defender import ScanAndReimageCompromisedMachines, ExternalRandomEvents # noqa: E402
from cyberbattle.utils.log_utils import setup_logging # noqa: E402
from cyberbattle.utils.heuristic_utils import heuristic_models # noqa: E402

script_dir = Path(__file__).parent

# Parse option (only performances available for now) and save results
def parse_option(config, envs, logger, verbose):
    test_folder = os.path.join("test", config['test_folder'], "heuristic", "1")
    if not os.path.exists(os.path.join(config['run_folder'], test_folder)):
        os.makedirs(os.path.join(config['run_folder'], test_folder))
    heuristic_model = heuristic_models[config['heuristic']]
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if config['option'] == "heuristic_performances":
        (df, agent_owned_list, agent_discovered_list, agent_availability_list, agent_discovered_amount_list, agent_disrupted_list, agent_won_list,
                 random_agent_owned_list, random_agent_discovered_list, random_agent_availability_list, random_agent_discovered_amount_list, random_agent_disrupted_list, random_agent_won_list) = calculate_average_performances(
                    heuristic_model, envs, config['proportional_cutoff_coefficient'], num_episodes=config['num_episodes_per_checkpoint'], avoid_random=config['no_random'], goal=config['goal'], score_name=config['score_name'], logger=logger, verbose=verbose)

        save_folder = os.path.join(config['run_folder'], test_folder)
        if config['static_defender_agent']:
            defender_parameter = str(config['random_event_probability']*100) if config['static_defender_agent'] == "events" else str(config['detect_probability']*100) + "-" + str(config['scan_capacity']) + "-" + str(config['scan_frequency'])
        else:
            defender_parameter = '0'
        df.to_csv(os.path.join(save_folder, f"average_performances_{config['heuristic']}_{config['proportional_cutoff_coefficient']}_{config['num_episodes_per_checkpoint']}_{defender_parameter}_{current_time}.csv"),
                              index=False)
        outcomes = {"Owned nodes percentage": agent_owned_list,
                                                       "Discovered nodes percentage": agent_discovered_list,
                                                       "Network availability": agent_availability_list,
                                                       "Discovered amount percentage": agent_discovered_amount_list,
                                                       "Disrupted nodes percentage": agent_disrupted_list,
                                                       "Episodes won": agent_won_list,
                                                       "Random - Owned nodes percentage": random_agent_owned_list,
                                                       "Random - Discovered nodes percentage": random_agent_discovered_list,
                                                       "Random - Network availability": random_agent_availability_list,
                                                       "Random - Discovered amount percentage": random_agent_discovered_amount_list,
                                                       "Random - Disrupted nodes percentage": random_agent_disrupted_list,
                                                       "Random - Episodes won": random_agent_won_list
                                                       }
        print_save_performance_metrics(outcomes, config['heuristic'], save_folder, num_episodes=config['num_episodes_per_checkpoint'], proportional_cutoff_coefficient=config['proportional_cutoff_coefficient'], current_time=current_time, logger=logger, verbose=verbose)


# Load test environments and wrap it with the proper features (defender, normalizer, etc.)
def load_test_envs(run_folder, args, train_config, test_config, logger=None, stable_baselines=True):
    test_ids = []
    envs_folder = None
    original_envs_folder = None
    if args.load_default_test_envs:
        with open(os.path.join(run_folder, "split.yaml"), 'r') as file:
            yaml_info = yaml.safe_load(file)
        for elem in yaml_info['test_set']:
            test_ids.append(str(elem['id']))
        envs_folder = os.path.join(run_folder, "envs")
    elif args.load_custom_test_envs:
        original_envs_folder = os.path.join('..', 'data', 'env_samples', args.load_custom_test_envs)
        with open(os.path.join(original_envs_folder, "split.yaml"), 'r') as file:
            yaml_info = yaml.safe_load(file)
        for elem in yaml_info['test_set']:
            test_ids.append(str(elem['id']))
    elif args.load_custom_envs:
        original_envs_folder = os.path.join('..', 'data', 'env_samples', args.load_custom_envs)
        for elem in os.listdir(original_envs_folder):
            if os.path.isdir(os.path.join(original_envs_folder, elem)):
                test_ids.append(str(elem))
    # Setting up the proper GAE
    with open(train_config['graph_encoder_config_path'], 'r') as config_file:
        config_encoder = yaml.safe_load(config_file)
    with open(train_config['graph_encoder_spec_path'], 'r') as config_file:
        spec_encoder = yaml.safe_load(config_file)
    config_encoder.update(spec_encoder)
    train_config['node_embeddings_dimensions'] = config_encoder['model_config']['layers'][-1]['out_channels']
    train_config['proportional_cutoff_coefficient'] = test_config['proportional_cutoff_coefficient']
    graph_encoder = GAEEncoder(config_encoder['node_feature_vector_size'], config_encoder['model_config']['layers'],
                                   config_encoder['edge_feature_vector_size'])
    graph_encoder.load_state_dict(torch.load(train_config['graph_encoder_path']))
    graph_encoder.eval()

    # map to classes after saving the configuration
    if args.static_defender_agent:
        map_dict = {
            "events": ExternalRandomEvents(test_config['random_event_probability'], logger=logger, verbose=args.verbose),
            "reimage": ScanAndReimageCompromisedMachines(test_config['detect_probability'], test_config['scan_capacity'], test_config['scan_frequency'], logger=logger, verbose=args.verbose)
        }

        train_config['static_defender_agent'] = map_dict[args.static_defender_agent]
        train_config['static_defender_eviction_goal'] = test_config['static_defender_eviction_goal']

    if args.load_custom_test_envs or args.load_custom_envs: # reload right elements from the custom folder
        # load in a temporary folder that will be then eliminated to not overload file system
        tmp_folder = os.path.join("tmp", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
        for folder in tqdm(os.listdir(original_envs_folder), desc="Loading environments..."):
            if os.path.isdir(os.path.join(str(original_envs_folder), folder)) and folder.isdigit() and folder in test_ids:
                if train_config['pca_components'] == 768:
                    network_folder = os.path.join(str(original_envs_folder), folder, f"network_{train_config['nlp_extractor']}.pkl")
                else:
                    network_folder = os.path.join(str(original_envs_folder), folder, "pca", "num_components="+str(train_config['pca_components']),
                                                          f"network_{train_config['nlp_extractor']}.pkl")
                with open(network_folder, 'rb') as f:
                    network = pickle.load(f)
                train_config.pop('verbose', None)
                env = wrap_graphs_to_compressed_envs(network, logger=logger, verbose=args.verbose, **train_config)
                env.set_pca_components(train_config['pca_components'])
                env.set_graph_encoder(graph_encoder)
                with open(os.path.join(tmp_folder, f"{folder}.pkl"), 'wb') as f:
                     pickle.dump(env, f)
        envs_folder = tmp_folder
    test_ids = [x for x in test_ids if f"{x}.pkl" in os.listdir(envs_folder)]
    if args.verbose:
        logger.info(f"Focusing on {len(test_ids)} test set environments...")
    envs = RandomSwitchEnv(test_ids, train_config['switch_interval'],
                                 envs_folder=envs_folder, csv_folder=run_folder, save_to_csv=False,
                                 save_to_csv_interval=False,
                                 save_embeddings=False,
                                verbose=args.verbose)
    if stable_baselines: # SB3 implementation requires DummyVecEnv and allows normalization
        envs = DummyVecEnv([lambda: Monitor(envs)])
        envs = VecNormalize(envs, norm_obs=train_config['norm_obs'], norm_reward=False) # reward not used in testing phase
    return envs, envs_folder



def main():
    parser = argparse.ArgumentParser(description='Test the trained RL agent on C-CyberBattleSim environments.')
    parser.add_argument('--heuristic', default='highest_score_vuln', choices=['random_node_random_vuln_outcome', 'highest_score_vuln'],
                        help='Decide which heuristic to use')
    parser.add_argument('--environment_type', type=str, choices=['compressed'], default='compressed',
                        help='Type of environment to be used for training')  # to be extended in the future to LOCAL or DISCRETE or others
    parser.add_argument('--load_default_test_envs', default=False, action="store_true", help='Load test environments using default location')
    parser.add_argument('--load_custom_envs', required=False,
                        help='Path to the test folder customized (different from default test folder)')
    parser.add_argument('--load_custom_test_envs', required=False, help='Path to the test folder customized (different from default test folder), focusing only in its test set')
    parser.add_argument('--static_seed', action='store_true', default=False, help='Use a static seed for training')
    parser.add_argument('--load_seed', default="config",
                        help='Path of the folder where the seeds.yaml should be loaded from (e.g. previous experiment)')
    parser.add_argument('--no_random', default=False, action="store_true",
                        help='Avoid calculation of average performances for the random agent')
    parser.add_argument('--random_seed', action='store_true', default=False, help='Use random seeds for training')
    parser.add_argument('--option', default='heuristic_performances', choices=['heuristic_performances'], help='Decide which statistics to plot')
    parser.add_argument('--static_defender_agent', default=None, choices=['reimage', 'events', None],
                        help='Static defender agent to use')
    parser.add_argument('--test_config', type=str, default='config/test_config.yaml', help='Path to the test configuration YAML file')
    parser.add_argument('-v', '--verbose', default=2, type=int, help='Verbose level: 0 - no output, 1 - training/validation information, 2 - episode level information, 3 - iteration level information')
    parser.add_argument('--no_save_log_file', action='store_false', dest='save_log_file',
                        default=True, help='Disable logging to file; log only to terminal')
    args = parser.parse_args()

    if args.load_default_test_envs and args.load_custom_test_envs:
        raise ValueError("ERROR: Cannot load both default and custom test environments...")
    if not args.load_default_test_envs and not args.load_custom_test_envs and not args.load_custom_envs:
        raise ValueError("ERROR: Need to specify either default or custom test environments...")

    # a specific logs folder will be created for each heuristic run
    args.logs_folder = os.path.join(script_dir,'logs', args.heuristic + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    os.makedirs(args.logs_folder)
    logger = setup_logging(args.logs_folder, args.save_log_file)

    # Read YAML configuration files
    with open(args.test_config, 'r') as config_file:
        test_config = yaml.safe_load(config_file)
    for key, value in vars(args).items():
        test_config[key] = value

    if args.load_custom_test_envs:
        test_config['test_folder'] = copy.deepcopy(args.load_custom_test_envs).split("/")[-1]
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

    if args.verbose:
        logger.info("Option selected: {}".format(test_config['option']))

    train_config_file = os.path.join(script_dir, "config", 'heuristic_config.yaml')
    with open(train_config_file, 'r') as train_config_file:
        train_config = yaml.safe_load(train_config_file)
    index = 0
    # metric that heuristic should optimize
    for score_name in ["cvss", "impact_score", "exploitability_score"]:
        test_run_config = copy.deepcopy(test_config)
        test_run_config['heuristic'] = args.heuristic
        test_run_config['goal'] = None
        test_run_config['score_name'] = score_name
        run_folder = os.path.join(args.logs_folder, test_run_config['heuristic'] + "_" + str(index) + "_" + test_run_config['score_name'] + "_None") # reuse structure normal trainings
        test_envs, envs_folder = load_test_envs(run_folder, args, train_config, test_config, logger=logger)
        test_run_config['run_folder'] = run_folder
        parse_option(test_run_config, test_envs, logger, args.verbose)
        if args.load_custom_test_envs or args.load_custom_envs:
            remove_folder_and_files(envs_folder)

if __name__ == "__main__":
    main()
