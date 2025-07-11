# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    test_agent.py
    Script to test a RL algorithm on a C-CyberBattleSim Environment using the trained model.
    Several options are present to assess the agent's performance.
"""
import copy
import torch
import argparse
import re
import numpy as np
import yaml
from tqdm import tqdm
import pickle
import datetime
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import sys
import os
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.utils.train_utils import algorithm_models # noqa: E402
from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv  # noqa: E402
from cyberbattle.gae.model import GAEEncoder  # noqa: E402
from cyberbattle.utils.envs_utils import wrap_graphs_to_compressed_envs, wrap_graphs_to_global_envs, wrap_graphs_to_local_envs  # noqa: E402
from cyberbattle.utils.test_utils import run_and_save_action_choices, calculate_average_performances, calculate_random_agent_average_performances, print_save_performance_metrics, play_agent_multiple_episodes_until_done, plot_action_distribution  # noqa: E402
from cyberbattle.utils.math_utils import set_seeds # noqa: E402
from cyberbattle.utils.file_utils import remove_folder_and_files  # noqa: E402
from cyberbattle._env.static_defender import ScanAndReimageCompromisedMachines, ExternalRandomEvents  # noqa: E402
from cyberbattle.utils.log_utils import setup_logging  # noqa: E402

script_dir = Path(__file__).parent

def parse_option(config, envs, logger, verbose):
    # Determine where the checkpoint to load should be a training or a validation one
    if config['val_checkpoints']:
        if verbose:
            logger.info("Focusing on validation checkpoints...")
        checkpoints_folder = "validation"
    else:
        if verbose:
            logger.info("Focusing on training checkpoints...")
        checkpoints_folder = "checkpoints"
    all_outcomes = {}

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

        outcomes = {} # to be averaged with the other runs

        for checkpoint_index in range(len(checkpoints)):
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            checkpoint_path = checkpoints[checkpoint_index]
            if "steps" in checkpoint_path: # training case (saved periodically)
                checkpoint_id = \
                    checkpoint_path.split(checkpoints_folder + "/" + str(config['run_id']) + "/checkpoint_")[1].split(
                        "_steps")[0]
            elif "reward" in checkpoint_path: # validation case (saved if overcome best known validation average reward)
                checkpoint_id = \
                    checkpoint_path.split(checkpoints_folder + "/" + str(config['run_id']) + "/checkpoint_")[
                        1].split("_reward")[0]
            else:
                checkpoint_id = None
            checkpoint_short_name = "best_val"
            model = algorithm_models[config['algorithm']].load(checkpoint_path)
            if verbose:
                logger.info("Focusing on the checkpoint: %s", checkpoint_path)
            # Action distribution case
            if config['option'] == "actions_distribution":
                vulnerabilities_chosen, vulnerabilities_types_chosen, outcomes_chosen = run_and_save_action_choices(model, envs, config['proportional_cutoff_coefficient'],
                                                      logger=logger, num_episodes=config['num_episodes_per_checkpoint'], verbose=verbose)

                with open(os.path.join(config['run_folder'], test_folder, str(config['run_id']),
                                        f"vulnerabilities_choices_{checkpoint_short_name}_{config['proportional_cutoff_coefficient']}_{config['num_episodes_per_checkpoint']}_{current_time}.yaml"), 'w', newline='') as file:
                    yaml.dump(vulnerabilities_chosen, file)
                with open(os.path.join(config['run_folder'], test_folder, str(config['run_id']),
                                        f"vulnerabilities_types_choices_{checkpoint_short_name}_{config['proportional_cutoff_coefficient']}_{config['num_episodes_per_checkpoint']}_{current_time}.yaml"), 'w', newline='') as file:
                    yaml.dump(vulnerabilities_types_chosen, file)
                with open(os.path.join(config['run_folder'], test_folder, str(config['run_id']),
                                        f"outcomes_choices_{checkpoint_short_name}_{config['proportional_cutoff_coefficient']}_{config['num_episodes_per_checkpoint']}_{current_time}.yaml"), 'w', newline='') as file:
                    yaml.dump(outcomes_chosen, file)

                plt = plot_action_distribution(outcomes_chosen, checkpoint_id, "Outcomes")
                fig_name = os.path.join(config['run_folder'], test_folder, str(config['run_id']),
                                        f"outcomes_distribution_{checkpoint_short_name}_{config['proportional_cutoff_coefficient']}_{config['num_episodes_per_checkpoint']}_{current_time}.png")
                plt.tight_layout()
                plt.savefig(fig_name)
                plt.close()
                plt = plot_action_distribution(vulnerabilities_types_chosen, checkpoint_id, "Vulnerabilities types")
                fig_name = os.path.join(config['run_folder'], test_folder, str(config['run_id']),
                                        f"vulnerabilities_types_distribution_{checkpoint_short_name}_{config['proportional_cutoff_coefficient']}_{config['num_episodes_per_checkpoint']}_{current_time}.png")
                plt.tight_layout()
                plt.savefig(fig_name)
                plt.close()
                plt = plot_action_distribution(vulnerabilities_chosen, checkpoint_id, "Vulnerabilities")
                fig_name = os.path.join(config['run_folder'], test_folder, str(config['run_id']),
                                        f"vulnerabilities_distribution_{checkpoint_short_name}_{config['proportional_cutoff_coefficient']}_{config['num_episodes_per_checkpoint']}_{current_time}.png")
                plt.tight_layout()
                plt.savefig(fig_name)
                plt.close()
                if checkpoint_short_name not in all_outcomes:
                    all_outcomes[checkpoint_short_name] = {"vulnerabilities_chosen": vulnerabilities_chosen,
                                                   "vulnerabilities_types_chosen": vulnerabilities_types_chosen,
                                                   "outcomes_chosen": outcomes_chosen}
                else:
                    all_outcomes[checkpoint_short_name]["vulnerabilities_chosen"].extend(vulnerabilities_chosen)
                    all_outcomes[checkpoint_short_name]["vulnerabilities_types_chosen"].extend(vulnerabilities_types_chosen)
                    all_outcomes[checkpoint_short_name]["outcomes_chosen"].extend(outcomes_chosen)
            # Average performance case
            elif config['option'] == "agent_performances":
                (df,  agent_owned_list, agent_discovered_list, agent_availability_list, agent_discovered_amount_list, agent_disrupted_list, agent_won_list,
                 random_agent_owned_list, random_agent_discovered_list, random_agent_availability_list, random_agent_discovered_amount_list, random_agent_disrupted_list, random_agent_won_list) = calculate_average_performances(
                    model, envs,  config['proportional_cutoff_coefficient'], num_episodes=config['num_episodes_per_checkpoint'], avoid_random=config['no_random'], logger=logger, verbose=verbose)

                save_folder = os.path.join(config['run_folder'], test_folder, str(config['run_id']))
                if config['static_defender_agent']:
                    defender_parameter = str(config['random_event_probability']*100) if config['static_defender_agent'] == "events" else str(config['detect_probability']*100) + "-" + str(config['scan_capacity']) + "-" + str(config['scan_frequency'])
                else:
                    defender_parameter = '0'
                if config['load_custom_val_envs']:
                    df.to_csv(os.path.join(save_folder,
                                           f"average_performances_{checkpoint_short_name}_{config['proportional_cutoff_coefficient']}_validationset_{config['num_episodes_per_checkpoint']}_{defender_parameter}_{current_time}.csv"),
                              index=False)
                if config['load_custom_val_envs']:
                    df.to_csv(os.path.join(save_folder,
                                           f"average_performances_{checkpoint_short_name}_{config['proportional_cutoff_coefficient']}_validationset_{config['num_episodes_per_checkpoint']}_{defender_parameter}_{current_time}.csv"),
                              index=False)
                elif config['load_custom_train_envs']:
                    df.to_csv(os.path.join(save_folder,
                                           f"average_performances_{checkpoint_short_name}_{config['proportional_cutoff_coefficient']}_trainingset_{config['num_episodes_per_checkpoint']}_{defender_parameter}_{current_time}.csv"),
                              index=False)
                else:
                    df.to_csv(os.path.join(save_folder, f"average_performances_{checkpoint_short_name}_{config['proportional_cutoff_coefficient']}_{config['num_episodes_per_checkpoint']}_{defender_parameter}_{current_time}.csv"),
                              index=False)
                outcomes[checkpoint_short_name] = {"Owned nodes percentage": agent_owned_list,
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
                if checkpoint_short_name not in all_outcomes:
                    all_outcomes[checkpoint_short_name] = {"Owned nodes percentage": agent_owned_list,
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
                else:
                    all_outcomes[checkpoint_short_name]["Owned nodes percentage"].extend(agent_owned_list)
                    all_outcomes[checkpoint_short_name]["Discovered nodes percentage"].extend(agent_discovered_list)
                    all_outcomes[checkpoint_short_name]["Network availability"].extend(agent_availability_list)
                    all_outcomes[checkpoint_short_name]["Discovered amount percentage"].extend(agent_discovered_amount_list)
                    all_outcomes[checkpoint_short_name]["Disrupted nodes percentage"].extend(agent_disrupted_list)
                    all_outcomes[checkpoint_short_name]["Episodes won"].extend(agent_won_list)
                    all_outcomes[checkpoint_short_name]["Random - Owned nodes percentage"].extend(random_agent_owned_list)
                    all_outcomes[checkpoint_short_name]["Random - Discovered nodes percentage"].extend(random_agent_discovered_list)
                    all_outcomes[checkpoint_short_name]["Random - Network availability"].extend(random_agent_availability_list)
                    all_outcomes[checkpoint_short_name]["Random - Discovered amount percentage"].extend(random_agent_discovered_amount_list)
                    all_outcomes[checkpoint_short_name]["Random - Disrupted nodes percentage"].extend(random_agent_disrupted_list)
                    all_outcomes[checkpoint_short_name]["Random - Episodes won"].extend(random_agent_won_list)
                print_save_performance_metrics(outcomes[checkpoint_short_name], checkpoint_short_name, save_folder, num_episodes=config['num_episodes_per_checkpoint'], proportional_cutoff_coefficient=config['proportional_cutoff_coefficient'], current_time=current_time, logger=logger, verbose=verbose)
            # Trajectories generation case
            elif config['option'] == "save_trajectories":
                if verbose:
                    logger.info("Saving the trajectories taken by the agent..")
                envs.envs[0].update_csv_folder(os.path.join(config['run_folder'], test_folder, str(config['run_id'])),
                                                    f"trajectories_{checkpoint_short_name}_{config['proportional_cutoff_coefficient']}_{config['num_episodes_per_checkpoint']}_{current_time}.csv")
                play_agent_multiple_episodes_until_done(envs, model, config['proportional_cutoff_coefficient'], config['num_episodes_per_checkpoint'])
                if verbose:
                    logger.info("Trajectories saved in the CSV file %s", os.path.join(config['run_folder'], test_folder, str(config['run_id']),
                                                                                     f"trajectories_{checkpoint_short_name}_{config['proportional_cutoff_coefficient']}_{config['num_episodes_per_checkpoint']}_{current_time}.csv"))
    return all_outcomes

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
    elif args.load_custom_val_envs:
        original_envs_folder = os.path.join('..', 'data', 'env_samples', args.load_custom_val_envs)
        original_envs_folder = os.path.join('..', 'data', 'env_samples', args.load_custom_val_envs)
        with open(os.path.join(original_envs_folder, "split.yaml"), 'r') as file:
            yaml_info = yaml.safe_load(file)
        for elem in yaml_info['validation_set']:
            test_ids.append(str(elem['id']))
    elif args.load_custom_train_envs:
        original_envs_folder = os.path.join('..', 'data', 'env_samples', args.load_custom_train_envs)
        with open(os.path.join(original_envs_folder, "split.yaml"), 'r') as file:
            yaml_info = yaml.safe_load(file)
        for elem in yaml_info['training_set']:
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
    else:
        train_config['static_defender_agent'] = None # if not desired during testing remove from train_config to avoid errors

    if args.load_custom_test_envs or args.load_custom_envs or args.load_custom_val_envs or args.load_custom_train_envs: # reload right elements from the custom folder
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
                if args.environment_type == 'global':
                    env = wrap_graphs_to_global_envs(network, logger=logger, verbose=args.verbose, **train_config)
                elif args.environment_type == "local":
                    env = wrap_graphs_to_local_envs(network, logger, **train_config)
                else: #if args.environment_type == "continuous":
                    env = wrap_graphs_to_compressed_envs(network, logger, **train_config)
                    env.set_graph_encoder(graph_encoder)
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
    parser.add_argument('-f', '--logs_folder', required=True, help='Path to the specific logs folder with runs')
    parser.add_argument('--run_set', default="all", type=str, help='Run folder name to gather the correct run metrics (or "all" for all periodically)')
    parser.add_argument('--algorithm', choices=['ppo', 'a2c', 'rppo', 'trpo', 'ddpg', 'sac', 'td3', 'tqc'], default='trpo', help='Algorithm to use ')
    parser.add_argument('--environment_type', type=str, choices=['continuous', 'local', 'global'], default='continuous',
                        help='Type of environment to be used for training')  # to be extended in the future to LOCAL or DISCRETE or others
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
    parser.add_argument('--option', default='actions_distribution', choices=['random_agent_performances', 'actions_distribution', 'agent_performances', 'save_trajectories'], help='Decide which statistics to plot')
    parser.add_argument('--no_random', default=False, action="store_true",
                        help='Avoid calculation of average performances for the random agent')
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
    if args.no_random and args.option == "random_agent_performances":
        raise ValueError("ERROR: Cannot avoid random agent performances for the random agent option...")
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
    if test_config['option'] == "random_agent_performances":
        if isinstance(test_config['run_set'], list):
            for run in test_config['run_set']:
                if run == "test":
                    continue
                if args.verbose:
                    logger.info("Run ID: %s", run)
                test_run_config = copy.deepcopy(test_config)
                test_run_config['run_id'] = run
                run_folder = os.path.join(args.logs_folder, run)
                test_run_config['run_folder'] = run_folder
                train_config_file = os.path.join(run_folder, 'train_config.yaml')
                with open(train_config_file, 'r') as train_config_file:
                    train_config = yaml.safe_load(train_config_file)
                test_envs, envs_folder = load_test_envs(run_folder, args, train_config, test_config, logger=logger, stable_baselines=False)
                if not os.path.exists(os.path.join(str(run_folder), "test", test_run_config['test_folder'], "random")):
                    os.makedirs(os.path.join(str(run_folder), "test", test_run_config['test_folder'], "random"))
                df = calculate_random_agent_average_performances(test_envs, test_run_config['proportional_cutoff_coefficient'], logger, test_run_config['num_episodes_per_checkpoint'], args.verbose)
                current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                df.to_csv(os.path.join(str(run_folder), "test", test_run_config['test_folder'], "random",
                                       f"random_performances_{test_run_config['proportional_cutoff_coefficient']}_{test_run_config['num_episodes_per_checkpoint']}_{current_time}.csv"), index=False)
                if args.load_custom_test_envs or args.load_custom_envs or args.load_custom_val_envs or args.load_custom_train_envs:
                    remove_folder_and_files(envs_folder)
        else:
            if args.verbose:
                logger.info("Run ID: %s", test_config['run_set'])
            run_folder = os.path.join(args.logs_folder, test_config['run_set'])
            train_config_file = os.path.join(str(run_folder), 'train_config.yaml')
            with open(train_config_file, 'r') as train_config_file:
                train_config = yaml.safe_load(train_config_file)
            test_envs, envs_folder = load_test_envs(run_folder, args, train_config, test_config, logger=logger,stable_baselines=False)
            test_config['run_id'] = test_config['run_set']
            if not os.path.exists(os.path.join(str(run_folder), "test", "random", test_config['test_folder'])):
                os.makedirs(os.path.join(str(run_folder), "test", "random", test_config['test_folder']))
            df = calculate_random_agent_average_performances(test_envs, test_config['proportional_cutoff_coefficient'],
                                                             logger, test_config['num_episodes_per_checkpoint'],
                                                             args.verbose)
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            df.to_csv(os.path.join(str(run_folder), "test", "random", test_config['test_folder'],
                                   f"random_performances_{test_config['proportional_cutoff_coefficient']}_{test_config['num_episodes_per_checkpoint']}_{current_time}.csv"),
                      index=False)
            if args.load_custom_test_envs or args.load_custom_envs or args.load_custom_val_envs or args.load_custom_train_envs:
                remove_folder_and_files(envs_folder)
    else: # general case of checkpoints and not random
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
                outcomes = parse_option(test_run_config, test_envs, logger, args.verbose)
                runs_outcomes.append(outcomes) # Potentially use runs_outcomes to average the outcomes in the future
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
            _ = parse_option(test_config, test_envs, logger, args.verbose)
            if args.load_custom_test_envs or args.load_custom_envs or args.load_custom_val_envs or args.load_custom_train_envs:
                remove_folder_and_files(envs_folder)

if __name__ == "__main__":
    main()
