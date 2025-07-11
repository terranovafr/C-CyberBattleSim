# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    train_gae.py
    This file contains the module used to train the GAE model.
    The model is trained to compress the node and edge feature vectors of a graph representing a cyber environment
    into embeddings of a proper dimensionality. The model is trained using evolving representations of different
    graphs representing the environment.
"""

import argparse
import torch
from torch_geometric.utils import from_networkx
import pickle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import numpy as np
import sys
from torch_geometric.data import DataLoader
from pathlib import Path
from tqdm import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.utils.envs_utils import wrap_graphs_to_gae_envs # noqa: E402
from cyberbattle.utils.log_utils import setup_logging # noqa: E402
from gae_utils import compute_backward_batch_train_loss # noqa: E402
from model import GAE # noqa: E402
from gae_utils import validate # noqa: E402
from cyberbattle.utils.math_utils import set_seeds # noqa: E402
from cyberbattle._env.cyberbattle_env_switch_gae import RandomSwitchGAEEnv # noqa: E402
from cyberbattle.utils.file_utils import save_yaml, read_split_file, load_yaml # noqa: E402

script_dir = Path(__file__).parent
torch.set_default_dtype(torch.float32)

# Function used to load a folder where the environments are stored and a sample environment to load information about encodings' sizes
def load_envs(config, envs_folder, nlp_extractor, logger):
    sample_env = None
    env = None
    if config['load_envs']: # Added with the idea of adding new modes in the future, like random generation live etc.
        # Networks relative to env_samples
        file_path = os.path.join(script_dir, '..', 'data', 'env_samples', config['load_envs'])
        # wrap in the environments
        config['num_environments'] = 0
        for index, folder in tqdm(enumerate(os.listdir(file_path)), desc="Loading networks"):
            if os.path.isdir(os.path.join(file_path, folder)) and folder.isdigit():
                config['num_environments'] += 1
                if config['pca_components'] == config['default_vulnerability_embeddings_size']: # no reduction
                    network_folder = os.path.join(file_path, folder, f"network_{nlp_extractor}.pkl")
                else:
                    network_folder = os.path.join(file_path, folder, "pca", "num_components="+str(config['pca_components']),
                                                  f"network_{nlp_extractor}.pkl")
                    assert os.path.exists(network_folder), f"Network folder {network_folder} does not exist, ensure that the dimensionality reduction has been applied with the number of components specified"
                with open(network_folder, 'rb') as f:
                    network = pickle.load(f)
                env = wrap_graphs_to_gae_envs(network, logger=logger, **config)
                with open(os.path.join(envs_folder, f"{folder}.pkl"), 'wb') as f:
                    pickle.dump(env, f)
        sample_env = env # last one as sample
    assert sample_env is not None, "No environments loaded, please check the path or the nlp extractor specified, ensuring it is used also during environment generation"
    return sample_env

# Initialize model and optimizer, with model initialized with the graph from the sample environment (determining feature vector attributes)
def init_model(sample_env, config):
    sample_env.reset()
    graph = sample_env.evolving_visible_graph
    indices_size = len(from_networkx(graph).x[0])
    config['continuous_indices'] = [ i for i in range(indices_size - sample_env.continuous_indices, indices_size-1)]
    config['binary_indices'] = [ i for i in range(sample_env.binary_indices)]
    config['multi_class_info'] = sample_env.multi_class_info
    config['node_feature_vector_size'] = sample_env.node_feature_vector_size
    config['edge_feature_vector_size'] = sample_env.edge_feature_vector_size
    model = GAE(config['node_feature_vector_size'], config['model_config']['layers'], config['edge_feature_vector_size'], config['binary_indices'], config['multi_class_info'], config['continuous_indices'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    return model, optimizer, config

# Holdout setup for training and validation environments
def setup_holdout(config, envs_folder, logs_folder):
    train_ids, val_ids = read_split_file(config['holdout'], os.path.join(script_dir, '..', 'data', 'env_samples', config['load_envs']), logs_folder, config['num_environments'])
    if len(train_ids) == 0:
        raise ValueError("No training environments found in the holdout split. Please check your split file.")
    train_envs = RandomSwitchGAEEnv(envs_ids=train_ids, switch_interval=config['switch_interval'],
                                     envs_folder=envs_folder, verbose=config['verbose'])
    if config['holdout']:
        if len(val_ids) == 0:
            raise ValueError("No validation environments found in the holdout split. Please check your split file.")
        val_envs = RandomSwitchGAEEnv(envs_ids=val_ids, switch_interval=config['val_switch_interval'],
                                       envs_folder=envs_folder, verbose=config['verbose'])
    else:
        val_envs = None
    return train_envs, val_envs

# Training and evaluation function for the model
def train_and_eval(config, train_envs, val_envs, model, optimizer, writer, logger):
    batch = []
    done = True
    total_loss = None
    average_val_loss = None
    for iteration in range(config['train_iterations']): # Number of different graph configurations to which the GAE is exposed
        if done:
            train_envs.reset()
        G = train_envs.get_evolving_visible_graph()
        # Sample a valid action and use it
        source_node, target_node, vulnerability_ID, outcome_desired = train_envs.sample_valid_action()
        done = train_envs.step_graph(source_node, target_node, vulnerability_ID, outcome_desired)
        # the graph will evolve to a new configuration with a valid action
        data = from_networkx(G)
        if 'vulnerabilities_embeddings' not in data: # case of no edge yet, simulate edge with empty embedding
            data.vulnerabilities_embeddings = torch.zeros(data.num_edges, config['edge_feature_vector_size'])
        batch.append(data)
        if len(batch) % config['batch_size'] == 0:
            batch_loader = DataLoader(batch, batch_size=config['batch_size'], shuffle=True)

            for batch_data in batch_loader:  # type: ignore
                total_loss, adj_loss, feature_loss, edge_feature_loss, diversity_loss, binary_cat_loss, multi_cat_loss, cont_loss = compute_backward_batch_train_loss(
                    model, batch_data, optimizer, **config)
                if total_loss is None:
                    done = True # graph blocked
                    train_envs.done = True
                    break
                if config['verbose']:
                    logger.info(f"Training iteration {iteration} - Total loss: {total_loss} Adj loss: {adj_loss} Node feature vector loss: {feature_loss} Edge feature vector loss: {edge_feature_loss} Diversity loss: {diversity_loss} Node feature vector binary cat loss: {binary_cat_loss} Node feature vector multi cat loss: {multi_cat_loss} Node feature vector cont loss: {cont_loss}")
                writer.add_scalar('train/total_loss', total_loss, iteration)
                if config['weights']['adj_weight'] > 0:
                    writer.add_scalar('train/adj_loss', adj_loss, iteration)
                if config['weights']['node_feature_vector_weight'] > 0:
                    writer.add_scalar('train/node_feature_vector_loss', feature_loss, iteration)
                if config['weights']['edge_feature_vector_weight'] > 0:
                    writer.add_scalar('train/edge_feature_vector_loss', edge_feature_loss, iteration)
                if config['weights']['diversity_weight'] > 0:
                    writer.add_scalar('train/diversity_loss', diversity_loss, iteration)
                if config['weights']['node_feature_vector_binary_cat_weight'] > 0:
                    writer.add_scalar('train/node_feature_vector/binary_cat_loss', binary_cat_loss, iteration)
                if config['weights']['node_feature_vector_multi_cat_weight'] > 0:
                    writer.add_scalar('train/node_feature_vector/multi_cat_loss', multi_cat_loss, iteration)
                if config['weights']['node_feature_vector_cont_weight'] > 0:
                    writer.add_scalar('train/node_feature_vector/cont_loss', cont_loss, iteration)
                batch = [] # TODO: ensure if emptying the batch is the best strategy
                break  # take only one batch

        # validation if holdout is set
        if val_envs and iteration % config['val_interval'] == 0:
            if config['verbose']:
                logger.info(
                    f"Validation phase at iteration {iteration}")
            average_val_loss, average_adj_loss, average_feature_loss, average_edge_feature_loss, average_diversity_loss, average_binary_cat_loss, average_multi_cat_loss, average_cont_loss = validate(model, val_envs, writer, config, iteration)
            if config['verbose']:
                logger.info(f"Validation - Average val loss: {average_val_loss} Average adj loss: {average_adj_loss} Average node feature vector loss: {average_feature_loss} Average edge feature vector loss: {average_edge_feature_loss} Average diversity loss: {average_diversity_loss} Average node feature vector binary cat loss: {average_binary_cat_loss} Average node feature vector multi cat loss: {average_multi_cat_loss} Average node feature vector cont loss: {average_cont_loss}")

    if not val_envs:
        score = total_loss
    else:
        score = average_val_loss

    return model, score

# Execute several runs of model training
def execute_runs(config, original_logs_folder, envs_folder, nlp_extractor, logger):
    scores = []
    for run in range(config['num_runs']):
        if config['num_runs'] == 1:
            logs_folder = os.path.join(original_logs_folder, nlp_extractor)
        else:
            logs_folder = os.path.join(original_logs_folder, nlp_extractor, f"run_{run+1}",)
        os.makedirs(logs_folder, exist_ok=True)
        writer = SummaryWriter(str(logs_folder))
        set_seeds(config['seeds'][run])
        sample_env = load_envs(config, envs_folder, nlp_extractor, logger)
        model, optimizer, config = init_model(sample_env, config)

        # Saving configuration yaml file with all information related to the training
        filename = os.path.join(str(logs_folder), "model_spec.yaml")
        indices_keys = ['continuous_indices', 'binary_indices', 'multi_class_info', 'node_feature_vector_size', 'edge_feature_vector_size']
        indices_config = {key: config[key] for key in indices_keys}
        save_yaml(indices_config, logs_folder, filename)
        copy_config = config.copy()
        for key in indices_keys:
            copy_config.pop(key, None)
        filename = os.path.join(str(logs_folder), "train_config_encoder.yaml")
        save_yaml(copy_config, logs_folder, filename)
        if config['verbose']:
            logger.info("Setting up holdout environments")

        train_envs, val_envs = setup_holdout(config, envs_folder, logs_folder)
        model, score = train_and_eval(config, train_envs, val_envs, model, optimizer, writer, logger)

        writer.close()
        torch.save(model.encoder.state_dict(), str(os.path.join(str(logs_folder), 'encoder.pth')))
        torch.save(model, str(os.path.join(str(logs_folder), 'model.pth')))
        scores.append(score) # average val loss OR train loss if not holdout
    return np.mean(scores)

def main(config, logs_folder, envs_folder, logger):
    scores_nlps = []
    # For every NLP extractor, train the model and take the mean performance
    for nlp_extractor in config['nlp_extractors']:
        # Adjusting the weights for the different NLP extractors, since their embeddings will have different ranges
        if config['verbose']:
            logger.info("Training run with NLP extractor %s", nlp_extractor)
        for scaler in config['nlp_extractors_scalers']:
            if nlp_extractor == scaler:
                for metric in config['nlp_extractors_scalers'][scaler]:
                    if config['verbose']:
                        logger.info("Rescaling weights for metric %s for nlp extractor %s", metric, nlp_extractor)
                    config['weights'][metric] = config['weights'][metric] / config['nlp_extractors_scalers'][scaler][metric]
        nlp_envs_folder = os.path.join(envs_folder, nlp_extractor)
        os.makedirs(nlp_envs_folder, exist_ok=True)
        score = execute_runs(config, logs_folder, nlp_envs_folder, nlp_extractor, logger)
        scores_nlps.append(score)
        if config['verbose']:
            logger.info("Final average validation loss for NLP extractor: %s %f", nlp_extractor, score)
    return np.mean(np.array(scores_nlps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a GNN Autoencoder')
    parser.add_argument('--train_config', type=str, default=os.path.join('config', 'train_config.yaml'), help='Path to the configuration YAML file')
    parser.add_argument('--name', default=False, help='Name of the logs folder related to the run')
    parser.add_argument('--static_seeds', action='store_true', default=False, help='Use a static seed for training')
    parser.add_argument('--load_seeds', default="config",
                        help='Path of the folder where the seeds.yaml should be loaded from (e.g. previous experiment)')
    parser.add_argument('--random_seeds', action='store_true', default=False, help='Use random seeds for training')
    parser.add_argument('--yaml', default=False, help='Read configuration file from YAML file of a previous training')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs to perform')
    parser.add_argument('--holdout', default=False, action="store_true", help='Switch between graphs')
    parser.add_argument('--load_envs', default=False, type=str, help='Path to the .pkl file containing the graph')
    parser.add_argument('--no_save_log_file', action='store_false', dest='save_log_file',
                        default=True, help='Disable logging to file; log only to terminal')
    parser.add_argument('-v', '--verbose', default=2, type=int, help='Verbose level: 0 - no output, 1 - training/validation information, 2 - episode level information, 3 - iteration level information')
    parser.add_argument('-nlp', '--nlp_extractors', default=['distilbert', 'roberta', 'CySecBERT', 'SecBERT', 'SecRoBERTa', 'SecureBERT', 'gpt2', 'bert'], nargs='+', type=str,
                        help='Name of the NLP extractor(s) to be used for generating embeddings to compress. If multiple are specified, they will be used separately and results will be merged.')
    parser.add_argument('-pca', '--pca_components', default=None, type=int, help='Invoke with the use of PCA for the feature vectors')
    args = parser.parse_args()

    general_config = load_yaml(os.path.join(script_dir, "..", "..", "config.yaml"))
    if not args.load_envs:
        args.load_envs = general_config['default_environments_path']

    # Creating logs folder
    if args.name:
        logs_folder = os.path.join(script_dir, 'logs/', args.name + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        logs_folder = os.path.join(script_dir, 'logs/', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(logs_folder, exist_ok=True)  # Ensure logs folder exists
    envs_folder = os.path.join(logs_folder, "envs") # folder where environments will be stored and copied

    logger = setup_logging(logs_folder, log_to_file=args.save_log_file)

    if args.verbose:
        logger.info("Logs folder: %s", os.path.basename(logs_folder.rstrip('/')))

    # Potential seed setting before the generation of environment
    if args.static_seeds:
        if args.verbose:
            logger.info("Setting static seeds for each run of training")
        seeds_runs = [42 for _ in range(args.num_runs)]
    elif args.random_seeds:
        if args.verbose:
            logger.info("Setting random seeds for training")
        seeds_runs = [np.random.randint(1000) for _ in range(args.num_runs)]
    else: # args.load_seeds:
        if args.verbose:
            logger.info("Loading seeds from seeds file %s", args.load_seeds)
        seeds_loaded = load_yaml(os.path.join(args.load_seeds, 'seeds.yaml'))
        seeds_runs = seeds_loaded['seeds'][:args.num_runs]

    if args.yaml:
        if args.verbose:
            logger.info("Reading YAML file from a previous training for reproducibility: %s", args.yaml)
        # Read eventually YAML configuration file of a previous training replacing all default config and args
        config = load_yaml(os.path.join(script_dir, "logs", args.yaml, "train_config_encoder.yaml"))
        config.update({"yaml": args.yaml})
        config.update({"verbose": args.verbose})
        config.update({"save_log_file": args.save_log_file})
    else:
        # Read default YAML configuration files and merge
        config = load_yaml(os.path.join(script_dir,args.train_config))
        # not updated if loaded configuration file from previous experiment
        config.update({"seeds": seeds_runs})
        config.update(vars(args))
        if not args.pca_components:
            config['pca_components'] = config['default_vulnerability_embeddings_size']
        else:
            config['pca_components'] = args.pca_components

    main(config, logs_folder, envs_folder, logger)
