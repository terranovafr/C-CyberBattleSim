# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    envs_utils.py
    This file contains the utilities used to wrap and save environments across their stages (networkx, model, environment), as well as other utilities
"""

import pickle
import sys
import os
import copy
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
import cyberbattle._env.cyberbattle_env_gae as cyberbattle_env_gae # noqa: E402
import cyberbattle._env.cyberbattle_env_compressed as cyberbattle_env_compressed # noqa: E402
import cyberbattle._env.cyberbattle_env_global as cyberbattle_env_global # noqa: E402
import cyberbattle._env.cyberbattle_env_local as cyberbattle_env_local # noqa: E402


# Wrap 1 or more graphs to the environment using compressed environment
def wrap_graphs_to_compressed_envs(nets, logger, **kwargs):
    if isinstance(nets, list):
        envs = []
        for net in nets:
            cyber_env = cyberbattle_env_compressed.CyberBattleCompressedEnv(initial_environment=net, logger=logger,**kwargs)
            envs.append(cyber_env)
        return envs
    else:
        cyber_env = cyberbattle_env_compressed.CyberBattleCompressedEnv(initial_environment=nets, logger=logger, **kwargs)
        return cyber_env

# Wrap 1 or more graphs to the environment using global environment
def wrap_graphs_to_global_envs(nets, logger, **kwargs):
    if isinstance(nets, list):
        envs = []
        for net in nets:
            cyber_env = cyberbattle_env_global.CyberBattleGlobalEnv(initial_environment=net, logger=logger,**kwargs)
            envs.append(cyber_env)
        return envs
    else:
        cyber_env = cyberbattle_env_global.CyberBattleGlobalEnv(initial_environment=nets, logger=logger, **kwargs)
        return cyber_env

# Wrap 1 or more graphs to the environment using local environment
def wrap_graphs_to_local_envs(nets, logger, **kwargs):
    if isinstance(nets, list):
        envs = []
        for net in nets:
            cyber_env = cyberbattle_env_local.CyberBattleLocalEnv(initial_environment=net, logger=logger,**kwargs)
            envs.append(cyber_env)
        return envs
    else:
        cyber_env = cyberbattle_env_local.CyberBattleLocalEnv(initial_environment=nets, logger=logger, **kwargs)
        return cyber_env

# Wrap 1 or more graphs to the environment using world model environment
def wrap_graphs_to_gae_envs(nets, **kwargs):
    if isinstance(nets, list):
        envs = []
        for net in nets:
            cyber_env = cyberbattle_env_gae.CyberBattleGAEEnv(initial_environment=net,**kwargs)
            envs.append(cyber_env)

        return envs
    else:
        cyber_env = cyberbattle_env_gae.CyberBattleGAEEnv(initial_environment=nets, **kwargs)
        return cyber_env

# Function to save networkx graphs
def save_network_or_model_as_pickle(data, folder_path, name=None):
    os.makedirs(folder_path, exist_ok=True)
    if name:
        file_path = os.path.join(folder_path, 'network_'+name+'.pkl')
    else:
        file_path = os.path.join(folder_path, 'network.pkl')
    with open(file_path, 'wb') as networks_file:
        pickle.dump(data, networks_file)

# Save models as pickle after updating the feature vectors using a specific nlp extractor in order to be used after
# Saved in different files in order to ensure modularity and avoid very heavy file
def save_model_nlp_extractors_versions(env, nlp_extractor_models, folder_path, graph_id):
    os.makedirs(folder_path, exist_ok=True)
    for nlp_extractor in nlp_extractor_models:
        env_copy = copy.deepcopy(env)
        env_copy.update_feature_vectors(nlp_extractor)
        save_network_or_model_as_pickle(env_copy, os.path.join(folder_path, str(graph_id)), nlp_extractor)
