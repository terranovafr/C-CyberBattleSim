# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    cyberbattle_env_global.py
    Class containing the sub-class of the CyberBattleEnv with the global discrete environment.
    This environment represents the discrete global view of the CyberBattle simulation,
    where the observation is a vector containing the concatenated feature vectors of all nodes in the graph.
    The action space is the cartesian product of discrete choices combining all the options (source node, target node, vulnerability, outcome).
"""

import time
from typing import Dict
import networkx as nx
import numpy as np
import sys
import os
from typing import TypedDict
from gym import spaces
import numpy
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle._env.cyberbattle_env import CyberBattleEnv # noqa: E402
from cyberbattle.simulation import model # noqa: E402
from cyberbattle.utils.encoding_utils import map_outcome_to_string # noqa: E402
from cyberbattle.utils.data_utils import flatten_dict_with_arrays, flatten # noqa: E402


# Format of the info dict returned by the step function
StepInfo = TypedDict(
    'StepInfo', {
        'description': str,
        'duration_in_ms': float,
        'step_count': int,
        'network_availability': float,
        'source_node': str,
        'target_node': str,
        'source_node_tag': str,
        'target_node_tag': str,
        'vulnerability': str,
        'vulnerability_type': str,
        'outcome': str,
        'outcome_class': model.VulnerabilityOutcome,
        'end_episode_reason': int,
    })


class CyberBattleGlobalEnv(CyberBattleEnv):
    """OpenAI Gym environment interface to the CyberBattle simulation.

    # Observation
        Graph global vector obtained concatenating the feature vectors of the nodes in the graph

    # Actions
        Source node x target node x vulnerability x outcome as discrete choice
    """

    @property
    def name(self) -> str:
        return "CyberBattleGlobalEnv"

    def __init__(self,
                 pca_components = 768, # used to determine the feature vector size of each node
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.env_type = "global"
        # build global discrete action space with all possible actions
        self.vulnerability_list = self.get_vulnerabilities_list()
        self.num_vulnerabilities = len(self.vulnerability_list)
        self.nodes_dict = self.get_nodes_dict()
        self.flattened_action_space = []
        for node in self.nodes_dict:
            for (target_node, vulnerability, outcomes) in self.vulnerability_list:
                    for outcome in outcomes:
                        self.flattened_action_space.append((node, target_node, vulnerability, outcome))
        self.action_space = spaces.Discrete(len(self.flattened_action_space))

        if self.verbose > 1:
            self.logger.info("Action space: " + str(self.action_space))

        # Done only once at the beginning to create the vulnerability embeddings
        self.create_vulnerabilities_embeddings()
        self.vulnerability_embeddings_dimensions = pca_components
        self.node_feature_vector_size = self.get_node_feature_vector(self.starter_node) # just to initialize the feature vector

        # Create the observation space proportional to the number of nodes and the size of the node feature vector (graph size specific)
        self.observation_space = spaces.Dict({"graph": spaces.Box(low=-100, high=100, shape=(self.num_nodes * len(self.node_feature_vector_size),), dtype=np.float64)})
        if self.verbose > 1:
            self.logger.info("Observation space: " + str(self.observation_space))

    # Store for each target node the list of vulnerabilities and the outcomes valid
    def get_vulnerabilities_list(self):
        self.vulnerabilities_list = []
        for node in self.get_nodes():
            node_info = self.get_node(node)
            for vulnerability in node_info.vulnerabilities:
                outcomes = []
                for result in node_info.vulnerabilities[vulnerability].results:
                    outcomes.append(result.outcome)
                if vulnerability not in self.vulnerabilities_list:
                    self.vulnerabilities_list.append((node, vulnerability, outcomes))
        return self.vulnerabilities_list

    # Create a dictionary mapping each node to the index in the flattened action space
    def get_nodes_dict(self):
        self.nodes_dict = {}
        for node in self.get_nodes():
            if node not in self.nodes_dict:
                self.nodes_dict[node] = len(self.nodes_dict)
        return self.nodes_dict

    # Reset function calling the base environment reset and determining initial observation vector
    def reset(self, **kwargs):
        super().reset_env()
        self.reset_evolving_visible_graph()
        self.observation = [0 for _ in range(self.num_nodes * len(self.node_feature_vector_size))]
        for index, node in enumerate(self.evolving_visible_graph.nodes):
            x = self.evolving_visible_graph.nodes[node]['x']
            self.observation[index * len(x): (index + 1) * len(x)] = x
        return {"graph": numpy.array(self.observation, dtype=numpy.float32)}

    # Create the embeddings for the vulnerabilities, used to create the feature vector of the nodes
    def create_vulnerabilities_embeddings(self):
        self.vulnerabilities_embeddings = {}
        for node in self.environment.nodes:
            for vulnerability_ID in self.get_node(node).vulnerabilities:
                self.vulnerabilities_embeddings[vulnerability_ID] = self.get_node(node).vulnerabilities[
                    vulnerability_ID].embedding

    # Reset the evolving visible graph, clearing it and adding the initial node
    def reset_evolving_visible_graph(self):
        self.evolving_visible_graph = nx.DiGraph()
        self.evolving_visible_graph.clear()
        self.add_node_evolving_visible_graph(self.starter_node) # initial node

    # Get the feature vector of a node, flattening it to a numpy array
    def get_node_feature_vector(self, node_id):
        node_features_dict = self.convert_node_info_to_observation(self.get_node(node_id))
        flattened_node_features_dict = flatten_dict_with_arrays(node_features_dict)
        node_features_array = numpy.array(
            flatten([flattened_node_features_dict[key] for key in flattened_node_features_dict]), dtype=numpy.float32)
        return node_features_array

    # Add a node to the evolving visible graph, initializing its feature vector
    def add_node_evolving_visible_graph(self, node_id):
        self.evolving_visible_graph.add_node(node_id, x=self.get_node_feature_vector(node_id))

    # Update the feature vector of a node in the evolving visible graph
    def update_node_evolving_visible_graph(self, node_id):
        self.evolving_visible_graph.nodes[node_id].update({'x': self.get_node_feature_vector(node_id)})

    # Create the feature vector of nodes encoding properly all elements
    def convert_node_info_to_observation(self, node_info) -> Dict:
        firewall_config_array = [
            0 for _ in range(2 * self.max_services_per_node)
        ]

        if node_info.visible:
            # include firewall information if visibility acquired on the node
            for config in node_info.firewall.incoming:
                permission = config.permission.value
                if self.get_service_index(config.port, node_info) != -1:
                    firewall_config_array[self.get_service_index(config.port, node_info)] = permission
            for config in node_info.firewall.outgoing:
                permission = config.permission.value
                if self.get_service_index(config.port, node_info) != -1:
                    firewall_config_array[self.max_services_per_node + self.get_service_index(config.port,
                                                                                                           node_info)] = permission
        # include listening services information
        listening_services_running_array = [0 for _ in range(
            self.max_services_per_node)]  # array indicating if each service is listening or not
        listening_services_fv_array = [0.0 for _ in range(self.vulnerability_embeddings_dimensions)]

        if node_info.visible:
            # fill services info in case of visibility on the node
            for i, service in enumerate(node_info.services):
                feature_vector = service.feature_vector
                listening_services_running_array[i] = int(service.running)
                for j in range(self.vulnerability_embeddings_dimensions):
                    listening_services_fv_array[j] += feature_vector[j]
            if len(node_info.services) > 0:
                for i in range(self.vulnerability_embeddings_dimensions):
                    listening_services_fv_array[i] /= len(node_info.services)

        # include mean of vulnerabilities embeddings ( to have a single array independent of the number of vulnerabilities)
        mean_vulnerabilities_embedding = [0.0 for _ in range(self.vulnerability_embeddings_dimensions)]
        if len(node_info.vulnerabilities) > 0:
            for vulnerability in node_info.vulnerabilities:
                mean_vulnerabilities_embedding = [
                    x + y for x, y in
                    zip(mean_vulnerabilities_embedding, self.vulnerabilities_embeddings[vulnerability])
                ]
            mean_vulnerabilities_embedding = [embedding / len(node_info.vulnerabilities) for embedding in
                                              mean_vulnerabilities_embedding]

        return {
            'firewall_config_array': firewall_config_array,
            'listening_services_running_array': listening_services_running_array,
            'visible': int(node_info.visible),
            'persistence': int(node_info.persistence),
            'data_collected': int(node_info.data_collected),
            'data_exfiltrated': int(node_info.data_exfiltrated),
            'defense_evasion': int(node_info.defense_evasion),
            'reimageable': int(node_info.reimageable),
            'privilege_level': int(node_info.privilege_level),
            'status': node_info.status.value,
            'value': node_info.value,
            'sla_weight': node_info.sla_weight,
            'listening_services_fv_array': listening_services_fv_array,
            'mean_vulnerabilities_embedding': mean_vulnerabilities_embedding
        }

    # Wrapper function used in case it is required to call only the step onn the env without the compressed logic
    def step_env(self, source_node, target_node, vulnerability_ID, outcome):
        super().step_attacker_env(source_node, target_node, vulnerability_ID, outcome)
        return self.done or self.truncated

    # Step function that maps action index to discrete action and perform step in the environment
    def step(self, action_index):
        start_time = time.time()
        # map discrete choice to the action
        source_node, target_node, vulnerability_ID, outcome = self.calculate_discrete_action(action_index)
        super().step_attacker_env(source_node, target_node, vulnerability_ID, outcome)
        # eventually update the evolving visible graph
        self.update_evolving_visible_graph_after_step(target_node)

        if self.verbose > 2:
            self.logger.info("Reward of the step: %s", self.reward)
        info = StepInfo(
            description='CyberBattleEnvGlobal step info',
            duration_in_ms=time.time() - start_time,
            step_count=self.stepcount,
            source_node=source_node,
            target_node=target_node,
            source_node_tag= self.get_node(source_node).tag,
            target_node_tag= self.get_node(target_node).tag,
            vulnerability=vulnerability_ID,
            vulnerability_type=self.vulnerability_type,
            network_availability=self.network_availability,
            outcome_class=self.outcome,
            outcome=map_outcome_to_string(self.outcome),
            end_episode_reason=self.end_episode_reason,
        )
        # prepare the new observation vector
        self.observation = [0 for _ in range(self.num_nodes * len(self.node_feature_vector_size))]
        for index, node in enumerate(self.discovered_nodes):
            x = self.evolving_visible_graph.nodes[node]['x']
            self.observation[index * len(x): (index + 1) * len(x)] = x
        self.observation = {"graph": numpy.array(self.observation, dtype=numpy.float32)}
        return self.observation, self.reward, self.done or self.truncated, info

    # Update the graph that should be used as observation
    def update_evolving_visible_graph_after_step(self, target_node):
        for node in self.discovered_nodes:
            if node not in self.evolving_visible_graph.nodes():
                self.add_node_evolving_visible_graph(node)

        # If an action that should modify the node feature vectors is issued, modify the graph
        if (isinstance(self.outcome, model.Discovery) or isinstance(self.outcome, model.Collection) or isinstance(
                self.outcome, model.Persistence)
                or isinstance(self.outcome, model.PrivilegeEscalation) or isinstance(self.outcome,
                                                                                     model.Exfiltration) or isinstance(
                    self.outcome, model.DefenseEvasion)
                or isinstance(self.outcome, model.DenialOfService) or isinstance(self.outcome,
                                                                                 model.LateralMove) or isinstance(
                    self.outcome, model.CredentialAccess)):
            self.update_node_evolving_visible_graph(target_node)

    # Calculate the discrete action from the action index
    def calculate_discrete_action(self, action):
        source_node, target_node, vulnerability_ID, outcome_class = self.flattened_action_space[action]
        return source_node, target_node, vulnerability_ID, outcome_class

    # Sample a random action from the action space
    def sample_random_action(self):
        return self.action_space.sample()

    # Set the number of PCA components to use for the vulnerability embeddings
    def set_pca_components(self, pca_components, default_value=768):
        if not pca_components:
            self.vulnerability_embeddings_dimensions = default_value
        else:
            self.vulnerability_embeddings_dimensions = pca_components
