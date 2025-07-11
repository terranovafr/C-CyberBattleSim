# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    cyberbattle_env_compressed.py
    Class containing the sub-class of the CyberBattleEnv with the compressed environment.
    This environment is suited for graph and vulnerabilities invariant agents that are independent of the topology of application.
    The graph is compressed into an embedding by the GAE starting from node embeddings and the action space is a concatenation of embeddings:
    - source node embedding: GAE embedding
    - target node embedding: GAE embedding
    - vulnerability embedding: NLP extracted embedding
    - outcome embedding: one-hot encoding of the possible outcomes
    The action space is continuous and the closest action is selected based on a distance metric.
"""

import time
from typing import Dict
import networkx as nx
import numpy as np
import sys
import os
from collections import defaultdict
import copy
from typing import TypedDict
import torch
from gym import spaces
import numpy
from scipy.spatial import distance as distance_cosine
from torch_geometric.utils import from_networkx
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle._env.cyberbattle_env import CyberBattleEnv # noqa: E402
from cyberbattle.simulation.model import Collection, CredentialAccess, Discovery, Reconnaissance, DenialOfService, PrivilegeEscalation, Persistence, LateralMove, Exfiltration, \
    DefenseEvasion # noqa: E402
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
        'min_distance_action': float
    })


class CyberBattleCompressedEnv(CyberBattleEnv):
    """OpenAI Gym environment interface to the CyberBattle simulation.

    # Observation
        Graph embedding (and eventually node embedding of the node of interest if goal is node-specific) + number of discovered nodes + number of owned nodes

    # Actions
        Source node embedding x target node embedding x vulnerability language embedding x outcome one-hot embedding
    """

    @property
    def name(self) -> str:
        return "CyberBattleCompressedEnv"

    def __init__(self,
                 edge_feature_aggregations = None, # which aggregation(s) to use for the edge embeddings in the graph, determine also the dimension of the edge embeddings
                 graph_embeddings_aggregations = None, # which aggregation(s) to use for the node embeddings in the graph, determine also the dimension of the node embeddings
                 node_embeddings_dimensions=64, # dimension of the node embeddings, determined by the GAE Encoder output size
                 outcome_dimensions=9,  # number of possible outcomes
                 discrete_features=None, # additional features to be added to the observation space
                 # whether PCA reduction has been performed on the vulnerability embeddings during graph generation
                 pca_components=768,
                 distance_metric='cosine', # in the action space to determine the closest action
                 sample_subset_samples=False, # use a sample of actions in the action space at every timestep to reduce the number of points and hence the distance calculation time load
                 remove_all_obstacles=False, # flag that removes the "obstacles" for a given goal, e.g. all disruption actions for control games
                 remove_main_obstacles=False, # flag that removes the "main obstacles" for a given goal, e.g. all disruption actions that kill the starter node for control games
                 precise_action_space_positions=False, # flag to set whether each time node embeddings change, the action space should be updated (precise but time intensive)
                 precise_graph_encoding=False, # flag to set whether each time node embeddings change, the graph should be re-encoded (precise but time intensive)
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.env_type = "compressed"
        edge_feature_aggregations = edge_feature_aggregations or ["mean"]
        graph_embeddings_aggregations = graph_embeddings_aggregations or ["mean", "max", "min"]
        discrete_features = discrete_features or ["owned_nodes", "discovered_nodes"]
        # aggregations to be used to put embeddings on the edges
        self.edge_feature_aggregations = edge_feature_aggregations
        # aggregations to be used to put embeddings on the nodes
        self.graph_embeddings_aggregations = graph_embeddings_aggregations
        self.node_embeddings_dimensions = node_embeddings_dimensions  # default choice for GAE
        self.distance_metric = distance_metric
        self.outcome_dimensions = outcome_dimensions
        self.discrete_features = discrete_features
        # determine whether PCA was used in order to calculate the right dimensions
        self.vulnerability_embeddings_dimensions = pca_components
        self.sample_subset_samples = sample_subset_samples
        self.remove_all_obstacles = remove_all_obstacles
        self.remove_main_obstacles = remove_main_obstacles
        self.precise_action_space_positions = precise_action_space_positions
        self.precise_graph_encoding = precise_graph_encoding

        # action space is continuous within the cartesian product of the embeddings of the source node, target node, vulnerability, and outcome (one-hot embedding of 9 discrete outcomes)
        self.action_space = spaces.Box(low=-4, high=4, # using -4, +4 assuming the last layer ensure normalization and most points lie in the normal space
                                           shape=(self.node_embeddings_dimensions * 2 + self.vulnerability_embeddings_dimensions + self.outcome_dimensions,),
                                           dtype=numpy.float32)

        if self.verbose > 1:
            self.logger.info("Action space: " + str(self.action_space))

        if self.goal.endswith("node"):
            # if node-specific goal, add the interest node embedding to the observation space together with the graph embeddings
            graph_box_space = spaces.Box(
                    low=-16, high=16,
                    shape=(self.node_embeddings_dimensions * len(self.graph_embeddings_aggregations) + self.node_embeddings_dimensions,),
                    dtype=numpy.float64
                )
        else:
            # if not node-specific goal, only the graph embeddings are present in the observation space as continuous vector
            graph_box_space = spaces.Box(
                    low=-16, high=16,
                    shape=(self.node_embeddings_dimensions * len(self.graph_embeddings_aggregations),),
                    dtype=numpy.float64
                )

        # observation space is a dictionary with the graph embeddings and the discrete features desired among options available
        self.observation_space = spaces.Dict({
            "graph_embeddings": graph_box_space,
            "discrete_features": spaces.Box(
                low=0, high=300,
                shape=(len(self.discrete_features),),
                dtype=numpy.float64
            ),
        })

        if self.verbose > 1:
            self.logger.info("Observation space: " + str(self.observation_space))

        self.graph_encoder_time = 0
        self.action_calculation_time = 0
        self.action_space_creation_time = 0
        self.update_evolving_visible_graph_time = 0
        self.inner_step_time = 0
        self.balance_action_space_time = 0
        # embeddings created just once and re-use in order to avoid recalculation
        self.create_vulnerabilities_embeddings()
        self.create_vulnerabilities_embeddings_per_node_type()

    # Reset function calling the original reset and preparing the overlay graph to encode and continuous observation
    def reset(self, **kwargs):
        if self.verbose > 1:
            if self.graph_encoder_time != 0:
                self.logger.info(f"Graph embedding time in the episode: {self.graph_encoder_time}")
                self.logger.info(f"Action calculation time in the episode: {self.action_calculation_time}")
                self.logger.info(f"Action space creation time in the episode: {self.action_space_creation_time}")
                self.logger.info(f"Update evolving visible graph time in the episode: {self.update_evolving_visible_graph_time}")
                self.logger.info(f"Inner step time in the episode: {self.inner_step_time}")
        super().reset_env()
        # keep track of the processed pairs (source node, target node) for which actions have been added to the continuous action space to avoid to re-add them
        self.processed_pairs = set()
        self.reset_evolving_visible_graph()
        self.action_embeddings = {}
        self.exploited_vulnerabilities_per_node_pairs = {}
        self.graph_encoder_time = 0
        self.action_calculation_time = 0
        self.action_space_creation_time = 0
        self.update_evolving_visible_graph_time = 0
        self.balance_action_space_time = 0
        self.inner_step_time = 0
        start_time = time.time()
        self.node_embeddings, self.observation = self.encode(self.evolving_visible_graph) # at the beginning only source node
        self.graph_encoder_time += time.time() - start_time
        start_time = time.time()
        self.create_continuous_action_space() # at the beginning only local vulnerabilities of source node
        self.action_space_creation_time += time.time() - start_time
        self.edges = []
        self.observation = {
            "graph_embeddings": self.observation,
            "discrete_features": self.create_discrete_features()
        }
        return self.observation

    # Reset the evolving visible graph to the initial state with only the starter node
    def reset_evolving_visible_graph(self):
        self.evolving_visible_graph = nx.DiGraph()
        self.evolving_visible_graph.clear()
        self.add_node_evolving_visible_graph(self.starter_node) # initial node

    # Function to get the feature vector of a node flattened as an array
    def get_node_feature_vector(self, node_id):
        node_features_dict = self.convert_node_info_to_observation(self.get_node(node_id))
        flattened_node_features_dict = flatten_dict_with_arrays(node_features_dict)
        node_features_array = numpy.array(
            flatten([flattened_node_features_dict[key] for key in flattened_node_features_dict]), dtype=numpy.float32)
        return node_features_array

    # Function to add a node to the evolving visible graph with its feature vector
    def add_node_evolving_visible_graph(self, node_id):
        self.evolving_visible_graph.add_node(node_id, x=self.get_node_feature_vector(node_id))

    # Function to update the node in the evolving visible graph with its feature vector
    def update_node_evolving_visible_graph(self, node_id):
        self.evolving_visible_graph.nodes[node_id].update({'x': self.get_node_feature_vector(node_id)})

    # Function to add an edge to the evolving visible graph with the vulnerabilities embeddings
    def add_edge_evolving_visible_graph(self, source_node, target_node, vuln_key):
        aggregation_functions = {
            "mean": np.mean,
            "sum": np.sum
        }
        if source_node not in self.evolving_visible_graph.nodes():
            self.add_node_evolving_visible_graph(source_node)
        if target_node not in self.evolving_visible_graph.nodes():
            self.add_node_evolving_visible_graph(target_node)
        self.edges.append((source_node, target_node, vuln_key))
        if self.evolving_visible_graph.has_edge(source_node, target_node):
            # re-merge vulnerabilities with aggregators if a vulnerability is already present such that edge feature vector is aggregation of vulnerabilities exploited
            if not self.exploited_vulnerabilities_per_node_pairs.get(source_node).get(target_node):
                self.exploited_vulnerabilities_per_node_pairs[source_node][target_node] = []
            self.exploited_vulnerabilities_per_node_pairs[source_node][target_node].append(self.vulnerabilities_embeddings[vuln_key])
            edge_embedding = []
            for edge_aggregation in self.edge_feature_aggregations:
                edge_embedding.append(aggregation_functions[edge_aggregation](self.exploited_vulnerabilities_per_node_pairs[source_node][target_node], axis=0))
            self.evolving_visible_graph[source_node][target_node]["vulnerabilities_embeddings"] = np.concatenate(edge_embedding)
            return True
        else:
            # no edge already exists between two nodes hence create first edge
            self.evolving_visible_graph.add_edge(source_node, target_node)
            self.exploited_vulnerabilities_per_node_pairs[source_node] = {}
            self.exploited_vulnerabilities_per_node_pairs[source_node][target_node] = [self.vulnerabilities_embeddings[vuln_key]]
            edge_embedding = []
            for edge_aggregation in self.edge_feature_aggregations:
                edge_embedding.append(
                    aggregation_functions[edge_aggregation](self.exploited_vulnerabilities_per_node_pairs[source_node][target_node],
                                                            axis=0))
            self.evolving_visible_graph[source_node][target_node]["vulnerabilities_embeddings"] = np.concatenate(
                edge_embedding)
            return True

    # Function leveraging the graph parameter and the GAE encoder to encode the graph and gather node embeddings
    def encode(self, graph):
        # Use the GAE Encoder to encode the graph
        node_embeddings = {}
        if self.goal.endswith("node"): # node-specific goal game
            graph = copy.deepcopy(graph) # if goal node is not in the graph, add it in order to have a pure node embedding without mixing with the graph neighbors
            if self.interest_node not in graph.nodes():
                self.add_node_evolving_visible_graph(self.interest_node)
        data = from_networkx(graph)
        if 'vulnerabilities_embeddings' not in data: # case where the graph has no edges
            data.vulnerabilities_embeddings = torch.zeros(self.vulnerability_embeddings_dimensions, dtype=torch.float32)
        data.x = data.x.float()
        data.vulnerabilities_embeddings = data.vulnerabilities_embeddings.float()

        z = self.graph_encoder(data.x, data.edge_index, data.vulnerabilities_embeddings)

        running_nodes = [node for node in graph.nodes() if
                         self.get_node(node).status == model.MachineStatus.Running]

        if not running_nodes: # if no running nodes, return empty embeddings
            empty_embedding = np.zeros(self.node_embeddings_dimensions, dtype=np.float32)
            concatenated_result = np.concatenate([empty_embedding for _ in self.graph_embeddings_aggregations])
            if self.goal.endswith("node"):
                concatenated_result = np.concatenate([concatenated_result, empty_embedding])
            return node_embeddings, concatenated_result

        # Get the embeddings for the running nodes
        for node in running_nodes:
            node_index = list(graph.nodes()).index(node)
            node_embedding = z[node_index].detach().numpy()
            node_embeddings[node] = node_embedding

        embeddings_array = np.array([node_embeddings[node] for node in node_embeddings], dtype=np.float32)
        # perform aggregations across all node embeddings to get the graph embedding
        graph_embeddings = []
        for agg_type in self.graph_embeddings_aggregations:
            if agg_type == "mean":
                graph_embeddings.append(np.average(embeddings_array, axis=0))
            elif agg_type == "sum":
                graph_embeddings.append(np.sum(embeddings_array, axis=0))
            elif agg_type == "min":
                graph_embeddings.append(np.min(embeddings_array, axis=0))
            elif agg_type == "max":
                graph_embeddings.append(np.max(embeddings_array, axis=0))
            else:
                raise ValueError(f"Unknown aggregation type: {agg_type}")

        # if node-specific goal, add the interest node embedding to the observation vector
        observation_embedding = np.concatenate(graph_embeddings)
        if self.goal.endswith("node"):
            if self.interest_node in running_nodes:
                observation_embedding = np.concatenate([observation_embedding, node_embeddings[self.interest_node]])
            else:
                observation_embedding = np.concatenate([observation_embedding, np.zeros(self.node_embeddings_dimensions, dtype=np.float32)])
            if self.interest_node not in self.discovered_nodes and self.interest_node in node_embeddings: # remove if fictious interest node was added
                node_embeddings.pop(self.interest_node)
        return node_embeddings, observation_embedding

    # discrete features to be added to the observation vector to provide additional information to understand semantics of graph embedding
    def create_discrete_features(self):
        discrete_features = []
        # should be selected among these options
        if 'discovered_nodes' in self.discrete_features:
            discrete_features.append(len(self.discovered_nodes))
        if 'owned_nodes' in self.discrete_features:
            discrete_features.append(len(self.owned_nodes))
        return numpy.array(discrete_features)

    # Create the feature vector of nodes encoding properly all elements
    def convert_node_info_to_observation(self, node_info) -> Dict:
        firewall_config_array = [
            0 for _ in range(2 * self.max_services_per_node)
        ]

        if node_info.visible:
            # include firewall information if visibility acquired on the node
            for config in node_info.firewall.incoming:
                permission = config.permission.value
                if self.get_service_index(config.port, node_info) != -1 and self.get_service_index(config.port, node_info) < self.max_services_per_node:
                    firewall_config_array[self.get_service_index(config.port, node_info)] = permission
            for config in node_info.firewall.outgoing:
                permission = config.permission.value
                if self.get_service_index(config.port, node_info) != -1 and self.get_service_index(config.port, node_info) < self.max_services_per_node:
                    firewall_config_array[self.max_services_per_node + self.get_service_index(config.port,
                                                                                                           node_info)] = permission
        # include listening services information
        listening_services_running_array = [0 for _ in range(
            self.max_services_per_node)]  # array indicating if each service is listening or not
        listening_services_fv_array = [0.0 for _ in range(self.vulnerability_embeddings_dimensions)]

        if node_info.visible:
            # fill services info in case of visibility on the node
            for i, service in enumerate(node_info.services):
                if i >= self.max_services_per_node:
                    break
                feature_vector = service.feature_vector
                listening_services_running_array[i] = int(service.running)
                for j in range(self.vulnerability_embeddings_dimensions):
                    listening_services_fv_array[j] += feature_vector[j]
            if len(node_info.services) > 0:
                for i in range(self.vulnerability_embeddings_dimensions):
                    listening_services_fv_array[i] /= len(node_info.services)

        # include mean of vulnerabilities embeddings ( to have a single array independent of the number of vulnerabilities)
        # GAE requires all node feature vectors to be of the same size, hence we need to pool the vulnerabilities embeddings
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

    # Wrapper function used in case it is required to call only the step on the env without the compressed logic
    def step_env(self, source_node, target_node, vulnerability_ID, outcome):
        super().step_attacker_env(source_node, target_node, vulnerability_ID, outcome)
        return self.done or self.truncated

    # Step function that takes an action vector, calls the original step function, and update the graph
    # New observation and reward are computed and returned
    def step(self, action_vector):
        start_time = time.time()
        # find the closest action to the action vector
        source_node, target_node, vulnerability_ID, outcome, distance = self.find_closest_action_embedding(copy.deepcopy(action_vector))
        self.action_calculation_time += time.time() - start_time
        start_time = time.time()
        super().step_attacker_env(source_node, target_node, vulnerability_ID, outcome)
        self.inner_step_time += time.time() - start_time
        start_time = time.time()
        # eventually update the evolving visible graph
        self.update_evolving_visible_graph_after_step(source_node, target_node, vulnerability_ID)
        self.update_evolving_visible_graph_time += time.time() - start_time
        if self.action_changes_evolving_visible_graph(outcome) or self.static_defender_agent: # if the action changes the graph, re-encode the graph, or if the defender acted so we do not know what it has done
            if self.verbose > 2:
                if self.action_changes_evolving_visible_graph(outcome):
                    self.logger.info("Re-encoding the graph since there was one action that changed the graph")
                elif self.static_defender_agent:
                    self.logger.info("Re-encoding the graph since the defender may have been acted with modifying actions")

            start_time = time.time()
            # need to re-encode if graph has changed
            self.node_embeddings, self.observation = self.encode(self.evolving_visible_graph)
            self.graph_encoder_time += time.time() - start_time
            self.observation = {
                "graph_embeddings": self.observation,
                "discrete_features": self.create_discrete_features()
            }
            start_time = time.time()
            # potentialy add new points in the continuous action space
            if self.action_changes_evolving_visible_graph(outcome):
                if self.precise_action_space_positions:
                    self.create_continuous_action_space(nodes_to_recalculate=[source_node, target_node])
                else:
                    self.create_continuous_action_space() #nodes_to_recalculate=[source_node, target_node])
            elif self.static_defender_agent:
                if self.precise_action_space_positions:
                    self.create_continuous_action_space(nodes_to_recalculate=self.changed_nodes)
                else:
                    self.create_continuous_action_space()
            self.action_space_creation_time += time.time() - start_time
         # add term proportional to the distance (negative coefficient)
        self.reward += self.penalties_dict['distance_penalty'] * distance
        if self.verbose > 2:
            self.logger.info("Penalty (distance penalty) : += %s * %s", self.penalties_dict['distance_penalty'], distance)
        if self.verbose > 2:
            self.logger.info("Reward of the step: %s", self.reward)
        info = StepInfo(
            description='CyberBattleEnvCompressed step info',
            duration_in_ms=time.time() - start_time,
            step_count=self.stepcount,
            source_node=source_node,
            target_node=target_node,
            source_node_tag= self.get_node(source_node).tag,
            target_node_tag= self.get_node(target_node).tag,
            vulnerability=vulnerability_ID,
            vulnerability_type=self.vulnerability_type,
            network_availability=self.network_availability,
            outcome_class=outcome,
            outcome=map_outcome_to_string(outcome),
            end_episode_reason=self.end_episode_reason,
            min_distance_action=distance
        )
        return self.observation, self.reward, self.done or self.truncated, info

    # Function determining if a certain outcome changes the evolving visible graph
    def action_changes_evolving_visible_graph(self, outcome):
        if self.precise_graph_encoding:
            return not (isinstance(outcome, model.InvalidAction) or isinstance(outcome, model.NoVulnerability)
                    or isinstance(outcome, model.NoEnoughPrivilege) or isinstance(outcome, model.UnsuccessfulAction)
                    or isinstance(outcome, model.OutcomeNonPresent) or isinstance(outcome, model.NonListeningPort)
                    or isinstance(outcome, model.FirewallBlock) or isinstance(outcome, model.NoNeededAction)
                    or isinstance(outcome, model.RepeatedResult) or isinstance(outcome, model.NonRunningMachine))
        else:
            return isinstance(outcome, model.LateralMove) or isinstance(outcome, model.DenialOfService) or isinstance(outcome, model.Reconnaissance)

    # Function updating the evolving visible graph after a step, adding nodes and edges if needed
    def update_evolving_visible_graph_after_step(self, source_node, target_node, vulnerability_ID):
        # Update the graph that should turn into a graph embedding
        for node in self.discovered_nodes:
            if node not in self.evolving_visible_graph.nodes():
                self.add_node_evolving_visible_graph(node)

        # If an action that should modify the node feature vectors is issued, modify the graph embedding since the graph should change
        if (isinstance(self.outcome, model.Discovery) or isinstance(self.outcome, model.Collection) or isinstance(
                self.outcome, model.Persistence)
                or isinstance(self.outcome, model.PrivilegeEscalation) or isinstance(self.outcome,
                                                                                     model.Exfiltration) or isinstance(
                    self.outcome, model.DefenseEvasion)
                or isinstance(self.outcome, model.DenialOfService) or isinstance(self.outcome,
                                                                                 model.LateralMove) or isinstance(
                    self.outcome, model.CredentialAccess)):
            self.update_node_evolving_visible_graph(target_node)

        # Add edges to the evolving visible graph
        if self.reward > 0:
            self.add_edge_evolving_visible_graph(source_node, target_node, vulnerability_ID)

    # Create continuous action space with all the actions represented as embeddings
    def create_continuous_action_space(self, nodes_to_recalculate=None):
        self.outcome_counts = defaultdict(int)  # Track counts for each outcome
        self.outcome_embeddings = defaultdict(list)  # Store action keys per outcome

        running_owned_nodes = {node: self.node_embeddings[node] for node in self.owned_nodes
                               if self.get_node(node).status == model.MachineStatus.Running}
        running_discovered_nodes = {node: self.node_embeddings[node] for node in self.discovered_nodes
                                    if self.get_node(node).status == model.MachineStatus.Running}

        for source_node, source_node_embedding in running_owned_nodes.items():
            for target_node, target_node_embedding in running_discovered_nodes.items():
                # if the current action involved some nodes, their embeddings may have changed
                if nodes_to_recalculate:
                    # check if some node between source and target is connected to node to recalculate in the evolving visible graph
                    if not (any(nx.has_path(self.evolving_visible_graph, source_node, node) for node in nodes_to_recalculate) or
                            any(nx.has_path(self.evolving_visible_graph, target_node, node) for node in nodes_to_recalculate)):
                        # in case it is not this the case we can skip calculation if already processed
                        if (source_node, target_node) in self.processed_pairs:
                            continue  # Skip redundant processing
                else: # process all if not already processed
                    if (source_node, target_node) in self.processed_pairs:
                        continue  # Skip redundant processing
                if source_node == target_node:
                    self.__add_vulnerabilities_to_action_space(source_node, source_node_embedding, target_node,
                                                   target_node_embedding, "local")

                # # in case of local vulnerability, add also all the remote ones, assuming the attacker can use a personal device as source node
                self.__add_vulnerabilities_to_action_space(source_node, source_node_embedding, target_node, target_node_embedding,
                                              "remote")
                if (source_node, target_node) not in self.processed_pairs:
                    self.processed_pairs.add((source_node, target_node))

        # Subset of actions at every timestep in order to reduce the number of points in the action space and hence distance computation
        start_time = time.time()
        if self.sample_subset_samples:
            self.__balance_action_space_by_outcome()
        self.balance_action_space_time += time.time() - start_time

    # Function to add a specific set of vulnerabilities  given a current pair of nodes to the continuous action space
    def __add_vulnerabilities_to_action_space(self, source_node, source_node_embedding, target_node, target_node_embedding,
                                 vulnerability_type):

        for vulnerability in self.vulnerabilities_embeddings_per_node_type[target_node][vulnerability_type]:
            action_key = (source_node, target_node, vulnerability["vulnerability_ID"], vulnerability["outcome"])

            if source_node == target_node and isinstance(vulnerability["outcome"], model.LateralMove) or isinstance(vulnerability["outcome"], model.CredentialAccess):
                continue  # invalid case removed that can hold since type is read by CVSS vector and outcome is instead forecasted

            # remove all obstacles that could have negative outcome, embeddings our knowledge on the action space
            if self.remove_all_obstacles and self.goal in ["control", "discovery", "control_node", "discovery_node"]:
                if isinstance(vulnerability["outcome"], model.DenialOfService):
                    continue

            # remove the main obstacles that may stop the episode
            if self.remove_main_obstacles: # regardless of the goal
                if isinstance(vulnerability["outcome"], model.DenialOfService) and target_node == self.starter_node:
                    continue

            if self.remove_main_obstacles and self.goal.endswith("node") and self.goal != "disruption_node":
                if isinstance(vulnerability["outcome"], model.DenialOfService) and target_node == self.interest_node:
                    continue

            # overwrite if changed
            self.action_embeddings[action_key] = np.concatenate((source_node_embedding, target_node_embedding, vulnerability['embedding']))

    # Keep a susbet of actions per type in the action space for distance computation issues as approximated solution, trying to keep the balance between the different outcomes
    def __balance_action_space_by_outcome(self):
        outcome_counts = defaultdict(list)
        for action_key in self.action_embeddings:
            outcome = action_key[-1]
            outcome_counts[type(outcome)].append(action_key)
        reduced_embeddings = {}
        for outcome, actions in outcome_counts.items():
            if len(actions) > self.sample_subset_samples:
                actions_to_keep_indices = np.random.choice(len(actions), self.sample_subset_samples, replace=False)
                actions_to_keep = [actions[i] for i in actions_to_keep_indices]
            else:
                actions_to_keep = actions
            for action in actions_to_keep:
                reduced_embeddings[action] = self.action_embeddings[action]
        self.action_embeddings = reduced_embeddings

    # Function to find the closest action embedding to a given action vector using the specified distance metric
    def find_closest_action_embedding(self, action_vector, no_output=False):
        metric_mapping = {
            'l1': lambda x, y: np.linalg.norm(x - y, ord=1, axis=1),
            'l2': lambda x, y: np.linalg.norm(x - y, ord=2, axis=1),
            'inf': lambda x, y: np.linalg.norm(x - y, ord=np.inf, axis=1),
            'cosine': lambda x, y: distance_cosine.cdist(x, y, 'cosine').flatten()
        }

        if self.distance_metric not in metric_mapping:
            raise ValueError(f"Unsupported metric '{self.distance_metric}'. Use 'l1', 'l2', 'inf', or 'cosine'.")

        embeddings_array = np.array(list(self.action_embeddings.values()))
        vector_segment = np.atleast_2d(np.array(action_vector, dtype=np.float32))
        distances = metric_mapping[self.distance_metric](vector_segment, embeddings_array)
        min_index = np.argmin(distances)
        action, distance = list(self.action_embeddings.keys())[min_index], distances[min_index]
        closest_source_node_index, closest_target_node_index, vulnerability_index, outcome_type = action
        if self.verbose > 2 and not no_output:
            self.logger.info("Closest action -> source node: %s, target node: %s, vulnerability: %s, outcome: %s, distance: %s",
                             closest_source_node_index, closest_target_node_index, vulnerability_index, outcome_type, distance)
        return closest_source_node_index, closest_target_node_index, vulnerability_index, outcome_type, distance

    # Function to map the outcome to a one-hot encoding based on its type and vulnerability type
    def map_outcome_to_onehot(self, vulnerability_type, outcome):
        labels = {
            "local": [DenialOfService, Discovery, Collection, Exfiltration,
                      Reconnaissance, DefenseEvasion, Persistence,
                      PrivilegeEscalation],
            "remote": [DenialOfService, Discovery, Collection, Exfiltration,
                       Reconnaissance, DefenseEvasion, Persistence,
                       CredentialAccess, LateralMove]
        }
        if isinstance(outcome, model.Execution):
            return None
        if vulnerability_type not in labels:
            raise ValueError("Vulnerability type must be either 'local' or 'remote'.")
        if type(outcome) not in labels[vulnerability_type]:
            return None
        index = labels[vulnerability_type].index(type(outcome))
        one_hot = [0] * self.outcome_dimensions
        one_hot[index] = 1
        return one_hot

    # Function to create the vulnerabilities embeddings from the environment nodes
    def create_vulnerabilities_embeddings(self):
        self.vulnerabilities_embeddings = {}
        for node in self.environment.nodes:
            for vulnerability_ID in self.get_node(node).vulnerabilities:
                self.vulnerabilities_embeddings[vulnerability_ID] = self.get_node(node).vulnerabilities[vulnerability_ID].embedding

    # Function used to create the vulnerabilities embeddings per node type, distinguishing between local and remote vulnerabilities
    def create_vulnerabilities_embeddings_per_node_type(self):
        # distinguish the vulns per node and per type
        self.vulnerabilities_embeddings_per_node_type = {}
        for node in self.environment.nodes:
            if node not in self.vulnerabilities_embeddings_per_node_type:
                self.vulnerabilities_embeddings_per_node_type[node] = {"local": [], "remote": []}
            for vulnerability_ID in self.get_node(node).vulnerabilities:
                embedding = self.get_node(node).vulnerabilities[vulnerability_ID].embedding
                for result in self.get_node(node).vulnerabilities[vulnerability_ID].results:
                    outcome_embedding = self.map_outcome_to_onehot(result.type_str, result.outcome)
                    if outcome_embedding is None:
                        continue
                    self.vulnerabilities_embeddings_per_node_type[node][result.type_str].append({
                            "vulnerability_ID": vulnerability_ID,
                            "outcome": result.outcome,
                            "embedding": np.concatenate((embedding, outcome_embedding))
                    })

    def sample_random_action(self):
        return self.action_space.sample()

    # Function to set the graph encoder used to encode the graph and get the node embeddings
    def set_graph_encoder(self, graph_encoder):
        self.graph_encoder = graph_encoder

    # Function to set the PCA components for the vulnerability embeddings, which also updates the action space accordingly
    def set_pca_components(self, pca_components, default_value=768):
        if not pca_components:
            self.vulnerability_embeddings_dimensions = default_value
        else:
            self.vulnerability_embeddings_dimensions = pca_components
        self.action_space = spaces.Box(low=-4, high=4,
                                       shape=(self.node_embeddings_dimensions * 2 + self.vulnerability_embeddings_dimensions + self.outcome_dimensions,),
                                       dtype=numpy.float32)
