# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    cyberbattle_env_gae.py
     Wrapper that creates several configuration evolutions of the evolving visible graph to be used in the GAE training or validation.
"""

from typing import Dict
import networkx as nx
import numpy as np
import sys
import os
import numpy
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle._env.cyberbattle_env import CyberBattleEnv # noqa: E402
from cyberbattle.simulation import  model # noqa: E402
from cyberbattle.utils.data_utils import flatten_dict_with_arrays, flatten # noqa: E402


class CyberBattleGAEEnv(CyberBattleEnv):
    @property
    def name(self) -> str:
        return "CyberBattleGAEEnv"

    def __init__(self,
                 edge_feature_aggregations = None, # aggregations to be used to merge embeddings of the edges
                 pca_components = None, # eventual pca reduction of the vulnerabilities embeddings
                 **kwargs
                 ):
        super().__init__(**kwargs)
        edge_feature_aggregations = edge_feature_aggregations or ["mean"]
        self.env_type = "gae"  # type of environment used for GAE training
        # numer of components to be used if the dimensionality of the vulnerabilities embeddings has been reduced with PCA
        if pca_components:
            self.vulnerability_embeddings_dimensions = pca_components
        else:
            self.vulnerability_embeddings_dimensions = 768

        # Load information for the GAE to set its output layer
        self.continuous_indices = self.vulnerability_embeddings_dimensions + self.vulnerability_embeddings_dimensions + 2  # vulnerability embedding, service name embedding, and 2 other elements (value, sla_weight)
        # elements to be used in the multi-class output layer with the number of classes
        self.multi_class_info = {
            'privilege level': 3,
            'status': 3
        }
        # rest of indices are binary
        self.binary_indices = 3 * self.max_services_per_node + 6
        self.edge_feature_aggregations = edge_feature_aggregations
        self.create_vulnerabilities_embeddings()
        self.reset()

    # Simple step: advance logic with action and update evolving visible graph
    def step_graph(self, source_node, target_node, vulnerability_ID, outcome_desired):
        super().step_attacker_env(source_node, target_node, vulnerability_ID, outcome_desired)
        # Update the graph that should turn into a graph embedding
        self.update_evolving_visible_graph_after_step(source_node, target_node, vulnerability_ID)
        return self.done or self.truncated

    # Function that update the evolving visible graph after a step
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

    # Reset function invoking the environment one and resetting the evolving visible graph
    def reset(self, **kwargs): # reset function for GAE training
        super().reset_env()
        self.exploited_vulnerabilities_per_node_pairs = {} # use for the GAE to avoid to resample actions already taken
        self.reset_evolving_visible_graph()
        self.action_embeddings = {}
        self.blocked_graph = False
        self.edges = []

    # Reset the evolving visible graph with only starter node
    def reset_evolving_visible_graph(self):
        # Reset graph and set some attributes used during GAE training
        self.evolving_visible_graph = nx.DiGraph()
        self.evolving_visible_graph.clear()
        self.add_node_evolving_visible_graph(self.starter_node)
        self.node_feature_vector_size = len(self.get_node_feature_vector(self.starter_node))
        self.edge_feature_vector_size = self.vulnerability_embeddings_dimensions # just one aggregation

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

    # Function to get the feature vector of a node given its ID
    def get_node_feature_vector(self, node_id):
        node_features_dict = self.convert_node_info_to_observation(self.get_node(node_id))
        flattened_node_features_dict = flatten_dict_with_arrays(node_features_dict)
        node_features_array = numpy.array(
            flatten([flattened_node_features_dict[key] for key in flattened_node_features_dict]), dtype=numpy.float32)
        return node_features_array

    # Function to add a node to the evolving visible graph
    def add_node_evolving_visible_graph(self, node_id):
        self.evolving_visible_graph.add_node(node_id, x=self.get_node_feature_vector(node_id))

    # Function to update the node feature vector in the evolving visible graph
    def update_node_evolving_visible_graph(self, node_id):
        self.evolving_visible_graph.nodes[node_id].update({'x': self.get_node_feature_vector(node_id)})

    # Function to create the vulnerabilities embeddings dictionary
    def create_vulnerabilities_embeddings(self):
        self.vulnerabilities_embeddings = {}
        for node in self.environment.nodes:
            for vulnerability_ID in self.get_node(node).vulnerabilities:
                self.vulnerabilities_embeddings[vulnerability_ID] = self.get_node(node).vulnerabilities[vulnerability_ID].embedding

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
            # remerge vulnerabilities with aggregator if a vulnerability is already present
            if not self.exploited_vulnerabilities_per_node_pairs.get(source_node).get(target_node):
                self.exploited_vulnerabilities_per_node_pairs[source_node][target_node] = []
            self.exploited_vulnerabilities_per_node_pairs[source_node][target_node].append(
                self.vulnerabilities_embeddings[vuln_key])
            edge_embedding = []
            for edge_aggregation in self.edge_feature_aggregations:
                edge_embedding.append(aggregation_functions[edge_aggregation](
                    self.exploited_vulnerabilities_per_node_pairs[source_node][target_node], axis=0))
            self.evolving_visible_graph[source_node][target_node]["vulnerabilities_embeddings"] = np.concatenate(
                edge_embedding)
            return True
        else:
            self.evolving_visible_graph.add_edge(source_node, target_node)
            self.exploited_vulnerabilities_per_node_pairs[source_node] = {}
            self.exploited_vulnerabilities_per_node_pairs[source_node][target_node] = [
                self.vulnerabilities_embeddings[vuln_key]]
            edge_embedding = []
            for edge_aggregation in self.edge_feature_aggregations:
                edge_embedding.append(
                    aggregation_functions[edge_aggregation](
                        self.exploited_vulnerabilities_per_node_pairs[source_node][target_node],
                        axis=0))
            self.evolving_visible_graph[source_node][target_node]["vulnerabilities_embeddings"] = np.concatenate(
                edge_embedding)
            return True
