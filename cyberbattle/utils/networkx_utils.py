# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    networkx_utils.py
    This file contains the utilities related to networkx graphs.
"""

import networkx as nx

# Iterates over the nodes in a networkx
def iterate_network_nodes(network: nx.graph.Graph):
    for nodeid, nodevalue in network.nodes.items():
        node_data = nodevalue['data']
        yield nodeid, node_data

# Calculates the average shortest path length across all pair of nodes in a directed graph
def calculate_average_shortest_path_length(digraph):
    try:
        all_pairs_shortest_lengths = dict(nx.all_pairs_shortest_path_length(digraph))
        max_possible_value = len(all_pairs_shortest_lengths) - 1  # The maximum number of steps between any two nodes = number of nodes - 1

        total_shortest_path_length = 0
        num_pairs = 0
        not_reachable = 0

        nodes_list = list(digraph.nodes)
        for source, target_lengths in all_pairs_shortest_lengths.items():
            target_lengths = {key: value for key, value in target_lengths.items()}
            for node in nodes_list:
                if node not in target_lengths:
                    not_reachable += 1
                    # If the node is not reachable from the source, use double the maximum possible value
                    target_lengths[node] = max_possible_value*2
            for target, length in target_lengths.items():
                if target != source:
                    total_shortest_path_length += length
                    num_pairs += 1
        if num_pairs == 0:
            return 0, 0, all_pairs_shortest_lengths
        reachability_metric = 1 - (not_reachable / (len(all_pairs_shortest_lengths) * len(all_pairs_shortest_lengths)))
        average_shortest_path_length = total_shortest_path_length / num_pairs
        connectivity_metric = 1 - (average_shortest_path_length / (2*max_possible_value))
        # filtering the dict to include None if a node is not reachable, used later
        for source, target_lengths in all_pairs_shortest_lengths.items():
            target_lengths = {key: value for key, value in target_lengths.items()}
            all_pairs_shortest_lengths[source] = target_lengths

        for source, target_lengths in all_pairs_shortest_lengths.items():
            for node in nodes_list:
                if node not in target_lengths.keys():
                    all_pairs_shortest_lengths[source][node] = None

        return reachability_metric, connectivity_metric, all_pairs_shortest_lengths
    except nx.NetworkXError:
        # If the graph is not weakly connected, handle the exception
        return 0, 0, None
