# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    heuristic_utils.py
    This file contains the utilities used to handle different heuristics to select actions
"""

import numpy as np
import random
from cyberbattle.simulation import model
import torch
from collections import defaultdict

outcomes = {
    "control": [model.LateralMove, model.CredentialAccess, model.Reconnaissance],
    "discovery": [model.Discovery, model.Collection, model.Reconnaissance, model.Exfiltration],
    "disruption": [model.DenialOfService, model.Reconnaissance, model.LateralMove, model.CredentialAccess]
}

# Heuristic function to select the highest score vulnerability with unused outcomes
def highest_score_vuln(env, used_vulnerabilities, score_type="cvss"):
    while True:
        # Get all discovered nodes
        random_owned_node = np.random.choice(env.envs[0].current_env.owned_nodes)
        discovered_nodes = env.envs[0].current_env.discovered_nodes

        # Gather all (discovered_node, vulnerability) pairs, filtering out fully used vulnerabilities
        candidate_vulnerabilities = []
        for node in discovered_nodes:
            node_data = env.envs[0].current_env.get_node(node)
            all_vulnerabilities = list(node_data.vulnerabilities.keys())

            # Exclude vulnerabilities where all outcomes have already been used
            for vuln in all_vulnerabilities:
                vulnerability_info = node_data.vulnerabilities[vuln]
                if (vuln, None) in used_vulnerabilities[node]:
                    continue
                remaining_outcomes = [
                    result.outcome for result in vulnerability_info.results
                    if (vuln, result.outcome) not in used_vulnerabilities[node]
                ]
                if remaining_outcomes:  # Only consider vulnerabilities with unused outcomes
                    score = getattr(vulnerability_info, score_type, None)
                    candidate_vulnerabilities.append((node, vuln, score))

        # If no candidates remain, exit the loop
        if not candidate_vulnerabilities:
            raise ValueError("No vulnerabilities with unused outcomes remain.")

        # Select the (node, vulnerability) pair with the highest impact score
        max_score = max(candidate_vulnerabilities, key=lambda x: x[2])[2]

        # Filter candidates with the maximum impact score
        top_candidates = [
            candidate for candidate in candidate_vulnerabilities if candidate[2] == max_score
        ]

        # Randomly select one of the top candidates
        selected_node, selected_vulnerability, _ = random.choice(top_candidates)

        # Get the vulnerability information for the selected node and vulnerability
        node_data = env.envs[0].current_env.get_node(selected_node)
        vulnerability_info = node_data.vulnerabilities[selected_vulnerability]

        # Shuffle the results to introduce randomness in outcomes
        random.shuffle(vulnerability_info.results)
        results_copy = [
            result for result in vulnerability_info.results
            if (selected_vulnerability, result.outcome) not in used_vulnerabilities[selected_node]
        ]

        # Remove DenialOfService outcomes for starter nodes
        if selected_node == env.envs[0].current_env.starter_node:
            results_copy = [
                result for result in results_copy
                if type(result.outcome) is not model.DenialOfService
            ]

        # If no valid results remain, mark this vulnerability as fully used and retry
        if not results_copy:
            used_vulnerabilities[selected_node].add((selected_vulnerability, None))  # Mark vuln as exhausted
            continue

        # Select a random valid result
        selected_result = random.choice(results_copy)

        # Mark the vulnerability and outcome as used
        used_vulnerabilities[selected_node].add((selected_vulnerability, selected_result.outcome))

        # Return the selected nodes, vulnerability, and outcome
        return random_owned_node, selected_node, selected_vulnerability, selected_result.outcome, used_vulnerabilities

# Heuristic function to select a random node and a random vulnerability outcome
def random_node_random_vuln_outcome(env, goal):
    while True:
        random_owned_node = np.random.choice(env.envs[0].current_env.owned_nodes)
        random_discovered_node = np.random.choice(env.envs[0].current_env.discovered_nodes)
        node_data = env.envs[0].current_env.get_node(random_discovered_node)
        random_vulnerability = random.choice(list(node_data.vulnerabilities.keys()))
        vulnerability_info = node_data.vulnerabilities[random_vulnerability]
        random.shuffle(vulnerability_info.results)
        for result in vulnerability_info.results:
            if type(result.outcome) in outcomes[goal]:
                return random_owned_node, random_discovered_node, random_vulnerability, result.outcome


def play_heuristic_episode_until_done(env, model, goal=None, score_name=None):
    used_vulnerabilities = defaultdict(set)
    _ = env.envs[0].reset()
    while True:
        with torch.no_grad():
            source_node, target_node, vulnerability, outcome, used_vulnerabilities = model(env, goal, used_vulnerabilities, score_name)
        env.envs[0].step_attacker_env(source_node, target_node, vulnerability, outcome)
        if env.envs[0].done or env.envs[0].truncated:
            break
    owned_nodes, discovered_nodes, _, disrupted_nodes, num_nodes, reachable_count, discoverable_count, disruptable_count, network_availability, _, _, discovered_amount, discoverable_amount, episode_won = \
    env.envs[0].get_statistics()
    return owned_nodes / (reachable_count + 1), discovered_nodes / (discoverable_count + 1), network_availability, discovered_amount / (discoverable_amount + 1), disrupted_nodes / (disruptable_count + 1), episode_won



heuristic_models = {
    "random_node_random_vuln_outcome": random_node_random_vuln_outcome,
    "highest_score_vuln": highest_score_vuln
}
