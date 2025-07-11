# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    test_utils.py
    This file contains the utilities related to the tester module (all testing functions).
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from cyberbattle.utils.math_utils import bootstrap_ci
from cyberbattle.utils.heuristic_utils import heuristic_models
from collections import defaultdict

desired_outcome_sets_by_goal = {
    'control': {'LateralMove-Credential'},
    'discovery': {'Reconnaissance'},
    'disruption': {'DenialOfService'}
}

# Bar plot of the count of actions chosen separated by type (vulnerability type, vulnerability ID, outcome)
def plot_action_distribution(actions_list, checkpoint_name, actions_type='Vulnerability type'):
    action_names = list(dict.fromkeys(actions_list))
    action_indices = np.arange(len(action_names))

    # Count the occurrences of each action
    action_counts = np.zeros(len(action_names))
    for action in actions_list:
        action_counts[list(action_names).index(action)] += 1

    plt.bar(action_indices, action_counts, color='red')
    plt.xticks(action_indices, action_names, rotation=45, ha='right')
    plt.xlabel(actions_type, fontsize=14)
    plt.ylabel('Number of Times Chosen', fontsize=14)
    plt.title(f'Action Distribution - Checkpoint {checkpoint_name}', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ion()
    plt.ioff()
    return plt

# Run episodes and save action choices statistics (outcome, vulnerability type, etc.)
def run_and_save_action_choices(model, gym_env, proportional_cutoff_coefficient, logger, num_episodes=500, verbose=1):
    outcomes_chosen = []
    vulnerability_types_chosen = []
    vulnerabilities_chosen = []
    if verbose:
        logger.info("Computing action choices of the agent...")
    for _ in tqdm(range(num_episodes), desc='Saving action choices of each episode...'):
        state = gym_env.reset()
        if proportional_cutoff_coefficient: # it can be different after N episodes for the switcher, hence resetting every episode
            gym_env.envs[0].set_proportional_cutoff_coefficient(proportional_cutoff_coefficient)
        while True:
            with torch.no_grad():
                action, _ = model.predict(state)
            next_state, reward, done, info = gym_env.step(action)
            vulnerability_types_chosen.append(info[0]['vulnerability_type'])
            vulnerabilities_chosen.append(info[0]['vulnerability'])
            outcomes_chosen.append(info[0]['outcome'])
            state = next_state
            if done:
                break
    if verbose:
        logger.info("Number of episodes: %d", num_episodes)
        logger.info("Number of actions chosen: %d", len(outcomes_chosen))
        logger.info("Number of unique actions chosen: %d", len(set(outcomes_chosen)))
        for action_type in ['vulnerability_types', 'vulnerabilities', 'outcomes']:
            unique_actions = set(locals()[f"{action_type}_chosen"])
            logger.info(f"Number of unique {action_type} chosen: %d", len(unique_actions))
    return vulnerabilities_chosen, vulnerability_types_chosen, outcomes_chosen

# Average performances calculation (control, discovery, disruption) of the agent and the random agent
def calculate_average_performances(model, gym_env, proportional_cutoff_coefficient, logger, goal=None, score_name=None, num_episodes=50, avoid_random=False, verbose=1):
    episode_list = []
    agent_owned_list = []
    agent_discovered_list = []
    agent_availability_list = []
    agent_discovered_amount_list = []
    agent_disrupted_list = []
    agent_won_list = []
    random_agent_owned_list = []
    random_agent_discovered_list = []
    random_agent_availability_list = []
    random_agent_discovered_amount_list = []
    random_agent_disrupted_list = []
    random_agent_won_list = []

    stats_data = []

    if verbose:
        logger.info("Computing average performances of the agent...")
        if not avoid_random:
            logger.info("Computing average performances of the random agent as well...")

    for episode in tqdm(range(num_episodes), desc='Saving performances of each episode...'):
        # Set the cut-off for the environment if needed
        if proportional_cutoff_coefficient:
            gym_env.envs[0].set_proportional_cutoff_coefficient(proportional_cutoff_coefficient)

        # Play random actions to calculate random agent performance
        if not avoid_random:
            random_agent_percentage_owned, random_agent_percentage_discovered, random_agent_network_availability, random_agent_discovered_amount, random_agent_disrupted, random_agent_won = play_random_agent_episode_until_done(gym_env)

            stats_data.append({
                'episode': episode,
                'agent': 'random',
                'owned_nodes_percentage': random_agent_percentage_owned,
                'discovered_nodes_percentage': random_agent_percentage_discovered,
                'network_availability': random_agent_network_availability,
                'discovered_amount_percentage': random_agent_discovered_amount,
                'disrupted_nodes_percentage': random_agent_disrupted,
                'episode_won': random_agent_won
            })

            random_agent_owned_list.append(random_agent_percentage_owned)
            random_agent_discovered_list.append(random_agent_percentage_discovered)
            random_agent_availability_list.append(random_agent_network_availability)
            random_agent_discovered_amount_list.append(random_agent_discovered_amount)
            random_agent_disrupted_list.append(random_agent_disrupted)
            random_agent_won_list.append(random_agent_won)

        if model in heuristic_models.values():
            agent_percentage_owned, agent_percentage_discovered, agent_network_availability, agent_discovered_amount, agent_disrupted, agent_won = play_heuristic_episode_until_done(
                gym_env, model,score_name)
        else:
            agent_percentage_owned, agent_percentage_discovered, agent_network_availability, agent_discovered_amount, agent_disrupted, agent_won = play_agent_episode_until_done(
                gym_env, model)

        episode_list.append(episode)
        agent_owned_list.append(agent_percentage_owned)
        agent_discovered_list.append(agent_percentage_discovered)
        agent_availability_list.append(agent_network_availability)
        agent_discovered_amount_list.append(agent_discovered_amount)
        agent_disrupted_list.append(agent_disrupted)
        agent_won_list.append(agent_won)

        stats_data.append({
            'episode': episode,
            'agent': 'agent',
            'owned_nodes_percentage': agent_percentage_owned,
            'discovered_nodes_percentage': agent_percentage_discovered,
            'network_availability': agent_network_availability,
            'discovered_amount_percentage': agent_discovered_amount,
            'disrupted_nodes_percentage': agent_disrupted,
            'episodes_won': agent_won
        })

    df = pd.DataFrame(stats_data, columns=['episode', 'agent', 'owned_nodes_percentage', 'discovered_nodes_percentage', 'network_availability', 'discovered_amount_percentage', 'disrupted_nodes_percentage', 'episodes_won'])
    return (df, agent_owned_list, agent_discovered_list, agent_availability_list, agent_discovered_amount_list, agent_disrupted_list, agent_won_list,
            random_agent_owned_list, random_agent_discovered_list, random_agent_availability_list, random_agent_discovered_amount_list, random_agent_disrupted_list, random_agent_won_list)

# Function to create only random agent statistics
def calculate_random_agent_average_performances(envs, proportional_cutoff_coefficient, logger, num_episodes=100, verbose=1):
    episode_list = []
    random_agent_owned_list = []
    random_agent_discovered_list = []
    random_agent_discovered_amount_list = []
    random_agent_disrupted_list = []
    random_agent_availability_list = []
    random_agent_won_list = []
    stats_data = []

    if verbose:
        logger.info("Computing average performances of the random agent...")

    for episode in tqdm(range(num_episodes), desc="Saving performances of the random agent in each episode..."):
        envs.reset()
        if proportional_cutoff_coefficient:
            envs.set_proportional_cutoff_coefficient(proportional_cutoff_coefficient)

        while (True):
            action = envs.action_space.sample()
            next_state, _, done, _, _ = envs.step(action)
            if done:
                break
        owned_nodes, discovered_nodes, _, disrupted_nodes, num_nodes, reachable_count, discoverable_count, disruptable_count, network_availability, _, _, discovered_amount, discoverable_amount, episode_won = envs.get_statistics()
        episode_list.append(episode)
        random_agent_owned_list.append((owned_nodes / (reachable_count+1)))
        random_agent_discovered_list.append((discovered_nodes / (discoverable_count+1)))
        random_agent_discovered_amount_list.append((discovered_amount / (discoverable_amount+1)))
        random_agent_disrupted_list.append((disrupted_nodes / (disruptable_count + 1)))
        random_agent_availability_list.append(network_availability)
        random_agent_won_list.append(episode_won)
        stats_data.append({
            'episode': episode,
            'agent': 'random',
            'owned_nodes_percentage': (owned_nodes / (reachable_count+1)),
            'discovered_nodes_percentage': (discovered_nodes /  (discoverable_count+1)),
            'discovered_amount_percentage': (discovered_amount / (discoverable_amount+1)),
            'disrupted_nodes_percentage': (disrupted_nodes / (disruptable_count + 1)),
            'network_availability': network_availability,
            'episode_won_percentage': episode_won
        })

    df = pd.DataFrame(stats_data,columns=['episode', 'agent', 'owned_nodes_percentage', 'discovered_nodes_percentage', 'discovered_amount_percentage', 'disrupted_nodes_percentage', 'network_availability'])

    average_owned_percentage, lower_bound_owned_percentage, upper_bound_owned_percentage = bootstrap_ci(random_agent_owned_list)
    average_discovered_percentage, lower_bound_discovered_percentage, upper_bound_discovered_percentage = bootstrap_ci(random_agent_discovered_list)
    average_discovered_amount_percentage, lower_bound_discovered_amount_percentage, upper_bound_discovered_amount_percentage = bootstrap_ci(random_agent_discovered_amount_list)
    average_disrupted_percentage, lower_bound_disrupted_percentage, upper_bound_disrupted_percentage = bootstrap_ci(random_agent_disrupted_list)
    average_network_availability, lower_bound_network_availability, upper_bound_network_availability = bootstrap_ci(random_agent_availability_list)
    average_episode_won, lower_bound_episode_won, upper_bound_episode_won = bootstrap_ci(random_agent_won_list)
    if verbose:
        logger.info("Random agent statistics:")
        logger.info("--------------------")
        logger.info(f"Average owned percentage: {average_owned_percentage} [{lower_bound_owned_percentage}, {upper_bound_owned_percentage}]")
        logger.info(f"Average discovered percentage: {average_discovered_percentage} [{lower_bound_discovered_percentage}, {upper_bound_discovered_percentage}]")
        logger.info(f"Average discovered amount percentage: {average_discovered_amount_percentage} [{lower_bound_discovered_amount_percentage}, {upper_bound_discovered_amount_percentage}]")
        logger.info(f"Average disrupted percentage: {average_disrupted_percentage} [{lower_bound_disrupted_percentage}, {upper_bound_disrupted_percentage}]")
        logger.info(f"Average network availability: {average_network_availability} [{lower_bound_network_availability}, {upper_bound_network_availability}]")
        logger.info(f"Average episode won: {average_episode_won} [{lower_bound_episode_won}, {upper_bound_episode_won}]")
        logger.info("--------------------")
    return df

# One episode of a random agent
def play_random_agent_episode_until_done(env):
    env.reset()
    while(True):
        action = env.action_space.sample()
        next_state, _, done, _ = env.step([action])
        if done:
            break
    owned_nodes, discovered_nodes, _, disrupted_nodes, num_nodes, reachable_count, discoverable_count, disruptable_count, network_availability, _, _, discovered_amount, discoverable_amount, episode_won = env.envs[0].get_statistics()
    return owned_nodes / (reachable_count + 1), discovered_nodes / (discoverable_count + 1), network_availability, discovered_amount / (discoverable_amount + 1), disrupted_nodes / (disruptable_count + 1), episode_won

# One episode of the agent
def play_agent_episode_until_done(env, model):
    state = env.reset()
    while True:
        with torch.no_grad():
            action, _ = model.predict(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break

    owned_nodes, discovered_nodes, _, disrupted_nodes, num_nodes, reachable_count, discoverable_count, disruptable_count, network_availability, _, _, discovered_amount, discoverable_amount, episode_won = \
    env.envs[0].get_statistics()
    return owned_nodes / (reachable_count + 1), discovered_nodes / (discoverable_count + 1), network_availability, discovered_amount / (discoverable_amount + 1), disrupted_nodes / (disruptable_count + 1), episode_won

def play_heuristic_episode_until_done(env, model, score_name=None):
    used_vulnerabilities = defaultdict(set)
    _ = env.envs[0].reset()
    while True:
        with torch.no_grad():
            source_node, target_node, vulnerability, outcome, used_vulnerabilities = model(env, used_vulnerabilities, score_name)
        env.envs[0].step_attacker_env(source_node, target_node, vulnerability, outcome)
        if env.envs[0].done or env.envs[0].truncated:
            break
    owned_nodes, discovered_nodes, _, disrupted_nodes, num_nodes, reachable_count, discoverable_count, disruptable_count, network_availability, _, _, discovered_amount, discoverable_amount, episode_won = \
    env.envs[0].get_statistics()
    return owned_nodes / (reachable_count + 1), discovered_nodes / (discoverable_count + 1), network_availability, discovered_amount / (discoverable_amount + 1), disrupted_nodes / (disruptable_count + 1), episode_won


# Run the agent for a number of episodes (used to generate trajectories)
def play_agent_multiple_episodes_until_done(env, model, proportional_cutoff_coefficient, num_episodes=500):
    for _ in tqdm(range(num_episodes), desc='Playing agent episodes...'):
        if proportional_cutoff_coefficient:
            env.envs[0].set_proportional_cutoff_coefficient(proportional_cutoff_coefficient)
        state = env.reset()
        while True:
            with torch.no_grad():
                action, _ = model.predict(state)
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                break


# Overall print and save of the indicators of the performance metrics
def print_save_performance_metrics(dict, checkpoint_short_name, logs_folder, logger, num_episodes=100, proportional_cutoff_coefficient=10, current_time="", verbose=1):
    average_owned_percentage, lower_bound_owned_percentage, upper_bound_owned_percentage = bootstrap_ci(dict['Owned nodes percentage'])
    average_discovered_percentage, lower_bound_discovered_percentage, upper_bound_discovered_percentage = bootstrap_ci(dict['Discovered nodes percentage'])
    average_network_availability, lower_bound_network_availability, upper_bound_network_availability = bootstrap_ci(dict['Network availability'])
    average_discovered_amount_percentage, lower_bound_discovered_amount_percentage, upper_bound_discovered_amount_percentage = bootstrap_ci(dict['Discovered amount percentage'])
    average_disrupted_percentage, lower_bound_disrupted_percentage, upper_bound_disrupted_percentage = bootstrap_ci(dict['Disrupted nodes percentage'])
    average_won_percentage, lower_bound_won_percentage, upper_bound_won_percentage = bootstrap_ci(dict['Episodes won'])
    average_random_owned_percentage, lower_bound_random_owned_percentage, upper_bound_random_owned_percentage = bootstrap_ci(dict['Random - Owned nodes percentage'])
    average_random_discovered_percentage, lower_bound_random_discovered_percentage, upper_bound_random_discovered_percentage = bootstrap_ci(dict['Random - Discovered nodes percentage'])
    average_random_network_availability, lower_bound_random_network_availability, upper_bound_random_network_availability = bootstrap_ci(dict['Random - Network availability'])
    average_random_discovered_amount_percentage, lower_bound_random_discovered_amount_percentage, upper_bound_random_discovered_amount_percentage = bootstrap_ci(dict['Random - Discovered amount percentage'])
    average_random_disrupted_percentage, lower_bound_random_disrupted_percentage, upper_bound_random_disrupted_percentage = bootstrap_ci(dict['Random - Disrupted nodes percentage'])
    average_random_won_percentage, lower_bound_random_won_percentage, upper_bound_random_won_percentage = bootstrap_ci(dict['Random - Episodes won'])
    if verbose:
        logger.info("--------------------")
        logger.info(f"Checkpoint {checkpoint_short_name}:")
        logger.info(f"Average owned percentage: {average_owned_percentage} [{lower_bound_owned_percentage}, {upper_bound_owned_percentage}]")
        logger.info(f"Average discovered percentage: {average_discovered_percentage} [{lower_bound_discovered_percentage}, {upper_bound_discovered_percentage}]")
        logger.info(f"Average network availability: {average_network_availability} [{lower_bound_network_availability}, {upper_bound_network_availability}]")
        logger.info(f"Average discovered amount percentage: {average_discovered_amount_percentage} [{lower_bound_discovered_amount_percentage}, {upper_bound_discovered_amount_percentage}]")
        logger.info(f"Average disrupted percentage: {average_disrupted_percentage} [{lower_bound_disrupted_percentage}, {upper_bound_disrupted_percentage}]")
        logger.info(f"Average episodes won percentage: {average_won_percentage} [{lower_bound_won_percentage}, {upper_bound_won_percentage}]")
        logger.info(f"Average random agent owned percentage: {average_random_owned_percentage} [{lower_bound_random_owned_percentage}, {upper_bound_random_owned_percentage}]")
        logger.info(f"Average random agent discovered percentage: {average_random_discovered_percentage} [{lower_bound_random_discovered_percentage}, {upper_bound_random_discovered_percentage}]")
        logger.info(f"Average random agent network availability: {average_random_network_availability} [{lower_bound_random_network_availability}, {upper_bound_random_network_availability}]")
        logger.info(f"Average random agent discovered amount percentage: {average_random_discovered_amount_percentage} [{lower_bound_random_discovered_amount_percentage}, {upper_bound_random_discovered_amount_percentage}]")
        logger.info(f"Average random agent disrupted percentage: {average_random_disrupted_percentage} [{lower_bound_random_disrupted_percentage}, {upper_bound_random_disrupted_percentage}]")
        logger.info(f"Average random agent episodes won percentage: {average_random_won_percentage} [{lower_bound_random_won_percentage}, {upper_bound_random_won_percentage}]")
        logger.info("--------------------")
    with open(os.path.join(logs_folder, f"average_performances_{checkpoint_short_name}_{proportional_cutoff_coefficient}_{num_episodes}_{current_time}.txt"), 'w') as file:
        file.write(f"Checkpoint {checkpoint_short_name}:\n")
        file.write(f"Average owned percentage: {average_owned_percentage} [{lower_bound_owned_percentage}, {upper_bound_owned_percentage}]\n")
        file.write(f"Average discovered percentage: {average_discovered_percentage} [{lower_bound_discovered_percentage}, {upper_bound_discovered_percentage}]\n")
        file.write(f"Average network availability: {average_network_availability} [{lower_bound_network_availability}, {upper_bound_network_availability}]\n")
        file.write(f"Average discovered amount percentage: {average_discovered_amount_percentage} [{lower_bound_discovered_amount_percentage}, {upper_bound_discovered_amount_percentage}]\n")
        file.write(f"Average disrupted percentage: {average_disrupted_percentage} [{lower_bound_disrupted_percentage}, {upper_bound_disrupted_percentage}]\n")
        file.write(f"Average episodes won percentage: {average_won_percentage} [{lower_bound_won_percentage}, {upper_bound_won_percentage}]\n")
        file.write(f"Average random agent owned percentage: {average_random_owned_percentage} [{lower_bound_random_owned_percentage}, {upper_bound_random_owned_percentage}]\n")
        file.write(f"Average random agent discovered percentage: {average_random_discovered_percentage} [{lower_bound_random_discovered_percentage}, {upper_bound_random_discovered_percentage}]\n")
        file.write(f"Average random agent network availability: {average_random_network_availability} [{lower_bound_random_network_availability}, {upper_bound_random_network_availability}]\n")
        file.write(f"Average random agent discovered amount percentage: {average_random_discovered_amount_percentage} [{lower_bound_random_discovered_amount_percentage}, {upper_bound_random_discovered_amount_percentage}]\n")
        file.write(f"Average random agent disrupted percentage: {average_random_disrupted_percentage} [{lower_bound_random_disrupted_percentage}, {upper_bound_random_disrupted_percentage}]\n")
        file.write(f"Average random agent episodes won percentage: {average_random_won_percentage} [{lower_bound_random_won_percentage}, {upper_bound_random_won_percentage}]\n")
