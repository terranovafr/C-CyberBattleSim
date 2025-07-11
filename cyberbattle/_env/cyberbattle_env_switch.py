# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    cyberbattle_env_switch.py
     Wrapper that randomly switch the environment given a certain interval to ensure that the agent learns generalizable behavior being exposed to multiple scenarios
"""
import os.path
import gymnasium
import numpy as np
import csv
import copy
from cyberbattle.simulation import model
import pickle
import time
from cyberbattle.utils.gym_utils import convert_gym_to_gymnasium_space

class RandomSwitchEnv(gymnasium.Env):
    # Decide for a current environment and wraps it accordingly, then switch to a new one every switch_interval episodes
    def __init__(self, envs_ids, switch_interval=50, envs_folder=None, envs_list=None,
                 save_to_csv=False, save_to_csv_interval=1,
                 csv_folder=None, save_embeddings=True, verbose = 1):
        # two options are used, or ids of files and folder provided to load pkl file or list with the actual objects
        self.envs_ids = envs_ids
        self.envs_folder = envs_folder
        self.envs_list = envs_list  # alternative to loading it every time, but memory intensive
        self.episode_count = 0
        self.switch_interval = switch_interval
        self.steps_in_current_episode = 0
        self.verbose = verbose
        self._switch_environment() # first initial environment selection and reset
        # represent a gym environment with the same action space and observation space of the ones in the list wrapped
        # should be gymnasium spaces
        self.action_space = convert_gym_to_gymnasium_space(self.current_env.action_space)
        self.observation_space = convert_gym_to_gymnasium_space(self.current_env.observation_space)
        self.save_to_csv = save_to_csv
        if self.current_env.env_type != "compressed" and self.save_to_csv:
            raise ValueError("Only continuous environments are supported for saving to csv")
        self.save_to_csv_interval = save_to_csv_interval
        self.file = None
        # option to periodically save logs (optionally)
        if self.save_to_csv and csv_folder is not None:
            self.setup_csv(csv_folder, save_embeddings)
        self.done = False
        self.num_envs = 1
        self.episode_start_time = time.time()

    def render(self, mode='human'):
        return self.current_env.render(mode)

    def close(self):
        self.current_env.close()

    # Function to load the environment from the list or from the folder
    def load_environment(self):
        if hasattr(self, 'current_env'):
            del self.current_env # enforce garbage collection
        while True:
            if self.envs_list is not None: # if list with objects present
                self.current_env = self.envs_list[self.current_env_index]
                break
            else: # if ids and folder present

                try:
                    with open(os.path.join(self.envs_folder, str(self.current_env_index) + ".pkl"), 'rb') as file:
                        self.current_env = pickle.load(file)
                        break
                except model.NoSuitableStarterNode: # if the environment is not suitable (does not have a node that respect the required criterias), switch to another one
                    if self.verbose:
                        self.current_env.logger.error("Error while loading environment")

    # Switch to a random new environment among the list
    def _switch_environment(self):
        self.current_env_index = np.random.choice(self.envs_ids)
        self.load_environment()
        if self.verbose and self.current_env.logger:
            self.current_env.logger.info("Switched to environment %s", self.current_env_index)

    # Step function that executes the action in the current environment, eventually saving the logs to csv file
    def step(self, action):
        previously_discovered_nodes = copy.deepcopy(self.current_env.discovered_nodes)
        previously_owned_nodes = copy.deepcopy(self.current_env.owned_nodes)
        previously_alive_nodes = copy.deepcopy(self.current_env.get_alive_nodes())
        if self.current_env.env_type == "compressed":
            source_node, target_node, vulnerability_ID, outcome, distance = self.current_env.find_closest_action_embedding(copy.deepcopy(action), no_output=True)
            previous_source_node_str = self.get_str_info(self.current_env.get_node(source_node))
            previous_target_node_str = self.get_str_info(self.current_env.get_node(target_node))
        else:
            previous_source_node_str = ""
            previous_target_node_str = ""

        observation, reward, self.done, info = self.current_env.step(action)

        # If requested, save to csv file the transition
        if self.save_to_csv and self.episode_count % self.save_to_csv_interval == 0 and self.episode_count != 0:
            if self.save_embeddings:
                self.add_to_csv_with_embeddings(previously_discovered_nodes, previously_owned_nodes,
                                                previously_alive_nodes, previous_source_node_str,
                                                previous_target_node_str, action)
            else:
                self.add_to_csv(previously_discovered_nodes, previously_owned_nodes, previously_alive_nodes,
                                previous_source_node_str, previous_target_node_str)

        self.truncated = self.current_env.truncated
        self.steps_in_current_episode += 1
        self.current_observation = observation
        if self.done:
            # if episode finished, log statistics as fields since gathered by callbacks eventually, before the environment optionally switches
            self.owned_nodes, self.discovered_nodes,  self.not_discovered_nodes, self.disrupted_nodes, self.num_nodes, self.reachable_count, self.discoverable_count, self.disruptable_count, self.network_availability, self.reimaged_nodes, self.num_events, self.discovered_amount, self.discoverable_amount, self.episode_won = self.current_env.get_statistics()
            self.overall_reimaged = self.current_env.overall_reimaged
            self.evicted = self.current_env.evicted
            self.episode_count += 1
            self.steps_in_current_episode = 0
            self.calculate_action_time = 0
            self.episode_duration = time.time() - self.episode_start_time
            if self.current_env.verbose > 1:
                self.current_env.logger.info("Episode duration: %s", self.episode_duration)
            self.episode_start_time = time.time()
        return observation, reward, self.done, self.truncated, info

    # Reset function resetting the current environment and check for switching
    def reset(self, **kwargs):
        self.steps_in_current_episode = 0
        # determine whether a switch is necessary
        self._check_switch()
        while True:
            try:
                self.current_observation = self.current_env.reset()
                break
            except model.NoSuitableStarterNode: # sometimes in some environment no node can respect minimum criterias required
                if self.verbose:
                    self.current_env.logger.info(
                        "Forced to switch environment since the current has no suitable starter node!")
                self._switch_environment()
                continue

        self.done = False
        return self.current_observation, {}

    # Step function for the attacker environment, it executes the action and updates the statistics
    def step_attacker_env(self, source_node, target_node, vulnerability, outcome):
        # get information before step function to load the transition into a csv folder
        self.current_env.step_attacker_env(source_node, target_node, vulnerability, outcome)

        self.truncated = self.current_env.truncated
        self.done = self.current_env.done
        self.steps_in_current_episode += 1
        if self.done or self.truncated:
            # if episode finished, log statistics as fields since gathered by callbacks eventually
            self.owned_nodes, self.discovered_nodes,  self.not_discovered_nodes, self.disrupted_nodes, self.num_nodes, self.reachable_count, self.discoverable_count, self.disruptable_count, self.network_availability, self.reimaged_nodes, self.num_events, self.discovered_amount, self.discoverable_amount, self.episode_won = self.current_env.get_statistics()
            self.overall_reimaged = self.current_env.overall_reimaged
            self.evicted = self.current_env.evicted
            self.episode_count += 1
            self.steps_in_current_episode = 0
            self.calculate_action_time = 0
            self.episode_duration = time.time() - self.episode_start_time
            if self.current_env.verbose > 1:
                self.current_env.logger.info("Episode duration: %s", self.episode_duration)
            self.episode_start_time = time.time()
        return self.done or self.truncated



    # Provide statistics of the last iteration (called at the end of the episode)
    def get_statistics(self):
        return self.owned_nodes, self.discovered_nodes, self.not_discovered_nodes, self.disrupted_nodes, self.num_nodes, self.reachable_count, self.discoverable_count, self.disruptable_count, self.network_availability, self.reimaged_nodes, self.num_events, self.discovered_amount, self.discoverable_amount, self.episode_won

    # Set cut-off number of iterations per episode
    def set_cut_off(self, cut_off):
        self.current_env.set_cut_off(cut_off)

    # Set proportional cut-off coefficient for episode iterations (K)
    def set_proportional_cutoff_coefficient(self, coefficient):
        self.current_env.set_proportional_cutoff_coefficient(coefficient)

    # Get the current evolving visible graph of the environment
    def get_evolving_visible_graph(self):
        return self.current_env.evolving_visible_graph

    # Sample valid action from the current environment
    def sample_valid_action(self):
        return self.current_env.sample_valid_action()

    # Sample random action from the current environment
    def sample_random_action(self):
        return self.current_env.sample_random_action()

    # Check if the current environment needs to be switched based on the switch interval
    def _check_switch(self):
        if (self.episode_count + 1) % (self.switch_interval + 1) == 0:
            self._switch_environment()

    # Setup the CSV file for logging the transitions
    def setup_csv(self, csv_folder, save_embeddings, filename="logs.csv"):
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)
        filename = os.path.join(csv_folder, filename)
        self.file = open(filename, mode='w', newline='')  # Keep the file open
        self.csv_writer = csv.writer(self.file)
        self.save_embeddings = save_embeddings
        if not self.save_embeddings:
            # save also embeddings for visualization (though creating heavy files)
            self.csv_writer.writerow(
                ["Environment", "Episode", "Iteration", "Discovered Nodes", "Owned Nodes", "Alive nodes", "Source node",
                 "Target node", "Vulnerability ID", "Outcome Mapped", "Reward", "Outcome", "Done",
                 "Source Node Details", "Target Node Details", "Previous Source Node Details",
                 "Previous Target Node Details", "Edges"])
        else:
            self.csv_writer.writerow(
                ["Environment", "Episode", "Iteration", "Discovered Nodes", "Owned Nodes", "Alive nodes",
                 "Source node", "Target node", "Vulnerability ID", "Outcome Mapped", "Reward", "Outcome",
                 "Done",
                 "Source Node Details", "Target Node Details", "Previous Source Node Details",
                 "Previous Target Node Details", "Edges",
                 "Action Embeddings", "Action Choice"])

    # Update the CSV folder and filename for logging, also set the save embeddings flag
    def update_csv_folder(self, csv_folder, filename="logs.csv", save_embeddings=False, save_to_csv_interval=1):
        if self.current_env.env_type != "compressed":
            raise ValueError("Only compressed environments are supported for saving to csv")
        self.save_to_csv = True

        self.save_to_csv_interval = save_to_csv_interval
        self.save_embeddings = save_embeddings
        if csv_folder is not None:
            self.setup_csv(csv_folder, save_embeddings, filename)

    # Add a transition to the CSV file
    def add_to_csv(self, previously_discovered_nodes, previously_owned_nodes, previously_alive_nodes, previous_source_node_str, previous_target_node_str):
        if isinstance(self.current_env.outcome, model.InvalidAction):
            outcome_str = self.current_env.outcome.reason
        elif isinstance(self.current_env.outcome, model.PrivilegeEscalation):
            outcome_str = "Privilege Escalation to " + str(self.current_env.outcome.level)
        else:
            outcome_str = self.current_env.outcome
        # Source node information
        source_node_str = self.get_str_info(self.current_env.get_node(self.current_env.source_node))
        # Target node information
        target_node_str = self.get_str_info(self.current_env.get_node(self.current_env.target_node))
        edges_str = ""
        for edge in self.current_env.edges:
            edges_str += edge[0] + ":" + edge[1] + ":" + edge[2] + ","
        edges_str = edges_str[:-1]
        self.csv_writer.writerow(
                [self.current_env_index, self.episode_count, self.steps_in_current_episode, previously_discovered_nodes,
                 previously_owned_nodes, previously_alive_nodes, self.current_env.source_node,
                 self.current_env.target_node, self.current_env.vulnerability_ID,
                 self.current_env.outcome_desired, self.current_env.reward, outcome_str, self.done, source_node_str,
                 target_node_str, previous_source_node_str, previous_target_node_str, edges_str])
        self.file.flush()

    # Add a transition to the CSV file with embeddings
    def add_to_csv_with_embeddings(self, previously_discovered_nodes, previously_owned_nodes, previously_alive_nodes, previous_source_node_str, previous_target_node_str, action_choice):
        if isinstance(self.current_env.outcome, model.InvalidAction):
            outcome_str = self.current_env.outcome.reason
        elif isinstance(self.current_env.outcome, model.PrivilegeEscalation):
            outcome_str = "Privilege Escalation to " + str(self.current_env.outcome.level)
        else:
            outcome_str = self.current_env.outcome
        # Source node information
        source_node_str = self.get_str_info(self.current_env.get_node(self.current_env.source_node))
        # Target node information
        target_node_str = self.get_str_info(self.current_env.get_node(self.current_env.target_node))
        edges_str = ""
        for edge in self.current_env.edges:
            edges_str += edge[0] + ":" + edge[1] + ":" + edge[2] + ","
        edges_str = edges_str[:-1]
        action_embeddings_str = ""
        for action_embedding in self.current_env.action_embeddings:
            action_embeddings_str += str(self.current_env.action_embeddings[action_embedding]) + ","
        action_embeddings_str = action_embeddings_str[:-1]
        self.csv_writer.writerow(
            [self.current_env_index, self.episode_count, self.steps_in_current_episode, previously_discovered_nodes, previously_owned_nodes, previously_alive_nodes, self.current_env.source_node, self.current_env.target_node, self.current_env.vulnerability_ID,
                self.current_env.outcome_desired, self.current_env.reward, outcome_str, self.done, source_node_str, target_node_str, previous_source_node_str, previous_target_node_str, edges_str, action_embeddings_str, action_choice])
        self.file.flush()

    # Get string information about the node, used for logging and CSV output
    def get_str_info(self, node_info):
        flag_str = ("status : " + str(node_info.status) + " / tag : " + str(node_info.tag) + " / value : " + str(
            node_info.value) + " / privilege level : " + str(
            node_info.privilege_level) + " / has data : " + str(
            node_info.has_data) + " / data collected : " + str(
            node_info.data_collected) + " / data exfiltrated : " + str(
            node_info.data_exfiltrated) + " / visible : " + str(
            node_info.visible) + " / persistence : " + str(
            node_info.persistence) + " / defense evasion : " + str(
            node_info.defense_evasion) + " / level at access : " + str(
            node_info.level_at_access) + " / services : ")
        for service in node_info.services:
            incoming_flag = 0
            outgoing_flag = 0
            for firewall_rule in node_info.firewall.incoming:
                if firewall_rule.port == service.name and firewall_rule.permission == model.RulePermission.ALLOW:
                    incoming_flag = 1
            for firewall_rule in node_info.firewall.outgoing:
                if firewall_rule.port == service.name and firewall_rule.permission == model.RulePermission.ALLOW:
                    outgoing_flag = 1
            flag_str += str(service.name) + " " + str(service.running) + " " + str(incoming_flag) + " " + str(outgoing_flag) + " "
        flag_str += " / vulnerabilities : "
        for vulnerability_ID in node_info.vulnerabilities:
            vulnerability = node_info.vulnerabilities[vulnerability_ID]
            flag_str += str(vulnerability_ID) + "  "
            for result in vulnerability.results:
                flag_str += result.type_str + "--" + result.outcome_str + "  "
        return flag_str
