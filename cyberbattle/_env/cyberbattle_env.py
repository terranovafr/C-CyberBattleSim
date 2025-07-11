# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    cyberbattle_env.py
    Class containing the main logic of the simulation game without providing a Gym Env full interface and defining an observation and action space.
    The observation and action spaces are defined by subclasses of this class.
"""

import time
from typing import Optional, List
import random
import gym as gym
import sys
import os
import copy
import numpy
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.simulation import attacker_actions, model, static_defender_actions # noqa: E402
from cyberbattle._env.static_defender import DefenderAgent, ScanAndReimageCompromisedMachines # noqa: E402
from cyberbattle.simulation.model import VulnerabilityType # noqa: E402
from cyberbattle.utils.file_utils import load_yaml # noqa: E402

class CyberBattleEnv(gym.Env):
    """ The simulation starts from a random initial node or a fixed one.
        The simulation ends if either the attacker reaches its goal, the defender reaches its goal, or a maximum number of actions is reached,
        or if one of the defender's constraints is not met (e.g. SLA).
    """

    metadata = {'render.modes': ['human']}

    @property
    def name(self) -> str:
        return "CyberBattleEnv"

    def __init__(self,
                 initial_environment: model.Model,
                 winning_reward=0,
                 losing_reward=-100,
                 random_starter_node=True,
                 absolute_reward=False, # reward = max(0, reward) at each timestep
                 stop_at_goal_reached=False, # stop the simulation when the attacker reaches its goal, done = True
                 rewards_dict=None,
                 penalties_dict=None,
                 isolation_filter_threshold=0.1, # minimum % of the network needed to be discoverable, ownable, disruptable from a node to use it as a starter node
                 max_services_per_node=10,
                 goal="control", # goal, can be general or specific node oriented
                 switch_interest_node_interval=1, # switch the interest node periodically with this period of episodes
                 interest_node_value = 1000,
                 static_defender_agent: Optional[DefenderAgent] = None,
                 static_defender_eviction_goal=True, # if the defender goal is to evict the attacker from the network
                 episode_iterations=100,  # cut-off for the episode
                 proportional_cutoff_coefficient=0, # other cut-off proportional to the number of nodes in the topology, -1 if not present
                 max_num_trials_find_feasible_starter_node=1000, # maximum number of trials to find a feasible starter node
                 verbose=0, # 0 nothing, 1 print only training information, 2 print also episode information, 3 print also single iteration information
                 logger=None,
                 **kwargs
                 ):
        self.environment = None
        self.winning_reward = winning_reward
        self.losing_reward = losing_reward
        self.random_starter_node = random_starter_node
        self.absolute_reward = absolute_reward
        self.stop_at_goal_reached = stop_at_goal_reached
        self.rewards_dict = rewards_dict
        self.penalties_dict = penalties_dict
        self.isolation_filter_threshold = isolation_filter_threshold
        self.max_services_per_node = max_services_per_node
        if goal is not None:
            self.goal = goal.lower()
        else:
            self.goal = goal
        self.interest_node_value = interest_node_value
        self.switch_interest_node_interval = switch_interest_node_interval
        self.static_defender_agent = static_defender_agent
        self.static_defender_eviction_goal = static_defender_eviction_goal
        self.episode_iterations = episode_iterations
        self.proportional_cutoff_coefficient = proportional_cutoff_coefficient
        self.max_num_trials_find_feasible_starter_node = max_num_trials_find_feasible_starter_node
        self.verbose = verbose
        self.logger = logger
        self.__initial_environment: model.Model = initial_environment
        self.done = False
        self.num_nodes = len(self.__initial_environment.network.nodes)
        self.episode_id = 0

        # in case of an experiment targeting a specific node, switch periodically
        if self.switch_interest_node_interval and self.goal.lower().endswith("node"):
            self.switch_interest_node()

        self.action_execution_time = 0
        self.defender_execution_time = 0
        self.check_end_time = 0

        self.action_space = gym.spaces.Discrete(1) # fictious
        self.observation_space = gym.spaces.Discrete(1) # fictious

        # Read default dicts if not provided to avoid errors (e.g. GAE not using rewards)
        if not self.rewards_dict:
            self.rewards_dict = load_yaml(os.path.join(project_root, "cyberbattle", "agents", "config", "rewards_config.yaml"))["rewards_dict"][self.goal.lower()]
        if not self.penalties_dict:
            self.penalties_dict = load_yaml(os.path.join(project_root, "cyberbattle", "agents", "config", "rewards_config.yaml"))["penalties_dict"][self.goal.lower()]

        # Reset environment to start the simulation
        self.reset_env()

    # Periodically switch the interest node if games with node of interest are played
    def switch_interest_node(self):
        if self.episode_id % self.switch_interest_node_interval == 0:
            self.interest_node = random.choice(list(self.__initial_environment.network.nodes))
            if self.verbose > 1:
                self.logger.info("Switching interest node to %s", self.interest_node)

    # Function to reset the environment, called at the beginning of each episode
    def reset_env(self):
        if self.verbose > 1:
            if self.action_execution_time != 0: # if the episode has really concluded and this is not the first time the reset is called
                self.logger.info("Action execution time in the episode: %s", self.action_execution_time)
                self.logger.info("Defender execution time in the episode: %s", self.defender_execution_time)
                self.logger.info("Check end time in the episode: %s", self.check_end_time)
        start_time = time.time()
        if self.environment:  # ensure garbage collection of the object
            del self.environment

        # deep copy because modifications could be done
        self.environment = copy.deepcopy(self.__initial_environment.network)
        self.access_graph = self.__initial_environment.access_graph
        self.knows_graph = self.__initial_environment.knows_graph
        self.dos_graph = self.__initial_environment.dos_graph
        try:
            self.pick_starter_node()
        except model.NoSuitableStarterNode:  # if no suitable starter node found, push the exception to the switcher that will choose another environment
            self.logger.info("No suitable starter node found, pushing the exception to the switcher")
            raise model.NoSuitableStarterNode("Could not find a suitable starter node")
        self.network_availability = 1.0
        self.episode_id += 1
        self.evicted = False
        self.done = False
        self.num_iterations = 0
        self.discovered_amount = 0
        self.overall_reimaged = []
        self.num_events = 0
        self.discovered_nodes: List[model.NodeID] = []
        self.owned_nodes: List[model.NodeID] = []
        self.episode_rewards: List[float] = []
        self.sampled_actions = [] # in case the episode is used to sample valid actions
        # Set the actuator used to execute actions in the simulation environment
        self._actuator = attacker_actions.AttackerAgentActions(self.environment,
                                                penalties_dict=self.penalties_dict,
                                                rewards_dict=self.rewards_dict,
                                                logger=self.logger,
                                                verbose=self.verbose,
                                                )
        self._defender_actuator = static_defender_actions.StaticDefenderAgentActions(self.environment, logger=self.logger, verbose=self.verbose)
        self.stepcount = 0
        self.done = False

        # start with only the starting node as discovered/owned node
        self.discovered_nodes.append(self.starter_node)
        self.owned_nodes.append(self.starter_node)

        self.reset_time = time.time() - start_time
        if self.verbose > 1:
            self.logger.info("Reset cyberbattleenv time: %s", self.reset_time)
        self.action_execution_time = 0
        self.defender_execution_time = 0
        self.check_end_time = 0

    # Function to reset the network to its initial state
    def pick_starter_node(self) -> None:
        if not self.random_starter_node:
            self.starter_node, _ = list(self.environment.nodes(data=True))[0] # take the first node in order
        else:  # Random starter node at every episode
            # While loop to be sure to select a starter node respecting the necessary conditions
            self.num_trials = 0 # if possible within a maximum number of trials
            while True:
                # stop looking for non-isolated nodes in the topology after many attempts
                self.num_trials += 1
                if self.num_trials > self.max_num_trials_find_feasible_starter_node:
                    raise model.NoSuitableStarterNode("Could not find a suitable starter node")

                self.starter_node, _ = list(self.environment.nodes(data=True))[random.randrange(len(self.environment.nodes))]
                if self.verbose > 2:
                    self.logger.info("Selected starter node %s as trial", self.starter_node)
                # check if the node reaches the minimum number required
                self.shortest_paths_starter_control = copy.deepcopy(self.__initial_environment.access_shortest_paths[self.starter_node])
                threshold_count = self.isolation_filter_threshold * len(self.shortest_paths_starter_control) # minimum amount required to reach
                # control goal
                self.shortest_paths_starter_control.pop(self.starter_node) # remove the node itself, always reachable in 0 steps
                self.ownable_count = sum(1 for value in self.shortest_paths_starter_control.values() if value is not None)
                # discovery goal
                self.shortest_paths_starter_discovery = copy.deepcopy(self.__initial_environment.knows_shortest_paths[self.starter_node])
                self.shortest_paths_starter_discovery.pop(self.starter_node) # remove the node itself, always discoverable in 0 steps
                self.discoverable_count = sum(1 for value in self.shortest_paths_starter_discovery.values() if value is not None)
                # disruption goal
                self.shortest_paths_starter_disruption = copy.deepcopy(self.__initial_environment.dos_shortest_paths[self.starter_node])
                self.shortest_paths_starter_disruption.pop(self.starter_node)
                self.disruptable_count = sum(1 for value in self.shortest_paths_starter_disruption.values() if value is not None)
                # Check isolation based on the specific goal considered for this instance of the environment
                if self.goal == "disruption":
                    if self.disruptable_count < threshold_count:
                        # pick new starter node
                        if self.verbose > 2:
                            self.logger.info("This node is isolated, can disrupt maximum %s while it should disrupt minimum %s", self.disruptable_count, threshold_count)
                        continue
                    # set maximum number disruptable if not isolated
                    if self.proportional_cutoff_coefficient:
                        self.proportional_nodes = self.disruptable_count
                elif self.goal == "control":
                    if self.ownable_count < threshold_count:
                        # pick new starter node
                        if self.verbose > 2:
                            self.logger.info("This node is isolated, can control maximum %s while it should control minimum %s",
                                        self.ownable_count, threshold_count)
                        continue
                    # set maximum number controllable if not isolated
                    if self.proportional_cutoff_coefficient:
                        self.proportional_nodes = self.ownable_count
                elif self.goal == "discovery":
                    if self.discoverable_count < threshold_count:
                        # pick new starter node
                        if self.verbose > 2:
                            self.logger.info(
                                "This node is isolated, can discover maximum %s while it should discover minimum %s",
                                self.discoverable_count, threshold_count)
                        continue
                    # set maximum number discoverable if not isolated
                    if self.proportional_cutoff_coefficient:
                        self.proportional_nodes = self.discoverable_count
                elif self.goal.endswith("node"):
                    if self.starter_node == self.interest_node: # invalid choice, otherwise it has already won
                        continue
                    # in node-specific games the starter node must be able to control/discover/disrupt the node of interest
                    if self.goal == "control_node":
                        if self.interest_node not in self.shortest_paths_starter_control or self.shortest_paths_starter_control[self.interest_node] is None:
                            if self.verbose > 2:
                                self.logger.info("Node of interest not controllable from starter node")
                            continue
                        if self.proportional_cutoff_coefficient:
                            self.proportional_nodes = self.ownable_count
                    elif self.goal == "discovery_node":
                        if self.interest_node not in self.shortest_paths_starter_discovery or \
                            self.shortest_paths_starter_discovery[self.interest_node] is None:
                            if self.verbose > 2:
                                self.logger.info("Node of interest not discoverable from starter node")
                            continue
                        if self.proportional_cutoff_coefficient:
                            self.proportional_nodes = self.discoverable_count
                    elif self.goal == "disruption_node":
                        if self.interest_node not in self.shortest_paths_starter_disruption or \
                            self.shortest_paths_starter_disruption[self.interest_node] is None:
                            if self.verbose > 2:
                                self.logger.info("Node of interest not disruptable from starter node")
                            continue
                        if self.proportional_cutoff_coefficient:
                            self.proportional_nodes = self.disruptable_count
                    # interest node value to be set much higher than the others
                    self.set_node_property(self.interest_node, "value", self.interest_node_value)
                # determine the amount of discovery actions that can be achieved (nodes, data, visibility, ...)
                self.discoverable_amount = 0
                for node in self.environment.nodes:
                    if node in self.shortest_paths_starter_discovery or node == self.starter_node:
                        self.discoverable_amount += 1  # count the discovery of the node
                    if node in self.shortest_paths_starter_control or node == self.starter_node:
                        node_info = self.get_node(node)
                        if node_info.has_data:
                            self.discoverable_amount += 2  # data to collect and exfiltrate
                        if not node_info.visible:
                            self.discoverable_amount += 1  # need to increase the amount of visibility in the node, discovering properties
                self.num_trials = 0  # reset once here
                for node in self.environment.nodes:
                    self.set_node_property(node, "agent_installed", False)
                break
            if self.verbose > 1:
                self.logger.info("Starter node %s selected", self.starter_node)
            # property common to all games
            self.set_node_property(self.starter_node, "agent_installed", True)

    # Logic used by all step function, regardless if it is the DRL training or GAE training
    def step_attacker_env(self, source_node, target_node, vulnerability_ID, outcome_desired):
        if self.done:
            self.logger.warning("New episode must be started with env.reset()")
            raise RuntimeError("New episode must be started with env.reset()")
        self.stepcount += 1
        start_time = time.time()
        if self.verbose > 2:
            self.logger.info("Step %s: source node %s, target node %s, vulnerability %s, outcome %s", self.stepcount, source_node, target_node, vulnerability_ID, outcome_desired)
        if source_node == target_node:  # use local (or remote) vulnerability on the node in case of the same source and target node
            if self.verbose > 2:
                self.logger.info("The vulnerability selected is LOCAL")
            result, self.vulnerability_type = self._actuator.exploit_local_vulnerability(source_node, vulnerability_ID, outcome_desired), "local"
        else:  # the nodes are different, use remote vulnerability
            if self.verbose > 2:
                self.logger.info("The vulnerability selected is REMOTE")
            result, self.vulnerability_type = self._actuator.exploit_remote_vulnerability(source_node, target_node, vulnerability_ID, outcome_desired), "remote"
        self.action_execution_time += time.time() - start_time
        self.reward = result.reward

        # update lists handling the episode based on the outcome
        self.outcome = self.update_episode_by_outcome(result.outcome, target_node)

        # In case of node-specific games, the reward is zeroed if the interest node was reachable directly by any other source node and the attacker is not using it
        if self.goal.endswith("node"):
            if self.interest_node in self.discovered_nodes and target_node != self.interest_node:
                if self.verbose > 2:
                    self.logger.info("Zeroing reward since the interest node %s was reachable and not used", self.interest_node)
                self.reward = 0
            # else it remains the same

        start_time = time.time()
        # Execute the defender step if involved
        if self.static_defender_agent:
            self.static_defender_step()

        self.defender_execution_time += time.time() - start_time

        start_time = time.time()
        # Check whether there has been some ending conditions
        self.end_episode_reason = 0
        self.truncated = False
        if not self.done:
            if self.attacker_goal_reached(): # attacker goal reached (vary per goal)
                if self.verbose > 2:
                    self.logger.info("Attacker won the episode! Assigning winning reward %.1f..", self.winning_reward)
                if self.goal == "disruption" or self.stop_at_goal_reached:
                    # in case of disruption we need to stop in any case, if it won it means it has killed them all
                    self.done = True
                self.reward = self.winning_reward
                self.end_episode_reason = 1  # attacker goal reached
            elif self.check_end_game(): # losing conditions depend also on the goal
                if self.verbose > 2:
                    self.logger.info("Game lost due to end game condition reached! Assigning losing reward %.1f..", self.losing_reward)
                self.done = True
                self.reward = self.losing_reward
                self.end_episode_reason = 2  # lost game
            elif self.proportional_cutoff_coefficient and self.proportional_cutoff_reached(): # cut-off proportional to the number of nodes
                if self.verbose > 2:
                    self.logger.info("Proportional cut-off %d reached, stopping the episode..", self.proportional_cutoff_coefficient)
                self.truncated = True
                self.end_episode_reason = 3 # cut-off
            elif self.num_iterations >= self.episode_iterations: # cut-off constant regardless of the number of nodes
                if self.verbose > 2:
                    self.logger.info("Episode iterations limit %d reached, stopping the episode..", self.episode_iterations)
                self.truncated = True
                self.end_episode_reason = 3

        self.check_end_time += time.time() - start_time

        if self.absolute_reward:
            self.reward = max(0, self.reward)
            if self.verbose > 2:
                self.logger.info("Reward absolute value: %s", self.reward)

        self.source_node = source_node
        self.target_node = target_node
        self.vulnerability_ID = vulnerability_ID
        self.outcome_desired = outcome_desired
        self.outcome_obtained = result.outcome
        self.network_availability = len([node for node in self.discovered_nodes if
                                      self.get_node(node).status == model.MachineStatus.Running]) / len(self.discovered_nodes)
        if self.verbose > 2:
            self.logger.info("Network availability after step: %f", self.network_availability)
        self.episode_rewards.append(self.reward)
        self.num_iterations += 1

    # Function to update the environment data structures based on the outcome of the action taken
    def update_episode_by_outcome(self, outcome, target_node):
        if isinstance(outcome, model.Reconnaissance):
            # update discovered nodes
            newly_discovered_nodes_count = 0
            outcome.new_nodes = []
            for node in outcome.nodes:
                if node not in self.discovered_nodes:
                    self.discovered_nodes.append(node)
                    newly_discovered_nodes_count += 1
                    outcome.new_nodes.append(node)
            self.discovered_amount += newly_discovered_nodes_count
        elif isinstance(outcome, model.CredentialAccess) or isinstance(outcome, model.LateralMove):
            # update owned nodes
            self.owned_nodes.append(target_node)
        elif isinstance(outcome, model.Collection) or isinstance(outcome, model.Exfiltration) or isinstance(outcome, model.Discovery):
            self.discovered_amount += 1  # discovered or exfiltrated some data (updated before for the number of nodes)
        return outcome

    # Function to perform one step using a static defender heuristic agent
    def static_defender_step(self):
        self._defender_actuator.on_attacker_step_taken() # the defender can reimage nodes or add external events
        new_events, self.changed_nodes = self.static_defender_agent.step(self, self._defender_actuator, self.stepcount)
        self.num_events += new_events
        if isinstance(self.static_defender_agent, ScanAndReimageCompromisedMachines):
            self.last_reimaged = self.changed_nodes
            self.overall_reimaged.extend(self.changed_nodes)
            if self.changed_nodes != []:
                for node in self.changed_nodes:
                    self.owned_nodes.remove(node)
            # Reiterate all nodes each time to see if they have been reimaged and persistence flag was set, hence we re-owned them
            for node in self.environment.nodes:
                node_info = self.get_node(node)
                if node_info.agent_installed and node_info.status == model.MachineStatus.Running and node not in self.owned_nodes:
                    self.owned_nodes.append(node)
        # no modification for ExternalRandomEvents static defender

    # Check if static defender goal is reached, i.e. all owned nodes are evicted
    def defender_goal_reached(self) -> bool:
        return self.static_defender_eviction_goal and len(self.owned_nodes) == 0

    # Function to check whether the episode has ended in a way that the agent lost
    def check_end_game(self):
        # in node-specific case, they should not kill the node of interest
        if self.goal == "control_node" or self.goal == "discovery_node":
            if self.get_node(self.interest_node).status == model.MachineStatus.Stopped:
                if self.verbose > 1:
                    self.logger.info("Episode ended since node of interest %s has been killed!", self.interest_node)
                return True
        # general condition for all cases
        # if disruption goal, it still fails if it kills all nodes including starter node before disrupting the maximum it can
        at_least_one_running_node = False
        for node in self.owned_nodes:
            if self.get_node(node).status == model.MachineStatus.Running:
                at_least_one_running_node = True
        if not at_least_one_running_node:
            if self.verbose > 1:
                self.logger.info("Episode ended since all nodes have been killed!")
        return not at_least_one_running_node

    # Function to check if the proportional cut-off has been reached, i.e. the number of iterations is greater than the number of nodes times the coefficient
    def proportional_cutoff_reached(self):
        if self.num_iterations >= self.proportional_nodes * self.proportional_cutoff_coefficient:
            return True
        return False

    # Function to set the proportional cutoff coefficient
    def set_proportional_cutoff_coefficient(self, coefficient):
        self.proportional_cutoff_coefficient = coefficient

    # Function to check whether the game is ended by the attacker reaching its goal, which depends on the goal of the game
    def attacker_goal_reached(self) -> bool:
        if self.goal == "control":
            root_nodes = [node for node in self.owned_nodes if self.get_node(node).privilege_level == model.PrivilegeLevel.ROOT and node != self.starter_node]
            if len(root_nodes) == self.ownable_count:
                if self.verbose > 1:
                    self.logger.info("Attacker won the episode since all nodes controlled at root level...")
                return True # in control games, episode finished when all nodes potentially controllable are controlled
        elif self.goal == "disruption":
            stopped_nodes = [node for node in self.discovered_nodes if self.get_node(node).status == model.MachineStatus.Stopped and node != self.starter_node]
            if len(stopped_nodes) == self.disruptable_count: # in disruption, if all nodes potentially disruptable are disrupted
                if self.verbose > 1:
                    self.logger.info("Attacker won the episode since all nodes except the starter node have been disrupted...")
                return True
        elif self.goal == "discovery":
            discovered_nodes = [node for node in self.discovered_nodes if node != self.starter_node]
            data_owning_nodes = [node for node in self.discovered_nodes if self.get_node(node).has_data]
            data_collected_not_exfiltred = [node for node in self.discovered_nodes if self.get_node(node).data_collected and not self.get_node(node).data_exfiltrated]
            if len(discovered_nodes) == self.discoverable_count and len(data_owning_nodes) == 0 and len(data_collected_not_exfiltred) == 0:
                if self.verbose > 1:
                    self.logger.info("Attacker won the episode since all nodes and data have been discovered and exfiltrated...")
                return True # in discovery games, episode finished when all nodes are discovered and no data is left
        elif self.goal == "control_node":
            if self.interest_node in self.owned_nodes and self.get_node(self.interest_node).privilege_level == model.PrivilegeLevel.ROOT:
                if self.verbose > 1:
                    self.logger.info("Attacker won the episode since the node of interest has been controlled at root level...")
                return True # the goal is to control the node of interest
        elif self.goal == "discovery_node":
            if self.interest_node in self.discovered_nodes:
                node_info = self.get_node(self.interest_node)
                if node_info.visible:
                    if node_info.has_data:
                        if node_info.data_collected and node_info.data_exfiltrated:
                            if self.verbose > 1:
                                self.logger.info(
                                    "Attacker won the episode since the node of interest has been discovered and data exfiltrated...")
                            return True # the goal is to discover the node of interest and all its elements if it has them
                    else:
                        if node_info.visible:
                            if self.verbose > 1:
                                self.logger.info("Attacker won the episode since the node of interest has been discovered (data was also not present)...")
                            return True
                        return True # case with no internal element
        elif self.goal == "disruption_node":
            if self.get_node(self.interest_node).status == model.MachineStatus.Stopped:
                if self.verbose > 1:
                    self.logger.info("Attacker won the episode since the node of interest has been disrupted...")
                return True # node to disrupt is disrupted
        return False

    # Called at the end of the episode to gather statistics:
    def get_statistics(self):
        owned_nodes = [node_id for node_id, node_data in self.environment.nodes.items() if node_data["data"].agent_installed]
        discovered_nodes = self.discovered_nodes
        not_discovered_nodes = [node_id for node_id, node_data in self.environment.nodes.items() if node_id not in self.discovered_nodes]
        disrupted_nodes = [node_id for node_id in self.discovered_nodes if self.get_node(node_id).status == model.MachineStatus.Stopped]
        self.network_availability = len([node for node in self.discovered_nodes if
                                         self.get_node(node).status == model.MachineStatus.Running]) / len(self.discovered_nodes)
        return len(owned_nodes), len(discovered_nodes), len(not_discovered_nodes), len(disrupted_nodes), self.num_nodes, self.ownable_count, self.discoverable_count, self.disruptable_count, self.network_availability, len(self.overall_reimaged), self.num_events, self.discovered_amount, self.discoverable_amount, self.attacker_goal_reached()

    # Function to get the list of alive nodes, i.e. nodes that are running
    def get_alive_nodes(self):
        return [node_id for node_id in self.discovered_nodes if self.get_node(node_id).status == model.MachineStatus.Running]

    # Function to get the list of nodes in the environment
    def get_nodes(self):
        return self.environment.nodes

    # Function to get the environment graph, used to access the network topology
    def get_graph(self):
        return self.environment

    # Function to get the actuator used to execute actions in the simulation environment
    def get_actuator(self):
        return self._actuator

    # Function to get the information of a specific node
    def get_node(self, node_id):
        return self.environment.nodes(data=True)[node_id]["data"]

    # Function to get the index of a service in a node's services list
    def get_service_index(self, port_name: model.PortName, node_info) -> int:
        for service in node_info.services:
            if service.name == port_name:
                return node_info.services.index(service)
        return -1

    # Function to get the list of discovered but not yet owned nodes
    def get_discovered_not_owned_nodes(self):
        discovered_not_owned_nodes = [node_id for node_id in self.discovered_nodes if node_id not in self.owned_nodes]
        return discovered_not_owned_nodes

    # Function to get the list of reachable nodes from the starter node based on the goal
    def get_reachable_nodes(self, goal="control"):
        reachable_nodes = []
        if goal == "control":
            paths = self.__initial_environment.access_shortest_paths
        elif goal == "discovery":
            paths = self.__initial_environment.knows_shortest_paths
        elif goal == "disruption":
            paths = self.__initial_environment.dos_shortest_paths
        else:
            return None # invalid goal for paths
        for node in paths[self.starter_node]:
            reachable_nodes.append(node)
        return reachable_nodes

    # Function to get the initial environment used to create the CyberBattleEnv instance
    def get_initial_environment(self):
        return self.__initial_environment

    # Function to set a property of a node, e.g. visibility, data collected, etc.
    def set_node_property(self, node_id, property_name, value):
        # Check if the node exists
        if node_id in self.environment.nodes:
            node_data = self.environment.nodes[node_id]["data"]

            # Use setattr to dynamically set the attribute on the node's data
            if hasattr(node_data, property_name):
                setattr(node_data, property_name, value)
            else:
                raise AttributeError(f"Property '{property_name}' not found in NodeInfo.")
        else:
            raise KeyError(f"Node with ID {node_id} not found in the environment.")

    # FUnction to set the cut-off for the episode, i.e. the maximum number of iterations, alternative to proportional cut-off
    def set_cut_off(self, cut_off):
        self.episode_iterations = cut_off

    # Function to set the winning reward
    def set_winning_reward(self, winning_reward):
        self.winning_reward = winning_reward

    # Function to set the goal of the attacker agent
    def set_goal(self, goal):
        self.goal = goal

    # Function to set the threshold determining a minimum amount of reachable nodes to not consider starter node isolated
    def set_isolation_filter_threshold(self, threshold):
        self.isolation_filter_threshold = threshold

    # Function to set the random starter node flag
    def set_random_starter_node(self, random_starter_node):
        self.random_starter_node = random_starter_node

    # Function to set the stop at goal reached flag
    def set_stop_at_goal_reached(self, stop_at_goal_reached):
        self.stop_at_goal_reached = stop_at_goal_reached

    # Function to print the information of a specific node
    def print_node_info(self, node_index, node_info):
        if self.get_node(node_index).agent_installed:
            print("discovery status: owned")
        elif node_index in self.discovered_nodes:
            print("discovery status: discovered")
        else:
            print("discovery status: not discovered")
        for key, value in node_info.items():
            print(f'{key}: {value}')
        print()

    # Function to print the information of all nodes in the environment
    def print_nodes_info(self):
        owned_nodes = [node_id for node_id, node_data in self.environment.nodes() if node_data.agent_installed]
        discovered_nodes = [node_id for node_id in self.discovered_nodes if
                            not self.get_node(node_id).agent_installed]
        not_discovered_nodes = [node_id for node_id, node_data in self.environment.nodes() if
                                not node_data.agent_installed and node_id not in self.discovered_nodes]
        max_width = max(len(owned_nodes), len(discovered_nodes), len(not_discovered_nodes))
        owned_texts = []
        discovered_texts = []
        not_discovered_texts = []
        for i in range(max_width):
            if i < len(owned_nodes):
                owned = owned_nodes[i]
                owned_color = "\033[0m"
                owned_texts.append(f"{owned_color}{owned}\033[0m")
            else:
                owned_texts.append("")

            if i < len(discovered_nodes):
                discovered = discovered_nodes[i]
                discovered_color =  "\033[0m"
                discovered_texts.append(f"{discovered_color}{discovered}\033[0m")
            else:
                discovered_texts.append("")

            if i < len(not_discovered_nodes):
                not_discovered = not_discovered_nodes[i]
                not_discovered_texts.append(not_discovered)
            else:
                not_discovered_texts.append("")

        owned_text = "Owned: { " + ", ".join(owned_texts) + " }"
        discovered_text = "Discovered: { " + ", ".join(discovered_texts) + " }"
        not_discovered_text = "Not discovered: { " + ", ".join(not_discovered_texts) + " }"

        print(owned_text)
        print(discovered_text)
        print(not_discovered_text)

    # Used to sample valid actions (e.g. to expose the GAE to many configurations)
    def sample_valid_action(self):
        selected = False
        valid_nodes = list(self.discovered_nodes)
        pairs = []
        vulnerabilities_selected = []
        source_node_index, target_node_index, vulnerability_ID, vulnerability_outcome_class, interested_node_index, vulnerability_type, vulnerability_outcome_str = None, None, None, None, None, None, None
        while True:
            # update list once nodes may be terminated
            for node in valid_nodes:
                if not self.get_node(node).status == model.MachineStatus.Running:
                    valid_nodes.remove(node)
            if len(valid_nodes) == 0:
                self.blocked_graph = True
                return None, None, None, None
            owned_nodes = [node for node in valid_nodes if self.get_node(node).agent_installed]
            if len(pairs) >= len(owned_nodes) * len(valid_nodes): # once all nodes selected at least once, can stop, it is enough
                break
            if len(owned_nodes) == 0:
                self.blocked_graph = True
                return None, None, None, None
            source_node_index = random.choice(owned_nodes)
            target_node_index = random.choice(valid_nodes)
            pairs.append((source_node_index, target_node_index))
            # Once selected the source and target nodes, select the vulnerability
            # Use logic to select a valid one
            if source_node_index == target_node_index:
                for vulnerability_ID in self.get_node(source_node_index).vulnerabilities:
                    vulnerability = self.get_node(source_node_index).vulnerabilities[vulnerability_ID]
                    if vulnerability.privileges_required:
                        if not self.get_node(source_node_index).privilege_level >= vulnerability.privileges_required:
                            continue
                    for result in vulnerability.results:
                        if self.get_node(source_node_index).privilege_level == model.PrivilegeLevel.ROOT and isinstance(result.outcome, model.PrivilegeEscalation):
                            continue
                        if not self.get_node(source_node_index).has_data and isinstance(result.outcome, model.Collection):
                            continue
                        if self.get_node(source_node_index).visible and isinstance(result.outcome, model.Discovery):
                            continue
                        if self.get_node(source_node_index).defense_evasion and isinstance(result.outcome, model.DefenseEvasion):
                            continue
                        if (self.get_node(source_node_index).data_exfiltrated or not self.get_node(source_node_index).data_collected) and isinstance(result.outcome, model.Exfiltration):
                            continue
                        if self.get_node(source_node_index).persistence and isinstance(result.outcome, model.Persistence):
                            continue
                        vulnerabilities_selected.append((source_node_index, target_node_index, source_node_index, vulnerability.vulnerability_ID, result.type_str, result.outcome_str, result.outcome))
            else:
                for vulnerability_ID in self.get_node(target_node_index).vulnerabilities:
                    vulnerability = self.get_node(target_node_index).vulnerabilities[vulnerability_ID]
                    if vulnerability.privileges_required:
                        if not self.get_node(target_node_index).privilege_level >= vulnerability.privileges_required:
                            continue
                    target_node_is_listening = vulnerability.port in [i.name for i in self.get_node(target_node_index).services if i.running]
                    if not target_node_is_listening:
                        continue
                    for result in vulnerability.results:
                        if result.type == VulnerabilityType.REMOTE:
                            if self.get_node(target_node_index).agent_installed and (isinstance(result.outcome, model.CredentialAccess) or isinstance(result.outcome, model.LateralMove)):
                                continue
                            if not self.get_node(target_node_index).has_data and isinstance(result.outcome, model.Collection):
                                continue
                            if self.get_node(target_node_index).visible and isinstance(result.outcome,
                                                                                                     model.Discovery):
                                continue
                            if self.get_node(target_node_index).defense_evasion and isinstance(result.outcome, model.DefenseEvasion):
                                continue
                            if (self.get_node(target_node_index).data_exfiltrated or not self.get_node(target_node_index).data_collected) and isinstance(result.outcome, model.Exfiltration):
                                continue
                            if self.get_node(target_node_index).persistence and isinstance(result.outcome, model.Persistence):
                                continue
                            if self.access_graph.has_edge(source_node_index, target_node_index):
                                edge_data = self.access_graph.get_edge_data(source_node_index, target_node_index)
                                vulnerabilities = edge_data.get('vulnerabilities', [])
                                vulnerabilities_IDs = [ vulnerability[0] for vulnerability in vulnerabilities ]
                                vulnerability_present = vulnerability_ID in vulnerabilities_IDs
                                if vulnerability_present:
                                    vulnerabilities_selected.append((source_node_index, target_node_index, target_node_index, vulnerability.vulnerability_ID, result.type_str, result.outcome_str, result.outcome))
        if vulnerabilities_selected:
            # Prioritize vulnerabilities selected according to a proper logic
            selected = False

            non_dos_vulnerabilities = []
            dos_vulnerabilities = []
            for vulnerability in vulnerabilities_selected:
                source_node_index, target_node_index, interested_node_index, vulnerability_ID, vulnerability_type, vulnerability_outcome_str, vulnerability_outcome_class = vulnerability
                if source_node_index == target_node_index and vulnerability_outcome_str.lower() == 'dos':
                    continue
                if vulnerability_outcome_str.lower() == 'dos':
                    dos_vulnerabilities.append(vulnerability)
                else:
                    non_dos_vulnerabilities.append(vulnerability)
            # prioritizing non DOS, since they disrupt elements
            while non_dos_vulnerabilities:
                vulnerability = random.choice(non_dos_vulnerabilities)
                source_node_index, target_node_index, interested_node_index, vulnerability_ID, vulnerability_type, vulnerability_outcome_str, vulnerability_outcome_class = vulnerability
                non_dos_vulnerabilities.remove(vulnerability)
                if (interested_node_index, vulnerability_ID, vulnerability_type,
                    vulnerability_outcome_str) not in self.sampled_actions:
                    selected = True
                    break
            if not selected:
                # If no non-DOS vulnerability was suitable, try DOS vulnerabilities
                while dos_vulnerabilities:
                    vulnerability = random.choice(dos_vulnerabilities)
                    source_node_index, target_node_index, interested_node_index, vulnerability_ID, vulnerability_type, vulnerability_outcome_str, vulnerability_outcome_class = vulnerability
                    dos_vulnerabilities.remove(vulnerability)
                    if (interested_node_index, vulnerability_ID, vulnerability_type,
                        vulnerability_outcome_str) not in self.sampled_actions:
                        selected = True
                        break
                if not selected:
                    self.blocked_graph = True
        else:
            self.blocked_graph = True
        if selected:
            if self.verbose > 2:
                self.logger.info("Sampled action: source node %s, target node %s, vulnerability %s, outcome %s", source_node_index, target_node_index, vulnerability_ID, vulnerability_outcome_class)
            self.sampled_actions.append((interested_node_index, vulnerability_ID, vulnerability_type, vulnerability_outcome_str))
            return source_node_index, target_node_index, vulnerability_ID, vulnerability_outcome_class
        else:
            return None, None, None, None

    # Function to update the environment used by the CyberBattleEnv instance
    def update_environment(self, environment):
        self.environment = environment

    # Function to seed the environment, setting the random seed for reproducibility
    def seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            self._seed = seed
            random.seed(seed)
            numpy.random.seed(seed)
            return

    def close(self) -> None:
        return None
