# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    callbacks.py
    Script containing the classes for logging training and validation statistics and handling early stopping
    The classes are coded in the format required by the stable-baselines3 library
"""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import sys
import os
from cyberbattle.simulation import model as m
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
# Callback logging some training statistics additional to the one provided by stable-baselines3
class TrainingCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.env = env
        self.verbose = verbose

    def _on_training_start(self) -> None:
        self.local_remote_actions_counts = np.zeros(2)
        self.local_remote_actions_success = np.zeros(2)
        self.vulnerability_counts = {}
        self.vulnerability_success = {}
        self.outcome_actions_counts = {}
        self.target_node_tag_counts = {}
        self.source_nodes = []
        self.target_nodes = []
        self.min_distance_actions = []
        self.invalid_actions = 0

    def _on_step(self) -> bool:
        # Gather at every step the statistics and update variables
        done = self.locals["dones"][0]
        info = self.locals["infos"][0]

        self.source_nodes.append(info['source_node'])
        self.source_nodes = list(set(self.source_nodes))
        self.target_nodes.append(info['target_node'])
        self.target_nodes = list(set(self.target_nodes))

        self.vulnerability_counts[info['vulnerability']] = self.vulnerability_counts.get(info['vulnerability'], 0) + 1
        self.target_node_tag_counts[info['target_node_tag']] = self.target_node_tag_counts.get(info['target_node_tag'], 0) + 1
        self.outcome_actions_counts[info['outcome']] = self.outcome_actions_counts.get(info['outcome'], 0) + 1
        if 'min_distance_action' in info:
            self.min_distance_actions.append(info['min_distance_action'])

        if issubclass(type(info['outcome_class']), m.InvalidAction): # big class having as sub-classess all invalid cases
            self.invalid_actions += 1
        else:
            self.vulnerability_success[info['vulnerability']] = self.vulnerability_success.get(info['vulnerability'], 0) + 1
        if info['vulnerability_type'] == "local":
            self.local_remote_actions_counts[0] += 1
            if not issubclass(type(info['outcome_class']), m.InvalidAction):
                self.local_remote_actions_success[0] += 1
        elif info['vulnerability_type'] == "remote":
            self.local_remote_actions_counts[1] += 1
            if not issubclass(type(info['outcome_class']), m.InvalidAction):
                self.local_remote_actions_success[1] += 1

        # Log only once done is true the final episode statistics
        if done:
            owned_nodes, discovered_nodes, not_discovered_nodes, disrupted_nodes, num_nodes, reachable_count, discoverable_count, disruptable_count, network_availability, reimaged_nodes, num_events, discovered_amount, discoverable_amount, _ = self.env.envs[0].get_statistics()
            self.logger.record("train/Owned nodes", owned_nodes)
            self.logger.record("train/Discovered nodes", discovered_nodes)
            self.logger.record("train/Discoverable amount", discoverable_amount)
            self.logger.record("train/Discovered amount", discovered_amount)
            self.logger.record("train/Not discovered nodes", not_discovered_nodes)
            self.logger.record("train/Total nodes", num_nodes)
            self.logger.record("train/Disrupted nodes", disrupted_nodes)
            self.logger.record("train/Reimaged nodes", reimaged_nodes)
            self.logger.record("train/Network availability", network_availability)
            self.logger.record("train/Number of events", num_events)
            self.min_distance_actions = [x for x in self.min_distance_actions if x is not None]
            if len(self.min_distance_actions) == 0:
                self.logger.record("train/Minimum distance action", 0)
            self.logger.record("train/Minimum distance action", np.mean(self.min_distance_actions))
            if info['end_episode_reason'] == 1:
                self.logger.record("train/Episode goal reached", 1)
            else:
                self.logger.record("train/Episode goal reached", 0)
            if info['end_episode_reason'] == 2:
                self.logger.record("train/Episode lost", 1)
            else:
                self.logger.record("train/Episode lost", 0)

            owned_percentage = owned_nodes / num_nodes
            discovered_percentage = discovered_nodes / num_nodes
            reimaged_percentage = reimaged_nodes / num_nodes
            disrupted_percentage = disrupted_nodes / num_nodes

            self.logger.record("train/Owned nodes percentage", owned_percentage)
            self.logger.record("train/Discovered nodes percentage", discovered_percentage)
            self.logger.record("train/Discovered amount percentage", discovered_amount / discoverable_amount)
            self.logger.record("train/Reimaged nodes percentage", reimaged_percentage)
            self.logger.record("train/Disrupted nodes percentage", disrupted_percentage)
            self.logger.record("train/Relative owned nodes percentage",
                                   owned_nodes / (reachable_count+1)) #re-adding ther starter node
            self.logger.record("train/Relative discovered nodes percentage",
                                      discovered_nodes / (discoverable_count+1))
            self.logger.record("train/Relative discovered amount percentage",
                               (discovered_amount+1) / discoverable_amount) # starter node already included in the second term while not in the first
            self.logger.record("train/Relative disrupted nodes percentage",
                               disrupted_nodes / (disruptable_count+1))

            self.logger.record("train/Owned-discovered ratio", owned_nodes/discovered_nodes)
            self.logger.record("train/Number of source nodes", len(set(self.source_nodes)))
            self.logger.record("train/Number of target nodes", len(set(self.target_nodes)))
            self.logger.record("train/Number of unique vulnerabilities", len(self.vulnerability_counts))
            self.logger.record("actions/train/Invalid actions", self.invalid_actions)
            self.logger.record("actions/train/Success rate for all actions", sum(self.vulnerability_success.values()))
            self.logger.record("actions/train/Total actions", sum(self.vulnerability_counts.values()))
            self.logger.record("actions/train/Local actions count", self.local_remote_actions_counts[0])
            self.logger.record("actions/train/Remote actions count", self.local_remote_actions_counts[1])
            self.logger.record("actions/train/Success rate for local actions", self.local_remote_actions_success[0])
            self.logger.record("actions/train/Success rate for remote actions", self.local_remote_actions_success[1])

            # Log the outcomes preferred by the agent
            for outcome, count in self.outcome_actions_counts.items():
                self.logger.record(f"actions/train/Outcome {outcome} count", count)

            # Logging which target nodes have been preferred (based on their tag)
            for tag, count in self.target_node_tag_counts.items():
                self.logger.record(f"actions/train/Target nodes {tag} count", count)

            # Reset all variables for the next episode
            self.local_remote_actions_counts = np.zeros(2)
            self.local_remote_actions_success = np.zeros(2)
            self.source_nodes = []
            self.target_nodes = []
            self.actions_counts = {}
            self.actions_success = {}
            self.vulnerability_success = {}
            self.vulnerability_counts = {}
            self.invalid_actions = 0
            for outcome in self.outcome_actions_counts:
                self.outcome_actions_counts[outcome] = 0
            for tag in self.target_node_tag_counts:
                self.target_node_tag_counts[tag] = 0

        return True  # Continue training


# Callback logging some validation statistics and handling early stopping
class ValidationCallback(BaseCallback):
    def __init__(self, val_env, n_val_episodes, val_freq, validation_folder_path, output_logger, early_stopping=False, patience=10, verbose=0):
        super(ValidationCallback, self).__init__(verbose)
        self.val_env = val_env
        self.output_logger = output_logger
        self.n_val_episodes = n_val_episodes
        self.val_freq = val_freq  # in time steps
        self.validation_folder_path = validation_folder_path
        self.early_stopping = bool(early_stopping)
        self.patience = patience
        self.best_mean_reward = -np.inf
        self.val_timesteps = 0
        self.current_patience = 0
        self.verbose = verbose

    def _on_step(self) -> bool:
        # Perform validation every val_freq timesteps
        if ((self.val_timesteps % self.val_freq) == 0):
            if self.verbose:
                self.output_logger.info("Validation phase at training step %s (set to be every %s)", self.num_timesteps, self.val_freq)

            # Evaluate the model and log custom metrics after each validation episode
            custom_metrics = self._run_evaluation()
            self._log_tensorboard_custom_metrics(custom_metrics)

            # Save model checkpoint if validation reward is better than the best known
            val_reward = custom_metrics['episode_reward']

            if val_reward > self.best_mean_reward:
                self.model.save(os.path.join(self.validation_folder_path, f"checkpoint_{val_reward}_reward.zip"))
                if self.verbose:
                    self.output_logger.info("Saving new best model with reward %s (previous best reward %s)", val_reward, self.best_mean_reward)
                self.best_mean_reward = val_reward
                self.current_patience = 0
            else:
                # otherwise increase patience
                if self.verbose >= 2:
                    self.output_logger.info("Increasing patience: validation reward did not improve with %s (best reward %s)", val_reward, self.best_mean_reward)
                self.current_patience += 1
            if self.early_stopping:
                # early stopping condition reached
                if self.current_patience >= self.patience:
                    if self.verbose:
                        self.output_logger.info("Stopping training due to lack of improvement in validation reward after %s patience checks", self.patience)
                    return False  # Stop training

        self.val_timesteps += 1
        return True  # Continue training

    def _run_evaluation(self):
        # lists containing metrics across episodes to perform subsequent aggregations
        local_actions_count_list = []
        local_actions_success_list = []
        remote_actions_count_list = []
        remote_actions_success_list = []
        source_nodes_list = []
        target_nodes_list = []
        vulnerability_counts_list = []
        vulnerability_success_list = []
        invalid_actions_list = []
        reachable_list = []
        discoverable_list = []
        discoverable_amount_list = []
        discovered_amount_list = []
        disruptable_list = []
        owned_list = []
        discovered_list = []
        disrupted_list = []
        reimaged_list = []
        not_discovered_list = []
        episode_reward_list = []
        owned_percentage_list = []
        discovered_percentage_list = []
        disrupted_percentage_list = []
        number_steps_list = []
        target_node_tag_counts_list = []
        outcome_actions_counts_list = []
        network_availability_list = []
        episode_won_list = []
        episode_lost_list = []
        reimaged_percentage_list = []
        num_events_list = []
        min_distance_actions_list = []

        # Perform a certain number of validation episodes and log average statistics
        for _ in range(self.n_val_episodes):
            local_remote_actions_counts = np.zeros(2)
            local_remote_actions_success = np.zeros(2)
            target_node_tag_counts = {}
            source_nodes = []
            target_nodes = []
            vulnerability_counts = {}
            vulnerability_success = {}
            outcome_actions_counts = {}
            invalid_actions = 0
            min_distance_actions = []

            obs = self.val_env.reset()
            episode_rewards = 0.0
            done = False

            number_steps = 0
            info = None
            while not done:
                number_steps += 1
                action, _ = self.model.predict(obs)
                obs, reward, done, info = self.val_env.step(action)
                episode_rewards += reward
                info = info[0]
                if 'min_distance_action' in info:
                    min_distance_actions.append(info['min_distance_action'])
                source_nodes.append(info['source_node'])
                source_nodes = list(set(source_nodes))
                target_nodes.append(info['target_node'])
                target_nodes = list(set(target_nodes))
                vulnerability_counts[info['vulnerability']] = vulnerability_counts.get(info['vulnerability'], 0) + 1
                target_node_tag_counts[info['target_node_tag']] = target_node_tag_counts.get(info['target_node_tag'], 0) + 1
                outcome_actions_counts[info['outcome']] = outcome_actions_counts.get(
                    info['outcome'], 0) + 1
                if issubclass(type(info['outcome_class']), m.InvalidAction): # big class having as subclassess all invalid cases
                    invalid_actions += 1
                else:
                    vulnerability_success[info['vulnerability']] = vulnerability_success.get(info['vulnerability'], 0) + 1
                if info['vulnerability_type'] == "local":
                    local_remote_actions_counts[0] += 1
                    if not issubclass(type(info['outcome_class']), m.InvalidAction):
                        local_remote_actions_success[0] += 1
                elif info['vulnerability_type'] == "remote":
                    local_remote_actions_counts[1] += 1
                    if not issubclass(type(info['outcome_class']), m.InvalidAction):
                        local_remote_actions_success[1] += 1
            episode_won_list.append(info['end_episode_reason'] == 1)
            episode_lost_list.append(info['end_episode_reason'] == 2)
            local_actions_success_list.append(local_remote_actions_success[0])
            local_actions_count_list.append(local_remote_actions_counts[0])
            remote_actions_success_list.append(local_remote_actions_success[1])
            remote_actions_count_list.append(local_remote_actions_counts[1])
            source_nodes_list.append(len(set(source_nodes)))
            target_nodes_list.append(len(set(target_nodes)))
            vulnerability_counts_list.append(sum(vulnerability_counts.values()))
            vulnerability_success_list.append(sum(vulnerability_success.values()))
            invalid_actions_list.append(invalid_actions)
            outcome_actions_counts_list.append(outcome_actions_counts)
            target_node_tag_counts_list.append(target_node_tag_counts)
            min_distance_actions = [x for x in min_distance_actions if x is not None]
            if len(min_distance_actions) == 0:
                min_distance_actions_list.append(0)
            else:
                min_distance_actions_list.append(np.mean(min_distance_actions))
            owned_nodes, discovered_nodes, not_discovered_nodes, disrupted_nodes, num_nodes, reachable_count, discoverable_count, disruptable_count, network_availability, reimaged_nodes, num_events, discovered_amount, discoverable_amount, _ = self.val_env.envs[0].get_statistics()
            network_availability_list.append(network_availability)
            reachable_list.append(reachable_count + 1) # re include the starter node
            discoverable_list.append(discoverable_count + 1)
            discovered_amount_list.append(discovered_amount + 1 )
            discoverable_amount_list.append(discoverable_amount) # starter node already included
            disruptable_list.append(disruptable_count + 1)
            owned_percentage = owned_nodes / num_nodes
            discovered_percentage = discovered_nodes / num_nodes
            reimaged_percentage = reimaged_nodes / num_nodes
            disrupted_percentage = disrupted_nodes / num_nodes

            owned_list.append(owned_nodes)
            discovered_list.append(discovered_nodes)
            reimaged_list.append(reimaged_nodes)
            disrupted_list.append(disrupted_nodes)
            not_discovered_list.append(not_discovered_nodes)
            episode_reward_list.append(episode_rewards)
            owned_percentage_list.append(owned_percentage)
            discovered_percentage_list.append(discovered_percentage)
            disrupted_percentage_list.append(disrupted_percentage)
            number_steps_list.append(number_steps)
            reimaged_percentage_list.append(reimaged_percentage)
            num_events_list.append(num_events)


        stats = {
            "local_actions_count": np.mean(local_actions_count_list),
            "local_actions_success": np.mean(local_actions_success_list),
            "remote_actions_count": np.mean(remote_actions_count_list),
            "remote_actions_success": np.mean(remote_actions_success_list),
            "source_nodes": np.mean(source_nodes_list),
            "target_nodes": np.mean(target_nodes_list),
            "actions_counts": np.mean(vulnerability_counts_list),
            "actions_success": np.mean(vulnerability_success_list),
            "invalid_actions": np.mean(invalid_actions_list),
            "owned": np.mean(owned_list),
            "discovered": np.mean(discovered_list),
            "disrupted": np.mean(disrupted_list),
            "reachable": np.mean(reachable_list),
            "discoverable": np.mean(discoverable_list),
            "discoverable_amount": np.mean(discoverable_amount_list),
            "discovered_amount": np.mean(discovered_amount_list),
            "disruptable": np.mean(disruptable_list),
            "not_discovered": np.mean(not_discovered_list),
            "episode_reward": np.mean(episode_reward_list),
            "owned_percentage": np.mean(owned_percentage_list),
            "discovered_percentage": np.mean(discovered_percentage_list),
            "disrupted_percentage": np.mean(disrupted_percentage_list),
            "number_steps": np.mean(number_steps_list),
            "network_availability": np.mean(network_availability_list),
            "episodes_won": np.mean(episode_won_list),
            "episodes_lost": np.mean(episode_lost_list),
            "reimaged": np.mean(reimaged_list),
            "reimaged_percentage": np.mean(reimaged_percentage_list),
            "num_events": np.mean(num_events_list),
            "min_distance_action": np.mean(min_distance_actions_list)
        }
        for i in range(len(outcome_actions_counts_list)):
            for outcome, count in outcome_actions_counts_list[i].items():
                stats[outcome + "_outcomes_count"] = stats.get(outcome + "_outcomes_count", 0) + count

        for outcome, count in outcome_actions_counts_list[0].items():
            stats[outcome + "_outcomes_count"] = stats.get(outcome + "_outcomes_count", 0) / self.n_val_episodes

        overall_tags_list = []
        for tag_counts in target_node_tag_counts_list:
            for tag, count in tag_counts.items():
                if tag not in overall_tags_list:
                    overall_tags_list.append(tag)
                stats[tag + "_tags_count"] = stats.get(tag + "_tags_count", 0) + count

        for tag in overall_tags_list:
            stats[tag + "_tags_count"] = stats.get(tag + "_tags_count", 0) / self.n_val_episodes

        return stats

    # Log average validation statistics to tensorboard
    def _log_tensorboard_custom_metrics(self, custom_metrics):
        self.logger.record("validation/Episode reward (mean)", custom_metrics['episode_reward'])
        self.logger.record("validation/Owned nodes (mean)", custom_metrics['owned'])
        self.logger.record("validation/Discovered nodes (mean)", custom_metrics['discovered'])
        self.logger.record("validation/Discoverable amount (mean)", custom_metrics['discoverable_amount'])

        self.logger.record("validation/Disrupted nodes (mean)", custom_metrics['disrupted'])
        self.logger.record("validation/Not discovered nodes (mean)", custom_metrics['not_discovered'])
        self.logger.record("validation/Owned percentage (mean)", custom_metrics['owned_percentage'])
        self.logger.record("validation/Discovered percentage (mean)", custom_metrics['discovered_percentage'])
        self.logger.record("validation/Discovered amount percentage (mean)", custom_metrics['discovered_amount'])

        self.logger.record("validation/Disrupted percentage (mean)", custom_metrics['disrupted_percentage'])
        self.logger.record("validation/Owned-discovered ratio (mean)", custom_metrics['owned'] / custom_metrics['discovered'])
        self.logger.record("validation/Network availability (mean)", custom_metrics['network_availability'])
        self.logger.record("validation/Episodes won (mean)", custom_metrics['episodes_won'])
        self.logger.record("validation/Episodes lost (mean)", custom_metrics['episodes_lost'])
        self.logger.record("validation/Reimaged nodes (mean)", custom_metrics['reimaged'])
        self.logger.record("validation/Reimaged percentage (mean)", custom_metrics['reimaged_percentage'])
        self.logger.record("validation/Number of events (mean)", custom_metrics['num_events'])
        self.logger.record("validation/Minimum distance action (mean)", custom_metrics['min_distance_action'])

        self.logger.record("validation/Relative owned percentage (mean)",
                               custom_metrics['owned'] / custom_metrics['reachable'])
        self.logger.record("validation/Relative discovered percentage (mean)",
                                 custom_metrics['discovered'] / custom_metrics['discoverable'])
        self.logger.record("validation/Relative disrupted percentage (mean)",
                            custom_metrics['disrupted'] / custom_metrics['disruptable'])
        self.logger.record("validation/Relative discovered amount (mean)", custom_metrics['discovered_amount'] / custom_metrics['discoverable_amount'])

        self.logger.record("validation/Number of source nodes (mean)", custom_metrics['source_nodes'])
        self.logger.record("validation/Number of target nodes (mean)", custom_metrics['target_nodes'])
        self.logger.record("validation/Number of unique vulnerabilities (mean)", custom_metrics['actions_counts'])
        self.logger.record("validation/Number of steps (mean)", custom_metrics['number_steps'])

        self.logger.record("actions/validation/Invalid actions (mean)", custom_metrics['invalid_actions'])
        self.logger.record("actions/validation/Local actions count (mean)", custom_metrics['local_actions_count'])
        self.logger.record("actions/validation/Remote actions count (mean)", custom_metrics['remote_actions_count'])
        self.logger.record("actions/validation/Success rate for local actions (mean)", custom_metrics['local_actions_success'])
        self.logger.record("actions/validation/Success rate for remote actions (mean)", custom_metrics['remote_actions_success'])
        self.logger.record("actions/validation/Success rate for all actions (mean)", custom_metrics['actions_success'])

        for string, count in custom_metrics.items():
            if string.endswith("_outcomes_count"):
                outcome = string.split("_outcomes_count")[0]
                self.logger.record(f"actions/validation/Outcome {outcome} count (mean)", count)
            if string.endswith("_tags_count"):
                tag_name = string.split("_tags_count")[0]
                self.logger.record(f"validation/Target nodes {tag_name} count (mean)", count)
