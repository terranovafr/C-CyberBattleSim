# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    attacker_actions.py
    This file contains the class and associated methods for the AttackerAgentActions
    class which interacts directly with the environment.
"""
import random
from dataclasses import dataclass
import dataclasses
import datetime
from collections import OrderedDict
from typing import Iterator, List, NamedTuple, Optional, Set, Tuple, Dict
from cyberbattle.simulation import model

# Result from executing an action
class ActionResult(NamedTuple):
    reward: float
    outcome: Optional[model.VulnerabilityOutcome]

# Track information about nodes gathered throughout the simulation
@dataclass
class NodeTrackingInformation:
    # Map (vulnid, local_or_remote) to time of last attack.
    # local_or_remote is true for local, false for remote
    last_attack: Dict[model.VulnerabilityID, datetime.datetime] = dataclasses.field(
        default_factory=dict
    )
    # Last time the node got owned by the attacker agent
    last_owned_at: Optional[datetime.datetime] = None
    # All node properties discovered so far
    discovered_properties: Set[int] = dataclasses.field(default_factory=set) # TODO: Not used yet in C-CyberBattleSim

# Class that interacts with and makes changes to the environment.
class AttackerAgentActions:
    def __init__(self,
                 environment, # C-CyberBattleSim environment
                 logger,
                 penalties_dict,
                 rewards_dict,
                 verbose = False,
                 ):
        self._environment = environment
        self._discovered_nodes: "OrderedDict[model.NodeID, NodeTrackingInformation]" = (
            OrderedDict()
        )
        # Mark all owned nodes as discovered
        for i, node in environment.nodes.items():
            node = node["data"]
            if node.agent_installed:
                self.__mark_node_as_owned(i, node.level_at_access)

        self.logger = logger
        self.penalties_dict = penalties_dict
        self.rewards_dict = rewards_dict
        self.verbose = verbose

    def discovered_nodes(self) -> Iterator[Tuple[model.NodeID, model.NodeInfo]]:
        for node_id in self._discovered_nodes:
            yield (node_id, self._environment[node_id]["data"])

    def __mark_node_as_discovered(self, node_id: model.NodeID) -> bool:
        newly_discovered = node_id not in self._discovered_nodes
        if newly_discovered:
            self._discovered_nodes[node_id] = NodeTrackingInformation()
        return newly_discovered

    def __mark_node_as_owned(
        self,
        node_id: model.NodeID,
        level: Optional[model.PrivilegeLevel] = None,
    ) -> Tuple[Optional[datetime.datetime], bool]:
        node_info = self._environment.nodes(data=True)[node_id]["data"]

        last_owned_at, is_currently_owned = self.__is_node_owned_history(
            node_id, node_info
        )
        if node_id not in self._discovered_nodes:
            self._discovered_nodes[node_id] = NodeTrackingInformation()
        node_info.agent_installed = True
        node_info.privilege_level = model.escalate(
            node_info.privilege_level, level
        )
        self._environment.nodes[node_id].update({"data": node_info})
        self._discovered_nodes[node_id].last_owned_at = datetime.datetime.now() # Record that the node just got owned at the current time

        return last_owned_at, is_currently_owned # Return the time it was previously own (or None) and whether it was already owned.

    # Attempt to exploit a remote vulnerability from a source node to another node using the specified vulnerability.
    def exploit_remote_vulnerability(
        self,
        source_node_id: model.NodeID,
        target_node_id: model.NodeID,
        vulnerability_id: model.VulnerabilityID,
        outcome_desired
    ) -> ActionResult:
        if source_node_id not in self._environment.nodes:
            # It should be a valid source node
            raise ValueError(f"invalid node id '{source_node_id}'")
        if target_node_id not in self._environment.nodes:
            # It should be a valid target node
            raise ValueError(f"invalid target node id '{target_node_id}'")

        source_node_info: model.NodeInfo = self._environment.nodes(data=True)[source_node_id]["data"]
        target_node_info: model.NodeInfo = self._environment.nodes(data=True)[target_node_id]["data"]

        if not source_node_info.agent_installed:
            # The source node should have the agent owning it
            if self.verbose > 2:
                self.logger.info("Penalty (invalid action) - agent not installed on source node: %s", self.penalties_dict["invalid_action"])
            return ActionResult(reward=self.penalties_dict["invalid_action"], outcome=model.InvalidAction("Agent not installed on source node"))

        if target_node_id not in self._discovered_nodes:
            # The target node should have been discovered
            if self.verbose > 2:
                self.logger.info("Penalty (invalid action) - target node not discovered: %s", self.penalties_dict["invalid_action"])
            return ActionResult(reward=self.penalties_dict["invalid_action"], outcome=model.InvalidAction("Target node not discovered"))

        if source_node_info.status != model.MachineStatus.Running:
            # The source node should be running
            if self.verbose > 2:
                self.logger.info("Penalty (invalid action) - source node not in running state: %s", self.penalties_dict["invalid_action"])
            return ActionResult(reward=self.penalties_dict["invalid_action"], outcome=model.NonRunningMachine(source_or_target=0, reason="Source machine not running"))

        if target_node_info.status != model.MachineStatus.Running:
            # The target node should be running
            if self.verbose > 2:
                self.logger.info("Penalty (invalid action) - target node not in running state: %s", self.penalties_dict["invalid_action"])
            return ActionResult(reward=self.penalties_dict["invalid_action"], outcome=model.NonRunningMachine(source_or_target=1, reason="Target machine not running"))

        if vulnerability_id in target_node_info.vulnerabilities:
            vulnerability = target_node_info.vulnerabilities[vulnerability_id]
        else:
            # The target node should have the vulnerability selected
            if self.verbose > 2:
                self.logger.info("Penalty (no vulnerability in node) - vulnerability not present in target node: %s", self.penalties_dict["no_vulnerability_in_node"])
            return ActionResult(reward=self.penalties_dict["no_vulnerability_in_node"], outcome=model.NoVulnerability("Vulnerability not existing on target node"))

        if vulnerability.privileges_required:
            # The target node vulnerability exploit requires certain privileges
            if not target_node_info.privilege_level >= vulnerability.privileges_required:
                if self.verbose > 2:
                    self.logger.info("Penalty (no enough privileges) - no enough privileges on target node to exploit the vulnerability selected: %s", self.penalties_dict["no_enough_privileges"])
                return ActionResult(reward=self.penalties_dict["no_enough_privileges"], outcome=model.NoEnoughPrivilege("Not enough privileges to exploit the vulnerability"))

        outcome = None
        for result in vulnerability.results:
            if result.type is model.VulnerabilityType.REMOTE and type(result.outcome) is type(outcome_desired):
                outcome = result.outcome
                break
        if outcome is None:
            # The vulnerability should have the chosen outcome
            if self.verbose > 2:
                self.logger.info(
                    "Penalty (invalid action) - the outcome desired is not present for the vulnerability selected: %s",
                    self.penalties_dict["invalid_action"])
            return ActionResult(reward=self.penalties_dict["invalid_action"], outcome=model.OutcomeNonPresent("Outcome not present for the vulnerability"))

        target_node_is_listening = vulnerability.port in [i.name for i in target_node_info.services if i.running]
        if not target_node_is_listening:
            # The target node should be listening on the port of the service associated to the vulnerability
            if self.verbose > 2:
                self.logger.info(
                    "Penalty (scanning unopen port) - the target node is not listining on the port of the service associated to the vulnerability: %s",
                    self.penalties_dict["scanning_unopen_port"])
            return ActionResult(reward=self.penalties_dict["scanning_unopen_port"], outcome=model.NonListeningPort("Target node not listening on port"))

        if not source_node_info.defense_evasion and not self.__is_passing_firewall_rules(
            source_node_info.firewall.outgoing, vulnerability.port
        ):
            # The source node firewall should allow outgoing traffic
            if self.verbose > 2:
                self.logger.info(
                    "Penalty (blocked by local firewall) - the source node does not allow outgoing traffic on the port associated to the service of the vulnerability: %s",
                    self.penalties_dict["blocked_by_local_firewall"])
            return ActionResult(reward=self.penalties_dict["blocked_by_local_firewall"], outcome=model.FirewallBlock(incoming_or_outgoing=1, reason="Source node firewall blocking outgoing traffic"))

        if not target_node_info.defense_evasion and not self.__is_passing_firewall_rules(
            target_node_info.firewall.incoming, vulnerability.port
        ):
            # The target node firewall should allow incoming traffic
            if self.verbose > 2:
                self.logger.info(
                    "Penalty (blocked by remote firewall) - the target node does not allow incoming traffic on the port associated to the service of the vulnerability: %s",
                    self.penalties_dict["blocked_by_remote_firewall"])
            return ActionResult(reward=self.penalties_dict["blocked_by_remote_firewall"], outcome=model.FirewallBlock(incoming_or_outgoing=0, reason="Target node firewall blocking incoming traffic"))

        if random.random() >= vulnerability.rates.successRate:
            # The remote exploit should succeed based on the success rate
            if self.verbose > 2:
                self.logger.info(
                    "Penalty (success rate failed) - the vulnerability success rate has caused failure of remote exploit: %s",
                    self.penalties_dict["success_rate_failed"])
            return ActionResult(reward=self.penalties_dict["success_rate_failed"],
                                outcome=model.UnsuccessfulAction("Remote exploit failed because of the success rate"))

        total_reward = 0
        # NOTE: Controlling whether the vulnerability has been executed before on the node has been removed since now the agent could use the same vulnerability with another outcome and that would be a valid action.

        if isinstance(outcome, model.Collection):
            # Collection of data if available and not yet collected, setting the proper attributes on the target node
            if target_node_info.has_data:
                target_node_info.has_data = False # hence if already collected, it will not work again
                target_node_info.data_collected = True # used by exfiltration after
                total_reward += self.rewards_dict['data_collected_reward']
                if self.verbose > 2:
                    self.logger.info(
                        "Reward (collection data) : += %s", self.rewards_dict['data_collected_reward'])
                self._environment.nodes[target_node_id].update({"data": target_node_info})
            else:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (not data to collect) - no data in the target node: %s", self.penalties_dict["no_data_to_collect"])
                return ActionResult(reward=self.penalties_dict["no_data_to_collect"], outcome=model.NoNeededAction("No data to collect"))
        elif isinstance(outcome, model.Persistence):
            # Setting the persistence attribute on the target node if not already persistent
            if target_node_info.persistence:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (already persistent) - Agent already persistent in target node: %s", self.penalties_dict["already_persistent"])
                return ActionResult(reward=self.penalties_dict["already_persistent"], outcome=model.RepeatedResult("Trying to persist but already did"))
            target_node_info.persistence = True
            total_reward += self.rewards_dict['persistence_reward']
            if self.verbose > 2:
                self.logger.info(
                    "Reward (persistence) : += %s", self.rewards_dict['persistence_reward'])
            self._environment.nodes[target_node_id].update({"data": target_node_info})
        elif isinstance(outcome, model.DenialOfService):
            # Putting the target node in DOS if not already stopped
            if target_node_info.status == model.MachineStatus.Stopped:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (node already stopped) - Agent already performed DOS on the target node: %s", self.penalties_dict["node_already_stopped"])
                return ActionResult(reward=self.penalties_dict["node_already_stopped"], outcome=model.RepeatedResult("Trying to DOS but already did"))
            target_node_info.status = model.MachineStatus.Stopped
            total_reward += self.rewards_dict['dos_coefficient'] * target_node_info.value
            if self.verbose > 2:
                self.logger.info(
                    "Reward (put the target node in DOS): += %s * %s", self.rewards_dict['dos_coefficient'], target_node_info.value)
            self._environment.nodes[target_node_id].update({"data": target_node_info})
        elif isinstance(outcome, model.Discovery):
            # Discovering the target node if not already discovered (visibility)
            if target_node_info.visible:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (node already visible) - Agent already acquired visibility on the target node: %s",
                        self.penalties_dict["node_already_visible"])
                return ActionResult(reward=self.penalties_dict["node_already_visible"], outcome=model.RepeatedResult("Trying to discover but already did"))
            else:
                target_node_info.visible = True
                total_reward += self.rewards_dict['acquired_visibility_reward']
                if self.verbose > 2:
                    self.logger.info("Reward (acquired visibility on target node): += %s ", self.rewards_dict['acquired_visibility_reward'])
                self._environment.nodes[target_node_id].update({"data": target_node_info})
        elif isinstance(outcome, model.Exfiltration):
            # Exfiltrating data if available and not yet exfiltrated, setting the proper attributes on the target node
            if target_node_info.data_collected and not target_node_info.data_exfiltrated:
                target_node_info.data_exfiltrated = True
                total_reward += self.rewards_dict['data_exfiltrated_reward']
                if self.verbose > 2:
                    self.logger.info("Reward (exfiltrated data from target node): += %s ",
                                 self.rewards_dict['data_exfiltrated_reward'])
                self._environment.nodes[target_node_id].update({"data": target_node_info})
            else:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (no data to exfiltrate) - Agent has not collected data on target node or data already exfiltrated: %s",
                        self.penalties_dict["node_already_visible"])
                return ActionResult(reward=self.penalties_dict["no_data_to_exfiltrate"], outcome=model.NoNeededAction("No data to exfiltrate"))
        elif isinstance(outcome, model.DefenseEvasion):
            # Evading defense if not already evaded, setting the proper attributes on the target node
            if target_node_info.defense_evasion:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (already defense evasion) - Agent has already evaded defense in target node: %s",
                        self.penalties_dict["already_defense_evasion"])
                return ActionResult(reward=self.penalties_dict["already_defense_evasion"], outcome=model.RepeatedResult("Trying to evade defence but already did"))
            target_node_info.defense_evasion = True
            total_reward += self.rewards_dict['defense_evaded_reward']
            if self.verbose > 2:
                self.logger.info("Reward (acquired defense evasion on target node): += %s ",
                                 self.rewards_dict['defense_evaded_reward'])
            self._environment.nodes[target_node_id].update({"data": target_node_info})
        elif isinstance(outcome, model.Reconnaissance):
            # Discovering the nodes related to the vulnerability if not already discovered
            newly_discovered_nodes = 0
            newly_discovered_nodes_value = 0

            for node_id in outcome.nodes:
                if self.__mark_node_as_discovered(node_id):
                    if self.verbose > 2:
                        self.logger.info("Node %s discovered", node_id)
                    newly_discovered_nodes += 1
                    newly_discovered_nodes_value += self._environment.nodes(data=True)[node_id]["data"].value

            total_reward += self.rewards_dict['node_discovered_coefficient'] * newly_discovered_nodes
            if self.verbose > 2:
                self.logger.info("Reward (nodes discovered): += %s * %s", self.rewards_dict['node_discovered_coefficient'], newly_discovered_nodes)
        elif isinstance(outcome, model.PrivilegeEscalation):
            # Escalating privileges on the target node if not already escalated
            if target_node_info.privilege_level == model.PrivilegeLevel.NoAccess:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (trying to escalate in node not owned) - Agent does not own target node for which it wants privilege escalation: %s",
                        self.penalties_dict["privilege_escalation_in_node_not_owned"])
                return ActionResult(reward=self.penalties_dict["privilege_escalation_in_node_not_owned"], outcome=model.NoEnoughPrivilege("Trying to escalate in a node not owned"))
            if target_node_info.privilege_level >= outcome.level:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (trying to escalate to level already had) - Agent already had the level on target node: %s",
                        self.penalties_dict["privilege_escalation_to_level_already_had"])
                return ActionResult(reward=self.penalties_dict["privilege_escalation_to_level_already_had"], outcome=model.RepeatedResult("Trying to escalate to level already had"))
            last_owned_at, is_already_owned = self.__mark_node_as_owned(target_node_id, outcome.level)
            if is_already_owned:
                if self.verbose > 2:
                    self.logger.info("Privilege escalation on node %s to level %s", target_node_id, outcome.level)
            else:
                if self.verbose > 2:
                    self.logger.info("Owning and privilege escalation on node %s to level %s", target_node_id, outcome.level)
            if self.verbose > 2:
                self.logger.info("Reward (privilege escalation on target node): += %s ", self.rewards_dict['privilege_escalation_reward'])
            total_reward += self.rewards_dict['privilege_escalation_reward']
        elif isinstance(outcome, model.LateralMove) or isinstance(outcome, model.CredentialAccess):
            # Getting control of the target node if not already owned
            last_owned_at, is_already_owned = self.__mark_node_as_owned(target_node_id, target_node_info.level_at_access)
            if not is_already_owned:
                if self.verbose > 2:
                    self.logger.info("Owning and reaching level privilege %s on node %s", target_node_info.level_at_access, target_node_id)
            if is_already_owned:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (target node already owned): %s", self.penalties_dict["node_already_owned"])
                return ActionResult(reward=self.penalties_dict["node_already_owned"], outcome=model.RepeatedResult("Target node already owned"))
            if last_owned_at is None:
                if self.verbose > 2:
                    self.logger.info("Reward (owning target node): += %s * %s ",
                                     self.rewards_dict['value_coefficient'], target_node_info.value)
                total_reward += self.rewards_dict['value_coefficient'] * float(target_node_info.value)
        else:
            if self.verbose > 2:
                self.logger.info(
                    "Penalty (invalid outcome for vulnerability): %s", self.penalties_dict["outcome_not_valid"])
            return ActionResult(reward=self.penalties_dict["outcome_not_valid"], outcome=model.RemoteOutcomeInLocalNode("Invalid outcome, e.g. credential access or lateral move have been chosen."))
        self._discovered_nodes[target_node_id].last_attack[vulnerability_id] = datetime.datetime.now()
        # The reward is decreased by the cost of the vulnerability
        total_reward -= self.rewards_dict['cost_coefficient'] * vulnerability.cost
        if self.verbose > 2:
            self.logger.info("Vulnerability cost: -= %s * %s", self.rewards_dict['cost_coefficient'], vulnerability.cost)
        return ActionResult(reward=total_reward, outcome=outcome)

    def exploit_local_vulnerability(
        self, node_id: model.NodeID, vulnerability_id: model.VulnerabilityID, outcome_desired
    ) -> ActionResult:
        # exploits a local vulnerability on a node (source node == target node)
        # In our implementation is considered valid for a node to exploit locally a remote vulnerability with the assumption that the attacker can use a dummy virtual machine/terminal

        if node_id not in self._environment.nodes:
            raise ValueError(f"invalid node id '{node_id}'")

        node_info = self._environment.nodes(data=True)[node_id]["data"]
        if not node_info.agent_installed:
            # The source node should have the agent owning it
            if self.verbose > 2:
                self.logger.info("Penalty (invalid action) - agent not installed on source node: %s",
                                 self.penalties_dict["invalid_action"])
            return ActionResult(reward=self.penalties_dict["invalid_action"], outcome=model.InvalidAction("Agent not installed on source node"))

        if node_info.status != model.MachineStatus.Running:
            # The source node should be running
            if self.verbose > 2:
                self.logger.info("Penalty (invalid action) - source node not in running state: %s",
                                 self.penalties_dict["invalid_action"])
            return ActionResult(reward=self.penalties_dict["invalid_action"], outcome=model.NonRunningMachine(source_or_target=0, reason="Machine not running"))

        if vulnerability_id in node_info.vulnerabilities:
            vulnerability = node_info.vulnerabilities[vulnerability_id]
        else:
            # The node should have the vulnerability
            if self.verbose > 2:
                self.logger.info("Penalty (no vulnerability in node) - vulnerability not present in source node: %s",
                                 self.penalties_dict["no_vulnerability_in_node"])
            return ActionResult(reward=self.penalties_dict["no_vulnerability_in_node"], outcome=model.NoVulnerability("Vulnerability not present on the node"))

        if vulnerability.privileges_required:
            # The vulnerability exploit requires certain privileges
            if not node_info.privilege_level >= vulnerability.privileges_required:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (no enough privileges) - no enough privileges on source node to exploit the vulnerability selected: %s",
                        self.penalties_dict["no_enough_privileges"])
                return ActionResult(reward=self.penalties_dict["no_enough_privileges"], outcome=model.NoEnoughPrivilege(
                    "Not enough privileges to exploit the vulnerability"))

        outcome = None
        for result in vulnerability.results:
            if type(result.outcome) is type(outcome_desired): # suppose it can uses both local and remote vulnerabilities, hence no check on type
                outcome = result.outcome
                break
        if outcome is None:
            # The vulnerability should have the chosen outcome
            if self.verbose > 2:
                self.logger.info(
                    "Penalty (invalid action) - the outcome desired is not present for the vulnerability selected: %s",
                    self.penalties_dict["invalid_action"])
            return ActionResult(reward=self.penalties_dict["invalid_action"], outcome=model.OutcomeNonPresent("Outcome not present for the vulnerability"))

        if random.random() >= vulnerability.rates.successRate:
            # The local exploit should succeed based on the success rate
            if self.verbose > 2:
                self.logger.info(
                    "Penalty (success rate failed) - the vulnerability success rate has caused failure of local exploit: %s",
                    self.penalties_dict["success_rate_failed"])
            return ActionResult(reward=self.penalties_dict["success_rate_failed"],
                                outcome=model.UnsuccessfulAction("Local exploit failed because of the success rate"))

        total_reward = 0
        if isinstance(outcome, model.Collection):
        # Collecting data if available and not yet collected, setting the proper attributes on the node
            if node_info.has_data:
                node_info.has_data = False
                node_info.data_collected = True
                total_reward += self.rewards_dict['data_collected_reward']
                if self.verbose > 2:
                    self.logger.info(
                        "Reward (collection data) : += %s", self.rewards_dict['data_collected_reward'])
                self._environment.nodes[node_id].update({"data": node_info})
            else:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (not data to collect) - no data in the source node: %s",
                        self.penalties_dict["no_data_to_collect"])
                return ActionResult(reward=self.penalties_dict["no_data_to_collect"],
                                    outcome=model.NoNeededAction("No data to collect"))
        elif isinstance(outcome, model.Persistence):
            # Persisting on the node if not already persistent, setting the proper attributes on the node
            if node_info.persistence:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (already persistent) - Agent already persistent in source node: %s",
                        self.penalties_dict["already_persistent"])
                return ActionResult(reward=self.penalties_dict["already_persistent"], outcome=model.RepeatedResult("Trying to persist but already did"))
            node_info.persistence = True
            total_reward += self.rewards_dict['persistence_reward']
            if self.verbose > 2:
                self.logger.info(
                    "Reward (persistence) : += %s", self.rewards_dict['persistence_reward'])
            self._environment.nodes[node_id].update({"data": node_info})
        elif isinstance(outcome, model.DenialOfService): # if used as source node, not stopped of course
            node_info.status = model.MachineStatus.Stopped
            total_reward += self.rewards_dict['dos_coefficient'] * node_info.value
            if self.verbose > 2:
                self.logger.info(
                    "Reward (put the source node in DOS): += %s * %s", self.rewards_dict['dos_coefficient'],
                    node_info.value)
            self._environment.nodes[node_id].update({"data": node_info})
        elif isinstance(outcome, model.Discovery):
            # Discovering the node if not already discovered (visibility), setting the proper attributes on the node
            if node_info.visible:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (node already visible) - Agent already acquired visibility on source target node: %s",
                        self.penalties_dict["node_already_visible"])
                return ActionResult(reward=self.penalties_dict["node_already_visible"], outcome=model.RepeatedResult("Trying to discover but already did"))
            else:
                node_info.visible = True
                total_reward += self.rewards_dict['acquired_visibility_reward']
                if self.verbose > 2:
                    self.logger.info("Reward (acquired visibility on source node): += %s ",
                                     self.rewards_dict['acquired_visibility_reward'])
                self._environment.nodes[node_id].update({"data": node_info})
        elif isinstance(outcome, model.Exfiltration):
            # Exfiltrating data if available and not yet exfiltrated, setting the proper attributes on the node
            if node_info.data_collected and not node_info.data_exfiltrated:
                node_info.data_exfiltrated = True
                total_reward += self.rewards_dict['data_exfiltrated_reward']
                if self.verbose > 2:
                    self.logger.info("Reward (exfiltrated data from source node): += %s ",
                                     self.rewards_dict['data_exfiltrated_reward'])
                self._environment.nodes[node_id].update({"data": node_info})
            else:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (no data to exfiltrate) - Agent has not collected data on source node or data already exfiltrated: %s",
                        self.penalties_dict["node_already_visible"])
                return ActionResult(reward=self.penalties_dict["no_data_to_exfiltrate"], outcome=model.NoNeededAction("No data to exfiltrate"))
        elif isinstance(outcome, model.DefenseEvasion):
            # Evading defense if not already evaded, setting the proper attributes on the node
            if node_info.defense_evasion:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (already defense evasion) - Agent has already evaded defense in source node: %s",
                        self.penalties_dict["already_defense_evasion"])
                return ActionResult(reward=self.penalties_dict["already_defense_evasion"], outcome=model.RepeatedResult("Trying to evade but already did"))
            node_info.defense_evasion = True
            total_reward += self.rewards_dict['defense_evaded_reward']
            if self.verbose > 2:
                self.logger.info("Reward (acquired defense evasion on source node): += %s ",
                                 self.rewards_dict['defense_evaded_reward'])
            self._environment.nodes[node_id].update({"data": node_info})
        elif isinstance(outcome, model.Reconnaissance):
            # Discovering the nodes related to the vulnerability if not already discovered
            newly_discovered_nodes = 0
            newly_discovered_nodes_value = 0
            for node_id in outcome.nodes:
                if self.__mark_node_as_discovered(node_id):
                    if self.verbose > 2:
                        self.logger.info("Node %s discovered", node_id)
                    newly_discovered_nodes += 1
                    newly_discovered_nodes_value += self._environment.nodes(data=True)[node_id]["data"].value
            total_reward += self.rewards_dict['node_discovered_coefficient'] * newly_discovered_nodes
            if self.verbose > 2:
                self.logger.info("Reward (nodes discovered): += %s * %s",
                                 self.rewards_dict['node_discovered_coefficient'], newly_discovered_nodes)
        elif isinstance(outcome, model.PrivilegeEscalation):
            # Escalating privileges on the node if not already escalated
            if node_info.privilege_level >= outcome.level:
                if self.verbose > 2:
                    self.logger.info(
                        "Penalty (trying to escalate to level already had) - Agent already had the level on source node: %s",
                        self.penalties_dict["privilege_escalation_to_level_already_had"])
                return ActionResult(reward=self.penalties_dict["privilege_escalation_to_level_already_had"], outcome=model.RepeatedResult("Trying to escalate to level already had"))
            last_owned_at, is_already_owned = self.__mark_node_as_owned(node_id, outcome.level)
            if is_already_owned:
                if self.verbose > 2:
                    self.logger.info("Privilege escalation on node %s to level %s", node_id, outcome.level)
            else:
                if self.verbose > 2:
                    self.logger.info("Owning and privilege escalation on node %s to level %s", node_id,
                                     outcome.level)
            if self.verbose > 2:
                self.logger.info("Reward (privilege escalation on source node): += %s ",
                                 self.rewards_dict['privilege_escalation_reward'])
            total_reward += self.rewards_dict['privilege_escalation_reward']
        else:
            if self.verbose > 2:
                self.logger.info(
                    "Penalty (invalid outcome for vulnerability): %s", self.penalties_dict["outcome_not_valid"])
            return ActionResult(reward=self.penalties_dict["outcome_not_valid"], outcome=model.RemoteOutcomeInLocalNode("Invalid outcome, e.g. credential access or lateral move have been chosen."))

        self._discovered_nodes[node_id].last_attack[vulnerability_id] = datetime.datetime.now()
        # The reward is decreased by the cost of the vulnerability
        total_reward -= self.rewards_dict['cost_coefficient'] * vulnerability.cost
        if self.verbose > 2:
            self.logger.info("Vulnerability cost: -= %s * %s", self.rewards_dict['cost_coefficient'], vulnerability.cost)
        return ActionResult(reward=total_reward, outcome=outcome)

    # Determine if traffic on the specified port is permitted by the specified sets of firewall rules
    def __is_passing_firewall_rules(
        self, rules: List[model.FirewallRule], port_name: model.PortName
    ) -> bool:
        for rule in rules:
            if rule.port == port_name:
                if rule.permission == model.RulePermission.ALLOW:
                    return True
                else:
                    return False
        return True

    # Returns the last time the node got owned and whether it is still currently owned
    def __is_node_owned_history(self, target_node_id, target_node_data):
        last_previously_owned_at = (
            self._discovered_nodes[target_node_id].last_owned_at
            if target_node_id in self._discovered_nodes
            else None
        )

        is_currently_owned = last_previously_owned_at is not None and (
            target_node_data.last_reimaging is None
            or last_previously_owned_at >= target_node_data.last_reimaging
        )
        return last_previously_owned_at, is_currently_owned
