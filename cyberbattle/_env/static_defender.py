# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    static_defender.py
     Defender agents hard-coded that perform hard-coded actions periodically in the environment.
"""

import random
import numpy
from abc import abstractmethod
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.simulation import model # noqa: E402
from cyberbattle.simulation.static_defender_actions import StaticDefenderAgentActions # noqa: E402

# Abstract defender agent class
class DefenderAgent:
    @abstractmethod
    def step(self, environment, actions: StaticDefenderAgentActions, t: int): # step function to be implemented by subclasses
        return None


class ScanAndReimageCompromisedMachines(DefenderAgent):
    """A defender agent that scans a subset of network nodes
     detects presence of an attacker on a given node with
    some fixed probability and if detected re-image the compromised node.

    probability -- probability that an attacker agent is detected when scanned given that the attacker agent is present
    scan_capacity -- maximum number of machine that a defender agent can scan in one simulation step
    scan_frequency -- frequency of the scan in simulation steps
    """

    def __init__(self, probability, scan_capacity, scan_frequency, logger, verbose):
        self.probability = probability
        self.scan_capacity = scan_capacity
        self.scan_frequency = scan_frequency
        self.logger = logger
        self.verbose = verbose

    def step(self, environment, actions: StaticDefenderAgentActions, t: int):
        reimaged_nodes = []
        if t % self.scan_frequency == 0:
            # scan "scan_capacity" nodes at random
            scanned_nodes = random.choices(list(environment.get_nodes()), k=self.scan_capacity)
            for node_id in scanned_nodes:
                node_info = environment.get_node(node_id)
                if node_info.status == model.MachineStatus.Running and \
                        node_info.agent_installed and not node_info.defense_evasion:
                    is_malware_detected = numpy.random.random() <= self.probability # use probability to simulate detection
                    if is_malware_detected:
                        if node_info.reimageable:
                            actions.reimage_node(node_id)
                            reimaged_nodes.append(node_id)
        if self.verbose > 2:
            self.logger.info(f"Defender re-imaged nodes: {reimaged_nodes}")
        return len(reimaged_nodes), reimaged_nodes


class ExternalRandomEvents(DefenderAgent):
    """
        A 'defender' that randomly alters network node configuration, selecting for each node whether
        to stop a service, start a service, add a firewall rule or remove a firewall rule at each timestep,
        based on a certain probability per node.
    """

    def __init__(self, probability, logger, verbose):
        self.probability = probability
        self.logger = logger
        self.verbose = verbose

    def step(self, environment, actions: StaticDefenderAgentActions, t: int):
        events = 0
        nodes_changed = []
        for node_id in environment.get_nodes(): # for every node the action will be different
            function = random.choice(["start service", "firewall remove", "stop service", "firewall add"])
            if function == "stop service":
                new_events = self.stop_service_at_random(node_id, environment, actions)
            elif function == "firewall add":
                new_events = self.firewall_change_add(node_id, environment, actions)
            elif function == "start service":
                new_events = self.start_service_at_random(node_id, environment, actions)
            else: # firewall remove
                new_events = self.firewall_change_remove(node_id, environment, actions)
            events += new_events
            if new_events > 0:
                nodes_changed.append(node_id)
        if self.verbose > 2:
            self.logger.info(f"Defender performed {events} random events.")
        return events, nodes_changed

    # Randomly stop a service on a node
    def stop_service_at_random(self, node_id, environment, actions: StaticDefenderAgentActions):
        node_data = environment.get_node(node_id)
        if node_data.defense_evasion:
            return 0
        remove_service = numpy.random.random() <= self.probability
        if remove_service and len(node_data.services) > 0:
            service = random.choice(node_data.services)
            actions.stop_service(node_id, service.name)
            return 1
        return 0

    # Randomly start a service on a node
    def start_service_at_random(self, node_id, environment, actions: StaticDefenderAgentActions):
        node_data = environment.get_node(node_id)
        if node_data.defense_evasion:
            return 0
        remove_service = numpy.random.random() <= self.probability
        if remove_service and len(node_data.services) > 0:
            service = random.choice(node_data.services)
            actions.start_service(node_id, service.name)
            return 1
        return 0

    # Randomly add a firewall rule on a node
    def firewall_change_add(self, node_id, environment, actions: StaticDefenderAgentActions):
        node_data = environment.get_node(node_id)
        if node_data.defense_evasion:
           return 0
        add_rule = numpy.random.random() <= self.probability
        if add_rule:
            ports_list = []
            for service in node_data.services:
                ports_list.append(service.name)
            rule_to_add = model.FirewallRule(port=random.choice(ports_list),
                                                 permission=model.RulePermission.BLOCK)
            # Randomly decide if we will add an incoming or outgoing rule
            incoming = numpy.random.random() <= 0.5
            if incoming and rule_to_add not in node_data.firewall.incoming:
                actions.override_firewall_rule(node_id, rule_to_add.port, True, rule_to_add.permission)
                return 1
            elif not incoming and rule_to_add not in node_data.firewall.incoming:
                actions.override_firewall_rule(node_id, rule_to_add.port, False, rule_to_add.permission)
                return 1
        return 0

    # Randomly remove a firewall rule on a node
    def firewall_change_remove(self, node_id, environment, actions: StaticDefenderAgentActions):
        node_data = environment.get_node(node_id)
        if node_data.defense_evasion:
           return 0
        remove_rule = numpy.random.random() <= self.probability
        if remove_rule:
            ports_list = []
            for service in node_data.services:
                ports_list.append(service.name)
            rule_to_add = model.FirewallRule(port=random.choice(ports_list),
                                                 permission=model.RulePermission.ALLOW)
            # Randomly decide if we will remove an incoming or outgoing rule
            incoming = numpy.random.random() <= 0.5
            if incoming and rule_to_add not in node_data.firewall.incoming:
                actions.override_firewall_rule(node_id, rule_to_add.port, True, rule_to_add.permission)
                return 1
            elif not incoming and rule_to_add not in node_data.firewall.incoming:
                actions.override_firewall_rule(node_id, rule_to_add.port, False, rule_to_add.permission)
                return 1
        return 0
