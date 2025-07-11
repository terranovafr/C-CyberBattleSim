# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    static_defender_actions.py
    This file contains the class and associated methods for the StaticDefenderAgentActions
    class which interacts directly with the environment.
"""

import datetime
from typing import List, Dict
from cyberbattle.simulation.model import FirewallRule, MachineStatus
from cyberbattle.simulation import model

class StaticDefenderAgentActions:

    # Number of steps it takes to completely reimage a node
    REIMAGING_DURATION = 15

    def __init__(self, environment, logger, verbose=False):
        # map nodes being reimaged to the remaining number of steps to completion
        self.node_reimaging_progress: Dict[model.NodeID, int] = dict()

        # Last calculated availability of the network
        self.__network_availability: float = 1.0

        self.logger = logger
        self.verbose = verbose
        self._environment = environment

    @property
    def network_availability(self):
        return self.__network_availability

    def reimage_node(self, node_id: model.NodeID): # Re-image a computer node
        # Mark the node for re-imaging and make it unavailable until re-imaging completes
        self.node_reimaging_progress[node_id] = self.REIMAGING_DURATION

        node_info = self._environment.nodes(data=True)[node_id]["data"]
        if not node_info.reimageable:
            if self.verbose > 2:
                self.logger.info(f"Defender - Machine {node_id} not re-imageable")
            return

        if self.verbose > 2:
            self.logger.info(f"Defender - Re-imaging machine {node_id} for {self.REIMAGING_DURATION} steps")

        node_info.agent_installed = False # do not modify privilege level so it will be restored by swapping this flag
        node_info.status = model.MachineStatus.Imaging
        node_info.last_reimaging = datetime.datetime.now()
        self._environment.nodes[node_id].update({"data": node_info})

    def on_attacker_step_taken(self): #Function to be called each time a step is take in the simulation
        # Updates the re-imaging progress of each node
        for node_id in list(self.node_reimaging_progress.keys()):
            remaining_steps = self.node_reimaging_progress[node_id]
            if remaining_steps > 0:
                self.node_reimaging_progress[node_id] -= 1
            else:
                if self.verbose > 2:
                    self.logger.info(f"Defender - Machine {node_id} re-imaging completed")
                node_info = self._environment.nodes(data=True)[node_id]["data"]
                node_info.status = model.MachineStatus.Running
                if node_info.persistence:
                    node_info.agent_installed = True
                self.node_reimaging_progress.pop(node_id)
                self._environment.nodes[node_id].update({"data": node_info})

        # Calculate the network availability metric based on machines and services that are running
        total_node_weights = 0
        network_node_availability = 0
        for node_id, node_info in self._environment.nodes.items():
            node_info = node_info['data']
            total_service_weights = 0
            running_service_weights = 0
            for service in node_info.services:
                total_service_weights += service.sla_weight
                running_service_weights += service.sla_weight * int(service.running)

            if node_info.status == MachineStatus.Running:
                adjusted_node_availability = (1 + running_service_weights) / (
                    1 + total_service_weights
                )
            else:
                adjusted_node_availability = 0.0

            total_node_weights += node_info.sla_weight
            network_node_availability += (
                adjusted_node_availability * node_info.sla_weight
            )

        self.__network_availability = network_node_availability / total_node_weights
        assert self.__network_availability <= 1.0 and self.__network_availability >= 0.0

    # Firewall rules override
    def override_firewall_rule(
        self,
        node_id: model.NodeID,
        port_name: model.PortName,
        incoming: bool,
        permission: model.RulePermission,
    ):
        node_data = self._environment.nodes(data=True)[node_id]["data"]

        def add_or_patch_rule(rules) -> List[FirewallRule]:
            new_rules = []
            has_matching_rule = False
            for r in rules:
                if r.port == port_name:
                    has_matching_rule = True
                    new_rules.append(FirewallRule(r.port, permission))
                else:
                    new_rules.append(r)

            if not has_matching_rule:
                new_rules.append(model.FirewallRule(port_name, permission))
            return new_rules

        if incoming:
            if self.verbose > 2:
                self.logger.info(f"Defender - Overriding incoming firewall rule for {node_id} on port {port_name} to {permission}")
            node_data.firewall.incoming = add_or_patch_rule(node_data.firewall.incoming)
        else:
            if self.verbose > 2:
                self.logger.info(f"Defender - Overriding outgoing firewall rule for {node_id} on port {port_name} to {permission}")
            node_data.firewall.outgoing = add_or_patch_rule(node_data.firewall.outgoing)
        self._environment.nodes[node_id].update({"data": node_data})

    # Block traffic overriding the eventually present firewall rule
    def block_traffic(
        self, node_id: model.NodeID, port_name: model.PortName, incoming: bool
    ):
        return self.override_firewall_rule(
            node_id, port_name, incoming, permission=model.RulePermission.BLOCK
        )

    # Allow traffic overriding the eventually present firewall rule
    def allow_traffic(
        self, node_id: model.NodeID, port_name: model.PortName, incoming: bool
    ):
        return self.override_firewall_rule(
            node_id, port_name, incoming, permission=model.RulePermission.ALLOW
        )

    def stop_service(self, node_id: model.NodeID, port_name: model.PortName):
        node_data = self._environment.nodes(data=True)[node_id]["data"]
        if node_data.status != model.MachineStatus.Running:
            return
        for service in node_data.services:
            if service.name == port_name:
                service.running = False

        if self.verbose > 2:
            self.logger.info(
                f"Defender - Stopping service {port_name} on machine {node_id}")

    def start_service(self, node_id: model.NodeID, port_name: model.PortName):
        node_data = self._environment.nodes(data=True)[node_id]["data"]
        if node_data.status != model.MachineStatus.Running:
            return
        for service in node_data.services:
            if service.name == port_name:
                service.running = True

        if self.verbose > 2:
            self.logger.info(
                f"Defender - Starting service {port_name} on machine {node_id}")
