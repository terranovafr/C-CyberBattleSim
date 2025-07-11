# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    generate_network.py
    This file contains the function to generate a cyberbattle model from a nodes graph, set of probabilities or ranges, and a vulnerability classifier.
"""

import copy
from cyberbattle.simulation.model import FirewallConfiguration, FirewallRule, RulePermission, Rates
from cyberbattle.utils.encoding_utils import map_label_to_class
import networkx as nx
from cyberbattle.simulation import model as m
import random
from cyberbattle.utils.classifier_utils import VulnerabilityClassifier

# Function used to draw a probability from a uniform distribution based on a score
def scale_probability_range_with_score(probability_range, score, score_range=None):
    if not score_range:
        score_range = [0, 10]
    if score is None:
        return random.uniform(probability_range[0], probability_range[1])
    if score < 0:
        return 0
    min_val = probability_range[0]
    max_val = probability_range[1]

    mapped_value = min_val + (((score - score_range[0]) / (score_range[1] - score_range[0])) * (max_val - min_val))
    return mapped_value

# Function to add a vulnerability edge to the graph, ensuring that vulnerabilities are not duplicated
# Used when the agent exploits a vulnerability (local or remote)
def add_vulnerability_edge(graph, src, dst, vulnerability_id, scores):
    if graph.has_edge(src, dst):
        # If the edge already exists, append the new vulnerability if not already listed
        vulnerabilities = graph[src][dst].get('vulnerabilities', [])
        if vulnerability_id not in vulnerabilities:
            vulnerabilities.append((vulnerability_id, scores))
            graph[src][dst]['vulnerabilities'] = vulnerabilities
    else:
        # If no edge exists, create one with a list containing the current vulnerability
        graph.add_edge(src, dst, vulnerabilities=[(vulnerability_id, scores)])

# Function to map confidentiality impact to a real number for the success rate calculation
def map_confidentiality_impact_to_real(confidentiality_impact):
    if confidentiality_impact is None:
        return 0
    confidentiality_mapping = {
        'NONE': -1, # no impact, it will become probability 0
        'PARTIAL': 0.5,
        'LOW': 0.5,
        'COMPLETE': 1,
        'HIGH': 1
    }
    # two different standard versions use two different values for the same impact (PARTIAL/LOW and COMPLETE/HIGH)
    return confidentiality_mapping[confidentiality_impact]

# Function to map attack complexity to a real number for the success rate calculation
def map_attack_complexity_to_real(attack_complexity): # inversed because it is gonna determine the succes rate
    if attack_complexity is None:
        return 0
    attack_complexity_mapping = {
        'LOW': 1,
        'MEDIUM': 0.75,
        'HIGH': 0.5
    }
    return attack_complexity_mapping[attack_complexity]

# Function to map the attack vector of a vulnerability to a class in the cyberbattle model
def map_vulnerability_attack_vector_to_class(attack_vector):
    if attack_vector == "LOCAL":
        return m.VulnerabilityType.LOCAL, "local"
    elif attack_vector == "NETWORK":
        return m.VulnerabilityType.REMOTE, "remote"
    elif attack_vector == "ADJACENT_NETWORK":
        return m.VulnerabilityType.REMOTE, "remote"
    elif attack_vector == "PHYSICAL":
        return m.VulnerabilityType.LOCAL, "local"
    else:
        raise ValueError("ALERT: Vulnerability type", attack_vector, " not found ")

# Function to map the privileges required for a vulnerability to a class in the cyberbattle model
def map_privilege_required_to_class(privileges_required):
    if privileges_required is None:
        return m.PrivilegeLevel.LocalUser
    privilege_mapping = {
        'NONE': m.PrivilegeLevel.NoAccess,
        'LOW': m.PrivilegeLevel.LocalUser,
        'SINGLE': m.PrivilegeLevel.LocalUser,
        'HIGH': m.PrivilegeLevel.ROOT,
        'MULTIPLE': m.PrivilegeLevel.ROOT,
    }
    return privilege_mapping[privileges_required]

# Creates a cyberbattle model from a nodes graph, set of probabilities or ranges, and a vulnerability classifier
def cyberbattle_model_from_nodes_graph(
    nodes_graph: nx.DiGraph,
    vulnerability_classifier: VulnerabilityClassifier,
    firewall_rule_incoming_probability=0.2,
    firewall_rule_outgoing_probability=0.2,
    knows_neighbor_probability_range=None,
    data_presence_probability=0.5,
    partial_visibility_probability=0.5,
    need_to_escalate_probability=0.5,
    service_shutdown_probability=0.1,
    success_rate_probability_range=None,
    probing_detection_probability_range=None,
    exploit_detection_probability_range=None,
    value_range=(0, 100),
) -> [nx.DiGraph, int]:

    # default lists, done here since they are mutable objects
    if knows_neighbor_probability_range is None:
        knows_neighbor_probability_range = [0.2, 0.3]
    if success_rate_probability_range is None:
        success_rate_probability_range = [0.9, 1]
    if probing_detection_probability_range is None:
        probing_detection_probability_range = [0.1, 0.2]
    if exploit_detection_probability_range is None:
        exploit_detection_probability_range = [0.1, 0.2]

    # Creating three graphs for the model storing the ground truth
    access_graph = nx.DiGraph()
    knows_graph = nx.DiGraph()
    dos_graph = nx.DiGraph()

    # Adding nodes to the graphs
    for node in nodes_graph.nodes:
        access_graph.add_node(node)
        knows_graph.add_node(node)
        dos_graph.add_node(node)

    overall_ports = []
    for node in list(nodes_graph.nodes):
        # default elements
        node_ports = []
        services = []
        vuln_library = {}
        data_to_be_found = False
        partially_visible = False
        level_at_access = m.PrivilegeLevel.ROOT
        # Use the info in the original graph to determine how to create each node
        for service in nodes_graph.nodes[node]["services"]:
            overall_ports.append(service["port"])
            node_ports.append(service["port"])
            # Probability of each service being shutdown
            service_shutdown = random.random() < service_shutdown_probability
            if 'feature_vector' not in service:
                service['feature_vector'] = [0] * 768
            services.append(m.ListeningService(name=service["port"], product=service["product"], version=service["version"], feature_vector=service["feature_vector"], description=service["description"], running=not service_shutdown))
            if "vulnerabilities" not in service: # case where the service has no vulnerabilities
                continue
            for vulnerability in service["vulnerabilities"]:
                classes_selected = []
                # If classification is already done, use the classes (default)
                if 'classes' in vulnerability:
                    for class_elem in vulnerability['classes']:
                        vuln_outcome = map_label_to_class(class_elem["class"])
                        vuln_outcome_str = class_elem["class"]
                        probability = class_elem["probability"]
                        if vuln_outcome is None:
                            continue
                        else:
                            classes_selected.append((vuln_outcome, vuln_outcome_str, probability))
                else:
                    # If classification is not done, use the classifier (slower)
                    classes = vulnerability_classifier.predict(vulnerability["description"])
                    for class_tuple in classes:
                        vuln_outcome = map_label_to_class(class_tuple[0])
                        vuln_outcome_str = class_tuple[0]
                        probability = class_tuple[1]
                        if vuln_outcome is None:
                            continue
                        classes_selected.append((vuln_outcome, vuln_outcome_str, probability))

                results = []
                discovery_list = [] # used by Reconnaissance outcomes
                # Determining the type of vulnerability based on the attack vector (two standard versions)
                vulnerability_type, vulnerability_type_str = map_vulnerability_attack_vector_to_class(vulnerability['attack_vector'])

                # Determining other node properties based on the vulnerabilities contained
                for class_tuple in classes_selected:
                    vuln_outcome, vuln_outcome_str, probability = class_tuple
                    if isinstance(vuln_outcome, m.Collection):
                        if random.random() < data_presence_probability:
                            # If there is at least one vulnerability that can collect data, the node has data
                            data_to_be_found = True
                    elif isinstance(vuln_outcome, m.Reconnaissance):
                        knows_neighbor_probability = scale_probability_range_with_score(knows_neighbor_probability_range, map_confidentiality_impact_to_real(vulnerability['confidentiality_impact']))
                        for node_id in nodes_graph.nodes:
                             if node_id != node:
                                # Determining the set of nodes that can be discovered based on the probability
                                if random.random() < knows_neighbor_probability:
                                    discovery_list.append(node_id)
                        vuln_outcome = map_label_to_class(vuln_outcome_str, discovery_list) # add nodes as the outcome of the reconnaissance
                    elif isinstance(vuln_outcome, m.Discovery):
                        if random.random() < partial_visibility_probability:
                            # If there is at least one vulnerability that can acquire visibility, the node is partially visible
                            partially_visible = True
                    elif isinstance(vuln_outcome, m.PrivilegeEscalation):
                        if random.random() < need_to_escalate_probability:
                            # If there is at least one vulnerability that requires escalation, the node needs to escalate
                            level_at_access = m.PrivilegeLevel.LocalUser

                    # adding it to the resulting outcomes possible for the vulnerability
                    results.append(m.PredictedResult(type=vulnerability_type, type_str=vulnerability_type_str, outcome=vuln_outcome, outcome_str=vuln_outcome_str, probability=probability))

                exploit_detection_probability = random.uniform(exploit_detection_probability_range[0], exploit_detection_probability_range[1])
                probing_detection_probability = random.uniform(probing_detection_probability_range[0], probing_detection_probability_range[1])
                privileges_required = map_privilege_required_to_class(vulnerability["privileges_required"])
                cost = 10 - vulnerability['exploitability_score']
                # adding vulnerability to the library of the node
                vuln_library[vulnerability["ID"]] = m.VulnerabilityInfo(vulnerability_ID=vulnerability["ID"], description=vulnerability["description"], cost=cost,
                                        port=service["port"], results=results, embedding=vulnerability['feature_vector'], rates=Rates(successRate=scale_probability_range_with_score(success_rate_probability_range, map_attack_complexity_to_real(vulnerability['attack_complexity'])), exploitDetectionRate=exploit_detection_probability,
                                        probingDetectionRate=probing_detection_probability),
                                        attack_complexity=vulnerability["attack_complexity"], attack_vector=vulnerability["attack_vector"],
                                        privileges_required=privileges_required, user_interaction=vulnerability["user_interaction"], confidentiality_impact=vulnerability["confidentiality_impact"],
                                        integrity_impact=vulnerability["integrity_impact"], availability_impact=vulnerability["availability_impact"], base_score=vulnerability["base_score"],
                                                                       exploitability_score=vulnerability["exploitability_score"], impact_score=vulnerability["impact_score"], base_severity=vulnerability["base_severity"],)
        # Firewall by default is all open
        firewall_conf = FirewallConfiguration(
            [FirewallRule(port, RulePermission.ALLOW) for port in node_ports],
            [FirewallRule(port, RulePermission.ALLOW) for port in node_ports])
        category = nodes_graph.nodes[node]['category']
        nodes_graph.nodes[node].clear()
        nodes_graph.nodes[node].update({'data': m.NodeInfo(
            tag=category,
            services=services,
            value=random.randint(value_range[0], value_range[1]),
            agent_installed=False,
            firewall=copy.deepcopy(firewall_conf),
            vulnerabilities=vuln_library,
            has_data=data_to_be_found,
            visible=not partially_visible,
            level_at_access=level_at_access
        )})

    ports_list = list(set(overall_ports))

    # Assign firewall a-posteriori, generating probabilistically some firewall rules
    for node in list(nodes_graph.nodes):
        service_names = [service.name for service in nodes_graph.nodes[node]['data'].services]
        for index, rule in enumerate(nodes_graph.nodes[node]['data'].firewall.incoming):
            if rule.port in service_names:
                if random.random() < firewall_rule_incoming_probability:
                    nodes_graph.nodes[node]['data'].firewall.incoming[index].permission = RulePermission.BLOCK
                else:
                    nodes_graph.nodes[node]['data'].firewall.incoming[index].permission = RulePermission.ALLOW

        for index, rule in enumerate(nodes_graph.nodes[node]['data'].firewall.outgoing):
            if rule.port in service_names:
                if random.random() < firewall_rule_outgoing_probability:
                    nodes_graph.nodes[node]['data'].firewall.outgoing[index].permission = RulePermission.BLOCK
                else:
                    nodes_graph.nodes[node]['data'].firewall.outgoing[index].permission = RulePermission.ALLOW

    # Update discovery graph adding edges based on reconnaissance results (edge present if a target node can be discovered by source node)
    for node in list(nodes_graph.nodes):
        for vulnerability_id in nodes_graph.nodes[node]['data'].vulnerabilities:
            vulnerability = nodes_graph.nodes[node]['data'].vulnerabilities[vulnerability_id]
            results = vulnerability.results
            for result in results:
                if isinstance(result.outcome, m.Reconnaissance):
                    for node_id in result.outcome.nodes:
                        if node_id != node:
                            knows_graph.add_edge(node, node_id)

    # Update access graph and DOS graphs
    # Adding edges based on vulnerabilities that allow access/DOS to other nodes (edge present if a target node can be accessed/DOSed by source node)
    for node in list(nodes_graph.nodes):
        for vulnerability_id in nodes_graph.nodes[node]['data'].vulnerabilities:
            vulnerability = nodes_graph.nodes[node]['data'].vulnerabilities[vulnerability_id]
            # Storing scores in the edge, useful for statistics calculation later
            scores = {
                "Base score": vulnerability.base_score,
                "Impact score": vulnerability.impact_score,
                "Exploitability score": vulnerability.exploitability_score
            }
            results = vulnerability.results
            for result in results:
                if isinstance(result.outcome, m.LateralMove) or isinstance(result.outcome, m.CredentialAccess)\
                    or isinstance(result.outcome, m.DenialOfService):
                    firewall_incoming = nodes_graph.nodes[node]['data'].firewall.incoming
                    block_incoming=False
                    for rule in firewall_incoming:
                        if rule.port == vulnerability.port and rule.permission == RulePermission.BLOCK:
                            block_incoming = True
                            break
                    if block_incoming:
                        continue
                    for source_node in list(nodes_graph.nodes):
                        if source_node != node:
                            if nx.has_path(knows_graph, source_node, node):
                                firewall_outgoing_present = False
                                firewall_outgoing = nodes_graph.nodes[source_node]['data'].firewall.outgoing
                                for rule in firewall_outgoing:
                                    if rule.port == vulnerability.port and rule.permission == RulePermission.BLOCK:
                                        firewall_outgoing_present = True
                                        break
                                if not firewall_outgoing_present:
                                    # Adding edge if the source node knows the target node and the firewall (incoming and outgoing) allows the connection
                                    if isinstance(result.outcome, m.LateralMove) or isinstance(result.outcome, m.CredentialAccess):
                                        add_vulnerability_edge(access_graph, source_node, node, vulnerability_id, scores)
                                    if isinstance(result.outcome, m.DenialOfService):
                                        add_vulnerability_edge(dos_graph, source_node, node, vulnerability_id, scores)


    nodes_graph.clear_edges() # starting graph
    evolving_visible_graph = nx.DiGraph() # GNN graph

    return nodes_graph, knows_graph, access_graph, dos_graph, evolving_visible_graph, ports_list
