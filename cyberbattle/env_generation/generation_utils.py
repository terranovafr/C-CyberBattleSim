# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    generation_utils.py
    Script with utility functions for the generation of the environment and its components.
"""

import math
import random
import copy
import numpy as np
from cyberbattle.simulation.model import wrap_networkx_to_model

# Ensures the assigned port is unique for the node
def assign_unique_port(service, other_services):
    while any(svc['port'] == service['port'] for svc in other_services):
        service['port'] = random.randint(1, 65535)

# Helper function to extract the most recent vulnerability data based on CVSS version
def select_cvss_version(metrics):
    if 'cvssMetricV31' in metrics:
        return 3.1, metrics['cvssMetricV31'][0]
    elif 'cvssMetricV30' in metrics:
        return 3, metrics['cvssMetricV30'][0]
    elif 'cvssMetricV2' in metrics:
        return 2, metrics['cvssMetricV2'][0]
    return None, None

# Extracts and formats the vulnerability data with proper function based on CVSS version
def extract_vulnerability_data(vuln_id, vuln, cvss_metric, version):
    if version >= 3:
        return format_cvss_v3_vulnerability(vuln_id, vuln, cvss_metric)
    elif version == 2:
        return format_cvss_v2_vulnerability(vuln_id, vuln, cvss_metric)

# Function to generate valid probabilities for the model
def generate_valid_probabilities(G, config, max_attempts = 100):
    while True:
        model = wrap_networkx_to_model(G.copy(), **config)
        max_attempts -= 1
        if not (model.access_connectivity < config['minimum_access_connectivity_threshold'] or
                model.knows_connectivity < config['minimum_knows_connectivity_threshold'] or
                model.dos_connectivity < config['minimum_dos_connectivity_threshold']):
            break
        if max_attempts == 0:
            break
    if max_attempts == 0:
        return None
    return model

# Formats CVSS v3 vulnerabilities
def format_cvss_v3_vulnerability(vuln_id, vuln, cvss_metric):
    return {
        "ID": vuln_id,
        "description": vuln["description"],
        "ranking_epss": vuln.get("ranking_epss"),
        "attack_vector": cvss_metric['cvssData']['attackVector'],
        "attack_complexity": cvss_metric['cvssData']['attackComplexity'],
        "privileges_required": cvss_metric['cvssData']['privilegesRequired'],
        "user_interaction": cvss_metric['cvssData']['userInteraction'],
        "confidentiality_impact": cvss_metric['cvssData']['confidentialityImpact'],
        "integrity_impact": cvss_metric['cvssData']['integrityImpact'],
        "availability_impact": cvss_metric['cvssData']['availabilityImpact'],
        "base_score": cvss_metric['cvssData']['baseScore'],
        "exploitability_score": cvss_metric['exploitabilityScore'],
        "impact_score": cvss_metric['impactScore'],
        'base_severity': cvss_metric['cvssData']['baseSeverity'],
        'classes': copy.deepcopy(vuln["classes"]),
        'feature_vector': extract_feature_vectors(vuln)
    }

# Formats CVSS v2 vulnerabilities
def format_cvss_v2_vulnerability(vuln_id, vuln, cvss_metric):
    return {
        "ID": vuln_id,
        "description": vuln["description"],
        "ranking_epss": vuln.get("ranking_epss"),
        "attack_vector": cvss_metric['cvssData']['accessVector'],
        "attack_complexity": cvss_metric['cvssData']['accessComplexity'],
        "privileges_required": cvss_metric['cvssData']['authentication'],
        "confidentiality_impact": cvss_metric['cvssData']['confidentialityImpact'],
        "integrity_impact": cvss_metric['cvssData']['integrityImpact'],
        "availability_impact": cvss_metric['cvssData']['availabilityImpact'],
        "base_score": cvss_metric['cvssData']['baseScore'],
        "exploitability_score": cvss_metric['exploitabilityScore'],
        "user_interaction": "None",  # CVSS v2 doesn't have user interaction
        "impact_score": cvss_metric['impactScore'],
        'base_severity': cvss_metric['baseSeverity'],
        'classes': copy.deepcopy(vuln["classes"]),
        'feature_vector': extract_feature_vectors(vuln)
    }

# Extract feature vectors of all NLP extractor models
def extract_feature_vectors(elem):
    feature_vectors = {}
    for key in elem:
        if key.startswith("feature_vector"):
            model_name = key.split("_")[2] if key != "feature_vector" else "unknown"
            feature_vectors[model_name] = elem[key]
    return feature_vectors

# Helper function to group services by product name
def group_services_by_product(services):
    product_groups = {}
    for service in services:
        if 'product' not in service:
            continue
        product = normalize_product_name(service['product'])
        if product not in product_groups:
            product_groups[product] = []
        service['product_group'] = product
        product_groups[product].append(service)
    return product_groups

# Normalizes product names to handle cases like Grafana, ZTE, etc. (that have subproducts but should belong to same product group)
# This is done with the assumption that it is rare to find similar products in the same host (e.g. ZTE X and ZTE Y)
def normalize_product_name(product):
    product = product.split()[0]
    known_products = ['Grafana', 'ZTE', 'DrayTek', 'Boa', 'Apache']
    return product if product in known_products else product

# Calculates selection probabilities for each service based on frequency of internet-connected hosts
def calculate_selection_probabilities(services):
    weights = []
    for service in services:
        frequency = service.get('frequency', 1)
        weight = math.log(frequency + 1)
        service['selection_probability'] = weight
        weights.append(weight)

    total_weight = sum(weights)
    for service in services:
        service['selection_probability'] /= total_weight


# Selects the CVSS source based on a priority order or randomly
def select_cvss_source(cvss_metric, priority_order=None):
    if not priority_order:
        priority_order = ['nvd@nist.gov', 'security-advisories@github.com', 'cret@cert.org', 'secalert@redhat.com']

    if len(cvss_metric) == 1:
        return cvss_metric[0]  # Return the only item if there's just one

    # Iterate through the priority order to find a preferred source
    for preferred_source in priority_order:
        for source in cvss_metric:
            if source['source'] == preferred_source:
                return source  # Return the first match based on priority

    return random.choice(cvss_metric)  # Return random source if no match found

def collect_environment_statistics(env, graph_id, vulnerabilities_distribution):
    stats = {
        'environment_ID': graph_id,
        'firewall_rule_incoming_probability': env.firewall_rule_incoming_probability,
        'firewall_rule_outgoing_probability': env.firewall_rule_outgoing_probability,
        'knows_neighbor_probability': np.mean(env.knows_neighbor_probability_range),
        'need_to_escalate_probability': env.need_to_escalate_probability,
        'partial_visibility_probability':  env.partial_visibility_probability,
        'data_presence_probability': env.data_presence_probability,
        'service_shutdown_probability': env.service_shutdown_probability,
        'value_range_mean': np.mean(env.value_range),
        'success_rate_probability_range': np.mean(env.success_rate_probability_range),
        'knows_connectivity': env.knows_connectivity,
        'knows_reachability': env.knows_reachability,
        'access_connectivity': env.access_connectivity,
        'access_reachability': env.access_reachability,
        'dos_reachability': env.dos_reachability,
        'dos_connectivity': env.dos_connectivity,
        'num_vulnerabilities': len(vulnerabilities_distribution)
    }
    return stats


# Helper function to clean vulnerabilities based on feature vectors and CVSS metrics
def clean_vulnerabilities(vulns, logger):
    clean_vulns = []
    for vuln_id, vuln in vulns.items():
        if not any(key.startswith("feature_vector") for key in vuln): # error in preprocesssing
            logger.info("No feature vector, skipping vulnerability..")
            continue

        version, cvss_metric = select_cvss_version(vuln['metrics']) # select latest
        if not cvss_metric:
            continue

        clean_vuln = extract_vulnerability_data(vuln_id, vuln, cvss_metric, version)
        clean_vulns.append(clean_vuln)
    return clean_vulns

# Sample random vulnerabilities in a fictious service that will be representative of this number of vulnerabilities
def sample_random_vulnerabilities(overall_services, num_requested_vulnerabilities, logger):
    sampled_services = []
    sampled_vulnerabilities = []
    for service in overall_services:
        if 'vulns' in service:
            clean_vulns = clean_vulnerabilities(service['vulns'], logger)
            # remove from clean vulns those already sampled
            clean_vulns = [clean_vuln for clean_vuln in clean_vulns if clean_vuln["ID"] not in [vuln["ID"] for vuln in sampled_vulnerabilities]]
            if len(clean_vulns) > num_requested_vulnerabilities-len(sampled_vulnerabilities):
                sampled_vulnerabilities.extend(random.sample(clean_vulns, num_requested_vulnerabilities-len(sampled_vulnerabilities)))
            else:
                sampled_vulnerabilities.extend(clean_vulns)

        else:
            continue
        last_feature_vector = extract_feature_vectors(service)
        if len(sampled_vulnerabilities) >= num_requested_vulnerabilities:
            break
    fictious_service = {"port": random.randint(1, 65535), "product": "fictious",
                        "version": "fictious", "description": "Fictious service to allocate some vulnerabilities",
                        "vulnerabilities": sampled_vulnerabilities, "feature_vector": last_feature_vector}
    assign_unique_port(fictious_service, sampled_services)
    return fictious_service


def collect_vulnerability_data(env):
    vulnerabilities_stats = {
        'vulnerabilities_presence': {},
        'outcomes_presence': {},
        'outcomes_nodes_presence': {},
    }

    for node, node_info in env.nodes():
        vulnerabilities_stats['outcomes_nodes_presence'][node] = {}
        for vulnerability in node_info.vulnerabilities:
            # Count vulnerabilities
            if vulnerability not in vulnerabilities_stats['vulnerabilities_presence']:
                vulnerabilities_stats['vulnerabilities_presence'][vulnerability] = 0
            vulnerabilities_stats['vulnerabilities_presence'][vulnerability] += 1

            # Record possible outcomes
            for result in node_info.vulnerabilities[vulnerability].results:
                outcome_type = result.outcome_str
                if outcome_type not in vulnerabilities_stats['outcomes_nodes_presence'][node]:
                    vulnerabilities_stats['outcomes_nodes_presence'][node][outcome_type] = []
                if outcome_type not in vulnerabilities_stats['outcomes_presence']:
                    vulnerabilities_stats['outcomes_presence'][outcome_type] = 0
                vulnerabilities_stats['outcomes_nodes_presence'][node][outcome_type].append(vulnerability)
                vulnerabilities_stats['outcomes_presence'][outcome_type] += 1

    return vulnerabilities_stats
