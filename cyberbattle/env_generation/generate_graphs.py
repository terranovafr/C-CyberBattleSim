# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    generate_graphs.py
    Script to generate network graphs based on the configuration file and the environment database.
"""

import argparse
import sys
import os
import networkx as nx
import random
from datetime import datetime
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.env_generation.split_graphs import split_environments # noqa: E402
from cyberbattle.utils.data_utils import list_subset_another # noqa: E402
from cyberbattle.utils.log_utils import setup_logging # noqa: E402
from cyberbattle.utils.math_utils import generate_convex_combinations # noqa: E402
from cyberbattle.utils.file_utils import load_yaml, save_yaml, load_and_merge_json_entries_by_tag, save_csv # noqa: E402
from cyberbattle.utils.envs_utils import save_network_or_model_as_pickle, save_model_nlp_extractors_versions # noqa: E402
from cyberbattle.env_generation.generation_utils import clean_vulnerabilities, generate_valid_probabilities, group_services_by_product, calculate_selection_probabilities, extract_feature_vectors, assign_unique_port, collect_vulnerability_data, collect_environment_statistics # noqa: E402
script_dir = Path(__file__).parent

# Samples unique products (using weighted sampling) based on the product groups required
def sample_unique_products(services, category, num_samples):
    product_groups = group_services_by_product(services) # avoid sampling the same version in a "unique combination" since unprobable to have two versions of a service in the same host
    calculate_selection_probabilities(services) # use as selection probabilities the number of hosts output of the query

    all_services = [svc for group in product_groups.values() for svc in group]
    sampled_services, sampled_products = [], []

    # Use the selection probabilities to sample the services avoiding collisions in the same group
    while len(sampled_services) < num_samples: # until we have the number of samples
        assert len(all_services) > 0, "No services available for the category {}, please scrape more services or change mode (-p) to some category where you have at least one sample in the environment database products.".format(category)
        sampled_service = random.choices(
            all_services,
            weights=[svc['selection_probability'] for svc in all_services],
            k=1
        )[0] # weighted sampling, take the only element (k=1)
        if sampled_service['product_group'] not in sampled_products:

            sampled_services.append(sampled_service)
            sampled_products.append(sampled_service['product_group'])

    return sampled_services, sampled_products

# Cleans and assigns ports to sampled services for each node
def sample_services(categories, unique_combinations_per_category, homogeneity, logger):
    sampled_services, cleaned_sampled_services = {}, {}

    for node in categories:
        category = categories[node]
        # choose a unique combination of services for the node (of its category)
        sampled_services[node] = sample_service_for_node(unique_combinations_per_category[category], homogeneity)
        cleaned_sampled_services[node] = clean_sampled_services(sampled_services[node], logger)

    return cleaned_sampled_services


# Samples services based on homogeneity value
def sample_service_for_node(services, homogeneity):
    if homogeneity == 0:
        return services
    return random.choices(services, weights=[homogeneity] * len(services), k=1)[0]


# Cleans sampled services for a node by assigning ports and cleaning vulnerabilities
def clean_sampled_services(sampled_service, logger):
    cleaned_services = []

    for service in sampled_service:
        if 'vulns' in service:
            clean_vulns = clean_vulnerabilities(service['vulns'], logger)
            service_dict = {"port": random.randint(1, 65535), "product": service["product"], "version": service["version"], "description": service["description"],
                            "vulnerabilities": clean_vulns}
        else:
            service_dict = {"port": random.randint(1, 65535), "product": service["product"], "version": service["version"], "description": service["description"]}

        service_dict["feature_vector"] = extract_feature_vectors(service)
        assign_unique_port(service_dict, cleaned_services)
        cleaned_services.append(service_dict)

    return cleaned_services



# Helper function to initialize a graph and its related data structures
def initialize_graph():
    G = nx.Graph()
    categories = {}
    graph_stats = {}
    unique_combinations_per_category = {}
    unique_products_per_category = {}
    return G, categories, graph_stats, unique_combinations_per_category, unique_products_per_category

# Function to calculate graph homogeneity and unique combinations
def calculate_homogeneity(config, num_nodes):
    homogeneity = random.uniform(config['homogeneity_range'][0], config['homogeneity_range'][1])
    num_unique_combinations = int(1 / homogeneity) if homogeneity != 0 else num_nodes
    return homogeneity, num_unique_combinations

# Function to generate unique service combinations per category
def generate_service_combinations(config, categories_config, category_samples, num_unique_combinations):
    unique_combinations_per_category = {}
    unique_products_per_category = {}
    overall_num_services = {}

    for category in categories_config['categories']:
        if config['percentage_type'] != 'random' and config['percentage_type'] != category:
            continue
        unique_combinations_per_category[category] = []
        unique_products_per_category[category] = []
        if config['num_services_range'][0] == config['num_services_range'][1] == 1:
            category_combinations = len(group_services_by_product(category_samples[category]))
        else:
            if category not in categories_config['max_services']:
                categories_config['max_services'][category] = len(group_services_by_product(category_samples[category]))
            category_combinations = min(num_unique_combinations, 2 ** categories_config['max_services'][category] - 1)

        overall_num_services[category] = 0
        num_elements = 0

        while num_elements < category_combinations:
            if category in categories_config['max_services']:
                num_services = min(random.randint(config['num_services_range'][0], config['num_services_range'][1]),
                                   categories_config['max_services'][category])
            else:
                num_services = random.randint(config['num_services_range'][0], config['num_services_range'][1])
            sampled_services, sampled_products = sample_unique_products(category_samples[category], category, num_services)
            if not list_subset_another(unique_products_per_category[category], sampled_products):
                unique_combinations_per_category[category].append(sampled_services)
                unique_products_per_category[category].append(sampled_products)
                num_elements += 1
                overall_num_services[category] += num_services

    return unique_combinations_per_category, overall_num_services

# Function to assign categories and services to nodes
def assign_node_services(num_nodes, category_percentages, unique_combinations_per_category, homogeneity, logger):
    categories = {}
    for i in range(num_nodes):
        category = random.choices(list(category_percentages.keys()), weights=list(category_percentages.values()))[0]
        categories[i] = category
    services = sample_services(categories, unique_combinations_per_category, homogeneity, logger)
    return categories, services

# Function to add nodes and services to the graph
def add_nodes_to_graph(G, num_nodes, services, categories):
    for i in range(num_nodes):
        node_id = f"Node_{i + 1}"
        G.add_node(node_id, services=services[i], category=categories[i])


# Main function to generate network graphs
def generate_network_graphs(logs_folder, config, categories_config, category_samples, nlp_extractors, logger, verbose=1):
    overall_graphs_stats = []
    graph_id = 0

    while graph_id < config['num_graphs']:
        # Step 1: Initialize graph and related data
        G, categories, graph_stats, unique_combinations_per_category, unique_products_per_category = initialize_graph()
        if verbose > 1:
            logger.info("Generating network %d...", graph_id + 1)

        # Step 2: Start to sample graph characteristics
        num_nodes = random.randint(config['num_nodes_range'][0], config['num_nodes_range'][1])
        graph_stats.update({'num_nodes': num_nodes})
        category_percentages = {category: pct for category, pct in config['percentages'][graph_id].items()}
        graph_stats.update(
            {f"percentage_{category}": category_percentages[category] for category in category_percentages})
        homogeneity, num_unique_combinations = calculate_homogeneity(config, num_nodes)
        graph_stats['num_unique_combinations'] = num_unique_combinations
        if verbose > 1:
            logger.info(f"Number of nodes: {num_nodes}")
            logger.info(f"Category percentages: {category_percentages}")
            logger.info(f"Graph homogeneity: {homogeneity}")

        # Step 3: Generate service combinations for each category
        unique_combinations_per_category, overall_num_services = generate_service_combinations(config, categories_config, category_samples, num_unique_combinations)
        if verbose > 1:
            for category in unique_combinations_per_category:
                logger.info("Bigger set of services of category %s has length %d", category, len(unique_combinations_per_category[category]))
                services = []
                for service in unique_combinations_per_category[category]:
                    services.append((service[0]["product"], service[0]["version"]))
                logger.info(f"Services: {services}")

        # Step 4: Assign services and categories to nodes
        categories, services = assign_node_services(num_nodes, category_percentages, unique_combinations_per_category, homogeneity, logger)
        if verbose > 2:
            for node in services:
                services_list = []
                for dict in services[node]:
                    services_list.append((dict["product"], dict["version"]))
                logger.info(f"Node: {node} category {categories[node]} assigned services: {services_list}")

        # Step 5: Add nodes to the graph
        add_nodes_to_graph(G, num_nodes, services, categories)

        # Generating many variation of probabilities until satisfying the thresholds
        model = generate_valid_probabilities(G, config)
        if not model:
            if verbose > 1:
                logger.warning("Restarting the generation of the graph %d as the generated graph does not satisfy the connectivity thresholds", graph_id + 1)
            continue

        if verbose > 1:
            logger.info("Graph %d has been successfully generated with proper probabilities respecting connectivity thresholds", graph_id + 1)
            logger.info("Connectivity values: Access: %.2f, Knows: %.2f, DoS: %.2f", model.access_connectivity, model.knows_connectivity, model.dos_connectivity)

        graph_id += 1

        # Step 6: Save the generated graph and its statistics
        save_network_or_model_as_pickle(G, os.path.join(logs_folder, str(graph_id)), 'graph')
        graph_stats['environment_ID'] = graph_id
        overall_graphs_stats.append(graph_stats)
        vulnerabilities_stats = collect_vulnerability_data(model)
        graph_stats.update(collect_environment_statistics(model, graph_id, vulnerabilities_stats['vulnerabilities_presence']))
        save_yaml(vulnerabilities_stats, os.path.join(logs_folder, str(graph_id)),
                  'vulnerabilities_distribution.yaml')
        if verbose > 1:
            logger.info(f"Saving the graph %d for each feature extractor model in {nlp_extractors}", graph_id)
        save_model_nlp_extractors_versions(model, nlp_extractors, logs_folder, graph_id)

    if verbose:
        logger.info("Generation of %d graphs completed successfully!", config['num_graphs'])
    save_csv(overall_graphs_stats, logs_folder, "graphs_stats.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graphs generation script")
    parser.add_argument('--name', type=str, default=None, help='Name of the graph generation experiment to save the logs')
    parser.add_argument('-p', '--percentage_type', type=str, default='random', choices=['iot', 'routers', 'windows', 'unix', 'ics', 'mixed_equally', 'random'], help='Percentage of nodes per category')
    parser.add_argument('--num_graphs', type=int, default=5,
                        help='Number of topologies to generate')
    parser.add_argument('-nlp', '--nlp_extractors', nargs='+', type=str, default=[ 'bert', 'roberta', 'distilbert', 'gpt2', 'SecBERT', 'SecRoBERTa', 'SecureBERT', 'CySecBERT'], help='Feature extractor models to use')
    parser.add_argument('-nos', '--no_split', action='store_false', dest='split',
                        default=True, help='Disable splitting the generated environments into training, validation and test sets')
    parser.add_argument('-train', '--split_train', type=float, default=0.6, help='Percentage of data to use for training')
    parser.add_argument('-val', '--split_val', type=float, default=0.2, help='Percentage of data to use for validation')
    parser.add_argument('-gc', '--generation_config', type=str,
                        default=os.path.join("config", "generation_config.yaml"),
                        help='Path to the configuration YAML file')
    parser.add_argument('-cc','--categories_config', type=str,
                        default=os.path.join("config", "categories_config.yaml"),
                        help='Path to the configuration YAML file')
    parser.add_argument( '--complexity_criteria_file', type=str, default=os.path.join("config", "complexity_criteria.yaml"),
                        help='Path to the YAML file containing complexity criteria'),
    parser.add_argument('--no_save_log_file', action='store_false', dest='save_log_file',
                        default=True, help='Disable logging to file; log only to terminal')
    parser.add_argument('-v', '--verbose', default=2, type=int,
                        help='Verbose level: 0 - no output, 1 - generation general information, 2 - single graph generation information, 3 - single node generation information', choices=[0, 1, 2, 3])
    args = parser.parse_args()
    if args.split_train + args.split_val > 1:
        raise ValueError("The sum of training and validation percentages must be less than 1.")

    if not args.name:
        logs_folder = os.path.join(script_dir, '..', 'data', 'env_samples', "graphs_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        logs_folder = os.path.join(script_dir, '..', 'data', 'env_samples',
                                   "graphs_" + args.name + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(logs_folder, exist_ok=True)

    logger = setup_logging(logs_folder, log_to_file=args.save_log_file)

    generation_config = load_yaml(os.path.join(script_dir, args.generation_config))
    if args.verbose:
        logger.info(f"Reading graphs generation configuration from {args.generation_config}")
        logger.info(generation_config)
    categories_config = load_yaml(os.path.join(script_dir, args.categories_config))

    # Generate percentage of nodes in each topology
    if args.percentage_type == 'random':
        if args.verbose:
            logger.info("Generating random percentage of nodes per category in each graph...")
        generation_config['percentages'] = []
        samples = generate_convex_combinations(args.num_graphs, 5, generation_config['min_presence_each_category'])
        for sample in samples:
            generation_config['percentages'].append({
            'iot': sample[0],
            'routers': sample[1],
            'windows': sample[2],
            'unix': sample[3],
            'ics': sample[4],
        })
    else:
        if args.verbose:
            logger.info(f"Generating graphs with percentage type: {args.percentage_type}")
        generation_config['percentages'] = []
        for i in range(args.num_graphs):
            generation_config['percentages'].append(categories_config['percentages_types'][args.percentage_type]) # fixed percentage types defined in the config file

    if args.verbose > 1:
        logger.info("Generated percentages for each graph:")
        logger.info(generation_config['percentages'])

    generation_config['num_graphs'] = args.num_graphs
    generation_config['max_services_per_category'] = categories_config['max_services']
    save_yaml(generation_config, logs_folder, "generation_config.yaml")

    # loading the services data samples related to each category
    category_samples = {}
    paths_config = load_yaml(os.path.join(script_dir, "..", "..", "config.yaml"))
    args.data_folder = os.path.join(script_dir, "..", "data", "scrape_samples", paths_config['nvd_data_path'])
    if args.verbose:
        logger.info("Loading services data sample from \"%s\"", args.data_folder)
    if args.percentage_type == 'random':
        for category in categories_config['categories']:
            category_samples[category] = load_and_merge_json_entries_by_tag(tag=category, folder=os.path.join(script_dir, args.data_folder), path_filename_contains='extracted')
    else:
        category = args.percentage_type
        category_samples[category] = load_and_merge_json_entries_by_tag(tag=category, folder=os.path.join(script_dir, args.data_folder), path_filename_contains='extracted')

    generation_config['percentage_type'] = args.percentage_type
    generate_network_graphs(logs_folder, generation_config, categories_config, category_samples, args.nlp_extractors, logger, args.verbose)
    envs_stats_file = os.path.join(logs_folder, "graphs_stats.csv")

    # if splitting is enabled, split the environments based on complexity criteria
    if args.split:
        logger.info("Splitting the generated environments into %.2f training, %.2f validation and %.2f test", args.split_train, args.split_val, 1 - (args.split_train + args.split_val))
        args.complexity_criteria_file = os.path.join(script_dir, args.complexity_criteria_file)
        split_environments(envs_stats_file, os.path.join(script_dir, args.complexity_criteria_file), args.split_train, args.split_val)
