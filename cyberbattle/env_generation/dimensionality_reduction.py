# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    dimensionality_reduction.py
    Script to reduce the dimensionality of the feature vectors of the services and vulnerabilities in the scenarios using PCA
"""

import argparse
import os
import sys
from sklearn.decomposition import PCA
import numpy as np
import copy
from dataclasses import replace
from pathlib import Path
import pickle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.utils.file_utils import save_yaml # noqa: E402
from cyberbattle.utils.log_utils import setup_logging # noqa: E402
script_dir = Path(__file__).parent

# Collect all feature vectors in an environment in order to form the dataset needed for PCA
def collect_feature_vectors(env, feature_vectors):
    networkx = env.network
    for node in networkx.nodes:
        node_info = networkx.nodes[node]['data']
        for service in node_info.services:
            feature_vector = service.feature_vector
            feature_vectors.append(feature_vector)
        for vulnerability in node_info.vulnerabilities:
            vulnerability_info = node_info.vulnerabilities[vulnerability]
            feature_vector = vulnerability_info.embedding
            feature_vectors.append(feature_vector)
    return feature_vectors

# Compute the number of components needed to explain the variance threshold required
def compute_pca_with_threshold(feature_vectors, variance_threshold, model, logger, verbose):
    n_components = 2
    while True:
        pca = PCA(n_components=n_components)
        pca.fit(feature_vectors)
        if verbose:
            logger.info(f"Model {model} explained variance ratio {np.sum(pca.explained_variance_ratio_)} for {n_components} components")
        if np.sum(pca.explained_variance_ratio_) >= variance_threshold:
            break
        n_components += 1
    if verbose:
        logger.info(f"Model {model} requires {n_components} components to reach variance threshold {variance_threshold}")
    return pca, n_components

# Compute the variance threshold reached with the number of components provided
def compute_pca_with_components(feature_vectors, n_components, model, logger, verbose):
    feature_vectors = np.array(feature_vectors)
    pca = PCA(n_components=n_components)
    pca.fit(feature_vectors)
    if verbose:
        logger.info(
            f"Model {model} using PCA with {n_components} components reaches variance threshold {np.sum(pca.explained_variance_ratio_)}")
        logger.info(f"Explained variance ratio per component: {pca.explained_variance_ratio_}")
    return pca, n_components

# Iteratively reduce the dimensions of the feature vectors in the environment based on the PCA model
def reduce_dimensions(env, pca, n_components, folder_name, file_name, logger):
    def transform_feature_vector(pca, feature_vector):
        try:
            return pca.transform([feature_vector])[0]
        except Exception as e:
            logger.warning(f"Transformation failed for feature vector {feature_vector}: {e}")
            return feature_vector

    networkx = copy.deepcopy(env.network)
    del env.network

    for node_idx, node in enumerate(networkx.nodes):
        node_info = networkx.nodes[node]['data']
        # Transform the feature vectors of the services
        node_info.services = [
            replace(service, feature_vector=transform_feature_vector(pca, service.feature_vector))
            for service in node_info.services
        ]
        # Transform the feature vectors of the vulnerabilities
        transformed_vulnerabilities = {
            vuln_key: replace(vuln_info, embedding=transform_feature_vector(pca, vuln_info.embedding))
            for vuln_key, vuln_info in node_info.vulnerabilities.items()
        }
        node_info.vulnerabilities = transformed_vulnerabilities
        del networkx.nodes[node]['data']
        networkx.nodes[node]['data'] = node_info
    env.network = networkx
    # Check that the feature vectors have been transformed correctly
    for node in networkx.nodes:
        node_info = networkx.nodes[node]['data']
        for service in node_info.services:
            assert len(service.feature_vector) == n_components
        for vulnerability in node_info.vulnerabilities:
            vulnerability_info = node_info.vulnerabilities[vulnerability]
            assert len(vulnerability_info.embedding) == n_components

    output_folder = os.path.join(folder_name, "pca", f"num_components={str(n_components)}")
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, os.path.basename(file_name)), 'wb') as f:
        pickle.dump(env, f)


# The variance threshold option, if used, will just provide the stats (number of components needed) and not do the reduction
# The number of components option will provide the stats and do the reduction
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform dimensionality reduction on the feature vectors of the services and vulnerabilities of a given environment")
    parser.add_argument('-f', '--folder', type=str, required=True, help='Name of the graph generation experiment to save the logs')
    parser.add_argument('-vt', '--variance_threshold', type=float, default=None, help='Variance threshold for the PCA')
    parser.add_argument('-nc', '--number_components', type=int, default=None, help='Variance threshold for the PCA')
    parser.add_argument('-ar', '--avoid_reduction', action="store_true", default=False, help='Just show stats and do not do the reduction')
    parser.add_argument( '-nlp', '--nlp_extractors', nargs='+', type=str,
                        default=['bert', 'roberta', 'distilbert', 'gpt2', 'SecBERT', 'SecRoBERTa', 'SecureBERT',
                                 'CySecBERT'], help='Feature extractor models to use')
    parser.add_argument('--no_save_log_file', action='store_false', dest='save_log_file',
                        default=True, help='Disable logging to file; log only to terminal')
    parser.add_argument('-v', '--verbose', default=1, type=int,
                        help='Verbose level: 0 - no output, 1 - reduction information', choices=[0, 1])
    args = parser.parse_args()

    logs_folder = os.path.join(script_dir, "..", "data", "env_samples", args.folder)
    logger = setup_logging(logs_folder, log_to_file=args.save_log_file)

    if not args.variance_threshold and not args.number_components:
        logger.warning("Please provide either variance threshold or number of components")
        sys.exit(1)

    if args.verbose:
        logger.info(f"Performing reduction of graphs in folder: {logs_folder}")
        logger.info(f"Perform reduction targeting NLP extractor models: {args.nlp_extractors}")

    if args.variance_threshold:
        if args.verbose:
            logger.info(f"Indicating number of components for PCA reduction with variance threshold: {args.variance_threshold}")
        args.avoid_reduction = True  # when variance threshold specified, the number of components may be different per nlp extractor hence avoid reduction
    elif args.number_components:
        if args.verbose:
            logger.info(
                f"Performing PCA reduction with number of components: {args.number_components}")

    feature_vectors = {}
    for model_name in args.nlp_extractors:
        feature_vectors[model_name] = []

    if args.verbose:
        logger.info("Loading feature vectors from graphs...")
    # Iterate through the folders in the logs folder and collect the feature vectors from the environments
    for folder in os.listdir(logs_folder):
        if os.path.isdir(os.path.join(logs_folder, folder)):
            if folder.isdigit():
                if args.verbose:
                    logger.info(f"Processing {folder}/")
                for file in os.listdir(os.path.join(logs_folder, folder)):
                    if file.endswith(".pkl"):
                        model_name = file.split("_")[1].split(".")[0]
                        if model_name == "graph":
                            continue
                        env_file = os.path.join(logs_folder, folder, file)
                        with open(env_file, 'rb') as f:
                            env = pickle.load(f)
                        feature_vectors[model_name] = collect_feature_vectors(env, feature_vectors[model_name])
                        del env

    # For each NLP extractor, compute the PCA and save the stats
    pca_per_LLM = {}
    number_components_per_LLM = {}
    for nlp_extractor in args.nlp_extractors:
        if args.verbose:
            logger.info(f"Computing PCA for NLP extractor model: {nlp_extractor}...")
        if args.variance_threshold is not None:
            pca, number_components = compute_pca_with_threshold(feature_vectors[nlp_extractor], args.variance_threshold, nlp_extractor, logger, args.verbose)
            pca_per_LLM[nlp_extractor] = pca
            number_components_per_LLM[nlp_extractor] = number_components
        elif args.number_components is not None:
            pca, number_components = compute_pca_with_components(feature_vectors[nlp_extractor], args.number_components, nlp_extractor, logger, args.verbose)
            pca_per_LLM[nlp_extractor] = pca
            number_components_per_LLM[nlp_extractor] = number_components
        stats = {}
        stats['n_components'] = number_components_per_LLM[nlp_extractor]
        stats['explained_variances'] = pca_per_LLM[nlp_extractor].explained_variance_ratio_.tolist()
        if args.variance_threshold:
            output_folder = os.path.join(logs_folder, "pca_stats", nlp_extractor, "variance_threshold="+str(args.variance_threshold))
        else:
            output_folder = os.path.join(logs_folder, "pca_stats", nlp_extractor, "num_components="+str(number_components_per_LLM[nlp_extractor]))

        if args.verbose:
            logger.info(f"Saving stats to {output_folder}")
        save_yaml(stats, output_folder, "stats.yaml")
        pickle.dump(pca_per_LLM[nlp_extractor], open(os.path.join(output_folder, "dimensions_reducer.pkl"), 'wb'))

    # Save the PCA reduced environments for each NLP extractor if not avoiding reduction
    if not args.avoid_reduction:
        if args.verbose:
            logger.info("Performing PCA reduction...")
        for folder in os.listdir(logs_folder):
            if os.path.isdir(os.path.join(logs_folder,folder)) and folder.isdigit():
                if args.verbose:
                    logger.info("Reducing vectors of network: %s ", folder)
                for file in os.listdir(os.path.join(logs_folder, folder)):
                    if file.endswith(".pkl"):
                        model_name = file.split("_")[1].split(".")[0]
                        if model_name == "graph":
                            continue
                        file_name = os.path.join(logs_folder, folder, file)
                        with open(file_name, 'rb') as f:
                            env = pickle.load(f)
                        reduce_dimensions(env, pca_per_LLM[model_name], number_components_per_LLM[model_name], os.path.join(logs_folder, folder), file_name, logger)
