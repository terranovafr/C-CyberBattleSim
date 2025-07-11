# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    scrape_shodan.py
    Script to scrape Shodan for stats on services and their versions
"""

import shodan
import yaml
import sys
import os
from tqdm import tqdm
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.utils.file_utils import load_yaml, save_yaml # noqa: E402
script_parent_dir = Path(__file__).parent.parent

# Add tags to products based on mappings file (hard-coded)
def add_tags_to_products(products, mappings):
    for product in products:
        product_name = product.get('product', '')
        tags = []
        for context, names in mappings.items():
            if context != 'shodan': # avoid shodan configuration element
                if product_name in names:
                    tags.append(context)
        product['tags'] = tags

# Get top products_limit products resulting from a Shodan query
def get_top_products(shodan_api, mappings, products_limit, query, folder, logger, verbose):
    facets = [('product', products_limit)]

    try:
        results = shodan_api.count(query, facets=facets)
        top_products = results.get('facets', {}).get('product', [])[:products_limit]

        if not top_products:
            return []

        for product in top_products:
            product['product'] = product.pop('value')

        add_tags_to_products(top_products, mappings)
        save_yaml(top_products, folder, 'top_products.yaml')
        if verbose:
            logger.info(
                f"Top {products_limit} products fetched successfully in folder {os.path.basename(folder.rstrip('/'))}")
    except shodan.APIError as e:
        logger.warning(f'Shodan API error: {e}')
        return []


# Get top versions_limit versions for each product in the top products list of the Shodan query
def get_top_versions(shodan_api, versions_limit, base_query, folder, logger, verbose):
    product_versions = []
    try:
        with open(os.path.join(folder, "top_products.yaml"), 'r') as file:
            top_products = yaml.safe_load(file)
        for product in tqdm(top_products, desc='Fetching versions of the top products'):
            product_name = product['product']
            query = f'product:"{product_name}"' + f' {base_query}'  # Add the product name to the query
            facet_results = shodan_api.count(query, facets=[('version', versions_limit)])
            top_versions = facet_results.get('facets', {}).get('version', [])[:versions_limit]
            for version in top_versions:
                product_versions.append({
                    'product': product_name,
                    'tags': product['tags'],  # Add tags to the version data
                    'version': version['value'],
                    'count': version['count']
                })

        save_yaml(product_versions, folder, 'top_products_versions.yaml')
        if verbose:
            logger.info(
                f"Top {versions_limit} versions per product fetched successfully in folder {os.path.basename(folder.rstrip('/'))}")

        return product_versions

    except shodan.APIError as e:
        logger.warning(f'Shodan API error: {e}')
        return []

# main function to execute the scraping process
def main(mapping_file, auth, num_products, num_versions, folder, logger, verbose, query='has_vuln:true',):
    mappings = load_yaml(mapping_file)
    SHODAN_API_KEY = auth['shodan']['key']
    shodan_api = shodan.Shodan(SHODAN_API_KEY)
    get_top_products(shodan_api, mappings, num_products, query, folder, logger, verbose)
    get_top_versions(shodan_api, num_versions, query, folder, logger, verbose)
