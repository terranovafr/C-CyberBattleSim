# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

# This product uses the NVD API but is not endorsed or certified by the NVD.
"""
    scrape_nvd.py
    Script to scrape the NVD for vulnerabilities of the service versions provided
"""


import copy

from tqdm import tqdm
import requests
import time
from dateutil import parser
import sys
import os
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.utils.file_utils import load_pickle, save_pickle, load_yaml, save_yaml, save_json # noqa: E402
script_parent_dir = Path(__file__).parent.parent

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
CACHE_SAVE_INTERVAL = 1 # how often to save the cache to disk, in number of products processed

# Construct a CPE string from a product and version
def construct_cpe(product, version):
    if version == "$1":
        version = None
    if version is not None:
        return f"cpe:2.3:a:*:{product.lower()}:{version}"
    else:
        return f"cpe:2.3:a:*:{product.lower()}"

# Get the most recent non-deprecated CPE from the NVD data of a certain product
def get_most_recent_non_deprecated_cpe(data):
    most_recent_cpe = None
    most_recent_date = None
    for product in data['products']:
        cpe = product['cpe']
        if not cpe['deprecated']:
            last_modified_date = parser.parse(cpe['lastModified'])
            if most_recent_date is None or last_modified_date > most_recent_date:
                most_recent_date = last_modified_date
                most_recent_cpe = cpe
    return most_recent_cpe

# Search the NVD for vulnerabilities for a given CPE
def search_cve_for_cpe(cpe, headers, logger, verbose):
    base_url = 'https://services.nvd.nist.gov/rest/json/cves/2.0'
    params = {
        'cpeName': cpe,
        'resultsPerPage': 100,
        'startIndex': 0
    }
    cpe_vulnerabilities = {}
    vulnerabilities = []
    cpe_vulnerabilities['cpe'] = cpe

    while True:
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            if 'vulnerabilities' in data:
                for elem in data['vulnerabilities']:
                    cve_id = elem['cve']['id']
                    # find the first one in english
                    cve_description = ""
                    for description in elem['cve']['descriptions']:
                        if description['lang'] == 'en':
                            cve_description = description['value']
                            break
                    cve_metrics = elem['cve']['metrics']
                    vulnerability = {
                        'cve_id': cve_id,
                        'description': cve_description,
                        'metrics': cve_metrics
                    }
                    vulnerabilities.append(vulnerability)
                total_results = data['totalResults']
                if params['startIndex'] + params['resultsPerPage'] >= total_results:
                   break
                else:
                    params['startIndex'] += params['resultsPerPage']

            else:
                break

        except requests.exceptions.RequestException as e:
            if verbose > 0:
                logger.warning(f"Error querying NVD for CPE {cpe}: {e}, waiting to retry...")
            time.sleep(10)
            continue

    cpe_vulnerabilities['vulnerabilities'] = vulnerabilities
    return cpe_vulnerabilities

# Query the NVD for a CPE and get the most recent non-deprecated CPE
def query_nvd_cpe(cpe, headers, most_recent = True):
    base_url = "https://services.nvd.nist.gov/rest/json/cpes/2.0?"
    params = {
        "cpeMatchString": cpe
    }
    while True:
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors

            data = response.json()
            if data.get("products"):
                if most_recent:
                    most_recent_cpe = get_most_recent_non_deprecated_cpe(data)
                    return most_recent_cpe['cpeName']
                else:
                    return data['products']
            else:
                raise Exception("No matching CPEs found.")

        except requests.exceptions.RequestException:
            time.sleep(10)
            continue

# Find the largest version from a list of versions
def find_largest_version(versions):
    def version_key(version):
        return list(map(int, version.split('.')))

    def pad_version(version_parts, max_length):
        return version_parts + [0] * (max_length - len(version_parts))

    # Convert versions to lists of integers and find the max length
    version_lists = [version_key(version) for version in versions]
    max_length = max(len(v) for v in version_lists)

    # Pad versions to the max length
    padded_versions = [pad_version(v, max_length) for v in version_lists]

    # Find the largest padded version
    largest_version = max(padded_versions)

    return '.'.join(map(str, largest_version))

# Find the closest version to a given one from a list of versions (since the one we are interested in may not be included in NVD)
def find_closest_version(desired_version, available_versions):
    digit_present = False
    for version in available_versions:
        if version == "-":
            continue
        else:
            digit_present = True
    if not digit_present:
        return available_versions[0]
    def version_key(version):
        return list(map(int, version.split('.')))

    desired_parts = version_key(desired_version)
    closest_version = None
    best_differences_per_level = None

    for version in available_versions:
        try:
            version_parts = version_key(version)
            max_length = max(len(desired_parts), len(version_parts))

            # Pad the shorter version with zeros for proper comparison
            padded_desired = desired_parts + [0] * (max_length - len(desired_parts))
            padded_version = version_parts + [0] * (max_length - len(version_parts))

            differences_per_level = []
            for dp, vp in zip(padded_desired, padded_version):
                difference = abs(dp - vp)
                differences_per_level.append(difference)

            if best_differences_per_level is None:
                best_differences_per_level = copy.deepcopy(differences_per_level)
                closest_version = version
            else:
                for index in range(len(differences_per_level)):
                    if differences_per_level[index] < best_differences_per_level[index]:
                        best_differences_per_level = copy.deepcopy(differences_per_level)
                        closest_version = version
                        break
                    elif differences_per_level[index] > best_differences_per_level[index]:
                        break
                    else:
                        if best_differences_per_level[index] != 0:
                            closest_version = find_largest_version([closest_version, version])
                            break
                        else:
                            continue
        except ValueError:
            continue

    return closest_version


# If we have a well-known product (cpe available in mappings file), we can use the CPE from the mapping
def is_well_known_product(product, product_cpe_map, version=None):
    if version == "$1":
        version = None

    # Normalize product name to lower case for comparison
    normalized_product = product.lower()

    for key, base_cpe in product_cpe_map.items():
        if normalized_product == key or normalized_product.startswith(key):
            # Construct the CPE with or without the version
            cpe = base_cpe if version is None else f"{base_cpe}:{version}"
            return True, cpe

    return False, product

def main(auth, mappings_file,  cache_file, target_folder, logger, verbose):
    NVD_API_KEY = auth['nvd']['key']
    headers = {
        'apiKey': NVD_API_KEY
    }
    product_cpe_map = load_yaml(mappings_file)['product_cpe_map']

    # Loading the product versions from the proper top_products_versions.yaml file
    product_versions = load_yaml(os.path.join(script_parent_dir, 'data', 'scrape_samples', target_folder, 'top_products_versions.yaml'))
    products_info = []
    new_product_versions = []
    cache = load_pickle(cache_file)

    # Fetching vulnerabilities for each version or from cache or from the NVD
    # If the NVD does not have the version, we find the closest one
    for pv in tqdm(product_versions, desc='Fetching vulnerabilities for products'):
        product_info = {}
        product = pv['product']
        version = pv['version']
        cache_key = f"{product}_{version}"
        if cache_key in cache:
            vulnerabilities_cpe = cache[cache_key]
            if verbose > 1:
                logger.info(f"Using cached data for {product} {version}")
        else:
            if verbose > 1:
                logger.info(f"Scaping data for {product} {version}")
            if ".v" in version:
                version = version.split(".v")[0]
            well_known, cpe = is_well_known_product(product, product_cpe_map, version)
            if well_known:
                pass
            else:
                cpe = construct_cpe(product, version)
            if verbose > 1:
                logger.info(f"Constructed CPE: {cpe}")
            try:
                cpe = query_nvd_cpe(cpe, headers)
                if verbose > 1:
                    logger.info(f"Queried CPE: {cpe}")
            except Exception:
                well_known, cpe = is_well_known_product(product, product_cpe_map, None)
                if well_known:
                    pass
                else:
                    cpe = construct_cpe(product, None)
                try:
                    cpe = query_nvd_cpe(cpe, headers, most_recent=False)
                    versions_available = [elem['cpe']['cpeName'].split(":")[5] for elem in cpe]
                    closest_version = find_closest_version(version, versions_available)
                    for elem in cpe:
                        if elem['cpe']['cpeName'].split(":")[5] == closest_version:
                            cpe = elem['cpe']['cpeName']
                            break
                    if verbose > 1:
                        logger.info(f"Closest CPE: {cpe}")
                except Exception:
                    logger.warning("Error querying CPE without version for product:", product)
                    continue
            vulnerabilities_cpe = search_cve_for_cpe(cpe, headers, logger, verbose)
            cache[cache_key] = vulnerabilities_cpe
        product_info["product"] = pv['product']
        product_info["version"] = pv['version']
        product_info["frequency"] = pv['count']
        product_info["tags"] = pv['tags']
        product_info["cpe"] = vulnerabilities_cpe['cpe']
        product_info["description"] = pv['product'] + " " + pv['version']
        product_info["vulns"] = {}
        for vuln in vulnerabilities_cpe['vulnerabilities']:
            product_info["vulns"][vuln['cve_id']] = vuln
        products_info.append(product_info)
        pv['cpe'] = vulnerabilities_cpe['cpe']
        new_product_versions.append(pv)
        if len(new_product_versions) % CACHE_SAVE_INTERVAL == 0:
            save_pickle(cache, cache_file)

    save_yaml(product_versions, os.path.join(script_parent_dir,'data', 'scrape_samples', target_folder), "top_products_versions.yaml")
    save_json(products_info, os.path.join(script_parent_dir, 'data', 'scrape_samples', target_folder,"products_info.json"))
