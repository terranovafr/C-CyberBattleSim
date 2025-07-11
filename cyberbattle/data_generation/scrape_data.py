# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    scrape_data.py
    Script to scrape data from NVD and Shodan for creating the environment database
"""

# This product uses the NVD API but is not endorsed or certified by the NVD.
import argparse
from pathlib import Path
import sys
import os
from datetime import datetime
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
script_dir = Path(__file__).parent
from cyberbattle.data_generation.utils import scrape_shodan, scrape_nvd, feature_extraction, classify_vulns # noqa: E402
from cyberbattle.utils.log_utils import setup_logging # noqa: E402
from cyberbattle.utils.file_utils import load_yaml # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape data for creating topologies")
    parser.add_argument("-np", "--num_products", type=int, default=50, required=False,
                        help="Number of top products to retrieve.")
    parser.add_argument("-nv", "--num_versions", type=int, default=5, required=False,
                        help="Number of top versions to retrieve per product.")
    parser.add_argument("-nlp", "--nlp_extractors", type=str, default=['bert', 'roberta', 'distilbert', 'gpt2', 'SecBERT', 'SecRoBERTa', 'SecureBERT',
        'CySecBERT'], help="The models to use for feature extraction.", nargs='+')
    parser.add_argument("-q", "--query", type=str,
                        default="has_vuln:True", help="The query used for Shodan scraping.")
    parser.add_argument('--cache_file', type=str,
                        default=os.path.join('cache', 'cve_cache.pkl'), help='Cache file name')
    parser.add_argument("--mappings_file", type=str, default=os.path.join('config', 'mappings.yaml'),
                        required=False,
                        help="Configuration file name.")
    parser.add_argument("--auth_file", type=str, default=os.path.join('config', 'auth.yaml'),
                        required=False,
                        help="Authentication file name.")
    parser.add_argument('--no_save_log_file', action='store_false', dest='save_log_file',
                        default=True, help='Disable logging to file; log only to terminal')
    parser.add_argument('-v', '--verbose', default=1, type=int,
                        help='Verbose level: 0 - no output, 1 - scraping information, 2 - all information',
                        choices=[0, 1, 2])
    args = parser.parse_args()

    args.auth_file = os.path.join(script_dir, args.auth_file)
    args.mappings_file = os.path.join(script_dir, args.mappings_file)
    args.cache_file = os.path.join(script_dir, args.cache_file)

    auth = load_yaml(args.auth_file)

    folder = os.path.join(script_dir, '..', 'data', 'scrape_samples', datetime.now().strftime('%Y%m%d%H%M%S'))
    logger = setup_logging(folder, log_to_file=args.save_log_file)
    if args.verbose:
        logger.info("Step 1: Using Shodan to scrape statistics on top products and versions....")
    scrape_shodan.main(args.mappings_file, auth, args.num_products, args.num_versions, folder, logger, args.verbose)
    if args.verbose:
        logger.info("Step 2: Using NVD to scrape vulnerability data....")
    scrape_nvd.main(auth, args.mappings_file, args.cache_file, folder, logger, args.verbose)

    if args.verbose:
        logger.info("Step 3: Classifying vulnerabilities....")
    classify_vulns.main(auth, folder, logger, args.verbose)
    if args.verbose:
        logger.info("Step 4: Extracting embeddings based on the vulnerability descriptions....")
    feature_extraction.main(args.nlp_extractors, args.mappings_file, auth, folder, logger, args.verbose)
