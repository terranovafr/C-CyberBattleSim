# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    classify_vulns.py
    Script to classify vulnerabilities (apriori on the JSON file) based on their description using a pre-trained model.
"""

import sys
import os
from tqdm import tqdm
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.utils.classifier_utils import VulnerabilityClassifier # noqa: E402
from cyberbattle.utils.file_utils import load_json, save_json # noqa: E402
script_parent_dir = Path(__file__).parent.parent

# Process entries to classify vulnerabilities based on their description using the provided model
def process_entries(data, model, logger, verbose):
    # cache allows to reuse previously classified vulnerabilities repeated in another hosts / products
    cache = {} # we maintain a dynamic cache without saving it since the model can change between executions
    id = 0
    for entry in tqdm(data, desc="Processing entries"):
        id += 1
        entry_id = entry.get('id', id)
        entry['id'] = entry_id
        vuln_data = entry.get('vulns', '')
        for vuln_id in vuln_data:
            clean_vuln_str = vuln_data[vuln_id]['description']
            if vuln_id in cache:
                classes_selected = cache[vuln_id]
                vuln_data[vuln_id]['classes'] = classes_selected
            else:
                classes = model.predict(clean_vuln_str)
                classes_selected = []
                for class_prediction in classes:
                    vuln_outcome = class_prediction
                    probability = classes[class_prediction]
                    if vuln_outcome is None:
                        continue
                    classes_selected.append({"class":vuln_outcome, "probability":probability})
                cache[vuln_id] = classes_selected
            vuln_data[vuln_id]['classes'] = classes_selected
            if verbose > 1:
                logger.info(f"Vulnerability {vuln_id} description {vuln_data[vuln_id]['description']}")
                logger.info(f"Predicted classes: {classes_selected}")
    return data

def main(auth, folder, logger, verbose):
    hf_token = auth['huggingface']['key']
    vulnerability_classifier = VulnerabilityClassifier(hf_token, logger, verbose)
    data_file_path = os.path.join(folder, "products_info.json")
    classified_file_path = os.path.join(folder, "classified_data.json")

    data = load_json(data_file_path)
    if data is None:
        logger.warning("No source data file found. Exiting.")
        sys.exit(1)

    existing_data = process_entries(data, vulnerability_classifier, logger, verbose)
    save_json(existing_data, classified_file_path)
