# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    feature_extraction.py
    Script to extract embeddings with NLP models from vulnerabilities and services based on their descriptions.
"""

from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import sys
import os
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle.utils.file_utils import load_json, save_json, load_yaml # noqa: E402
script_parent_dir = Path(__file__).parent.parent

# Process entries to extract feature vectors from descriptions (services and vulns) using the specified NLP model
def process_entries(data, model, tokenizer, model_name, logger):
    cache = {}
    id = 0
    for entry in tqdm(data, desc="Processing entries"):
        id += 1
        entry_id = entry.get('id', id)
        entry['id'] = entry_id
        feature_key = 'feature_vector_' + model_name
        if feature_key in entry:
            continue  # Skip processing if feature vector already exists
        description = entry.get('description', '')
        if description in cache:
            entry[feature_key] = cache[description]
        else:
            inputs = tokenizer(description, return_tensors='pt', padding=True, truncation=True, max_length=512)
            if inputs['input_ids'].numel() == 0:  # Check if the tensor is empty
                logger.warning("Empty tensor after tokenization, skipping entry.")
                continue
            with torch.no_grad():
                outputs = model(**inputs)
            feature_vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
            entry[feature_key] = feature_vector
            cache[description] = feature_vector
        vuln_data = entry.get('vulns', '')
        for vuln_id in vuln_data:
            vuln_key = 'feature_vector_' + model_name
            if vuln_key in vuln_data[vuln_id]:
                continue  # Skip processing if feature vector already exists
            clean_vuln_str = vuln_data[vuln_id]['description']
            if clean_vuln_str in cache:
                vuln_data[vuln_id][vuln_key] = cache[clean_vuln_str]
                continue
            inputs = tokenizer(clean_vuln_str, return_tensors='pt', padding=True, truncation=True, max_length=512)
            if inputs['input_ids'].numel() == 0:  # Check if the tensor is empty
                logger.error("Empty tensor after tokenization, skipping entry.")
                continue
            with torch.no_grad():
                outputs = model(**inputs)
            feature_vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
            vuln_data[vuln_id][vuln_key] = feature_vector
            cache[clean_vuln_str] = feature_vector
    return data

# Looping vulnerabilities and services and extracting embeddings apriori with NLP models to avoid this intensive operation during training of the agent
def main(models, mappings_file, auth, folder, logger, verbose):
    hf_token = auth['huggingface']['key']
    model_identifiers = load_yaml(mappings_file)['model_identifiers']
    for model in models:
        if verbose:
            logger.info(f"Processing vulnerabilities and services embeddings using NLP model {model}..")

        extracted_file_path = os.path.join( folder, f"extracted_data_{model}.json")
        data_file_path = os.path.join(folder, "classified_data.json")

        existing_data = load_json(extracted_file_path)
        if existing_data is None:
            existing_data = load_json(data_file_path)
            if existing_data is None:
                logger.error("No source data file found. Exiting.")
                return
        if model not in model_identifiers:
            if verbose:
                logger.warning(f"Model {model} not supported. Please choose from: {', '.join(model_identifiers.keys())}")
            continue

        model_name = model_identifiers[model]
        feature_key = 'feature_vector_' + model

        # Check if the model has already been processed
        if any(feature_key in entry for entry in existing_data):
            if verbose:
                logger.info(f"Model {model} already processed. Skipping...")
            continue

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            model_class = AutoModel.from_pretrained(model_name, token=hf_token)
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer for {model_name}: {e}")
            continue

        if model_name == 'gpt2':
            tokenizer.pad_token = tokenizer.eos_token
        existing_data = process_entries(existing_data, model_class, tokenizer, model, logger)
        if verbose:
            logger.info(f"Feature extraction complete for model {model}..")
        save_json(existing_data, extracted_file_path)
