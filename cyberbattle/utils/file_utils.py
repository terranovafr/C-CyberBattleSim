# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    file_utils.py
    This file contains the utilities used to save and load files according to different formats as well as folder utilities.
"""

import yaml
import os
import json
from collections import defaultdict
from cyberbattle.utils.data_utils import merge_dicts
import csv
from datetime import datetime
import pickle
import numpy as np
import shutil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Delete a folder and its content
def remove_folder_and_files(folder):
    if os.path.exists(os.path.join(folder)):
        shutil.rmtree(folder)

# Find most recent subfolder in a directory based on name including date and time
def find_most_recent_folder(base_path, avoid_prefix="cache"):
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and not f.startswith(avoid_prefix)]
    folders.sort(key=lambda x: datetime.strptime(x, '%Y%m%d%H%M%S'), reverse=True)
    if folders:
        return folders[0]
    else:
        return None

# Extract a specific metric from a tensorboard log (x and y axis)
def extract_metric_data(log_dir, metric_name): # from tensorboard log
    event_acc = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    event_acc.Reload()
    if metric_name in event_acc.Tags()['scalars']:
        metric_events = event_acc.Scalars(metric_name)
        times = [event.step for event in metric_events]
        values = [event.value for event in metric_events]
        return np.array(times), np.array(values)
    else:
        raise ValueError(f"Metric {metric_name} not found in TensorBoard logs.")

# Load json file
def load_json(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Save data as json file
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Load pickle file
def load_pickle(pickle_file):
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            return data
    return {}

# Save data as pickle file
def save_pickle(data, pickle_file):
    if not os.path.exists(os.path.dirname(pickle_file)):
        os.makedirs(os.path.dirname(pickle_file))
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)

# Load configuration from a yaml file
def load_yaml(file_path):
    with open(file_path, 'r') as config_file:
        return yaml.safe_load(config_file)

# Save configuration as a yaml file
def save_yaml(data, folder_path, file_name="train_config.yaml"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    data_path = os.path.join(folder_path, file_name)
    with open(data_path, 'w') as config_file:
        yaml.safe_dump(data, config_file)

# Save data as csv file
def save_csv(csv_content, logs_folder, filename):
    with open(os.path.join(logs_folder, filename), 'w') as f:
        for key in csv_content[0].keys():
            f.write(f"{key},")
        f.write("\n")
        for csv_row in csv_content:
            for value in csv_row.values():
                f.write(f"{value},")
            f.write("\n")

# Read csv file
def read_csv(file_path):
    with open(file_path, 'r') as file:
        return [line for line in csv.reader(file)]

# Read split file to determine training and validation sets (used during training)
def read_split_file(holdout, nets_folder, logs_folder, num_environments):
    if holdout:
        with open(os.path.join(nets_folder, "split.yaml"), 'r') as file:
            yaml_info = yaml.safe_load(file)
            train_ids = []
            for elem in yaml_info['training_set']:
                train_ids.append(elem['id'])
            val_ids = []
            for elem in yaml_info['validation_set']:
                val_ids.append(elem['id'])
            # save in logs
            with open(os.path.join(logs_folder, "split.yaml"), 'w') as f:
                yaml.dump(yaml_info, f)
    else:
        train_ids = [i+1 for i in range(num_environments)]
        val_ids = None
    return train_ids, val_ids


# Load all the services related to a tag and merge them based on 'id'
# Application: merging dicts each having a feature vector of an NLP extractor into a single dict
def load_and_merge_json_entries_by_tag(tag, folder, path_filename_contains, tag_column='tags', id_column='id'):
    data_by_id = defaultdict(dict)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"The folder {folder} does not exist. Modify the environment database folder.")
    for filename in os.listdir(folder):
        if filename.endswith(".json") and filename.startswith(path_filename_contains): # json files with extracted embeddings
            file_path = os.path.join(folder, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                for item in data:
                    if tag in item[tag_column]:
                        id_ = item[id_column]
                        # Merge data based on 'id', in particular merging feature vectors stored in different files
                        if id_ in data_by_id:
                            merge_dicts(data_by_id[id_], item)
                        else:
                            data_by_id[id_] = item
    merged_data = list(data_by_id.values())
    return merged_data
