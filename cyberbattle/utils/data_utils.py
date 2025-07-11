# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    data_utils.py
    This file contains the utilities used to handle different data structures
"""

import numpy as np

def merge_dicts(dict1, dict2):
    # Recursively merge two dictionaries
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                merge_dicts(dict1[key], dict2[key])
            else:
                # If the key exists but is not a dict, we replace/merge based on the new value
                dict1[key] = dict2[key]
        else:
            dict1[key] = dict2[key]

# Check whether a list is a subset of another list (have all elements)
def list_subset_another(list_of_lists, new_list):
    existing_sets = [set(lst) for lst in list_of_lists]
    new_set = set(new_list)
    return new_set in existing_sets

# Flatten observation into Discrete or Box supported types as requested by StableBaselines3
def flatten_dict_with_arrays(input_dict):
    flattened_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, flatten it recursively
            flattened_subdict = flatten_dict_with_arrays(value)
            flattened_dict.update(
                {f"{key}_{sub_key}": sub_value for sub_key, sub_value in flattened_subdict.items()})
        elif isinstance(value, list):
            # If the value is a list, flatten it to individual components
            for i, sub_value in enumerate(value):
                if isinstance(sub_value, tuple):
                    for j, inner_sub_value in enumerate(sub_value):
                        flattened_dict[f"{key}_{i}_{j}"] = inner_sub_value
                else:
                    flattened_dict[f"{key}_{i}"] = sub_value
        else:
            # If the value is not a dictionary or list, include it as is
            flattened_dict[key] = value
    return flattened_dict

# Flatten a list of lists into a single list
def flatten(input_list):
    flat_list = []
    for item in input_list:
        if isinstance(item, (list, tuple, np.ndarray)):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list
