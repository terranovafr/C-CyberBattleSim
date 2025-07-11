# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    clean_logs_folder_checkpoints.py
    Script to remove all checkpoints except the largest one in `checkpoints/` and `validation/` folders, to avoid large folder sizes as the last and best checkpoint are those of interest typically.
"""

import os
import re
import argparse
script_dir = os.path.dirname(__file__)

def get_largest_checkpoint(folder, pattern, key_func):
    checkpoints = [f for f in os.listdir(folder) if re.match(pattern, f)]
    if not checkpoints:
        return None
    return max(checkpoints, key=key_func)

def clean_checkpoints(base_dir):
    for root, dirs, _ in os.walk(base_dir):
        for subdir in dirs:
            checkpoints_dir = os.path.join(root, subdir, "checkpoints")
            if os.path.exists(checkpoints_dir):
                for checkpoint_subfolder in os.listdir(checkpoints_dir):
                    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_subfolder)
                    if os.path.isdir(checkpoint_path):
                        print(f"Processing folder: {checkpoint_path}")
                        largest_checkpoint = get_largest_checkpoint(
                            checkpoint_path,
                            r"^checkpoint_(\d+)_steps\.zip$",
                            lambda x: int(re.search(r"(\d+)_steps", x).group(1))
                        )
                        if largest_checkpoint:
                            for file in os.listdir(checkpoint_path):
                                if file != largest_checkpoint:
                                    os.remove(os.path.join(checkpoint_path, file))
                                    print(f"Removed: {os.path.join(checkpoint_path, file)}")

            validation_dir = os.path.join(root, subdir, "validation")
            if os.path.exists(validation_dir):
                for validation_subfolder in os.listdir(validation_dir):
                    validation_path = os.path.join(validation_dir, validation_subfolder)
                    if os.path.isdir(validation_path):
                        print(f"Processing folder: {validation_path}")
                        largest_validation_checkpoint = get_largest_checkpoint(
                            validation_path,
                            r"^checkpoint_([-.\d]+)_reward\.zip$",
                            lambda x: float(re.search(r"([-.\d]+)_reward", x).group(1))
                        )
                        if largest_validation_checkpoint:
                            for file in os.listdir(validation_path):
                                if file != largest_validation_checkpoint:
                                    os.remove(os.path.join(validation_path, file))
                                    print(f"Removed: {os.path.join(validation_path, file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean up checkpoints by keeping only the largest in `checkpoints/` and `validation/` folders.")
    parser.add_argument('-f', '--folder', type=str, required=True, help='Path to the logs directory.')
    args = parser.parse_args()
    clean_checkpoints(os.path.join(script_dir, "logs", args.folder))
