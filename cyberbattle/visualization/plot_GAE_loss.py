# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    plot_GAE_loss.py
    Script to plot the GAE loss averaging across different nlp extractors
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorboard.backend.event_processing.event_accumulator as tb_event_accumulator
from datetime import datetime
import argparse

script_dir = os.path.dirname(__file__)

def load_tensorboard_data(logdir, start_step=1300):
    ea = tb_event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    scalars = {}
    for tag in ea.Tags()['scalars']:
        scalars[tag] = [
            event.value for event in ea.Scalars(tag) if event.step >= start_step
        ]

    return scalars


def plot_results(mean_train, stsd_train, mean_val, stsd_val, start_step, output_file):
    plt.figure(figsize=(12, 9))
    x_train = start_step + np.arange(len(mean_train)) * 32  # x-axis for train data
    x_val = start_step + np.arange(len(mean_val)) * 57 * 19  # x-axis for val data, scaled


    plt.plot(x_train, mean_train, color='blue', label='Train Loss')
    plt.fill_between(x_train, np.array(mean_train) - np.array(stsd_train),
                     np.array(mean_train) + np.array(stsd_train), color='blue', alpha=0.3)

    plt.plot(x_val, mean_val, color='red', label='Validation Loss')
    plt.fill_between(x_val, np.array(mean_val) - np.array(stsd_val),
                     np.array(mean_val) + np.array(stsd_val), color='red', alpha=0.3)

    plt.xlabel('Iteration', fontsize=38)
    plt.ylabel('Mean Loss & Std', fontsize=40)
    plt.title('Train/Validation GAE Loss', fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=34)
    plt.legend(fontsize=34)
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()


def main(folder, nlp_extractors, start_step, output_file):
    all_train_values = []
    all_val_values = []

    for nlp_extractor in nlp_extractors:
        logdir = os.path.join(script_dir, "../gae/logs", folder, nlp_extractor)
        print(f"Processing {logdir}")
        if not os.path.exists(logdir):
            print(f"Log directory {logdir} does not exist. Skipping this NLP extractor.")
            continue
        event_files = [f for f in os.listdir(logdir) if f.startswith('events.out.tfevents')]

        if event_files:
            event_file = os.path.join(logdir, event_files[0])
            data = load_tensorboard_data(event_file, start_step=start_step)
            if data:
                train_values = []
                val_values = []

                for tag, values in data.items():
                    if 'train/total_loss' in tag:
                        train_values = values
                    elif 'val/total_loss' in tag:
                        val_values = values

                # Ensure all lists are of the same length
                max_len = max(len(train_values), len(val_values))
                while len(all_train_values) < max_len:
                    all_train_values.append([])
                    all_val_values.append([])

                for i, value in enumerate(train_values):
                    all_train_values[i].append(value)

                for i, value in enumerate(val_values):
                    all_val_values[i].append(value)

    # Compute mean and standard deviation for aggregated data
    mean_train = []
    stsd_train = []
    mean_val = []
    stsd_val = []

    for train_timestep_values, val_timestep_values in zip(all_train_values, all_val_values):
        mean_train.append(np.mean(train_timestep_values))
        stsd_train.append(np.std(train_timestep_values))
        mean_val.append(np.mean(val_timestep_values))
        stsd_val.append(np.std(val_timestep_values))

    plot_results(mean_train, stsd_train, mean_val, stsd_val, start_step, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot LLMs')
    parser.add_argument('-f', '--folder', type=str, required=True, help='List of trial folders')
    parser.add_argument('-nlp', '--nlp_extractors', type=str,
                        choices=["bert", "distilbert", "roberta", "gpt2", "CySecBERT", "SecureBERT", "SecBERT",
                                 "SecRoBERTa"], default=["bert", "distilbert", "roberta", "gpt2", "CySecBERT", "SecureBERT", "SecBERT",
                                 "SecRoBERTa"], nargs='+',
                        help='NLP extractor to be used for extracting vulnerability embeddings')
    parser.add_argument('-t', '--start_step', type=int, default=1300, help='Starting step for plotting')
    args = parser.parse_args()

    output_folder = os.path.join(script_dir, "logs", f"GAE_plot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, 'GAE_train_val.pdf')
    main(args.folder, args.nlp_extractors, args.start_step, output_file)
