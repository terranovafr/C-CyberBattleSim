# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    log_utils.py
    This file contains the utilities related to logging.
"""

import logging
import os

# Setups logging for the different modules, using both terminal output and eventually file output
def setup_logging(logs_folder, log_to_file=True, log_filename='app.log', log_level=logging.INFO):
    os.makedirs(logs_folder, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers if any, to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a console handler for logging to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create a file handler for logging to a file
    if log_to_file:
        file_handler = logging.FileHandler(os.path.join(logs_folder, log_filename), mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
