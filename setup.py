#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""setup CyberBattle simulator module"""

import os
import setuptools
from typing import List

pwd = os.path.dirname(__file__)

def get_install_requires(requirements_txt) -> List[str]:
    """get the list of required packages"""
    install_requires = []
    with open(os.path.join(pwd, requirements_txt)) as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                install_requires.append(line)
    return install_requires


# main setup kw args
setup_kwargs = {
    "name": "ccyberbattlesim",
    "version": "1.0.0",
    "description": "Continuous CyberBattleSim, an extension of Microsoft CyberBattleSim for training RL agents on Cyber-Attack Paths Prediction",
    "author": "Franco Terranova",
    "packages": ['cyberbattle'],
    "author_email": "franco.terranova@inria.fr",
    "install_requires": get_install_requires("requirements.txt"),
    "classifiers": [
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    "zip_safe": True
}

if __name__ == "__main__":
    setuptools.setup(**setup_kwargs)
