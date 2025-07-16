.. _setup:

Installation and Setup
==========================

Follow the steps below to set up the project environment and download the required resources to launch the project pipeline.

Create the Python Environment
-------------------------

You can set up the Python environment using either Conda or pip.

**Option 1: Using Conda**

Run the following command to create a new environment with the required dependencies:

.. code-block:: bash

   conda env create -f environment.yml

**Option 2: Using pip**

Alternatively, you can install the dependencies with pip:

.. code-block:: bash

   pip install -r requirements.txt

Initialize Default Resources
-------------------------------

Once the environment is ready, you can either download the default files and models or reuse the exact resources used in the paper.

**Option 1: Use Default Setup**

Run the initialization script to download and set up the default resources:

.. code-block:: bash

   chmod +x init.sh
   ./init.sh

This will provide:

- The default environment database (the 50 most vulnerable services and  up to five of their most vulnerable versions at the timestep of 18 November 2024).
- A default set of scenarios (20 graph scenarios of 100 nodes each, built with domain randomization).
- A pretrained classifier model for mapping vulnerability descriptions to MITRE ATT&CK Tactics (SecureBERT fine-tuned for multi-label classification).
- A pretrained GAE model for creating graph and node embeddings (trained on 300 scenario graphs, leveraging an hyper-optimized architecture).

**Option 2: Reuse Resources from the Paper**

To replicate the exact experiments from the paper, follow the instructions in the `REPRODUCIBILITY.md` file. This approach ensures consistency with the results and configurations presented in the publication.
