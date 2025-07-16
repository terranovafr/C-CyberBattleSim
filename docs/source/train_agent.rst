.. _train-agent:

Training the DRL Agent
===========================

Our framework enables training agents using Deep Reinforcement Learning (DRL) algorithms across three types of observation and action spaces:

- **Global**: The traditional setting used in *CyberBattleSim* and most of the existing literature.
- **Local**: Introduced by the study `Leveraging Deep Reinforcement Learning for Cyber-Attack Paths Prediction`_ (Terranova et al.). This view focuses only on a pair of nodes as observation and the selection of a local vulnerability or switching action at each timestep.
- **Continuous**: Proposed in this project and detailed in our accompanying paper, enabling more expressive and flexible agent actions.

The framework integrates all major algorithms from the `Stable Baselines3 <https://github.com/DLR-RM/stable-baselines3>`_ library and is built for flexibility and advanced experimentation.

Training Command
----------------------

To launch training, use the following command:

.. code-block:: bash

    cd cyberbattle/agent

    python3 train_agent.py \
        --name LOGS_FOLDER_NAME \
        --num_runs NUM_RUNS \
        --environment_type {global, local, continuous} \
        --algorithm {TRPO, PPO, A2C, DQN, DDPG, SAC, TD3} \
        [--load_seeds SEEDS_FILE] \
        [--random_seeds] \
        [--load_envs ENVS_FOLDER] \
        [--holdout] \
        [--early_stopping PATIENCE] \
        --goal {control, discovery, disruption} \
        --nlp_extractor NLP_EXTRACTOR \
        [--finetune_model MODEL_PATH]

- ``--name``: Name of the logs folder.
- ``--num_runs``: Number of training runs (for robustness across different seeds).
- ``--environment_type``: Type of observation/action space (global, local, or continuous).
- ``--algorithm``: DRL algorithm from Stable Baselines3.
- ``--load_seeds`` / ``--random_seeds``: Specify a file containing seeds or generate randomly the seeds to use.
- ``--load_envs``: Folder with scenario files (if not set, default ones from root ``config.yaml`` file will be used).
- ``--holdout``: Use training/validation split.
- ``--early_stopping``: Patience before stopping on no validation improvement.
- ``--goal``: Agent threat model between ``control``, ``discovery``, or ``disruption`` (set can be extended).
- ``--nlp_extractor``: NLP model used for encoding vulnerabilities (defines part of the action space) and the content of the feature vectors forming the observation vector.
- ``--finetune_model``: Path to a pre-trained model to fine-tune, without starting from a randomly initialized model.

More detailed settings (e.g., agent architecture, episode config, reward shaping, normalization) can be configured via the following configutation files:

- ``cyberbattle/agent/config/train_config.yaml``
- ``cyberbattle/agent/config/algo_config.yaml``
- ``cyberbattle/agent/config/rewards_config.yaml``

Training Output
---------------------

After training, the log folder contains models, checkpoints, environments, and TensorBoard logs:

.. code-block:: text

    LOGS_FOLDER
    ├── app.log
    ├── checkpoints/
    │   └── 1/
    │       ├── checkpoint_10000_steps.zip
    │       └── ...
    ├── envs/
    │   ├── 1.pkl
    │   └── ...
    ├── seeds.yaml
    ├── split.yaml
    ├── train_config.yaml
    ├── TRPO_1/
    │   └── events.out.tfevents...
    └── validation/
        └── 1/
            ├── checkpoint_123.5_reward.zip
            └── ...

- **checkpoints/**: Saved models during training.
- **validation/**: Best validation checkpoints.
- **envs/**: Pickled environment scenarios.
- **YAML files**: Used configs and seeds for reproducibility.


Monitoring with TensorBoard
~~~~~~~~~~~~


To visualize training and validation logs:

.. code-block:: bash

    tensorboard --logdir LOGS_FOLDER_NAME

Then open the browser at the provided URL. Tracked metrics include:

- Exploited vulnerabilities (local/remote)
- Action selection stats
- Nodes controlled/discovered/disrupted
- Episode reward and length
- Action success rates
- ...

Batch Sampling
----------------------------------

To evaluate performance across all combinations of goals and NLP extractors you can use the following command:

.. code-block:: bash

    python3 sample_agent.py ...

The option and configuration files are the same of ``train_agent.py`` script, without the ``--goal`` and ``--nlp_extractor`` parameters.
This produces separate log folders for each combination:

.. code-block:: text

    LOGS_FOLDER/
    ├── ALGO_1_control_bert/
    ├── ALGO_2_discovery_roberta/
    └── ...

The configuration and command-line options are identical to the main training script.

Hyperparameter Optimization
-------------------------------

Hyperparameters can be optimized using Optuna with the following command:

.. code-block:: bash

    python3 cyberbattle/agent/hyperopt_agent.py ... \
        --num_trials NUM_TRIALS \
        --optimization_type {grid, random, tpe, cmaes, ...}

New parameters:

- ``--num_trials``: Number of optimization trials.
- ``--optimization_type``: Search strategy (e.g., ``tpe``, ``random``).

Hyperparameter search spaces are defined in ``cyberbattle/agent/config/hyperparams_ranges.yaml``.
The default optimization goal is to **maximize average validation reward**, sampling across goals and NLP extractors.

Visualization with Optuna Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To visualize the hyperparameter optimization process you can use the following command:

.. code-block:: bash

    optuna dashboard --storage sqlite:///cyberbattle/agent/logs/LOGS_FOLDER/ALGO_hyperopt.db

Access the dashboard via the provided URL to track trial performance, parameter importances, and convergence.

**Bibliography**

- `Leveraging Deep Reinforcement Learning for Cyber-Attack Paths Prediction <https://doi.org/10.1145/3678890.3678902>`_.
  Franco Terranova, Abdelkader Lahmadi, and Isabelle Chrisment. 2024. *Leveraging Deep Reinforcement Learning for Cyber-Attack Paths Prediction: Formulation, Generalization, and Evaluation*.
  In *Proceedings of the 27th International Symposium on Research in Attacks, Intrusions and Defenses (RAID '24)*. Association for Computing Machinery.