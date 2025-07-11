# Reproducibility Instructions

To reproduce the results of the paper and the experiments, follow the steps below.

Use the default scenarios and data provided in the associated [Zenodo data repository](https://zenodo.org/records/14604652/latest) to train the Graph Autoencoder (GAE) and DRL agents. If not already present, move the necessary files to the appropriate directories:

- `environment_database/*` → `cyberbattle/data/scrape_samples/default_data/`
- `scenarios/*` → `cyberbattle/env_samples/`
- `classifiers_training_testing/SecureBERT_chosen_classifier.bin` → `cyberbattle/models/classifier/pytorch_model.bin`
- *(Optional)* `gae_training/*` → `cyberbattle/gae/logs/default/`

Then, update the root `config.yaml` file with the correct paths:

- `nvd_data_path`: path to the environment database
- `vulnerability_classifier_path`: path to the classifier model
- `gae_path`: path to the GAE logs folder
- `default_environments_path`: path to the default scenarios

By default, the seeds used in the experiments are loaded from `seeds.yaml` (reference seeds used in the study), unless overridden via `--load_seeds`.

---

## GAE Hyperparameter Optimization (Optional)

Navigate to the GAE directory:

```bash
cd cyberbattle/gae
```
Run hyperparameter optimization with:
```bash
python3 hyperopt_gae.py --load_envs domain_randomization_500_graphs_10-100_nodes --name GAE --holdout
```

### GAE Training (Optional)
Once the best hyperparameters are found, update config/train_config.yaml accordingly. Train the GAE using all desired LMs as NLP extractors (8 are used by default):
```bash
python3 train_gae.py --load_envs domain_randomization_500_graphs_10-100_nodes --name GAE --holdout
```
The trained models will be saved in a new logs folder. Update the gae_path in the root config.yaml to point to this folder. The trained GAE will be used as the default from this point onward.

### DRL Agent Hyperparameter Optimization (Optional)
Ensure you are using the same classifier, GAE, and hyperparameters as used in the study. Move to the agents directory:
```bash
cd cyberbattle/agents
```
For each algorithm _ALGO_, run:
```bash
python3 hyperopt_agent.py --load_envs domain_randomization_500_graphs_10-100_nodes --holdout --name ALGO --algorithm ALGO --environment_type {continuous, local, global}
```
After optimization, update cyberbattle/config/algo_config.yaml with the best parameters.

### Scalability study
To run the scalability study across different environment types (continuous, global, local), manually set the best hyperparameters in sample_scalability_architectures.py. Then run:
```bash
python3 sample_scalability_spaces.py --environment_type {continuous, local, global}
```
The script uses the scenarios in the scalability/ folder. To visualize results:
```bash
cv ../visualization
python3 plot_scalability_spaces.py
```

### Generalization study
Use sample_agent.py to train one agent per (goal, LM) pair (24 combinations by default):
```bash
python3 sample_agent.py --load_envs domain_randomization_500_graphs_10-100_nodes --holdout --name ALGO --algorithm ALGO
```
The _config/train\_config.yaml_ file will be used to set the training parameters.
Once trained a model for each (DRL algorithm, goal, LM) tuple, the models can be used to generate statistics about trajectories performance and evaluate the agents.
To evaluate agent performance on test scenarios:
```bash
python3 test_agent.py --logs_folder LOGS_FOLDER --load_custom_envs domain_randomization_500_graphs_10-100_nodes --last_checkpoint --option agent_performances --val_checkpoints --no_random
```
The _config/test\_config.yaml_ file can be used to set the parameters of the test.
Results will be present in a _test_ folder created automatically inside each run folder.
Results can be visualized using the _plot_scores.py_ file in the visualization folder.
```bash
cd ../visualization
python3 plot_scores.py --f LOGS_FOLDERS -o rank -d RL # rank plot for RL algorithms
```
The following command can be used to test the agent when the defender is in the game.
To evaluate against a static defender (e.g., random event-based agent):
```bash
python3 test_agent.py --logs_folder LOGS_FOLDER --load_custom_envs domain_randomization_500_graphs_10-100_nodes --last_checkpoint --option agent_performances --val_checkpoints --no_random --static_defender_agent events
```
The parameters of the defender agent can be set in the _config/test\_config.yaml_ file.

### Deployment study

These steps reproduce the deployment study using the emulated network scenario. _(Note: Real network data is excluded due to privacy restrictions.)_

1. Re-train specialized agents for the deployment scenario and test them:
```bash
python3 sample_agent.py --load_envs emulated_network_100_nodes --name emulated --algorithm trpo
python3 test_agent.py --logs_folder LOG_FOLDER --algorithm trpo --load_custom_envs emulated_network_100_nodes --last_checkpoint --option agent_performances
```

2. Train on synthetic graphs and test on the deployment scenarios:

```bash
python3 sample_agent.py --load_envs syntethic_deployment_20_graphs_100_nodes --name syntethic --algorithm trpo --holdout
python3 test_agent.py --logs_folder LOG_FOLDER --algorithm trpo --load_custom_envs emulated_network_100_nodes --last_checkpoint --option agent_performances
```

3. Run heuristic agents on the deployment scenarios:
```bash
python3 test_heuristic.py --heuristic highest_score_vuln --load_custom_envs emulated_network_100_nodes --option heuristic_performances --no_random
```

4. Compute statistics/plots using a custom script for visualization.
