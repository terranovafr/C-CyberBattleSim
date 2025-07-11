# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    cyberbattle_env_switch_gae.py
     Wrapper that randomly switch the environment with logic similar to normal switcher but adapted for the GAE
"""

from cyberbattle._env.cyberbattle_env_switch import RandomSwitchEnv
import time

# Wrapper that randomly switch the environment given a certain interval to ensure that the agent is robust to environment changes
class RandomSwitchGAEEnv(RandomSwitchEnv):
    # Decide for a current environment and wraps it accordingly, then switch to a new one every switch_interval episodes
    def __init__(self, envs_ids, switch_interval=50, envs_folder=None, envs_list=None, verbose=1):
        super().__init__(envs_ids=envs_ids, switch_interval=switch_interval, envs_folder=envs_folder, envs_list=envs_list, verbose=verbose)
        self.envs_ids = envs_ids
        self.envs_folder = envs_folder
        self.envs_list = envs_list  # alternative to loading it every time
        self.verbose = verbose
        self._switch_environment()
        self.iteration_count = 0 # here we have iterations of graph configurations and not episodes
        self.switch_interval = switch_interval # here works for iterations and not episodes
        self.current_observation = None
        self.start_time = time.time()
        self.calculate_action_time = 0
        # represent a gym environment with the same action space and observation space of the ones in the list wrapped
        self.done = False

    # Override the check switch method to use iteration count instead of episode count
    def _check_switch(self):
        if (self.iteration_count + 1) % (self.switch_interval + 1) == 0: # check done on iterations
            self._switch_environment()

    # Override the step method to create the step function needed for GAE training and validation
    def step_graph(self, source_node, target_node, vulnerability_ID, outcome_desired):
        if self.current_env.blocked_graph: # no other actions to sample as valid
            self.done = True
        else:
            self.done = self.current_env.step_graph(source_node, target_node, vulnerability_ID, outcome_desired)
            self.truncated = self.current_env.truncated
            self.iteration_count += 1
            self._check_switch()
        return self.done

    # Reset functions and others are the same
