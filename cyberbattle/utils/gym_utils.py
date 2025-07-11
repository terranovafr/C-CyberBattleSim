# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    gym_utils.py
    This file contains the utilities related to the gym and gymnasium libraries.
"""

import gym
import gymnasium
import numpy as np

# Map gym spaces to gymnasium spaces for compatibility with libraries
def convert_gym_to_gymnasium_space(space):
    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=space.dtype if space.dtype is not None else np.float32
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gymnasium.spaces.Discrete(n=space.n)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return gymnasium.spaces.MultiDiscrete(nvec=space.nvec)
    elif isinstance(space, gym.spaces.MultiBinary):
        return gymnasium.spaces.MultiBinary(n=space.n)
    elif isinstance(space, gym.spaces.Dict):
        # Recursively convert the dictionary of spaces
        return gymnasium.spaces.Dict({
            key: convert_gym_to_gymnasium_space(subspace) for key, subspace in space.spaces.items()
        })
    elif isinstance(space, gym.spaces.Tuple):
        # Recursively convert the tuple of spaces
        return gymnasium.spaces.Tuple(tuple(convert_gym_to_gymnasium_space(subspace) for subspace in space.spaces))
    else:
        raise TypeError(f"Unsupported space type: {type(space)}")
