.. _static-defenders:

Static Defenders
=============================

Static defenders simulate predefined defensive mechanisms that respond to the agent's actions during training or evaluation. These defenders are adapted from the original CyberBattleSim project and are designed to introduce reactive behavior from the environment.
They operate **at every agent step**, triggered probabilistically.
.. note::

   When a static defender is active, the environment **is no longer Markovian**.
   This means the next state is not solely dependent on the current state and action, but also on the defender's internal logic.
   This non-Markovian nature can affect the **convergence properties of standard RL algorithms**, which typically assume the environment is Markovian.

Integrated Defender Types
-----------------------------

We currently support two types of static defenders:

1. **ScanAndReimageCompromisedMachines**
2. **ExternalRandomEvents**

These defenders differ in purpose and behavior, as described below.

**1. ScanAndReimageCompromisedMachines**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This defender simulates a security scan and reimaging process. It follows these steps:

- **Step 1**: Every `scan_frequency` actions, the defender randomly selects up to `scan_capacity` nodes in the environment.
- **Step 2**: For each selected node:
  - If the agent is present on the node:
  - It is detected with a certain `detection_probability`.
  - If the agent has **not** achieved defense evasion on that node, the node is **reimaged** (i.e., reset to a clean state).
- **Step 3**: The reimaging process takes `reimaging_duration` timesteps to complete. During this time, the node is offline.
- **Step 4**: Once the node comes back online, if the agent had established **persistence** before reimaging, it regains control of the node automatically.

This defender simulates a realistic response by blue teams to detected compromises.

**2. ExternalRandomEvents**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This defender introduces unpredictable environmental changes at each timestep:

- **Step 1**: After each agent action, every node in the network has a chance to be affected, based on a predefined `event_probability`.
- **Step 2**: If triggered, one of the following actions is randomly selected for the node:

  - Start a random service
  - Stop a random service
  - Add a firewall rule
  - Remove a firewall rule
- **Step 3**: The selected action is then applied to a random service or firewall rule on the node.

This defender introduces variability which may have positive or negative outcome for the attacker agent.

Defender Integration
------

Static defenders can be enabled during:

- Training
- Sampling
- Hyperparameter optimization
- Evaluation and testing

To enable a static defender, add the following argument when running your agent script:

.. code-block:: bash

   python3 cyberbattle/agent/FUNCTION_agent.py .... --static_defender_agent {reimage, events}

You can modify the defender's behavior by editing its parameters in the corresponding configuration file: ``cyberbattle/agents/config/{train,test}_config.yaml``.
