.. _reward-function:

Threat Models and Reward Function
=================================

The agent's reward function—and consequently the actions and outcomes it learns to prioritize—depends on the threat model or goal under which it is trained. Each threat model defines specific objectives and termination criteria for the episode. The table below summarizes the different goals supported by the framework:

.. list-table:: **Threat Models / Agent Goals**
   :widths: 25 25 25 25
   :header-rows: 1

   * - **Goal**
     - **Description**
     - **Prioritized Outcomes**
     - **Episode End Condition**
   * - **Control**
     - Gain control over as many nodes as possible
       in the network.
     - ``CredentialAccess``, ``LateralMove``,
       ``PrivilegeEscalation``
     - Terminates when all controllable nodes
       reachable from the starter node are
       owned with ROOT privilege.
   * - **Disruption**
     - Disable the largest possible portion
       of the network.
     - ``DenialOfService``
     - Terminates when all stoppable nodes
       reachable from the starter node are
       disrupted.
   * - **Discovery**
     - Achieve full visibility over the topology,
       collect all data, and exfiltrate it.
     - ``Reconnaissance``, ``Discovery``,
       ``Collection``, ``Exfiltration``
     - Terminates when all nodes reachable from
       the starter node are discovered, made
       visible, and their data collected and
       exfiltrated.

Reward Function Formulation
---------------------------

The agent’s learning is guided by a reward function that combines a weighted sum of bonuses and penalties:

.. math::

    R(o, a) = \sum_j w^B_j \cdot B_j(o, a) - \sum_k w^P_k \cdot P_k(o, a)

Where:

- :math:`o`: Agent’s current observation
- :math:`a`: Selected action
- :math:`B_j(o, a)`: Bonus terms rewarding desirable behavior (e.g., owning nodes, collecting data)
- :math:`P_k(o, a)`: Penalty terms for undesirable outcomes (e.g., hitting firewalls, unnecessary actions)
- :math:`w^B_j, w^P_k`: Weights determining the importance of each term

This formulation allows the agent to learn trade-offs appropriate for its goal.

Bonus and Penalty Terms
-----------------------

Bonus terms include both fixed rewards and scaled coefficients tied to metrics such as the number of owned, disrupted, or discovered nodes. Penalty terms account for various costs and limitations, such as:

- Exploitation cost based on CVSS Exploitability Score
- Distance in embedding space (to improve latent space structure)
- Actions blocked by firewalls
- Attempting invalid actions (e.g., scanning already scanned nodes)

The reward structure thus supports goal-aligned behavior through nuanced and flexible design.
The following table provides an overview of the types of coefficients, constant rewards, and penalty terms currently implemented:

.. list-table:: **Reward Function Components**
   :header-rows: 1

   * - **Weighted Coefficients**
     - **Constant Rewards**
     - **Penalty Terms**
   * - Node Ownership Value Coefficient
     - Data Collected Reward
     - Exploitation Cost
   * - Denial of Service Coefficient
     - Privilege Escalation Reward
     - No Enough Privileges
   * - Node Discovery Coefficient
     - Visibility Acquired Reward
     - Failed Success Rate
   * -
     - Data Exfiltration Reward
     - No Data to Collect
   * -
     - Defense Evasion Reward
     - No Data to Discover
   * -
     - Winning Episode Reward
     - No Data to Exfiltrate
   * -
     -
     - Already Persistent
   * -
     -
     - Machine Already Stopped
   * -
     -
     - Node Already Owned
   * -
     -
     - Node Already Visible
   * -
     -
     - Already Defense Evasion
   * -
     -
     - Scanning Unopen Port
   * -
     -
     - Privilege Escalation in Node Not Owned
   * -
     -
     - Privilege Escalation to Level Already Had
   * -
     -
     - Blocked by Local Firewall
   * -
     -
     - Blocked by Remote Firewall
   * -
     -
     - Distance Penalty
   * -
     -
     - Losing Episode Penalty

