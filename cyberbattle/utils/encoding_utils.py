# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    encoding_utils.py
    This file contains the utilities used to handle different encodings for the node features and outcomes.
"""

import torch
from cyberbattle.simulation import model

# Encoding choice for privilege level into an array for the node feature vector
def convert_privilege_to_indices(privilege_levels):
    privilege_levels_encoded = []
    for privilege_level in privilege_levels:
        privilege_level = int(privilege_level)
        if privilege_level == 0:
            privilege_levels_encoded.append(0)
        elif privilege_level == 1:
            privilege_levels_encoded.append(1)
        elif privilege_level == 3:
            privilege_levels_encoded.append(2)
    return torch.tensor(privilege_levels_encoded, dtype=torch.long)

# Encoding choice for status into an array for the node feature vector
def convert_status_to_indices(status_values):
    status_encoded = []
    for status in status_values:
        status = int(status)
        if status == 0:
            status_encoded.append(0)
        elif status == 1:
            status_encoded.append(1)
        elif status == 3:
            status_encoded.append(2)
    return torch.tensor(status_encoded, dtype=torch.long)

# Map outcome class into a string
def map_outcome_to_string(outcome):
    if isinstance(outcome, model.Reconnaissance):
        return "Reconnaissance"
    elif isinstance(outcome, model.DenialOfService):
        return "DenialOfService"
    elif isinstance(outcome, model.PrivilegeEscalation):
        return "PrivilegeEscalation"
    elif isinstance(outcome, model.Collection):
        return "Collection"
    elif isinstance(outcome, model.Discovery):
        return "Discovery"
    elif isinstance(outcome, model.CredentialAccess) or isinstance(outcome, model.LateralMove):
        return "LateralMove-Credential"
    elif isinstance(outcome, model.Persistence):
        return "Persistence"
    elif isinstance(outcome, model.Exfiltration):
        return "Exfiltration"
    elif isinstance(outcome, model.DefenseEvasion):
        return "DefenseEvasion"
    elif isinstance(outcome, model.FirewallBlock):
        return "FirewallBlock"
    elif isinstance(outcome, model.RepeatedResult):
        return "RepeatedResult"
    elif isinstance(outcome, model.NonRunningMachine):
        return "NonRunningMachine"
    elif isinstance(outcome, model.NoNeededAction):
        return "NoNeededAction"
    elif isinstance(outcome, model.NonListeningPort):
        return "NonListeningPort"
    elif isinstance(outcome, model.UnsuccessfulAction):
        return "UnsuccessfulAction"
    elif isinstance(outcome, model.OutcomeNonPresent):
        return "OutcomeNonPresent"
    elif isinstance(outcome, model.NoEnoughPrivilege):
        return "NoEnoughPrivilege"
    elif isinstance(outcome, model.NoVulnerability):
        return "NoVulnerability"
    elif isinstance(outcome, model.RemoteOutcomeInLocalNode):
        return "RemoteOutcomeInLocalNode"
    elif isinstance(outcome, model.Movement):
        return "MovementAnchors"
    elif isinstance(outcome, model.InvalidMovement):
        return "InvalidMovement"
    else:
        return None

# Map outcome string into a class
def map_string_to_outcome(outcome_string):
    if outcome_string == "Reconnaissance":
        return model.Reconnaissance(nodes=[]) # just sample class object
    elif outcome_string == "DenialOfService":
        return model.DenialOfService()
    elif outcome_string == "PrivilegeEscalation":
        return model.PrivilegeEscalation()
    elif outcome_string == "Collection":
        return model.Collection()
    elif outcome_string == "Discovery":
        return model.Discovery()
    elif outcome_string == "LateralMove-Credential":
        return model.LateralMove()
    elif outcome_string == "Persistence":
        return model.Persistence()
    elif outcome_string == "Exfiltration":
        return model.Exfiltration()
    elif outcome_string == "DefenseEvasion":
        return model.DefenseEvasion()
    elif outcome_string == "FirewallBlock":
        return model.FirewallBlock(incoming_or_outgoing=0)
    elif outcome_string == "RepeatedResult":
        return model.RepeatedResult()
    elif outcome_string == "NonRunningMachine":
        return model.NonRunningMachine()
    elif outcome_string == "NoNeededAction":
        return model.NoNeededAction()
    elif outcome_string == "NonListeningPort":
        return model.NonListeningPort()
    elif outcome_string == "UnsuccessfulAction":
        return model.UnsuccessfulAction()
    elif outcome_string == "OutcomeNonPresent":
        return model.OutcomeNonPresent()
    elif outcome_string == "NoEnoughPrivilege":
        return model.NoEnoughPrivilege()
    elif outcome_string == "NoVulnerability":
        return model.NoVulnerability()
    elif outcome_string == "RemoteOutcomeInLocalNode":
        return model.RemoteOutcomeInLocalNode()
    else:
        return None

def map_label_to_class(label, list=None):
    if label is None:
        return None
    label_mapping = {
        'local': model.VulnerabilityType.LOCAL,
        'remote': model.VulnerabilityType.REMOTE,
        'reconnaissance': model.Reconnaissance(list),
        'discovery': model.Discovery(),
        'persistence': model.Persistence(),
        'credential access': model.CredentialAccess(),
        'collection': model.Collection(),
        'privilege escalation': model.PrivilegeEscalation(level=model.PrivilegeLevel.ROOT),
        'DOS': model.DenialOfService(),
        'lateral move': model.LateralMove(),
        'execution': model.Execution(),
        'defense evasion': model.DefenseEvasion(),
        'exfiltration': model.Exfiltration(),
    }
    return label_mapping[label]
