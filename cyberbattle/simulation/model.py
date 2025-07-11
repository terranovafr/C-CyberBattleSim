# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    model.py
    Data model for the simulation environment including all classes used by the environment and the model environment itself which constructs the network of nodes and the vulnerabilities associated with them.
"""

from datetime import datetime
from typing import NamedTuple, List, Dict, Optional, Union, Tuple, Iterator
import dataclasses
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import networkx as nx
from cyberbattle.utils.networkx_utils import iterate_network_nodes, calculate_average_shortest_path_length
import random

VERSION_TAG = "CC-1.0.0" # C-CyberBattleSim + version

# Type aliases
NodeID = str
ID = str
NodeValue = int
PortName = int
VulnerabilityID = str
Probability = float
PropertyName = str


@dataclass
class ListeningService:
    """A service listening on a port of a given node hosting some vulnerabilities"""
    # Port to which the service is listening to
    name: PortName
    # product info
    product: str
    # product version info
    version: str
    # Feature vector for each NLP extractor (of the service description/response)
    feature_vector: Dict
    # description of the service response
    description: str = ""
    # whether the service is running or stopped
    running: bool = True
    # Weight used to evaluate the cost of not running the service
    sla_weight = 1.0

class Rates(NamedTuple):
    """Probabilities associated with a given vulnerability"""
    probingDetectionRate: Probability = 0.0
    exploitDetectionRate: Probability = 0.0
    successRate: Probability = 1.0

class VulnerabilityType(Enum):
    """Is the vulnerability exploitable locally or remotely?"""
    LOCAL = 0
    REMOTE = 1

class PrivilegeLevel(IntEnum):
    """Access privilege level on a given node"""
    NoAccess = 0
    LocalUser = 1
    ROOT = 3 # assigned 3 to leave space for an intermediary level

class VulnerabilityOutcome:
    """Outcome of exploiting a given vulnerability"""

class Movement(VulnerabilityOutcome):
    """Represent a general movement to another node"""

class LateralMove(VulnerabilityOutcome):
    """Lateral movement to the target node"""
    success: bool

class Collection(VulnerabilityOutcome):
    """Collect data on target node"""

class CredentialAccess(VulnerabilityOutcome):
    """Access the target node using some credentials found"""

class Discovery(VulnerabilityOutcome):
    """Discovery of properties of nodes (acquired visibility, e.g. OS and firewall)"""

class PrivilegeEscalation(VulnerabilityOutcome):
    """Privilege escalation outcome"""
    def __init__(self, level=PrivilegeLevel.ROOT):
        self.level = level

    @property
    def tag(self):
        # Escalation tag that gets added to node properties when the escalation level is reached for that node
        return f"privilege_{self.level}"

class InvalidAction(VulnerabilityOutcome):
    """This is for situations where the exploit fails (reason also explained by subclass)"""
    def __init__(self, reason=""):
        self.reason = reason

class FirewallBlock(InvalidAction):
    """Blocked by firewall (outgoing or incoming)"""
    def __init__(self, incoming_or_outgoing, reason=""):
        super().__init__(reason)
        self.incoming_or_outgoing = incoming_or_outgoing
        self.reason = reason

class NonListeningPort(InvalidAction):
    """Port not listening, hence if a vulnerability is related to a service on that port, it cannot be exploited"""
    def __init__(self, reason=""):
        super().__init__(reason)
        self.reason = reason

class OutcomeNonPresent(InvalidAction):
    """Outcome non present/possible for a vulnerability (whether it is possible is given by the multi-label classifier)"""
    def __init__(self, reason=""):
        super().__init__(reason)
        self.reason = reason

class NoEnoughPrivilege(InvalidAction):
    """No enough privilege on node to exploit vulnerability"""
    def __init__(self, reason=""):
        super().__init__(reason)
        self.reason = reason

class NonRunningMachine(InvalidAction):
    """Machine not running"""
    def __init__(self, source_or_target=0, reason=""):
        super().__init__(reason)
        self.source_or_target = source_or_target
        self.reason = reason

class UnsuccessfulAction(InvalidAction):
    """Action unsuccessful based on success rate"""
    def __init__(self, reason=""):
        super().__init__(reason)
        self.reason = reason

class RepeatedResult(InvalidAction):
    """Outcome already exploited"""
    def __init__(self, reason=""):
        super().__init__(reason)
        self.reason = reason

class NoNeededAction(InvalidAction):
    """This is for situations where the action is useless"""
    def __init__(self, reason=""):
        super().__init__(reason)
        self.reason = reason

class NoVulnerability(InvalidAction):
    """Vulnerability not present on the node"""
    def __init__(self, reason=""):
        super().__init__(reason)
        self.reason = reason

class RemoteOutcomeInLocalNode(InvalidAction):
    """Wrong outcome in local vulnerability (e.g. credential access)"""
    def __init__(self, reason=""):
        super().__init__(reason)
        self.reason = reason


class InvalidMovement(InvalidAction):
    """Invalid movement"""
    def __init__(self, reason=""):
        super().__init__(reason)
        self.reason = reason

class Reconnaissance(VulnerabilityOutcome):
    """A set of node IDs obtained by exploiting a vulnerability"""

    def __init__(self, nodes):
        if not nodes:
            nodes = []
        self.nodes = nodes

class DenialOfService(VulnerabilityOutcome):
    """Denial of service outcome"""

class NoOutcome(VulnerabilityOutcome):
    """No outcome"""

class Persistence(VulnerabilityOutcome):
    """Persistence outcome"""

class Execution(VulnerabilityOutcome):
    """Execution general outcome -> allows all the others"""

class Exfiltration(VulnerabilityOutcome):
    """Exfiltration outcome"""

class DefenseEvasion(VulnerabilityOutcome):
    """Defense Evasion outcome"""

VulnerabilityOutcomes = Union[
    Discovery, PrivilegeEscalation, Reconnaissance, Collection, LateralMove, Persistence, DenialOfService, InvalidAction, CredentialAccess]

# Map outcome to index of the outcome
def map_outcome_to_index(outcome):
    outcomes = [Discovery, PrivilegeEscalation, Reconnaissance, Collection, LateralMove, Persistence, DenialOfService, InvalidAction, CredentialAccess]
    outcome_to_index = {cls: idx for idx, cls in enumerate(outcomes)}
    return outcome_to_index.get(outcome.__class__)

class PredictedResult(NamedTuple):
    """The predicted outcome allowed by exploiting a vulnerability"""
    type: VulnerabilityType # gathered from CVSS vector
    type_str: str
    outcome: VulnerabilityOutcome # predicted by the multi-label classifier
    outcome_str: str
    probability: float # output probability of the multi label classifier

@dataclass
class VulnerabilityInfo:
    """Definition of a known vulnerability"""
    vulnerability_ID: str
    # port of the service that has the vulnerability
    port: PortName
    # an optional description of what the vulnerability is
    description: str
    # what happens when successfully exploiting the vulnerability (approximated outcomes by the multi-label classifier)
    results: List[PredictedResult] = field(default_factory=list)
    # rates of success/failure associated with this vulnerability
    rates: Rates = Rates()
    # points to information about the vulnerability
    URL: str = ""
    # some cost associated with exploiting this vulnerability (e.g.
    # brute force more costly than dumping credentials)
    cost: float = 1.0
    # dict of embeddings for all models
    embedding: Dict = field(default_factory=dict)
    # metrics from CVSS
    cvss: float = 0.0
    cvss_v2: float = 0.0
    epss: float = 0.0
    ranking_epss: float = 0.0
    attack_vector: str = ""
    attack_complexity: str = ""
    privileges_required: PrivilegeLevel = PrivilegeLevel.LocalUser
    user_interaction: str = ""
    confidentiality_impact: str = ""
    integrity_impact: str = ""
    availability_impact: str = ""
    base_score: float = 0.0
    exploitability_score: float = 0.0
    impact_score: float = 0.0
    base_severity: str = ""
    # a string displayed when the vulnerability is successfully exploited
    reward_string: str = ""

# A dictionary storing information about the vulnerabilities inside a node
VulnerabilityLibrary = Dict[VulnerabilityID, VulnerabilityInfo]

class RulePermission(Enum):
    """Determine if a rule is blocks or allows traffic"""
    ALLOW = 0
    BLOCK = 1

@dataclass
class FirewallRule:
    """A firewall rule"""
    # A port number/name
    port: PortName
    # permission on this port
    permission: RulePermission
    # An optional reason for the block/allow rule
    reason: str = ""

    def __eq__(self, other):
        if not isinstance(other, FirewallRule):
            return NotImplemented
        return (self.port, self.permission, self.reason) == (other.port, other.permission, other.reason)

    def __hash__(self):
        return hash((self.port, self.permission, self.reason))


@dataclass
class FirewallConfiguration:
    """Firewall configuration on a given node.
    Determine if traffic should be allowed or specifically blocked on a given port for outgoing and incoming traffic.
    The rules are process in order: the first rule matching a given port is applied and the rest are ignored.
    """
    outgoing: List[FirewallRule] = field(repr=True, default_factory=lambda: [])
    incoming: List[FirewallRule] = field(repr=True, default_factory=lambda: [])


class MachineStatus(Enum):
    """Machine running status"""
    Stopped = 0
    Running = 1
    Imaging = 2


@dataclass
class NodeInfo:
    """A computer node in the enterprise network"""
    # Node ID
    node_id: NodeID = ""
    # Attacker agent installed on the node? (aka the node is 'pwned')
    agent_installed: bool = False
    # Last time the node was reimaged
    last_reimaging: Optional[datetime] = None
    # String displayed when the node gets owned
    owned_string: str = ""
    # tag representing the category (e.g. Windows, IoT, ...)
    tag: str = ""
    # List of services the node is hosting
    services: List[ListeningService] = dataclasses.field(default_factory=list)
    # Intrinsic value of the node (translates into a reward if the node gets owned)
    value: NodeValue = 0
    # Eventual properties of the nodes, not yet used in the simulation
    properties: List[PropertyName] = dataclasses.field(default_factory=list)
    # List of known vulnerabilities for the node
    vulnerabilities: VulnerabilityLibrary = dataclasses.field(default_factory=dict)
    # Firewall configuration of the node
    firewall: FirewallConfiguration = field(default_factory=FirewallConfiguration)
    # Current privilege escalation level
    privilege_level: PrivilegeLevel = PrivilegeLevel.NoAccess
    # Can the node be re-imaged by a defender agent?
    reimageable: bool = True
    # Machine status: running or stopped
    status = MachineStatus.Running
    # Relative node weight used to calculate the cost of stopping this machine or its services
    sla_weight: float = 1.0
    # Is here data to be found using a collection vulnerability?
    has_data: bool = False
    # Has the data been collected ?
    data_collected: bool = False
    # Has the data been exfiltrated ?
    data_exfiltrated: bool = False
    # Partially visible or not?
    visible: bool = True
    # Persistence flag, ensuring that at reboot the agent will still be there
    persistence: bool = False
    # Defense evasion flag, ensuring that the agent will not be detected by firewall rules or can evade defender actions on node
    defense_evasion: bool = False
    # Privilege level when accessing the node
    level_at_access: PrivilegeLevel = PrivilegeLevel.LocalUser

def escalate(current_level, escalation_level: PrivilegeLevel) -> PrivilegeLevel:
    return PrivilegeLevel(max(int(current_level), int(escalation_level)))

class NoSuitableStarterNode(Exception):
    """Exception raised when no suitable starter node is found (no-one respects the isolation threshold)."""
    def __init__(self, message="No suitable starter node found."):
        self.message = message
        super().__init__(self.message)

@dataclass
class Model:
    """ A static single graph defining a network of computers """

    def __init__(self,
                 network=None,
                 creationTime=None,
                 lastModified=None,
                 version=VERSION_TAG,
                 firewall_rule_incoming_probability=0.2,
                 firewall_rule_outgoing_probability=0.2,
                 knows_neighbor_probability_range=None,
                 data_presence_probability=0.5,
                 partial_visibility_probability=0.5,
                 need_to_escalate_probability=0.5,
                 service_shutdown_probability=0.1,
                 success_rate_probability_range=None,
                 probing_detection_probability_range=None,
                 exploit_detection_probability_range=None,
                 value_range=(0, 100),
                 env_type="compressed_env",
                 knows_graph=None,
                 access_graph=None,
                 evolving_visible_graph=None,
                 vulnerability_classifier=None,
                 **kwargs):

        # Set the fields from arguments
        self.network: nx.DiGraph = network
        self.creationTime = creationTime if creationTime is not None else datetime.now()
        self.lastModified = lastModified if lastModified is not None else datetime.now()
        self.version = version
        self.firewall_rule_incoming_probability = firewall_rule_incoming_probability
        self.firewall_rule_outgoing_probability = firewall_rule_outgoing_probability
        self.knows_neighbor_probability_range = knows_neighbor_probability_range
        self.data_presence_probability = data_presence_probability
        self.partial_visibility_probability = partial_visibility_probability
        self.need_to_escalate_probability = need_to_escalate_probability
        self.service_shutdown_probability = service_shutdown_probability
        self.success_rate_probability_range = success_rate_probability_range
        self.probing_detection_probability_range = probing_detection_probability_range
        self.exploit_detection_probability_range = exploit_detection_probability_range
        self.value_range = value_range
        self.evolving_visible_graph = evolving_visible_graph
        self.env_type = env_type
        self.knows_graph = knows_graph
        self.access_graph = access_graph

        # imported here due to a circular import issue
        import cyberbattle.simulation.generate_network as g

        # Generate model from networkx graph and probabilities
        self.network, self.knows_graph, self.access_graph, self.dos_graph, self.evolving_visible_graph, self.ports_list = g.cyberbattle_model_from_nodes_graph(
                network,
                firewall_rule_incoming_probability=self.firewall_rule_incoming_probability,
                firewall_rule_outgoing_probability=self.firewall_rule_outgoing_probability,
                knows_neighbor_probability_range=self.knows_neighbor_probability_range,
                data_presence_probability=self.data_presence_probability,
                partial_visibility_probability=self.partial_visibility_probability,
                need_to_escalate_probability=self.need_to_escalate_probability,
                service_shutdown_probability=self.service_shutdown_probability,
                value_range=self.value_range,
                vulnerability_classifier=vulnerability_classifier,
                success_rate_probability_range=self.success_rate_probability_range,
                probing_detection_probability_range=self.probing_detection_probability_range,
                exploit_detection_probability_range=self.exploit_detection_probability_range,
        )

        # Calculate reachability, connectivity metric and dict of shortest paths for every pair of nodes
        self.knows_reachability, self.knows_connectivity, self.knows_shortest_paths = calculate_average_shortest_path_length(self.knows_graph)
        self.access_reachability, self.access_connectivity, self.access_shortest_paths = calculate_average_shortest_path_length(self.access_graph)
        self.dos_reachability, self.dos_connectivity, self.dos_shortest_paths = calculate_average_shortest_path_length(self.dos_graph)

    # Function used to update the feature vectors of the services and vulnerabilities embeddings using another feature extractor model
    def update_feature_vectors(self, feature_extractor_model=None):
        for node in self.network.nodes:
            for i, service in enumerate(self.network.nodes[node]['data'].services):
                self.network.nodes[node]['data'].services[i].feature_vector = service.feature_vector[feature_extractor_model]
                for vulnerability in self.network.nodes[node]['data'].vulnerabilities:
                    if isinstance(self.network.nodes[node]['data'].vulnerabilities[vulnerability].embedding, dict):
                        self.network.nodes[node]['data'].vulnerabilities[vulnerability].embedding = self.network.nodes[node]['data'].vulnerabilities[vulnerability].embedding[feature_extractor_model]
                    # otherwise already processed, case of multiple services in node with same vuln

    def nodes(self) -> Iterator[Tuple[NodeID, NodeInfo]]:
        return iterate_network_nodes(self.network)

    def get_node(self, node_id: NodeID) -> NodeInfo:
        node_info: NodeInfo = self.network.nodes[node_id]['data']
        return node_info


# Wrap networkx to the model needed to be wrapped to the environment
def wrap_networkx_to_model(net, firewall_rule_incoming_probability_range, firewall_rule_outgoing_probability_range,
                knows_neighbor_probability_range, need_to_escalate_probability_range, partial_visibility_probability_range,
                                data_presence_probability_range, value_range, service_shutdown_probability_range, **kwargs):
    if isinstance(net, tuple):
        net = net[0]
    firewall_rule_incoming_probability = random.uniform(firewall_rule_incoming_probability_range[0],
                                                            firewall_rule_incoming_probability_range[1])
    firewall_rule_outgoing_probability = random.uniform(firewall_rule_outgoing_probability_range[0],
                                                            firewall_rule_outgoing_probability_range[1])
    need_to_escalate_probability = random.uniform(need_to_escalate_probability_range[0], need_to_escalate_probability_range[1])
    partial_visibility_probability = random.uniform(partial_visibility_probability_range[0], partial_visibility_probability_range[1])
    data_presence_probability = random.uniform(data_presence_probability_range[0], data_presence_probability_range[1])
    service_shutdown_probability = random.uniform(service_shutdown_probability_range[0], service_shutdown_probability_range[1])

    probabilities = {
        'firewall_rule_incoming_probability': firewall_rule_incoming_probability,
        'firewall_rule_outgoing_probability': firewall_rule_outgoing_probability,
        'knows_neighbor_probability_range': knows_neighbor_probability_range,
        'value_range': value_range,
        'need_to_escalate_probability': need_to_escalate_probability,
        'partial_visibility_probability': partial_visibility_probability,
        'data_presence_probability': data_presence_probability,
        'service_shutdown_probability': service_shutdown_probability
    }
    kwargs.update(probabilities)
    model_env = Model(network=net, env_type="shodan_env", **kwargs)
    return model_env
