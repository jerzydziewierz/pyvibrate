"""Network and Node classes for circuit topology (immutable/functional style)."""

from __future__ import annotations
from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .simulator import SimFns


class Node(NamedTuple):
    """A node in the circuit (electrical connection point)."""
    name: str
    index: int  # index in the MNA matrix (0 = ground)


class ComponentRef(NamedTuple):
    """Reference to a component for later probing."""
    name: str
    kind: str  # "R", "C", "L", "VSource", etc.


class ComponentSpec(NamedTuple):
    """Specification for a component (topology and optional default values)."""
    name: str
    kind: str
    nodes: tuple[int, ...]  # node indices
    extra_vars: int  # number of extra MNA variables (e.g., current through VSource)
    defaults: tuple[tuple[str, float], ...] = ()  # ((param_name, default_value), ...)


class Network(NamedTuple):
    """
    Immutable circuit network topology.

    Build using functional style:
        net = Network()
        net, n1 = net.node("n1")
        net, r1 = R(net, n1, net.gnd, name="R1")
    """
    nodes: tuple[Node, ...] = (Node("gnd", 0),)
    components: tuple[ComponentSpec, ...] = ()

    @property
    def gnd(self) -> Node:
        """Ground node (reference, always 0V)."""
        return self.nodes[0]

    @property
    def num_nodes(self) -> int:
        """Number of non-ground nodes."""
        return len(self.nodes) - 1

    def node(self, name: str) -> tuple[Network, Node]:
        """
        Create a new node.

        Returns (new_network, node).
        """
        # Check if exists
        for n in self.nodes:
            if n.name == name:
                return self, n

        new_node = Node(name, len(self.nodes))
        new_net = self._replace(nodes=self.nodes + (new_node,))
        return new_net, new_node

    def add_component(self, spec: ComponentSpec) -> tuple[Network, ComponentRef]:
        """
        Add a component specification.

        Returns (new_network, component_ref).
        """
        new_net = self._replace(components=self.components + (spec,))
        ref = ComponentRef(spec.name, spec.kind)
        return new_net, ref

    def compile(self, dt: float) -> SimFns:
        """
        Create simulation functions from this network.

        Args:
            dt: Timestep in seconds

        Returns:
            SimFns with init, step, and probe functions
        """
        from .simulator import compile_network
        return compile_network(self, dt)
