"""Reusable subcircuit building blocks for frequency-domain analysis (functional style)."""

from __future__ import annotations

from .network import Network, Node


def Series(
    net: Network,
    n1: Node,
    n2: Node,
    elem1_factory,
    elem2_factory,
    prefix: str = "ser",
) -> tuple[Network, tuple]:
    """
    Connect two arbitrary two-port elements in series.

    Topology:
        n1 ──[elem1]──(n_mid)──[elem2]── n2

    This creates a floating subcircuit that does not require ground connection.
    Both elements can be any two-port component (R, C, L, or even other subcircuits).

    Args:
        net: Network to add to
        n1: Input terminal
        n2: Output terminal
        elem1_factory: Callable (net, node_a, node_b) -> (net, ref)
                      Factory function that creates first element
        elem2_factory: Callable (net, node_a, node_b) -> (net, ref)
                      Factory function that creates second element
        prefix: Name prefix for internal nodes

    Returns:
        (new_network, (ref1, ref2, n_mid))
        where n_mid is the internal connection node

    Example:
        # Series RC low-pass filter
        from pyvibrate.frequencydomain import Network, R, C
        from pyvibrate.frequencydomain.subcircuits import Series

        net = Network()
        net, n_in = net.node("in")
        net, n_out = net.node("out")

        net, (r_ref, c_ref, mid) = Series(
            net, n_in, n_out,
            lambda net, a, b: R(net, a, b, name="r1", value=1000.0),
            lambda net, a, b: C(net, a, b, name="c1", value=1e-6),
            prefix="rc_lpf"
        )

        # Add source and solve
        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
        solver = net.compile()
        sol = solver.solve_at(omega=2*3.14159*1000)
    """
    net, n_mid = net.node(f"{prefix}_mid")
    net, ref1 = elem1_factory(net, n1, n_mid)
    net, ref2 = elem2_factory(net, n_mid, n2)
    return net, (ref1, ref2, n_mid)


def Parallel(
    net: Network,
    n1: Node,
    n2: Node,
    elem1_factory,
    elem2_factory,
    prefix: str = "par",
) -> tuple[Network, tuple]:
    """
    Connect two arbitrary two-port elements in parallel.

    Topology:
        n1 ──┬──[elem1]──┬── n2
             └──[elem2]──┘

    This creates a floating subcircuit that does not require ground connection.
    Both elements can be any two-port component (R, C, L, or even other subcircuits).

    Args:
        net: Network to add to
        n1: First terminal
        n2: Second terminal
        elem1_factory: Callable (net, node_a, node_b) -> (net, ref)
                      Factory function that creates first element
        elem2_factory: Callable (net, node_a, node_b) -> (net, ref)
                      Factory function that creates second element
        prefix: Name prefix (currently unused but kept for consistency)

    Returns:
        (new_network, (ref1, ref2))

    Example:
        # Parallel RL impedance
        from pyvibrate.frequencydomain import Network, R, L
        from pyvibrate.frequencydomain.subcircuits import Series, Parallel

        net = Network()
        net, n_in = net.node("in")
        net, n_out = net.node("out")

        net, (r_ref, l_ref) = Parallel(
            net, n_in, n_out,
            lambda net, a, b: R(net, a, b, name="r1", value=100.0),
            lambda net, a, b: L(net, a, b, name="l1", value=1e-3),
            prefix="rl_par"
        )

        # Add source and solve
        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
        solver = net.compile()
        sol = solver.solve_at(omega=2*3.14159*1000)
    """
    net, ref1 = elem1_factory(net, n1, n2)
    net, ref2 = elem2_factory(net, n1, n2)
    return net, (ref1, ref2)
