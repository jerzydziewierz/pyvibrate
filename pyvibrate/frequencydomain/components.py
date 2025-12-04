"""Circuit component factory functions for frequency-domain analysis (functional style).

In frequency domain:
- R has admittance Y = 1/R (real)
- C has admittance Y = jωC (positive imaginary)
- L has admittance Y = 1/(jωL) = -j/(ωL) (negative imaginary)
"""

from __future__ import annotations

from .network import Network, Node, ComponentSpec, ComponentRef


def R(
    net: Network,
    node_a: Node,
    node_b: Node,
    *,
    name: str,
    value: float | None = None,
) -> tuple[Network, ComponentRef]:
    """
    Create a resistor.

    Admittance: Y = 1/R (real, frequency-independent)

    Args:
        net: Network to add to
        node_a: First terminal
        node_b: Second terminal
        name: Component name (required, used as key in params)
        value: Resistance in Ohms (optional default, can be overridden at solve time)

    Returns:
        (new_network, component_ref)

    Example:
        net, r1 = R(net, n1, n2, name="R1", value=1000.0)  # 1 kOhm
    """
    defaults = ((name, value),) if value is not None else ()
    spec = ComponentSpec(
        name=name,
        kind="R",
        nodes=(node_a.index, node_b.index),
        extra_vars=0,
        defaults=defaults,
    )
    return net.add_component(spec)


def ACSource(
    net: Network,
    node_p: Node,
    node_n: Node,
    *,
    name: str,
    value: float | None = None,
    phase: float | None = None,
) -> tuple[Network, ComponentRef]:
    """
    Create an AC voltage source.

    The source provides a complex voltage phasor: V = value * exp(j * phase)

    Args:
        net: Network to add to
        node_p: Positive terminal
        node_n: Negative terminal
        name: Component name (required, used as key in params)
        value: Voltage magnitude in Volts (optional default)
        phase: Phase angle in radians (optional default, 0 if not specified)

    Returns:
        (new_network, component_ref)

    Example:
        net, vs = ACSource(net, n1, gnd, name="vs", value=1.0)  # 1V at 0 degrees
        net, vs = ACSource(net, n1, gnd, name="vs", value=1.0, phase=0.5)  # 1V at ~28.6 degrees
    """
    defaults = ()
    if value is not None:
        defaults = defaults + ((name, value),)
    if phase is not None:
        defaults = defaults + ((f"{name}_phase", phase),)

    spec = ComponentSpec(
        name=name,
        kind="ACSource",
        nodes=(node_p.index, node_n.index),
        extra_vars=1,  # source current
        defaults=defaults,
    )
    return net.add_component(spec)


def C(
    net: Network,
    node_a: Node,
    node_b: Node,
    *,
    name: str,
    value: float | None = None,
) -> tuple[Network, ComponentRef]:
    """
    Create a capacitor.

    Admittance: Y = jωC (positive imaginary, increases with frequency)
    Impedance: Z = 1/(jωC) = -j/(ωC) (negative imaginary)

    Args:
        net: Network to add to
        node_a: First terminal
        node_b: Second terminal
        name: Component name (required, used as key in params)
        value: Capacitance in Farads (optional default, can be overridden at solve time)

    Returns:
        (new_network, component_ref)

    Example:
        net, c1 = C(net, n1, n2, name="C1", value=1e-6)  # 1 uF
    """
    defaults = ((name, value),) if value is not None else ()
    spec = ComponentSpec(
        name=name,
        kind="C",
        nodes=(node_a.index, node_b.index),
        extra_vars=0,
        defaults=defaults,
    )
    return net.add_component(spec)


def L(
    net: Network,
    node_a: Node,
    node_b: Node,
    *,
    name: str,
    value: float | None = None,
) -> tuple[Network, ComponentRef]:
    """
    Create an inductor.

    Admittance: Y = 1/(jωL) = -j/(ωL) (negative imaginary, decreases with frequency)
    Impedance: Z = jωL (positive imaginary)

    Args:
        net: Network to add to
        node_a: First terminal
        node_b: Second terminal
        name: Component name (required, used as key in params)
        value: Inductance in Henrys (optional default, can be overridden at solve time)

    Returns:
        (new_network, component_ref)

    Example:
        net, l1 = L(net, n1, n2, name="L1", value=1e-3)  # 1 mH
    """
    defaults = ((name, value),) if value is not None else ()
    spec = ComponentSpec(
        name=name,
        kind="L",
        nodes=(node_a.index, node_b.index),
        extra_vars=0,
        defaults=defaults,
    )
    return net.add_component(spec)


def PhaseShift(
    net: Network,
    in_pos: Node,
    in_neg: Node,
    out_pos: Node,
    out_neg: Node,
    *,
    name: str,
    tau: float | None = None,
) -> tuple[Network, ComponentRef]:
    """
    Create a pure phase shift element (ideal delay line).

    V_out = V_in * exp(-j * omega * tau)

    This is an ideal element with no impedance mismatch effects - just pure phase rotation.
    For transmission line effects with reflections, use TLine instead.

    Args:
        net: Network to add to
        in_pos: Positive input terminal
        in_neg: Negative input terminal (reference for input voltage)
        out_pos: Positive output terminal
        out_neg: Negative output terminal (reference for output voltage)
        name: Component name (required)
        tau: Time delay in seconds (optional default, can be overridden at solve time)

    Returns:
        (new_network, component_ref)

    Example:
        net, ps1 = PhaseShift(net, in_p, in_n, out_p, out_n, name="PS1", tau=1e-9)  # 1 ns delay
    """
    defaults = ()
    if tau is not None:
        defaults = defaults + ((f"{name}_tau", tau),)

    spec = ComponentSpec(
        name=name,
        kind="PhaseShift",
        nodes=(in_pos.index, in_neg.index, out_pos.index, out_neg.index),
        extra_vars=1,  # output current
        defaults=defaults,
    )
    return net.add_component(spec)


def VCVS(
    net: Network,
    out_pos: Node,
    out_neg: Node,
    ctrl_pos: Node,
    ctrl_neg: Node,
    *,
    name: str,
    gain: float | None = None,
) -> tuple[Network, ComponentRef]:
    """
    Create a Voltage-Controlled Voltage Source.

    V_out = gain * (V_ctrl_pos - V_ctrl_neg)

    This is an ideal dependent source with infinite input impedance
    and zero output impedance.

    Args:
        net: Network to add to
        out_pos: Positive output terminal
        out_neg: Negative output terminal
        ctrl_pos: Positive control input
        ctrl_neg: Negative control input
        name: Component name (required, used as key in params for gain)
        gain: Voltage gain (optional default, can be overridden at solve time)

    Returns:
        (new_network, component_ref)

    Example:
        net, e1 = VCVS(net, out_p, out_n, ctrl_p, ctrl_n, name="E1", gain=10.0)
    """
    defaults = ((name, gain),) if gain is not None else ()
    spec = ComponentSpec(
        name=name,
        kind="VCVS",
        nodes=(out_pos.index, out_neg.index, ctrl_pos.index, ctrl_neg.index),
        extra_vars=1,  # output current
        defaults=defaults,
    )
    return net.add_component(spec)


def TLine(
    net: Network,
    port1_pos: Node,
    port1_neg: Node,
    port2_pos: Node,
    port2_neg: Node,
    *,
    name: str,
    Z0: float | None = None,
    tau: float | None = None,
) -> tuple[Network, ComponentRef]:
    """
    Create a lossless transmission line with characteristic impedance and delay.

    Y-parameter model:
        Y11 = Y22 = -j*cot(omega*tau) / Z0
        Y12 = Y21 = j*csc(omega*tau) / Z0

    This model includes impedance mismatch and reflection effects, unlike PhaseShift
    which is a pure phase rotation with no reflections.

    Args:
        net: Network to add to
        port1_pos: Port 1 positive terminal
        port1_neg: Port 1 negative terminal
        port2_pos: Port 2 positive terminal
        port2_neg: Port 2 negative terminal
        name: Component name (required)
        Z0: Characteristic impedance in Ohms (optional default, can be overridden at solve time)
        tau: Propagation delay in seconds (optional default, can be overridden at solve time)

    Returns:
        (new_network, component_ref)

    Example:
        # 50 ohm coax with 5 ns electrical length
        net, tl1 = TLine(net, p1_p, p1_n, p2_p, p2_n, name="TL1", Z0=50.0, tau=5e-9)
    """
    defaults = ()
    if Z0 is not None:
        defaults = defaults + ((f"{name}_Z0", Z0),)
    if tau is not None:
        defaults = defaults + ((f"{name}_tau", tau),)

    spec = ComponentSpec(
        name=name,
        kind="TLine",
        nodes=(port1_pos.index, port1_neg.index, port2_pos.index, port2_neg.index),
        extra_vars=0,  # Pure Y-parameter model, no extra vars
        defaults=defaults,
    )
    return net.add_component(spec)
