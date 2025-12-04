"""Circuit component factory functions (functional style)."""

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

    Args:
        net: Network to add to
        node_a: First terminal
        node_b: Second terminal
        name: Component name (required, used as key in params)
        value: Resistance in Ohms (optional default, can be overridden at sim time)

    Returns:
        (new_network, component_ref)

    Example:
        net, r1 = R(net, n1, n2, name="R1", value=1000.0)  # 1 kΩ default
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

    Args:
        net: Network to add to
        node_a: First terminal (positive for voltage reference)
        node_b: Second terminal
        name: Component name (required, used as key in params)
        value: Capacitance in Farads (optional default, can be overridden at sim time)

    Returns:
        (new_network, component_ref)

    Example:
        net, c1 = C(net, n1, gnd, name="C1", value=1e-6)  # 1 µF default
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

    Args:
        net: Network to add to
        node_a: First terminal
        node_b: Second terminal
        name: Component name (required, used as key in params)
        value: Inductance in Henrys (optional default, can be overridden at sim time)

    Returns:
        (new_network, component_ref)

    Example:
        net, l1 = L(net, n1, n2, name="L1", value=1e-3)  # 1 mH default
    """
    defaults = ((name, value),) if value is not None else ()
    spec = ComponentSpec(
        name=name,
        kind="L",
        nodes=(node_a.index, node_b.index),
        extra_vars=1,  # inductor current
        defaults=defaults,
    )
    return net.add_component(spec)


def VSource(
    net: Network,
    node_p: Node,
    node_n: Node,
    *,
    name: str,
    value: float | None = None,
) -> tuple[Network, ComponentRef]:
    """
    Create a voltage source.

    Args:
        net: Network to add to
        node_p: Positive terminal
        node_n: Negative terminal
        name: Component name (required, used as key in controls/params)
        value: Voltage in Volts (optional default, can be overridden via controls at sim time)

    Returns:
        (new_network, component_ref)

    Example:
        net, vs = VSource(net, n1, gnd, name="vs", value=5.0)  # 5V default
    """
    defaults = ((name, value),) if value is not None else ()
    spec = ComponentSpec(
        name=name,
        kind="VSource",
        nodes=(node_p.index, node_n.index),
        extra_vars=1,  # source current
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
) -> tuple[Network, ComponentRef]:
    """
    Create a Voltage-Controlled Voltage Source.

    Output voltage = gain * (V(ctrl_pos) - V(ctrl_neg))
    Gain is provided via params dict at simulation time.

    Args:
        net: Network to add to
        out_pos: Positive output terminal
        out_neg: Negative output terminal
        ctrl_pos: Positive control input
        ctrl_neg: Negative control input
        name: Component name (required, used as key in params for gain)

    Returns:
        (new_network, component_ref)
    """
    spec = ComponentSpec(
        name=name,
        kind="VCVS",
        nodes=(out_pos.index, out_neg.index, ctrl_pos.index, ctrl_neg.index),
        extra_vars=1,  # source current
    )
    return net.add_component(spec)


def Switch(net: Network, node_a: Node, node_b: Node, *, name: str) -> tuple[Network, ComponentRef]:
    """
    Create a controllable switch.

    State (True/False) is provided via controls dict at simulation time.
    Optional params: {name}_r_on, {name}_r_off for custom resistances.

    Args:
        net: Network to add to
        node_a: First terminal
        node_b: Second terminal
        name: Component name (required, used as key in controls)

    Returns:
        (new_network, component_ref)
    """
    spec = ComponentSpec(
        name=name,
        kind="Switch",
        nodes=(node_a.index, node_b.index),
        extra_vars=0,
    )
    return net.add_component(spec)


def VoltageSwitch(
    net: Network,
    node_a: Node,
    node_b: Node,
    ctrl_pos: Node,
    ctrl_neg: Node,
    *,
    name: str,
) -> tuple[Network, ComponentRef]:
    """
    Create a voltage-controlled switch.

    Switch closes when V(ctrl_pos) - V(ctrl_neg) > threshold.
    Params: {name}_threshold, {name}_r_on, {name}_r_off, {name}_inverted

    Args:
        net: Network to add to
        node_a: Switch terminal A
        node_b: Switch terminal B
        ctrl_pos: Positive control input
        ctrl_neg: Negative control input
        name: Component name

    Returns:
        (new_network, component_ref)
    """
    spec = ComponentSpec(
        name=name,
        kind="VoltageSwitch",
        nodes=(node_a.index, node_b.index, ctrl_pos.index, ctrl_neg.index),
        extra_vars=0,
    )
    return net.add_component(spec)


def VCR(
    net: Network,
    node_a: Node,
    node_b: Node,
    ctrl_pos: Node,
    ctrl_neg: Node,
    *,
    name: str,
    r0: float | None = None,
    k: float | None = None,
) -> tuple[Network, ComponentRef]:
    """
    Create a Voltage-Controlled Resistor.

    Resistance is a linear function of control voltage:
        R = R0 + k * V_ctrl
    where V_ctrl = V(ctrl_pos) - V(ctrl_neg)

    Args:
        net: Network to add to
        node_a: First terminal
        node_b: Second terminal
        ctrl_pos: Positive control input
        ctrl_neg: Negative control input
        name: Component name
        r0: Base resistance in Ohms (optional default, can be overridden via params)
        k: Sensitivity in Ohms/Volt (optional default, can be overridden via params)

    Params at simulation time:
        {name}_r0: Base resistance (Ohms)
        {name}_k: Sensitivity (Ohms/Volt)

    Returns:
        (new_network, component_ref)

    Example:
        # VCR with R = 1000 + 100*V_ctrl
        net, vcr1 = VCR(net, n1, n2, ctrl_p, ctrl_n, name="VCR1", r0=1000.0, k=100.0)
    """
    defaults = ()
    if r0 is not None:
        defaults = defaults + ((f"{name}_r0", r0),)
    if k is not None:
        defaults = defaults + ((f"{name}_k", k),)

    spec = ComponentSpec(
        name=name,
        kind="VCR",
        nodes=(node_a.index, node_b.index, ctrl_pos.index, ctrl_neg.index),
        extra_vars=0,
        defaults=defaults,
    )
    return net.add_component(spec)


def DelayLine(
    net: Network,
    in_pos: Node,
    in_neg: Node,
    out_pos: Node,
    out_neg: Node,
    *,
    delay_samples: int = 1,
    name: str,
) -> tuple[Network, ComponentRef]:
    """
    Create a pure voltage delay line.

    Output voltage = input voltage delayed by N timesteps.
    V_out(t) = V_in(t - N*dt)

    Args:
        net: Network to add to
        in_pos: Positive input terminal
        in_neg: Negative input terminal
        out_pos: Positive output terminal
        out_neg: Negative output terminal
        delay_samples: Delay in timesteps (integer >= 1)
        name: Component name

    Returns:
        (new_network, component_ref)
    """
    delay_samples = max(1, delay_samples)
    # Buffer size: inherent 1-step delay from reading previous state,
    # buffer adds the rest
    buffer_size = max(1, delay_samples - 1) if delay_samples > 1 else 1

    spec = ComponentSpec(
        name=name,
        kind="DelayLine",
        nodes=(in_pos.index, in_neg.index, out_pos.index, out_neg.index, buffer_size),
        extra_vars=1,  # source current
    )
    return net.add_component(spec)
