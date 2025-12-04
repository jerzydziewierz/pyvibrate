"""Reusable subcircuit building blocks (functional style)."""

from __future__ import annotations
from typing import NamedTuple

from .network import Network, Node, ComponentRef
from .components import Switch


class HBridgeRefs(NamedTuple):
    """References to H-bridge components."""
    sw_ha: ComponentRef
    sw_la: ComponentRef
    sw_hb: ComponentRef
    sw_lb: ComponentRef
    prefix: str


def HBridge(
    net: Network,
    v_high: Node,
    v_low: Node,
    out_a: Node,
    out_b: Node,
    *,
    prefix: str = "hb",
) -> tuple[Network, HBridgeRefs]:
    """
    Create H-bridge subcircuit with 4 switches.

    Topology:
        v_high ──┬────────┬── v_high
                 │        │
               [HA]     [HB]
                 │        │
        out_a ───┴────────┴─── out_b
                 │        │
               [LA]     [LB]
                 │        │
        v_low ───┴────────┴─── v_low

    Control names are: {prefix}_ha, {prefix}_hb, {prefix}_la, {prefix}_lb
    Params for resistances: {prefix}_ha_r_on, {prefix}_ha_r_off, etc.

    Args:
        net: Network to add to
        v_high: High-side supply node
        v_low: Low-side supply node (usually gnd)
        out_a: Output A node
        out_b: Output B node
        prefix: Name prefix for switch controls

    Returns:
        (new_network, HBridgeRefs)
    """
    net, sw_ha = Switch(net, v_high, out_a, name=f"{prefix}_ha")
    net, sw_la = Switch(net, out_a, v_low, name=f"{prefix}_la")
    net, sw_hb = Switch(net, v_high, out_b, name=f"{prefix}_hb")
    net, sw_lb = Switch(net, out_b, v_low, name=f"{prefix}_lb")

    refs = HBridgeRefs(sw_ha, sw_la, sw_hb, sw_lb, prefix)
    return net, refs


def hbridge_controls(refs: HBridgeRefs, ha: bool, la: bool, hb: bool, lb: bool) -> dict:
    """
    Create controls dict for H-bridge state.

    Args:
        refs: H-bridge component references
        ha: High-side A switch (True=closed)
        la: Low-side A switch
        hb: High-side B switch
        lb: Low-side B switch

    Returns:
        Controls dict to merge into step() controls
    """
    prefix = refs.prefix
    return {
        f"{prefix}_ha": ha,
        f"{prefix}_la": la,
        f"{prefix}_hb": hb,
        f"{prefix}_lb": lb,
    }


def hbridge_drive_a_high(refs: HBridgeRefs) -> dict:
    """Drive output A to v_high, output B to v_low."""
    return hbridge_controls(refs, ha=True, la=False, hb=False, lb=True)


def hbridge_drive_b_high(refs: HBridgeRefs) -> dict:
    """Drive output A to v_low, output B to v_high."""
    return hbridge_controls(refs, ha=False, la=True, hb=True, lb=False)


def hbridge_freewheel_low(refs: HBridgeRefs) -> dict:
    """Freewheel through low-side switches."""
    return hbridge_controls(refs, ha=False, la=True, hb=False, lb=True)


def hbridge_freewheel_high(refs: HBridgeRefs) -> dict:
    """Freewheel through high-side switches."""
    return hbridge_controls(refs, ha=True, la=False, hb=True, lb=False)


def hbridge_all_off(refs: HBridgeRefs) -> dict:
    """All switches open (high impedance)."""
    return hbridge_controls(refs, ha=False, la=False, hb=False, lb=False)
