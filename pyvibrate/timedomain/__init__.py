"""PyVibrate Time-Domain Simulation Module.

This module provides JAX-compatible time-domain circuit simulation
using Modified Nodal Analysis (MNA) with trapezoidal integration.

Components:
    - R, C, L: Basic passive components
    - VSource: Voltage source
    - Switch, VoltageSwitch: Controllable switches
    - VCVS: Voltage-controlled voltage source
    - VCR: Voltage-controlled resistor
    - DelayLine: Pure voltage delay

Subcircuits:
    - HBridge: H-bridge power stage with 4 switches
"""

from .network import Network, Node, ComponentRef, ComponentSpec
from .components import R, C, L, VSource, Switch, VoltageSwitch, VCVS, VCR, DelayLine
from .simulator import SimState, SimFns, compile_network
from .subcircuits import (
    HBridge,
    HBridgeRefs,
    hbridge_controls,
    hbridge_drive_a_high,
    hbridge_drive_b_high,
    hbridge_freewheel_low,
    hbridge_freewheel_high,
    hbridge_all_off,
)

__all__ = [
    # Network building
    "Network",
    "Node",
    "ComponentRef",
    "ComponentSpec",
    # Components
    "R",
    "C",
    "L",
    "VSource",
    "Switch",
    "VoltageSwitch",
    "VCVS",
    "VCR",
    "DelayLine",
    # Simulation
    "SimState",
    "SimFns",
    "compile_network",
    # Subcircuits
    "HBridge",
    "HBridgeRefs",
    "hbridge_controls",
    "hbridge_drive_a_high",
    "hbridge_drive_b_high",
    "hbridge_freewheel_low",
    "hbridge_freewheel_high",
    "hbridge_all_off",
]
