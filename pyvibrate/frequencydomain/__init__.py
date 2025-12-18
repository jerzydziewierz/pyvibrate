"""PyVibrate Frequency-Domain Simulation Module.

This module provides JAX-compatible frequency-domain circuit analysis
using Modified Nodal Analysis (MNA) with complex admittances.

Components:
    - R, C, L: Basic passive components
    - ACSource: AC voltage source with magnitude and phase
    - ConstantTimeDelayVCVS: Ideal time delay element (active/energy source)
    - VCVS: Voltage-controlled voltage source
    - TLine: Transmission line with characteristic impedance and delay
"""

from .network import Network, Node, ComponentRef, ComponentSpec
from .components import R, ACSource, C, L, ConstantTimeDelayVCVS, VCVS, TLine
from .solver import Solver, Solution, compile_network
from .subcircuits import Series, Parallel

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
    "ACSource",
    "ConstantTimeDelayVCVS",
    "VCVS",
    "TLine",
    # Solver
    "Solver",
    "Solution",
    "compile_network",
    # Subcircuits
    "Series",
    "Parallel",
]
