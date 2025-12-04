"""PyVibrate - JAX-compatible circuit simulator.

This package provides two simulation domains:
    - timedomain: Transient simulation using MNA with trapezoidal integration
    - frequencydomain: Steady-state AC analysis with complex phasors

Usage:
    from pyvibrate.timedomain import Network, R, C, L, VSource, ...
    from pyvibrate.frequencydomain import Network, R, C, L, ACSource, ...
"""

__version__ = "0.1.0"
__all__ = ["timedomain", "frequencydomain", "__version__"]
