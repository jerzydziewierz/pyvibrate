"""Frequency-domain solver using complex MNA (Modified Nodal Analysis).

At each frequency omega, we solve:
    Y * V = I
where:
    Y is the complex admittance matrix
    V is the complex node voltage phasor vector
    I is the complex current injection vector

Components stamp their admittances into Y and sources stamp into I.
"""

from __future__ import annotations
from typing import NamedTuple, Callable
import jax.numpy as jnp
from jax import Array


class Solution(NamedTuple):
    """Result of solving the circuit at a single frequency."""
    omega: Array  # angular frequency (rad/s)
    voltages: Array  # complex node voltage phasors (excluding ground)
    currents: Array  # complex branch currents for components with extra_vars


class Solver(NamedTuple):
    """
    Compiled frequency-domain solver for a circuit.

    Contains functions to solve at single frequency or sweep frequencies.
    """
    network: object  # Network reference (for node lookups)
    solve_at: Callable  # (omega, params) -> Solution
    v: Callable  # (solution, node) -> complex voltage phasor
    i: Callable  # (solution, component_ref) -> complex current phasor
    z_in: Callable  # (solution, port_ref) -> complex impedance at port
    component_current_indices: dict  # {component_name: index in currents}
    defaults: dict  # {param_name: default_value}


def compile_network(network) -> Solver:
    """
    Compile a Network into frequency-domain solver functions.

    Args:
        network: Network with components

    Returns:
        Solver with solve_at, v, i, z_in functions
    """
    num_nodes = network.num_nodes  # excludes ground

    # Count extra variables (branch currents for voltage sources, etc.)
    extra_var_offset = num_nodes
    component_current_indices = {}
    for comp in network.components:
        if comp.extra_vars > 0:
            component_current_indices[comp.name] = extra_var_offset
            extra_var_offset += comp.extra_vars

    total_size = extra_var_offset

    # Collect defaults
    defaults = {}
    for comp in network.components:
        for param_name, default_value in comp.defaults:
            defaults[param_name] = default_value

    def solve_at(omega: float | Array, params: dict | None = None) -> Solution:
        """
        Solve the circuit at a single angular frequency.

        Args:
            omega: Angular frequency in rad/s (2*pi*f)
            params: Component parameters (merged with defaults)

        Returns:
            Solution with complex voltage phasors
        """
        if params is None:
            params = {}

        # Merge with defaults
        merged_params = {**defaults, **params}

        omega = jnp.asarray(omega, dtype=jnp.complex64).real

        # Build complex admittance matrix Y and current vector I
        Y = jnp.zeros((total_size, total_size), dtype=jnp.complex64)
        I = jnp.zeros(total_size, dtype=jnp.complex64)

        for comp in network.components:
            Y, I = _stamp_component(Y, I, comp, omega, merged_params,
                                    component_current_indices)

        # Solve Y * V = I
        solution_vec = jnp.linalg.solve(Y, I)

        # Extract node voltages (first num_nodes entries)
        voltages = solution_vec[:num_nodes]

        # Extract branch currents (remaining entries)
        currents = solution_vec[num_nodes:]

        return Solution(
            omega=jnp.asarray(omega),
            voltages=voltages,
            currents=currents,
        )

    def v(solution: Solution, node) -> Array:
        """Get complex voltage phasor at a node."""
        if node.index == 0:  # ground
            return jnp.array(0.0, dtype=jnp.complex64)
        return solution.voltages[node.index - 1]

    def i(solution: Solution, component_ref) -> Array:
        """Get complex current phasor through a component (for sources/inductors)."""
        if component_ref.name not in component_current_indices:
            raise ValueError(f"Component {component_ref.name} has no branch current")
        idx = component_current_indices[component_ref.name] - num_nodes
        return solution.currents[idx]

    def z_in(solution: Solution, source_ref) -> Array:
        """
        Get input impedance seen at an AC source.

        Z_in = V_source / I_source
        """
        # Find the source component
        source_comp = None
        for comp in network.components:
            if comp.name == source_ref.name:
                source_comp = comp
                break

        if source_comp is None:
            raise ValueError(f"Source {source_ref.name} not found")

        # Voltage across source
        n_pos, n_neg = source_comp.nodes[:2]
        v_pos = solution.voltages[n_pos - 1] if n_pos > 0 else 0.0
        v_neg = solution.voltages[n_neg - 1] if n_neg > 0 else 0.0
        v_source = v_pos - v_neg

        # Current through source
        i_source = i(solution, source_ref)

        # Z = V / I (negative because source current is defined as flowing out)
        return -v_source / i_source

    return Solver(
        network=network,
        solve_at=solve_at,
        v=v,
        i=i,
        z_in=z_in,
        component_current_indices=component_current_indices,
        defaults=defaults,
    )


def _stamp_component(Y: Array, I: Array, comp, omega: Array, params: dict,
                     current_indices: dict) -> tuple[Array, Array]:
    """Stamp a component into the Y matrix and I vector."""

    kind = comp.kind
    nodes = comp.nodes
    name = comp.name

    if kind == "R":
        # Y = 1/R between nodes
        R_val = params.get(name, 1.0)
        g = 1.0 / R_val  # conductance
        Y = _stamp_admittance(Y, nodes[0], nodes[1], g)

    elif kind == "C":
        # Y = j*omega*C between nodes
        C_val = params.get(name, 1.0)
        y = 1j * omega * C_val
        Y = _stamp_admittance(Y, nodes[0], nodes[1], y)

    elif kind == "L":
        # Y = 1/(j*omega*L) = -j/(omega*L) between nodes
        L_val = params.get(name, 1.0)
        # Avoid division by zero at DC
        omega_safe = jnp.where(omega == 0, 1e-12, omega)
        y = 1.0 / (1j * omega_safe * L_val)
        # At DC, inductor is short circuit - handled by extra variable
        Y = _stamp_admittance(Y, nodes[0], nodes[1], y)

    elif kind == "ACSource":
        # Voltage source with complex phasor value
        # V(n_pos) - V(n_neg) = V_source
        # Uses extra variable for current
        n_pos, n_neg = nodes
        idx = current_indices[name]

        # Get source voltage (complex phasor)
        v_mag = params.get(name, 1.0)
        v_phase = params.get(f"{name}_phase", 0.0)
        v_source = v_mag * jnp.exp(1j * v_phase)

        # Stamp:
        # Row for voltage equation: V_pos - V_neg - V_source = 0
        # Current flows from pos to neg inside source
        if n_pos > 0:
            Y = Y.at[idx, n_pos - 1].add(1.0)
            Y = Y.at[n_pos - 1, idx].add(1.0)
        if n_neg > 0:
            Y = Y.at[idx, n_neg - 1].add(-1.0)
            Y = Y.at[n_neg - 1, idx].add(-1.0)

        I = I.at[idx].add(v_source)

    elif kind == "VCVS":
        # Voltage-controlled voltage source
        # V_out = gain * V_ctrl
        # V(out_pos) - V(out_neg) = gain * (V(ctrl_pos) - V(ctrl_neg))
        out_pos, out_neg, ctrl_pos, ctrl_neg = nodes
        idx = current_indices[name]
        gain = params.get(name, 1.0)

        # Output voltage equation row
        if out_pos > 0:
            Y = Y.at[idx, out_pos - 1].add(1.0)
            Y = Y.at[out_pos - 1, idx].add(1.0)
        if out_neg > 0:
            Y = Y.at[idx, out_neg - 1].add(-1.0)
            Y = Y.at[out_neg - 1, idx].add(-1.0)

        # Control voltage contribution
        if ctrl_pos > 0:
            Y = Y.at[idx, ctrl_pos - 1].add(-gain)
        if ctrl_neg > 0:
            Y = Y.at[idx, ctrl_neg - 1].add(gain)

    elif kind == "ConstantTimeDelayVCVS":
        # Constant time delay element (ideal voltage-controlled delay)
        # V_out = V_in * exp(-j*omega*tau)
        # Implemented as VCVS with frequency-dependent complex gain
        # This is an active element that can provide energy
        in_pos, in_neg, out_pos, out_neg = nodes
        idx = current_indices[name]
        tau = params.get(f"{name}_tau", 0.0)

        # Complex gain (frequency-dependent phase shift)
        gain = jnp.exp(-1j * omega * tau)

        # Output voltage equation
        if out_pos > 0:
            Y = Y.at[idx, out_pos - 1].add(1.0)
            Y = Y.at[out_pos - 1, idx].add(1.0)
        if out_neg > 0:
            Y = Y.at[idx, out_neg - 1].add(-1.0)
            Y = Y.at[out_neg - 1, idx].add(-1.0)

        # Input voltage contribution (with phase shift)
        if in_pos > 0:
            Y = Y.at[idx, in_pos - 1].add(-gain)
        if in_neg > 0:
            Y = Y.at[idx, in_neg - 1].add(gain)

    elif kind == "TLine":
        # Transmission line with characteristic impedance Z0 and delay tau
        # Y-parameter model:
        # Y11 = Y22 = -j*cot(omega*tau) / Z0
        # Y12 = Y21 = j*csc(omega*tau) / Z0
        n1_pos, n1_neg, n2_pos, n2_neg = nodes
        Z0 = params.get(f"{name}_Z0", 50.0)
        tau = params.get(f"{name}_tau", 1e-9)

        theta = omega * tau
        # Avoid singularities at theta = 0 and theta = n*pi
        theta_safe = jnp.where(jnp.abs(theta) < 1e-10, 1e-10, theta)

        y11 = -1j * jnp.cos(theta_safe) / (jnp.sin(theta_safe) * Z0)  # -j*cot/Z0
        y12 = 1j / (jnp.sin(theta_safe) * Z0)  # j*csc/Z0

        # Stamp Y parameters as 2-port network
        # Port 1: n1_pos - n1_neg
        # Port 2: n2_pos - n2_neg
        Y = _stamp_2port_admittance(Y, n1_pos, n1_neg, n2_pos, n2_neg, y11, y12, y12, y11)

    return Y, I


def _stamp_admittance(Y: Array, n1: int, n2: int, y) -> Array:
    """Stamp admittance y between nodes n1 and n2 into Y matrix."""
    # Skip if both nodes are ground (n=0)
    # MNA matrix excludes ground node (index 0)
    if n1 > 0:
        Y = Y.at[n1 - 1, n1 - 1].add(y)
    if n2 > 0:
        Y = Y.at[n2 - 1, n2 - 1].add(y)
    if n1 > 0 and n2 > 0:
        Y = Y.at[n1 - 1, n2 - 1].add(-y)
        Y = Y.at[n2 - 1, n1 - 1].add(-y)
    return Y


def _stamp_2port_admittance(Y: Array, p1_pos: int, p1_neg: int, p2_pos: int, p2_neg: int,
                            y11, y12, y21, y22) -> Array:
    """
    Stamp 2-port Y-parameters into the admittance matrix.

    Port 1: p1_pos - p1_neg
    Port 2: p2_pos - p2_neg

    I1 = y11*V1 + y12*V2
    I2 = y21*V1 + y22*V2
    """
    # Y11 stamps
    Y = _stamp_admittance(Y, p1_pos, p1_neg, y11)

    # Y22 stamps
    Y = _stamp_admittance(Y, p2_pos, p2_neg, y22)

    # Y12 cross-coupling (port 2 voltage affects port 1 current)
    if p1_pos > 0 and p2_pos > 0:
        Y = Y.at[p1_pos - 1, p2_pos - 1].add(y12)
    if p1_pos > 0 and p2_neg > 0:
        Y = Y.at[p1_pos - 1, p2_neg - 1].add(-y12)
    if p1_neg > 0 and p2_pos > 0:
        Y = Y.at[p1_neg - 1, p2_pos - 1].add(-y12)
    if p1_neg > 0 and p2_neg > 0:
        Y = Y.at[p1_neg - 1, p2_neg - 1].add(y12)

    # Y21 cross-coupling (port 1 voltage affects port 2 current)
    if p2_pos > 0 and p1_pos > 0:
        Y = Y.at[p2_pos - 1, p1_pos - 1].add(y21)
    if p2_pos > 0 and p1_neg > 0:
        Y = Y.at[p2_pos - 1, p1_neg - 1].add(-y21)
    if p2_neg > 0 and p1_pos > 0:
        Y = Y.at[p2_neg - 1, p1_pos - 1].add(-y21)
    if p2_neg > 0 and p1_neg > 0:
        Y = Y.at[p2_neg - 1, p1_neg - 1].add(y21)

    return Y
