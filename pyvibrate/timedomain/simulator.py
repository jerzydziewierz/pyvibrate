"""
Vectorized, JIT-compatible time-domain simulator using MNA.

Key optimizations over simulator.py:
1. Pre-computed stamp patterns at compile time
2. Vectorized matrix assembly (no Python loops in step)
3. Array-based params/controls for JIT compatibility
4. jax.jit wrapped step function
"""

from __future__ import annotations
from typing import NamedTuple, Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from .network import Network, Node, ComponentSpec, ComponentRef


class SimState(NamedTuple):
    """
    Immutable simulation state (JAX pytree).
    """
    time: Array                # scalar
    node_voltages: Array       # (n_nodes,)
    cap_voltages: Array        # (n_caps,)
    cap_currents: Array        # (n_caps,)
    ind_currents: Array        # (n_inds,)
    ind_voltages: Array        # (n_inds,)
    delay_buffers: Array       # (total_buffer_size,) flattened delay line buffers
    delay_indices: Array       # (n_delays,) current index in each buffer


class CompiledNetwork(NamedTuple):
    """
    Pre-computed data structures for fast simulation.
    All indices and patterns computed at compile time.
    """
    # Matrix dimensions
    n_nodes: int
    n_total: int
    dt: float

    # Resistor stamps: patterns[i] is (n_total, n_total) matrix for resistor i
    # Resistors: G_total = sum(r_patterns * conductances[:, None, None])
    r_linear_idx: Array    # (n_resistors, 4) - flattened indices for 4 stamp entries
    r_signs: Array         # (n_resistors, 4) - signs for each entry (+1 or -1)
    r_param_idx: Array     # (n_resistors,) - index into params array
    n_resistors: int

    # Capacitor stamps (trapezoidal companion model)
    c_linear_idx: Array    # (n_caps, 4) - G matrix stamps (like resistor)
    c_signs: Array         # (n_caps, 4)
    c_b_idx: Array         # (n_caps, 2) - b vector stamps (current injection)
    c_b_signs: Array       # (n_caps, 2)
    c_node_a: Array        # (n_caps,) - positive node (for voltage extraction)
    c_node_b: Array        # (n_caps,) - negative node
    c_param_idx: Array     # (n_caps,) - index into params array
    n_caps: int

    # Inductor stamps (trapezoidal companion model)
    # Inductors add branch current as MNA variable
    l_g_linear_idx: Array  # (n_inds, 5) - G matrix: 4 coupling entries + 1 diagonal
    l_g_signs: Array       # (n_inds, 5)
    l_b_idx: Array         # (n_inds,) - b vector index for Veq
    l_mna_idx: Array       # (n_inds,) - MNA index for branch current
    l_node_a: Array        # (n_inds,) - positive node
    l_node_b: Array        # (n_inds,) - negative node
    l_param_idx: Array     # (n_inds,) - index into params array
    n_inds: int

    # Voltage source stamps
    vs_g_linear_idx: Array  # (n_vsources, 4) - G matrix coupling
    vs_g_signs: Array       # (n_vsources, 4)
    vs_b_idx: Array         # (n_vsources,) - b vector index
    vs_mna_idx: Array       # (n_vsources,) - MNA index for current
    vs_control_idx: Array   # (n_vsources,) - index into controls array (-1 if param)
    vs_param_idx: Array     # (n_vsources,) - index into params array (-1 if control)
    n_vsources: int

    # Switch stamps (resistor with dynamic conductance)
    sw_linear_idx: Array    # (n_switches, 4)
    sw_signs: Array         # (n_switches, 4)
    sw_r_on_idx: Array      # (n_switches,) - param index for r_on
    sw_r_off_idx: Array     # (n_switches,) - param index for r_off
    sw_control_idx: Array   # (n_switches,) - control index for closed state
    n_switches: int

    # Voltage-controlled switch stamps
    vsw_linear_idx: Array   # (n_vswitches, 4)
    vsw_signs: Array        # (n_vswitches, 4)
    vsw_r_on_idx: Array     # (n_vswitches,)
    vsw_r_off_idx: Array    # (n_vswitches,)
    vsw_threshold_idx: Array # (n_vswitches,)
    vsw_inverted_idx: Array # (n_vswitches,) - param index for inverted flag
    vsw_ctrl_pos: Array     # (n_vswitches,) - control positive node
    vsw_ctrl_neg: Array     # (n_vswitches,) - control negative node
    n_vswitches: int

    # VCVS stamps
    vcvs_g_linear_idx: Array  # (n_vcvs, 4)
    vcvs_g_signs: Array       # (n_vcvs, 4)
    vcvs_b_idx: Array         # (n_vcvs,)
    vcvs_ctrl_pos: Array      # (n_vcvs,)
    vcvs_ctrl_neg: Array      # (n_vcvs,)
    vcvs_gain_idx: Array      # (n_vcvs,) - param index
    n_vcvs: int

    # DelayLine stamps
    dl_g_linear_idx: Array  # (n_delays, 4)
    dl_g_signs: Array       # (n_delays, 4)
    dl_b_idx: Array         # (n_delays,)
    dl_in_pos: Array        # (n_delays,)
    dl_in_neg: Array        # (n_delays,)
    dl_buf_start: Array     # (n_delays,)
    dl_buf_size: Array      # (n_delays,)
    n_delays: int
    total_delay_buf: int

    # VCR (voltage-controlled resistor) stamps
    vcr_linear_idx: Array   # (n_vcrs, 4) - G matrix stamps (like resistor)
    vcr_signs: Array        # (n_vcrs, 4)
    vcr_r0_idx: Array       # (n_vcrs,) - param index for base resistance
    vcr_k_idx: Array        # (n_vcrs,) - param index for sensitivity
    vcr_ctrl_pos: Array     # (n_vcrs,) - control positive node
    vcr_ctrl_neg: Array     # (n_vcrs,) - control negative node
    n_vcrs: int

    # Name to index mappings (for user-friendly API)
    param_names: tuple[str, ...]
    control_names: tuple[str, ...]

    # Default values for params and controls
    param_defaults: tuple[float, ...]
    control_defaults: tuple[float, ...]

    # Component name to (kind, index_within_kind) for current probing
    # e.g., {"R1": ("R", 0), "L1": ("L", 0), "vs": ("VSource", 0)}
    component_index_map: dict[str, tuple[str, int]]

    # Original network for reference
    network: Network


class SimFns(NamedTuple):
    """Collection of pure simulation functions."""
    init: Callable[[dict], SimState]
    step: Callable[[dict, SimState, dict], SimState]
    step_arrays: Callable[[Array, SimState, Array], SimState]  # JIT-friendly version
    v: Callable[[SimState, Node], Array]
    i: Callable[[SimState, "ComponentRef"], Array]  # Current probe
    dt: float
    network: Network
    compiled: CompiledNetwork


def _compute_linear_idx(n_total: int, ia: int, ib: int) -> tuple[Array, Array]:
    """
    Compute linear indices and signs for resistor-style stamp.

    Resistor between nodes ia, ib (0 = ground):
    G[ia-1, ia-1] += g
    G[ib-1, ib-1] += g
    G[ia-1, ib-1] -= g
    G[ib-1, ia-1] -= g

    Returns (linear_indices, signs) where ground entries are marked with -1.
    """
    indices = []
    signs = []

    # Diagonal entries
    if ia > 0:
        indices.append((ia - 1) * n_total + (ia - 1))
        signs.append(1.0)
    else:
        indices.append(-1)
        signs.append(0.0)

    if ib > 0:
        indices.append((ib - 1) * n_total + (ib - 1))
        signs.append(1.0)
    else:
        indices.append(-1)
        signs.append(0.0)

    # Off-diagonal entries
    if ia > 0 and ib > 0:
        indices.append((ia - 1) * n_total + (ib - 1))
        signs.append(-1.0)
        indices.append((ib - 1) * n_total + (ia - 1))
        signs.append(-1.0)
    else:
        indices.append(-1)
        signs.append(0.0)
        indices.append(-1)
        signs.append(0.0)

    return jnp.array(indices, dtype=jnp.int32), jnp.array(signs)


def _compute_b_idx(ia: int, ib: int) -> tuple[Array, Array]:
    """Compute b vector indices and signs for current injection."""
    indices = []
    signs = []

    if ia > 0:
        indices.append(ia - 1)
        signs.append(1.0)
    else:
        indices.append(-1)
        signs.append(0.0)

    if ib > 0:
        indices.append(ib - 1)
        signs.append(-1.0)
    else:
        indices.append(-1)
        signs.append(0.0)

    return jnp.array(indices, dtype=jnp.int32), jnp.array(signs)


def _compute_vsource_idx(n_total: int, ip: int, in_: int, mna_idx: int) -> tuple[Array, Array]:
    """
    Compute linear indices for voltage source stamp.

    G[ip-1, mna_idx] = 1
    G[mna_idx, ip-1] = 1
    G[in-1, mna_idx] = -1
    G[mna_idx, in-1] = -1
    """
    indices = []
    signs = []

    if ip > 0:
        indices.append((ip - 1) * n_total + mna_idx)
        signs.append(1.0)
        indices.append(mna_idx * n_total + (ip - 1))
        signs.append(1.0)
    else:
        indices.append(-1)
        signs.append(0.0)
        indices.append(-1)
        signs.append(0.0)

    if in_ > 0:
        indices.append((in_ - 1) * n_total + mna_idx)
        signs.append(-1.0)
        indices.append(mna_idx * n_total + (in_ - 1))
        signs.append(-1.0)
    else:
        indices.append(-1)
        signs.append(0.0)
        indices.append(-1)
        signs.append(0.0)

    return jnp.array(indices, dtype=jnp.int32), jnp.array(signs)


def _compute_inductor_idx(n_total: int, ia: int, ib: int, mna_idx: int) -> tuple[Array, Array]:
    """
    Compute linear indices for inductor stamp.

    G[mna_idx, ia-1] = 1  (coupling)
    G[ia-1, mna_idx] = 1
    G[mna_idx, ib-1] = -1
    G[ib-1, mna_idx] = -1
    G[mna_idx, mna_idx] = -Req  (diagonal, sign handled separately)
    """
    indices = []
    signs = []

    if ia > 0:
        indices.append(mna_idx * n_total + (ia - 1))
        signs.append(1.0)
        indices.append((ia - 1) * n_total + mna_idx)
        signs.append(1.0)
    else:
        indices.append(-1)
        signs.append(0.0)
        indices.append(-1)
        signs.append(0.0)

    if ib > 0:
        indices.append(mna_idx * n_total + (ib - 1))
        signs.append(-1.0)
        indices.append((ib - 1) * n_total + mna_idx)
        signs.append(-1.0)
    else:
        indices.append(-1)
        signs.append(0.0)
        indices.append(-1)
        signs.append(0.0)

    # Diagonal entry (for -Req)
    indices.append(mna_idx * n_total + mna_idx)
    signs.append(-1.0)  # Will multiply by Req

    return jnp.array(indices, dtype=jnp.int32), jnp.array(signs)


def compile_network(net: Network, dt: float) -> SimFns:
    """
    Compile network into optimized simulation functions.

    All stamp patterns and indices computed here (once).
    The step function is JIT-compiled.
    """
    n_nodes = net.num_nodes

    # First pass: count components and assign MNA indices
    mna_extra_idx = n_nodes

    r_specs = []
    c_specs = []
    l_specs = []
    vs_specs = []
    sw_specs = []
    vsw_specs = []
    vcvs_specs = []
    dl_specs = []
    vcr_specs = []

    delay_buf_offset = 0

    for spec in net.components:
        if spec.kind == "R":
            r_specs.append(spec)
        elif spec.kind == "C":
            c_specs.append(spec)
        elif spec.kind == "L":
            l_specs.append((spec, mna_extra_idx))
            mna_extra_idx += 1
        elif spec.kind == "VSource":
            vs_specs.append((spec, mna_extra_idx))
            mna_extra_idx += 1
        elif spec.kind == "Switch":
            sw_specs.append(spec)
        elif spec.kind == "VoltageSwitch":
            vsw_specs.append(spec)
        elif spec.kind == "VCVS":
            vcvs_specs.append((spec, mna_extra_idx))
            mna_extra_idx += 1
        elif spec.kind == "DelayLine":
            buf_size = spec.nodes[4]
            dl_specs.append((spec, mna_extra_idx, delay_buf_offset, buf_size))
            delay_buf_offset += buf_size
            mna_extra_idx += 1
        elif spec.kind == "VCR":
            vcr_specs.append(spec)

    n_total = mna_extra_idx
    total_delay_buf = max(delay_buf_offset, 1)

    # Build name -> index mappings with default values
    param_names = []
    param_defaults = []
    control_names = []
    control_defaults = []
    param_idx_map = {}
    control_idx_map = {}

    def add_param(name: str, default: float = 0.0):
        param_idx_map[name] = len(param_names)
        param_names.append(name)
        param_defaults.append(default)

    def add_control(name: str, default: float = 0.0):
        control_idx_map[name] = len(control_names)
        control_names.append(name)
        control_defaults.append(default)

    def get_spec_default(spec: ComponentSpec, param_name: str, fallback: float) -> float:
        """Get default value from ComponentSpec.defaults, or use fallback."""
        for name, value in spec.defaults:
            if name == param_name:
                return value
        return fallback

    # Resistors: one param per resistor
    for spec in r_specs:
        default = get_spec_default(spec, spec.name, 0.0)
        add_param(spec.name, default)

    # Capacitors: one param per capacitor
    for spec in c_specs:
        default = get_spec_default(spec, spec.name, 0.0)
        add_param(spec.name, default)

    # Inductors: one param per inductor
    for spec, _ in l_specs:
        default = get_spec_default(spec, spec.name, 0.0)
        add_param(spec.name, default)

    # Voltage sources: control value (check spec.defaults for value)
    for spec, _ in vs_specs:
        default = get_spec_default(spec, spec.name, 0.0)
        add_control(spec.name, default)

    # Switches: r_on, r_off params; closed control
    for spec in sw_specs:
        add_param(f"{spec.name}_r_on", 1e-3)   # Default 1mΩ when closed
        add_param(f"{spec.name}_r_off", 1e6)   # Default 1MΩ when open
        add_control(spec.name, 0.0)  # Default open (False)

    # Voltage switches: r_on, r_off, threshold, inverted params
    for spec in vsw_specs:
        add_param(f"{spec.name}_r_on", 1e-3)
        add_param(f"{spec.name}_r_off", 1e6)
        add_param(f"{spec.name}_threshold", 0.0)
        add_param(f"{spec.name}_inverted", 0.0)  # 0.0 = False

    # VCVS: gain param
    for spec, _ in vcvs_specs:
        default = get_spec_default(spec, spec.name, 1.0)  # Default unity gain
        add_param(spec.name, default)

    # VCR: r0 and k params
    for spec in vcr_specs:
        r0_default = get_spec_default(spec, f"{spec.name}_r0", 1000.0)  # Default 1kΩ
        k_default = get_spec_default(spec, f"{spec.name}_k", 0.0)  # Default no sensitivity
        add_param(f"{spec.name}_r0", r0_default)
        add_param(f"{spec.name}_k", k_default)

    # Build stamp arrays
    # Resistors
    r_linear_idx_list = []
    r_signs_list = []
    r_param_idx_list = []

    for spec in r_specs:
        ia, ib = spec.nodes[0], spec.nodes[1]
        idx, signs = _compute_linear_idx(n_total, ia, ib)
        r_linear_idx_list.append(idx)
        r_signs_list.append(signs)
        r_param_idx_list.append(param_idx_map[spec.name])

    # Capacitors
    c_linear_idx_list = []
    c_signs_list = []
    c_b_idx_list = []
    c_b_signs_list = []
    c_node_a_list = []
    c_node_b_list = []
    c_param_idx_list = []

    for spec in c_specs:
        ia, ib = spec.nodes[0], spec.nodes[1]
        idx, signs = _compute_linear_idx(n_total, ia, ib)
        b_idx, b_signs = _compute_b_idx(ia, ib)
        c_linear_idx_list.append(idx)
        c_signs_list.append(signs)
        c_b_idx_list.append(b_idx)
        c_b_signs_list.append(b_signs)
        c_node_a_list.append(ia)
        c_node_b_list.append(ib)
        c_param_idx_list.append(param_idx_map[spec.name])

    # Inductors
    l_g_linear_idx_list = []
    l_g_signs_list = []
    l_b_idx_list = []
    l_mna_idx_list = []
    l_node_a_list = []
    l_node_b_list = []
    l_param_idx_list = []

    for spec, mna_idx in l_specs:
        ia, ib = spec.nodes[0], spec.nodes[1]
        idx, signs = _compute_inductor_idx(n_total, ia, ib, mna_idx)
        l_g_linear_idx_list.append(idx)
        l_g_signs_list.append(signs)
        l_b_idx_list.append(mna_idx)
        l_mna_idx_list.append(mna_idx)
        l_node_a_list.append(ia)
        l_node_b_list.append(ib)
        l_param_idx_list.append(param_idx_map[spec.name])

    # Voltage sources
    vs_g_linear_idx_list = []
    vs_g_signs_list = []
    vs_b_idx_list = []
    vs_mna_idx_list = []
    vs_control_idx_list = []

    for spec, mna_idx in vs_specs:
        ip, in_ = spec.nodes[0], spec.nodes[1]
        idx, signs = _compute_vsource_idx(n_total, ip, in_, mna_idx)
        vs_g_linear_idx_list.append(idx)
        vs_g_signs_list.append(signs)
        vs_b_idx_list.append(mna_idx)
        vs_mna_idx_list.append(mna_idx)
        vs_control_idx_list.append(control_idx_map[spec.name])

    # Switches
    sw_linear_idx_list = []
    sw_signs_list = []
    sw_r_on_idx_list = []
    sw_r_off_idx_list = []
    sw_control_idx_list = []

    for spec in sw_specs:
        ia, ib = spec.nodes[0], spec.nodes[1]
        idx, signs = _compute_linear_idx(n_total, ia, ib)
        sw_linear_idx_list.append(idx)
        sw_signs_list.append(signs)
        sw_r_on_idx_list.append(param_idx_map[f"{spec.name}_r_on"])
        sw_r_off_idx_list.append(param_idx_map[f"{spec.name}_r_off"])
        sw_control_idx_list.append(control_idx_map[spec.name])

    # Voltage switches
    vsw_linear_idx_list = []
    vsw_signs_list = []
    vsw_r_on_idx_list = []
    vsw_r_off_idx_list = []
    vsw_threshold_idx_list = []
    vsw_inverted_idx_list = []
    vsw_ctrl_pos_list = []
    vsw_ctrl_neg_list = []

    for spec in vsw_specs:
        ia, ib, cp, cn = spec.nodes[:4]
        idx, signs = _compute_linear_idx(n_total, ia, ib)
        vsw_linear_idx_list.append(idx)
        vsw_signs_list.append(signs)
        vsw_r_on_idx_list.append(param_idx_map[f"{spec.name}_r_on"])
        vsw_r_off_idx_list.append(param_idx_map[f"{spec.name}_r_off"])
        vsw_threshold_idx_list.append(param_idx_map[f"{spec.name}_threshold"])
        vsw_inverted_idx_list.append(param_idx_map[f"{spec.name}_inverted"])
        vsw_ctrl_pos_list.append(cp)
        vsw_ctrl_neg_list.append(cn)

    # VCVS
    vcvs_g_linear_idx_list = []
    vcvs_g_signs_list = []
    vcvs_b_idx_list = []
    vcvs_ctrl_pos_list = []
    vcvs_ctrl_neg_list = []
    vcvs_gain_idx_list = []

    for spec, mna_idx in vcvs_specs:
        op, on, cp, cn = spec.nodes[:4]
        idx, signs = _compute_vsource_idx(n_total, op, on, mna_idx)
        vcvs_g_linear_idx_list.append(idx)
        vcvs_g_signs_list.append(signs)
        vcvs_b_idx_list.append(mna_idx)
        vcvs_ctrl_pos_list.append(cp)
        vcvs_ctrl_neg_list.append(cn)
        vcvs_gain_idx_list.append(param_idx_map[spec.name])

    # Delay lines
    dl_g_linear_idx_list = []
    dl_g_signs_list = []
    dl_b_idx_list = []
    dl_in_pos_list = []
    dl_in_neg_list = []
    dl_buf_start_list = []
    dl_buf_size_list = []

    for spec, mna_idx, buf_start, buf_size in dl_specs:
        ip, in_, op, on = spec.nodes[:4]
        idx, signs = _compute_vsource_idx(n_total, op, on, mna_idx)
        dl_g_linear_idx_list.append(idx)
        dl_g_signs_list.append(signs)
        dl_b_idx_list.append(mna_idx)
        dl_in_pos_list.append(ip)
        dl_in_neg_list.append(in_)
        dl_buf_start_list.append(buf_start)
        dl_buf_size_list.append(buf_size)

    # VCRs (voltage-controlled resistors)
    vcr_linear_idx_list = []
    vcr_signs_list = []
    vcr_r0_idx_list = []
    vcr_k_idx_list = []
    vcr_ctrl_pos_list = []
    vcr_ctrl_neg_list = []

    for spec in vcr_specs:
        ia, ib, cp, cn = spec.nodes[:4]
        idx, signs = _compute_linear_idx(n_total, ia, ib)
        vcr_linear_idx_list.append(idx)
        vcr_signs_list.append(signs)
        vcr_r0_idx_list.append(param_idx_map[f"{spec.name}_r0"])
        vcr_k_idx_list.append(param_idx_map[f"{spec.name}_k"])
        vcr_ctrl_pos_list.append(cp)
        vcr_ctrl_neg_list.append(cn)

    # Convert to arrays (with padding for empty arrays)
    def _stack_or_empty(lst, shape):
        if lst:
            return jnp.stack(lst)
        return jnp.zeros(shape, dtype=jnp.int32 if len(shape) > 0 else jnp.float32)

    def _array_or_empty(lst, dtype=jnp.int32):
        if lst:
            return jnp.array(lst, dtype=dtype)
        return jnp.zeros((0,), dtype=dtype)

    # Build component index map for current probing
    component_index_map = {}
    for i, spec in enumerate(r_specs):
        component_index_map[spec.name] = ("R", i)
    for i, spec in enumerate(c_specs):
        component_index_map[spec.name] = ("C", i)
    for i, (spec, _) in enumerate(l_specs):
        component_index_map[spec.name] = ("L", i)
    for i, (spec, _) in enumerate(vs_specs):
        component_index_map[spec.name] = ("VSource", i)
    for i, spec in enumerate(sw_specs):
        component_index_map[spec.name] = ("Switch", i)
    for i, spec in enumerate(vsw_specs):
        component_index_map[spec.name] = ("VoltageSwitch", i)
    for i, (spec, _) in enumerate(vcvs_specs):
        component_index_map[spec.name] = ("VCVS", i)
    for i, (spec, _, _, _) in enumerate(dl_specs):
        component_index_map[spec.name] = ("DelayLine", i)
    for i, spec in enumerate(vcr_specs):
        component_index_map[spec.name] = ("VCR", i)

    compiled = CompiledNetwork(
        n_nodes=n_nodes,
        n_total=n_total,
        dt=dt,

        # Resistors
        r_linear_idx=_stack_or_empty(r_linear_idx_list, (0, 4)),
        r_signs=_stack_or_empty(r_signs_list, (0, 4)),
        r_param_idx=_array_or_empty(r_param_idx_list),
        n_resistors=len(r_specs),

        # Capacitors
        c_linear_idx=_stack_or_empty(c_linear_idx_list, (0, 4)),
        c_signs=_stack_or_empty(c_signs_list, (0, 4)),
        c_b_idx=_stack_or_empty(c_b_idx_list, (0, 2)),
        c_b_signs=_stack_or_empty(c_b_signs_list, (0, 2)),
        c_node_a=_array_or_empty(c_node_a_list),
        c_node_b=_array_or_empty(c_node_b_list),
        c_param_idx=_array_or_empty(c_param_idx_list),
        n_caps=len(c_specs),

        # Inductors
        l_g_linear_idx=_stack_or_empty(l_g_linear_idx_list, (0, 5)),
        l_g_signs=_stack_or_empty(l_g_signs_list, (0, 5)),
        l_b_idx=_array_or_empty(l_b_idx_list),
        l_mna_idx=_array_or_empty(l_mna_idx_list),
        l_node_a=_array_or_empty(l_node_a_list),
        l_node_b=_array_or_empty(l_node_b_list),
        l_param_idx=_array_or_empty(l_param_idx_list),
        n_inds=len(l_specs),

        # Voltage sources
        vs_g_linear_idx=_stack_or_empty(vs_g_linear_idx_list, (0, 4)),
        vs_g_signs=_stack_or_empty(vs_g_signs_list, (0, 4)),
        vs_b_idx=_array_or_empty(vs_b_idx_list),
        vs_mna_idx=_array_or_empty(vs_mna_idx_list),
        vs_control_idx=_array_or_empty(vs_control_idx_list),
        vs_param_idx=_array_or_empty([]),  # Not used - all voltage sources controlled
        n_vsources=len(vs_specs),

        # Switches
        sw_linear_idx=_stack_or_empty(sw_linear_idx_list, (0, 4)),
        sw_signs=_stack_or_empty(sw_signs_list, (0, 4)),
        sw_r_on_idx=_array_or_empty(sw_r_on_idx_list),
        sw_r_off_idx=_array_or_empty(sw_r_off_idx_list),
        sw_control_idx=_array_or_empty(sw_control_idx_list),
        n_switches=len(sw_specs),

        # Voltage switches
        vsw_linear_idx=_stack_or_empty(vsw_linear_idx_list, (0, 4)),
        vsw_signs=_stack_or_empty(vsw_signs_list, (0, 4)),
        vsw_r_on_idx=_array_or_empty(vsw_r_on_idx_list),
        vsw_r_off_idx=_array_or_empty(vsw_r_off_idx_list),
        vsw_threshold_idx=_array_or_empty(vsw_threshold_idx_list),
        vsw_inverted_idx=_array_or_empty(vsw_inverted_idx_list),
        vsw_ctrl_pos=_array_or_empty(vsw_ctrl_pos_list),
        vsw_ctrl_neg=_array_or_empty(vsw_ctrl_neg_list),
        n_vswitches=len(vsw_specs),

        # VCVS
        vcvs_g_linear_idx=_stack_or_empty(vcvs_g_linear_idx_list, (0, 4)),
        vcvs_g_signs=_stack_or_empty(vcvs_g_signs_list, (0, 4)),
        vcvs_b_idx=_array_or_empty(vcvs_b_idx_list),
        vcvs_ctrl_pos=_array_or_empty(vcvs_ctrl_pos_list),
        vcvs_ctrl_neg=_array_or_empty(vcvs_ctrl_neg_list),
        vcvs_gain_idx=_array_or_empty(vcvs_gain_idx_list),
        n_vcvs=len(vcvs_specs),

        # Delay lines
        dl_g_linear_idx=_stack_or_empty(dl_g_linear_idx_list, (0, 4)),
        dl_g_signs=_stack_or_empty(dl_g_signs_list, (0, 4)),
        dl_b_idx=_array_or_empty(dl_b_idx_list),
        dl_in_pos=_array_or_empty(dl_in_pos_list),
        dl_in_neg=_array_or_empty(dl_in_neg_list),
        dl_buf_start=_array_or_empty(dl_buf_start_list),
        dl_buf_size=_array_or_empty(dl_buf_size_list),
        n_delays=len(dl_specs),
        total_delay_buf=total_delay_buf,

        # VCRs
        vcr_linear_idx=_stack_or_empty(vcr_linear_idx_list, (0, 4)),
        vcr_signs=_stack_or_empty(vcr_signs_list, (0, 4)),
        vcr_r0_idx=_array_or_empty(vcr_r0_idx_list),
        vcr_k_idx=_array_or_empty(vcr_k_idx_list),
        vcr_ctrl_pos=_array_or_empty(vcr_ctrl_pos_list),
        vcr_ctrl_neg=_array_or_empty(vcr_ctrl_neg_list),
        n_vcrs=len(vcr_specs),

        # Mappings
        param_names=tuple(param_names),
        control_names=tuple(control_names),
        param_defaults=tuple(param_defaults),
        control_defaults=tuple(control_defaults),
        component_index_map=component_index_map,
        network=net,
    )

    # Create init function
    def init(params: dict) -> SimState:
        """Create initial state."""
        return SimState(
            time=jnp.array(0.0),
            node_voltages=jnp.zeros(n_nodes),
            cap_voltages=jnp.zeros(compiled.n_caps),
            cap_currents=jnp.zeros(compiled.n_caps),
            ind_currents=jnp.zeros(compiled.n_inds),
            ind_voltages=jnp.zeros(compiled.n_inds),
            delay_buffers=jnp.zeros(compiled.total_delay_buf),
            delay_indices=jnp.zeros(compiled.n_delays, dtype=jnp.int32),
        )

    # Create the JIT-compiled step function
    # Capture compiled network in closure - this is the key for JIT
    cn = compiled  # alias for clarity in the function below

    @jax.jit
    def step_arrays_impl(params: Array, state: SimState, controls: Array) -> SimState:
        """
        JIT-compiled step using arrays.

        The compiled network is captured in closure (compile-time constant).
        params, state, controls are dynamic (JAX traced).
        """
        n_total_val = cn.n_total
        n_nodes_val = cn.n_nodes
        dt_val = cn.dt

        # Initialize G and b as flat array (for scatter)
        G_flat = jnp.zeros(n_total_val * n_total_val)
        b = jnp.zeros(n_total_val)

        # --- Stamp resistors ---
        if cn.n_resistors > 0:
            conductances = 1.0 / params[cn.r_param_idx]  # (n_resistors,)
            # For each resistor, scatter conductance * sign to G_flat
            for i in range(cn.n_resistors):
                g = conductances[i]
                for j in range(4):
                    idx = cn.r_linear_idx[i, j]
                    sign = cn.r_signs[i, j]
                    G_flat = jnp.where(idx >= 0, G_flat.at[idx].add(g * sign), G_flat)

        # --- Stamp capacitors (trapezoidal companion) ---
        if cn.n_caps > 0:
            cap_vals = params[cn.c_param_idx]  # (n_caps,)
            geqs = 2.0 * cap_vals / dt_val
            ieqs = geqs * state.cap_voltages + state.cap_currents

            for i in range(cn.n_caps):
                geq = geqs[i]
                ieq = ieqs[i]

                # G stamps
                for j in range(4):
                    idx = cn.c_linear_idx[i, j]
                    sign = cn.c_signs[i, j]
                    G_flat = jnp.where(idx >= 0, G_flat.at[idx].add(geq * sign), G_flat)

                # b stamps
                for j in range(2):
                    idx = cn.c_b_idx[i, j]
                    sign = cn.c_b_signs[i, j]
                    b = jnp.where(idx >= 0, b.at[idx].add(ieq * sign), b)

        # --- Stamp inductors (trapezoidal companion) ---
        if cn.n_inds > 0:
            L_vals = params[cn.l_param_idx]  # (n_inds,)
            reqs = 2.0 * L_vals / dt_val
            veqs = reqs * state.ind_currents + state.ind_voltages

            for i in range(cn.n_inds):
                req = reqs[i]
                veq = veqs[i]

                # G stamps (4 coupling + 1 diagonal)
                for j in range(4):
                    idx = cn.l_g_linear_idx[i, j]
                    sign = cn.l_g_signs[i, j]
                    G_flat = jnp.where(idx >= 0, G_flat.at[idx].add(sign), G_flat)

                # Diagonal: -Req
                diag_idx = cn.l_g_linear_idx[i, 4]
                G_flat = G_flat.at[diag_idx].add(-req)

                # b stamp: -Veq
                b_idx = cn.l_b_idx[i]
                b = b.at[b_idx].add(-veq)

        # --- Stamp voltage sources ---
        if cn.n_vsources > 0:
            vs_values = controls[cn.vs_control_idx]  # (n_vsources,)

            for i in range(cn.n_vsources):
                v = vs_values[i]

                # G stamps
                for j in range(4):
                    idx = cn.vs_g_linear_idx[i, j]
                    sign = cn.vs_g_signs[i, j]
                    G_flat = jnp.where(idx >= 0, G_flat.at[idx].add(sign), G_flat)

                # b stamp
                b_idx = cn.vs_b_idx[i]
                b = b.at[b_idx].set(v)

        # --- Stamp switches ---
        if cn.n_switches > 0:
            r_ons = params[cn.sw_r_on_idx]
            r_offs = params[cn.sw_r_off_idx]
            closed = controls[cn.sw_control_idx]  # Should be 0.0 or 1.0

            rs = jnp.where(closed > 0.5, r_ons, r_offs)
            gs = 1.0 / rs

            for i in range(cn.n_switches):
                g = gs[i]
                for j in range(4):
                    idx = cn.sw_linear_idx[i, j]
                    sign = cn.sw_signs[i, j]
                    G_flat = jnp.where(idx >= 0, G_flat.at[idx].add(g * sign), G_flat)

        # --- Stamp voltage-controlled switches ---
        if cn.n_vswitches > 0:
            r_ons = params[cn.vsw_r_on_idx]
            r_offs = params[cn.vsw_r_off_idx]
            thresholds = params[cn.vsw_threshold_idx]
            inverted = params[cn.vsw_inverted_idx]

            # Get control voltages
            v_ctrl = jnp.zeros(cn.n_vswitches)
            for i in range(cn.n_vswitches):
                cp = cn.vsw_ctrl_pos[i]
                cn_node = cn.vsw_ctrl_neg[i]
                v_pos = jnp.where(cp > 0, state.node_voltages[cp - 1], 0.0)
                v_neg = jnp.where(cn_node > 0, state.node_voltages[cn_node - 1], 0.0)
                v_ctrl = v_ctrl.at[i].set(v_pos - v_neg)

            closed = v_ctrl > thresholds
            closed = jnp.where(inverted > 0.5, ~closed, closed)
            rs = jnp.where(closed, r_ons, r_offs)
            gs = 1.0 / rs

            for i in range(cn.n_vswitches):
                g = gs[i]
                for j in range(4):
                    idx = cn.vsw_linear_idx[i, j]
                    sign = cn.vsw_signs[i, j]
                    G_flat = jnp.where(idx >= 0, G_flat.at[idx].add(g * sign), G_flat)

        # --- Stamp VCVS ---
        if cn.n_vcvs > 0:
            gains = params[cn.vcvs_gain_idx]

            for i in range(cn.n_vcvs):
                cp = cn.vcvs_ctrl_pos[i]
                cn_node = cn.vcvs_ctrl_neg[i]
                v_pos = jnp.where(cp > 0, state.node_voltages[cp - 1], 0.0)
                v_neg = jnp.where(cn_node > 0, state.node_voltages[cn_node - 1], 0.0)
                v_out = gains[i] * (v_pos - v_neg)

                # G stamps
                for j in range(4):
                    idx = cn.vcvs_g_linear_idx[i, j]
                    sign = cn.vcvs_g_signs[i, j]
                    G_flat = jnp.where(idx >= 0, G_flat.at[idx].add(sign), G_flat)

                # b stamp
                b_idx = cn.vcvs_b_idx[i]
                b = b.at[b_idx].set(v_out)

        # --- Stamp delay lines ---
        new_delay_buffers = state.delay_buffers
        new_delay_indices = state.delay_indices

        if cn.n_delays > 0:
            for i in range(cn.n_delays):
                ip = cn.dl_in_pos[i]
                in_ = cn.dl_in_neg[i]
                v_in_pos = jnp.where(ip > 0, state.node_voltages[ip - 1], 0.0)
                v_in_neg = jnp.where(in_ > 0, state.node_voltages[in_ - 1], 0.0)
                v_in = v_in_pos - v_in_neg

                buf_start = cn.dl_buf_start[i]
                buf_size = cn.dl_buf_size[i]
                buf_idx = state.delay_indices[i]

                # Output from delay buffer (or passthrough if size <= 1)
                v_out = jnp.where(
                    buf_size > 1,
                    state.delay_buffers[buf_start + buf_idx],
                    v_in
                )

                # Update buffer
                new_delay_buffers = jnp.where(
                    buf_size > 1,
                    new_delay_buffers.at[buf_start + buf_idx].set(v_in),
                    new_delay_buffers
                )
                new_delay_indices = jnp.where(
                    buf_size > 1,
                    new_delay_indices.at[i].set((buf_idx + 1) % buf_size),
                    new_delay_indices
                )

                # G stamps
                for j in range(4):
                    idx = cn.dl_g_linear_idx[i, j]
                    sign = cn.dl_g_signs[i, j]
                    G_flat = jnp.where(idx >= 0, G_flat.at[idx].add(sign), G_flat)

                # b stamp
                b_idx = cn.dl_b_idx[i]
                b = b.at[b_idx].set(v_out)

        # --- Stamp VCRs (voltage-controlled resistors) ---
        if cn.n_vcrs > 0:
            r0_vals = params[cn.vcr_r0_idx]  # (n_vcrs,)
            k_vals = params[cn.vcr_k_idx]    # (n_vcrs,)

            for i in range(cn.n_vcrs):
                # Get control voltage from previous state
                cp = cn.vcr_ctrl_pos[i]
                cn_node = cn.vcr_ctrl_neg[i]
                v_pos = jnp.where(cp > 0, state.node_voltages[cp - 1], 0.0)
                v_neg = jnp.where(cn_node > 0, state.node_voltages[cn_node - 1], 0.0)
                v_ctrl = v_pos - v_neg

                # R = R0 + k * V_ctrl
                r = r0_vals[i] + k_vals[i] * v_ctrl
                g = 1.0 / r

                # G stamps (same as resistor)
                for j in range(4):
                    idx = cn.vcr_linear_idx[i, j]
                    sign = cn.vcr_signs[i, j]
                    G_flat = jnp.where(idx >= 0, G_flat.at[idx].add(g * sign), G_flat)

        # Reshape G and solve
        G = G_flat.reshape((n_total_val, n_total_val))
        x = jnp.linalg.solve(G, b)

        node_voltages = x[:n_nodes_val]

        # Update capacitor state
        if cn.n_caps > 0:
            cap_vals = params[cn.c_param_idx]
            geqs = 2.0 * cap_vals / dt_val

            va = jnp.where(cn.c_node_a > 0, node_voltages[cn.c_node_a - 1], 0.0)
            vb = jnp.where(cn.c_node_b > 0, node_voltages[cn.c_node_b - 1], 0.0)
            new_cap_voltages = va - vb
            new_cap_currents = geqs * (new_cap_voltages - state.cap_voltages) - state.cap_currents
        else:
            new_cap_voltages = jnp.zeros(0)
            new_cap_currents = jnp.zeros(0)

        # Update inductor state
        if cn.n_inds > 0:
            new_ind_currents = x[cn.l_mna_idx]
            va = jnp.where(cn.l_node_a > 0, node_voltages[cn.l_node_a - 1], 0.0)
            vb = jnp.where(cn.l_node_b > 0, node_voltages[cn.l_node_b - 1], 0.0)
            new_ind_voltages = va - vb
        else:
            new_ind_currents = jnp.zeros(0)
            new_ind_voltages = jnp.zeros(0)

        return SimState(
            time=state.time + dt_val,
            node_voltages=node_voltages,
            cap_voltages=new_cap_voltages,
            cap_currents=new_cap_currents,
            ind_currents=new_ind_currents,
            ind_voltages=new_ind_voltages,
            delay_buffers=new_delay_buffers,
            delay_indices=new_delay_indices,
        )

    def step_arrays(params: Array, state: SimState, controls: Array) -> SimState:
        """Step function with array inputs (JIT-friendly)."""
        return step_arrays_impl(params, state, controls)

    def step(params: dict, state: SimState, controls: dict) -> SimState:
        """
        Step function with dict inputs (user-friendly API).

        Converts dicts to arrays internally, using defaults for missing values.
        Controls can also be specified in params as fallback (for VSource compatibility).
        """
        # Convert params dict to array using defaults
        params_arr = jnp.array([
            params.get(name, default)
            for name, default in zip(compiled.param_names, compiled.param_defaults)
        ])

        # Convert controls dict to array using defaults
        # For controls, also check params as fallback (for VSource values)
        # Handle both float and bool values
        # Use jnp.asarray to preserve JAX tracers for differentiability
        controls_list = []
        for name, default in zip(compiled.control_names, compiled.control_defaults):
            # Check controls first, then params, then default
            if name in controls:
                val = controls[name]
            elif name in params:
                val = params[name]
            else:
                val = default
            if isinstance(val, bool):
                val = 1.0 if val else 0.0
            controls_list.append(val)
        controls_arr = jnp.asarray(controls_list, dtype=jnp.float32)

        return step_arrays(params_arr, state, controls_arr)

    def v(state: SimState, node: Node) -> Array:
        """Get voltage at a node."""
        if node.index == 0:
            return jnp.array(0.0)
        return state.node_voltages[node.index - 1]

    def i(state: SimState, component: ComponentRef) -> Array:
        """
        Get current through a component.

        Args:
            state: Current simulation state
            component: ComponentRef returned when creating the component

        Returns:
            Current in Amperes (positive direction depends on component type)

        Supported components:
            - L (Inductor): Returns stored branch current
            - C (Capacitor): Returns stored capacitor current
            - VSource: Returns MNA branch current
            - VCVS: Returns MNA branch current

        Not yet supported (would need params):
            - R, Switch, VoltageSwitch: Need V and R to compute I = V/R
        """
        kind, idx = cn.component_index_map[component.name]

        if kind == "L":
            return state.ind_currents[idx]
        elif kind == "C":
            return state.cap_currents[idx]
        elif kind == "VSource":
            # VSource current is computed during MNA solve but not stored in SimState.
            # Would need to add vs_currents array to SimState to support this.
            raise NotImplementedError(
                f"VSource current probing not yet implemented. "
                f"Use node voltage differences to estimate current through adjacent components."
            )
        elif kind == "VCVS":
            raise NotImplementedError(
                f"VCVS current probing not yet implemented."
            )
        elif kind in ("R", "Switch", "VoltageSwitch", "VCR"):
            raise NotImplementedError(
                f"{kind} current probing requires params (resistance). "
                f"Use I = (V_a - V_b) / R with sim.v() and your resistance value."
            )
        else:
            raise ValueError(f"Unknown component kind: {kind}")

    return SimFns(
        init=init,
        step=step,
        step_arrays=step_arrays,
        v=v,
        i=i,
        dt=dt,
        network=net,
        compiled=compiled,
    )
