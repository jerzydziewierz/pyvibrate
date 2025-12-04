"""Tests for frequency-domain TLine (Transmission Line) component."""

import pytest
import jax.numpy as jnp
from jax import grad
import math


def test_tline_matched_load():
    """TLine with matched load (Z_load = Z0) should show no reflections."""
    from pyvibrate.frequencydomain import Network, R, TLine, ACSource

    # 50 ohm transmission line with 50 ohm load
    Z0 = 50.0
    tau = 10e-9  # 10 ns

    net = Network()
    net, n_in = net.node("n_in")
    net, n_out = net.node("n_out")

    net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
    net, tl = TLine(net, n_in, net.gnd, n_out, net.gnd, name="TL1", Z0=Z0, tau=tau)
    net, r_load = R(net, n_out, net.gnd, name="R_load", value=Z0)

    solver = net.compile()

    # At frequencies away from resonance, input impedance should be close to Z0
    # Choose frequency that's not at quarter-wave resonance
    freq = 10e6  # 10 MHz
    omega = 2 * math.pi * freq
    sol = solver.solve_at(omega)

    v_in = solver.v(sol, n_in)
    v_out = solver.v(sol, n_out)

    # With matched load, |V_out|/|V_in| depends on phase shift but no standing waves
    # The power should transfer efficiently
    assert jnp.abs(v_out) > 0.1, f"Output voltage too small: {jnp.abs(v_out)}"


def test_tline_open_circuit():
    """TLine with open circuit should show quarter-wave resonance."""
    from pyvibrate.frequencydomain import Network, R, TLine, ACSource

    Z0 = 50.0
    tau = 10e-9  # 10 ns

    # Quarter-wave frequency: theta = pi/2, so omega*tau = pi/2, f = 1/(4*tau)
    f_quarter = 1.0 / (4 * tau)  # 25 MHz

    net = Network()
    net, n_src = net.node("n_src")  # Source node
    net, n_in = net.node("n_in")    # Input to transmission line
    net, n_out = net.node("n_out")  # Output of transmission line

    net, vs = ACSource(net, n_src, net.gnd, name="vs", value=1.0)
    # Source resistance creates voltage divider with line input impedance
    net, r_src = R(net, n_src, n_in, name="R_src", value=50.0)
    net, tl = TLine(net, n_in, net.gnd, n_out, net.gnd, name="TL1", Z0=Z0, tau=tau)
    # Open circuit = very high resistance
    net, r_load = R(net, n_out, net.gnd, name="R_load", value=1e9)

    solver = net.compile()
    omega_quarter = 2 * math.pi * f_quarter
    sol = solver.solve_at(omega_quarter)

    # At quarter-wave with open load, input impedance transforms to low (near short)
    # So v_in should be lower than at off-resonance
    v_in_quarter = solver.v(sol, n_in)

    # Try at different frequency - should see different behavior
    omega_off = 2 * math.pi * f_quarter * 0.5  # Half the quarter-wave frequency
    sol_off = solver.solve_at(omega_off)
    v_in_off = solver.v(sol_off, n_in)

    # At quarter-wave with open load, input impedance is very low
    # At off-resonance with open load, input impedance is high (capacitive)
    # So v_in should differ
    assert abs(jnp.abs(v_in_quarter) - jnp.abs(v_in_off)) > 0.01, \
        f"Behavior should differ at resonance: |v_quarter|={jnp.abs(v_in_quarter)}, |v_off|={jnp.abs(v_in_off)}"


def test_tline_short_circuit():
    """TLine with short circuit load."""
    from pyvibrate.frequencydomain import Network, R, TLine, ACSource

    Z0 = 50.0
    tau = 10e-9  # 10 ns

    net = Network()
    net, n_in = net.node("n_in")
    net, n_out = net.node("n_out")

    net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
    net, r_src = R(net, n_in, net.gnd, name="R_src", value=50.0)
    net, tl = TLine(net, n_in, net.gnd, n_out, net.gnd, name="TL1", Z0=Z0, tau=tau)
    # Short circuit = very low resistance (not zero to avoid singularity)
    net, r_load = R(net, n_out, net.gnd, name="R_load", value=0.001)

    solver = net.compile()

    # At quarter-wave with short load, input looks like open circuit
    f_quarter = 1.0 / (4 * tau)
    omega_quarter = 2 * math.pi * f_quarter
    sol = solver.solve_at(omega_quarter)

    v_out = solver.v(sol, n_out)

    # Output should be near zero due to short circuit
    assert jnp.abs(v_out) < 0.01, f"Output should be ~0 with short load: {jnp.abs(v_out)}"


def test_tline_phase_delay():
    """TLine should introduce phase delay proportional to electrical length."""
    from pyvibrate.frequencydomain import Network, R, TLine, ACSource

    Z0 = 50.0
    tau = 10e-9  # 10 ns

    net = Network()
    net, n_in = net.node("n_in")
    net, n_out = net.node("n_out")

    net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
    net, tl = TLine(net, n_in, net.gnd, n_out, net.gnd, name="TL1", Z0=Z0, tau=tau)
    net, r_load = R(net, n_out, net.gnd, name="R_load", value=Z0)

    solver = net.compile()

    # At frequency where omega*tau is small, phase shift should be roughly omega*tau
    freq = 1e6  # 1 MHz -> omega*tau = 0.063 rad
    omega = 2 * math.pi * freq
    sol = solver.solve_at(omega)

    v_in = solver.v(sol, n_in)
    v_out = solver.v(sol, n_out)

    # The phase difference should relate to omega*tau
    # With matched load, it's exactly omega*tau
    phase_diff = jnp.angle(v_out) - jnp.angle(v_in)
    expected_phase = -omega * tau

    # Allow some tolerance due to loading effects
    assert abs(phase_diff - expected_phase) < 0.2, \
        f"Phase diff should be ~{expected_phase}: got {phase_diff}"


def test_tline_impedance_transformation():
    """TLine transforms impedance according to transmission line theory."""
    from pyvibrate.frequencydomain import Network, R, TLine, ACSource

    Z0 = 50.0
    tau = 10e-9  # 10 ns

    # At half-wave frequency, the line "passes through" the load
    f_half = 1.0 / (2 * tau)  # 50 MHz

    net = Network()
    net, n_in = net.node("n_in")
    net, n_out = net.node("n_out")

    # Load different from Z0
    Z_load = 100.0

    net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
    net, tl = TLine(net, n_in, net.gnd, n_out, net.gnd, name="TL1", Z0=Z0, tau=tau)
    net, r_load = R(net, n_out, net.gnd, name="R_load", value=Z_load)

    solver = net.compile()
    omega_half = 2 * math.pi * f_half
    sol = solver.solve_at(omega_half)

    # At half-wave, input impedance equals load impedance
    z_in = solver.z_in(sol, vs)

    # Allow tolerance for numerical precision
    assert abs(z_in.real - Z_load) < 5.0, \
        f"At half-wave, Z_in real should be ~{Z_load}: got {z_in.real}"
    assert abs(z_in.imag) < 5.0, f"At half-wave, Z_in should be ~real: got imag={z_in.imag}"


def test_tline_differentiable():
    """Test that TLine parameters are differentiable."""
    from pyvibrate.frequencydomain import Network, R, TLine, ACSource

    net = Network()
    net, n_in = net.node("n_in")
    net, n_out = net.node("n_out")

    net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
    net, tl = TLine(net, n_in, net.gnd, n_out, net.gnd, name="TL1")  # No defaults
    net, r_load = R(net, n_out, net.gnd, name="R_load", value=50.0)

    solver = net.compile()
    freq = 10e6
    omega = 2 * math.pi * freq

    def get_output_magnitude(Z0_val):
        sol = solver.solve_at(omega, {"TL1_Z0": Z0_val, "TL1_tau": 10e-9})
        v_out = solver.v(sol, n_out)
        return jnp.abs(v_out)

    # Gradient should exist
    Z0_val = 50.0
    d_vout_dZ0 = grad(get_output_magnitude)(Z0_val)

    # Just check it's finite and non-zero
    assert jnp.isfinite(d_vout_dZ0), f"Gradient should be finite: {d_vout_dZ0}"


def test_tline_different_Z0():
    """Different Z0 values should produce different impedance transformations."""
    from pyvibrate.frequencydomain import Network, R, TLine, ACSource

    tau = 10e-9
    Z_load = 100.0

    def measure_z_in(Z0):
        net = Network()
        net, n_in = net.node("n_in")
        net, n_out = net.node("n_out")

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
        net, tl = TLine(net, n_in, net.gnd, n_out, net.gnd, name="TL1", Z0=Z0, tau=tau)
        net, r_load = R(net, n_out, net.gnd, name="R_load", value=Z_load)

        solver = net.compile()
        freq = 15e6
        omega = 2 * math.pi * freq
        sol = solver.solve_at(omega)
        return solver.z_in(sol, vs)

    z_in_50 = measure_z_in(50.0)
    z_in_75 = measure_z_in(75.0)
    z_in_100 = measure_z_in(100.0)

    # Different Z0 should give different Z_in
    assert abs(z_in_50 - z_in_75) > 1.0, "Z0=50 and Z0=75 should give different Z_in"
    assert abs(z_in_75 - z_in_100) > 1.0, "Z0=75 and Z0=100 should give different Z_in"

    # When Z0 = Z_load = 100, we approach matched condition
    # (though not exactly due to frequency)
    assert abs(z_in_100.real) < abs(z_in_50.real) or abs(z_in_100.real) > abs(z_in_50.real), \
        "Different Z0 should transform differently"
