"""Tests for frequency-domain L (inductor) component."""

import pytest
import jax.numpy as jnp
from jax import grad
import math


def test_l_impedance_is_positive_imaginary():
    """Inductor impedance should be purely positive imaginary: Z = j*omega*L."""
    from pyvibrate.frequencydomain import Network, L, ACSource

    net = Network()
    net, n1 = net.node("n1")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, l1 = L(net, n1, net.gnd, name="L1", value=1e-3)  # 1 mH

    solver = net.compile()

    freq = 1000.0  # 1 kHz
    omega = 2 * math.pi * freq
    sol = solver.solve_at(omega)

    z = solver.z_in(sol, vs)

    # Expected: Z = j * omega * L = j * 2*pi*1000 * 1e-3 = j * 6.28 ohm
    expected_z = 1j * omega * 1e-3

    assert abs(z.real) < 0.1, f"Inductor has real impedance: {z.real}"
    assert abs(z.imag - expected_z.imag) < 0.1, f"Inductor impedance wrong: {z.imag} vs {expected_z.imag}"


def test_l_impedance_increases_with_frequency():
    """Inductor impedance magnitude increases as frequency increases."""
    from pyvibrate.frequencydomain import Network, L, ACSource

    net = Network()
    net, n1 = net.node("n1")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, l1 = L(net, n1, net.gnd, name="L1", value=1e-3)

    solver = net.compile()

    freq_low = 100.0
    freq_high = 10000.0

    sol_low = solver.solve_at(2 * math.pi * freq_low)
    sol_high = solver.solve_at(2 * math.pi * freq_high)

    z_low = solver.z_in(sol_low, vs)
    z_high = solver.z_in(sol_high, vs)

    # Higher frequency should have higher impedance
    assert abs(z_high) > abs(z_low), f"|Z_high|={abs(z_high)} should be > |Z_low|={abs(z_low)}"
    # Factor of 100 in frequency -> factor of 100 in impedance
    ratio = abs(z_high) / abs(z_low)
    assert abs(ratio - 100.0) < 5.0, f"Impedance ratio should be ~100, got {ratio}"


def test_l_differentiable():
    """Test that inductor value is differentiable."""
    from pyvibrate.frequencydomain import Network, L, ACSource

    net = Network()
    net, n1 = net.node("n1")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, l1 = L(net, n1, net.gnd, name="L1")  # No default

    solver = net.compile()
    omega = 2 * math.pi * 1000.0  # 1 kHz

    def get_impedance_magnitude(L_val):
        sol = solver.solve_at(omega, {"L1": L_val})
        z = solver.z_in(sol, vs)
        return jnp.abs(z)

    # |Z| = omega*L, so d|Z|/dL = omega
    L_val = 1e-3
    dZ_dL = grad(get_impedance_magnitude)(L_val)

    expected_dZ_dL = omega
    assert abs(dZ_dL - expected_dZ_dL) / abs(expected_dZ_dL) < 0.01, \
        f"dZ/dL wrong: {dZ_dL} vs {expected_dZ_dL}"


def test_rl_series_impedance():
    """Test RL series combination: Z = R + j*omega*L."""
    from pyvibrate.frequencydomain import Network, R, L, ACSource

    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, r1 = R(net, n1, n2, name="R1", value=100.0)
    net, l1 = L(net, n2, net.gnd, name="L1", value=1e-3)

    solver = net.compile()

    freq = 1000.0
    omega = 2 * math.pi * freq
    sol = solver.solve_at(omega)

    z = solver.z_in(sol, vs)

    # Expected: Z = R + j*omega*L
    z_l = 1j * omega * 1e-3
    expected_z = 100.0 + z_l

    assert abs(z.real - expected_z.real) < 1.0, f"Real part wrong: {z.real} vs {expected_z.real}"
    assert abs(z.imag - expected_z.imag) < 1.0, f"Imag part wrong: {z.imag} vs {expected_z.imag}"


def test_l_phase_is_plus_90():
    """Inductor voltage should lead current by 90 degrees."""
    from pyvibrate.frequencydomain import Network, L, ACSource

    net = Network()
    net, n1 = net.node("n1")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, l1 = L(net, n1, net.gnd, name="L1", value=1e-3)

    solver = net.compile()
    omega = 2 * math.pi * 1000.0
    sol = solver.solve_at(omega)

    z = solver.z_in(sol, vs)

    # Phase of impedance should be +90 degrees (+pi/2)
    phase = jnp.angle(z)
    expected_phase = math.pi / 2

    assert abs(phase - expected_phase) < 0.01, f"Phase wrong: {phase} vs {expected_phase}"


def test_lc_resonance():
    """Test LC series resonance: at f_res, impedance is minimum (purely resistive at R=0)."""
    from pyvibrate.frequencydomain import Network, L, C, R, ACSource

    # Series LC with small R to avoid singular matrix
    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")
    net, n3 = net.node("n3")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, r1 = R(net, n1, n2, name="R1", value=1.0)  # Small series resistance
    net, l1 = L(net, n2, n3, name="L1", value=1e-3)  # 1 mH
    net, c1 = C(net, n3, net.gnd, name="C1", value=1e-6)  # 1 uF

    solver = net.compile()

    # Resonant frequency: f = 1/(2*pi*sqrt(LC)) = 1/(2*pi*sqrt(1e-3 * 1e-6)) = 5033 Hz
    f_res = 1 / (2 * math.pi * math.sqrt(1e-3 * 1e-6))
    omega_res = 2 * math.pi * f_res

    sol_res = solver.solve_at(omega_res)
    z_res = solver.z_in(sol_res, vs)

    # At resonance, inductive and capacitive reactances cancel
    # Z should be close to R (1 ohm)
    assert abs(z_res.imag) < 1.0, f"At resonance, imaginary part should be ~0: {z_res.imag}"
    assert abs(z_res.real - 1.0) < 0.5, f"At resonance, real part should be ~1 ohm: {z_res.real}"

    # Test off-resonance - impedance should be higher
    sol_low = solver.solve_at(omega_res / 2)
    sol_high = solver.solve_at(omega_res * 2)
    z_low = solver.z_in(sol_low, vs)
    z_high = solver.z_in(sol_high, vs)

    assert abs(z_low) > abs(z_res), "Impedance below resonance should be higher"
    assert abs(z_high) > abs(z_res), "Impedance above resonance should be higher"
