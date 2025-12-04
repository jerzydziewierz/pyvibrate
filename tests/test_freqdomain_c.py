"""Tests for frequency-domain C (capacitor) component."""

import pytest
import jax.numpy as jnp
from jax import grad
import math


def test_c_impedance_is_negative_imaginary():
    """Capacitor impedance should be purely negative imaginary: Z = -j/(omega*C)."""
    from pyvibrate.frequencydomain import Network, C, ACSource

    net = Network()
    net, n1 = net.node("n1")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, c1 = C(net, n1, net.gnd, name="C1", value=1e-6)  # 1 uF

    solver = net.compile()

    freq = 1000.0  # 1 kHz
    omega = 2 * math.pi * freq
    sol = solver.solve_at(omega)

    z = solver.z_in(sol, vs)

    # Expected: Z = -j / (omega * C) = -j / (2*pi*1000 * 1e-6) = -j * 159.15 ohm
    expected_z = -1j / (omega * 1e-6)

    assert abs(z.real) < 1.0, f"Capacitor has real impedance: {z.real}"
    assert abs(z.imag - expected_z.imag) < 0.5, f"Capacitor impedance wrong: {z.imag} vs {expected_z.imag}"


def test_c_impedance_decreases_with_frequency():
    """Capacitor impedance magnitude decreases as frequency increases."""
    from pyvibrate.frequencydomain import Network, C, ACSource

    net = Network()
    net, n1 = net.node("n1")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, c1 = C(net, n1, net.gnd, name="C1", value=1e-6)

    solver = net.compile()

    freq_low = 100.0
    freq_high = 10000.0

    sol_low = solver.solve_at(2 * math.pi * freq_low)
    sol_high = solver.solve_at(2 * math.pi * freq_high)

    z_low = solver.z_in(sol_low, vs)
    z_high = solver.z_in(sol_high, vs)

    # Higher frequency should have lower impedance
    assert abs(z_high) < abs(z_low), f"|Z_high|={abs(z_high)} should be < |Z_low|={abs(z_low)}"
    # Factor of 100 in frequency -> factor of 100 in impedance
    ratio = abs(z_low) / abs(z_high)
    assert abs(ratio - 100.0) < 5.0, f"Impedance ratio should be ~100, got {ratio}"


def test_c_differentiable():
    """Test that capacitor value is differentiable."""
    from pyvibrate.frequencydomain import Network, C, ACSource

    net = Network()
    net, n1 = net.node("n1")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, c1 = C(net, n1, net.gnd, name="C1")  # No default

    solver = net.compile()
    omega = 2 * math.pi * 1000.0  # 1 kHz

    def get_impedance_magnitude(C_val):
        sol = solver.solve_at(omega, {"C1": C_val})
        z = solver.z_in(sol, vs)
        return jnp.abs(z)

    # |Z| = 1/(omega*C), so d|Z|/dC = -1/(omega*C^2)
    C_val = 1e-6
    dZ_dC = grad(get_impedance_magnitude)(C_val)

    expected_dZ_dC = -1.0 / (omega * C_val**2)
    assert abs(dZ_dC - expected_dZ_dC) / abs(expected_dZ_dC) < 0.01, \
        f"dZ/dC wrong: {dZ_dC} vs {expected_dZ_dC}"


def test_rc_series_impedance():
    """Test RC series combination: Z = R - j/(omega*C)."""
    from pyvibrate.frequencydomain import Network, R, C, ACSource

    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, r1 = R(net, n1, n2, name="R1", value=100.0)
    net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)

    solver = net.compile()

    freq = 1000.0
    omega = 2 * math.pi * freq
    sol = solver.solve_at(omega)

    z = solver.z_in(sol, vs)

    # Expected: Z = R - j/(omega*C)
    z_c = -1j / (omega * 1e-6)
    expected_z = 100.0 + z_c

    assert abs(z.real - expected_z.real) < 1.0, f"Real part wrong: {z.real} vs {expected_z.real}"
    assert abs(z.imag - expected_z.imag) < 1.0, f"Imag part wrong: {z.imag} vs {expected_z.imag}"


def test_rc_parallel_impedance():
    """Test RC parallel combination: 1/Z = 1/R + j*omega*C."""
    from pyvibrate.frequencydomain import Network, R, C, ACSource

    net = Network()
    net, n1 = net.node("n1")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, r1 = R(net, n1, net.gnd, name="R1", value=100.0)
    net, c1 = C(net, n1, net.gnd, name="C1", value=1e-6)

    solver = net.compile()

    freq = 1000.0
    omega = 2 * math.pi * freq
    sol = solver.solve_at(omega)

    z = solver.z_in(sol, vs)

    # Expected: Y = 1/R + j*omega*C, Z = 1/Y
    Y_expected = 1/100.0 + 1j * omega * 1e-6
    z_expected = 1.0 / Y_expected

    assert abs(z.real - z_expected.real) / abs(z_expected.real) < 0.01
    assert abs(z.imag - z_expected.imag) / abs(z_expected.imag) < 0.01


def test_c_phase_is_minus_90():
    """Capacitor voltage should lag current by 90 degrees."""
    from pyvibrate.frequencydomain import Network, C, ACSource

    net = Network()
    net, n1 = net.node("n1")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, c1 = C(net, n1, net.gnd, name="C1", value=1e-6)

    solver = net.compile()
    omega = 2 * math.pi * 1000.0
    sol = solver.solve_at(omega)

    z = solver.z_in(sol, vs)

    # Phase of impedance should be -90 degrees (-pi/2)
    phase = jnp.angle(z)
    expected_phase = -math.pi / 2

    assert abs(phase - expected_phase) < 0.01, f"Phase wrong: {phase} vs {expected_phase}"
