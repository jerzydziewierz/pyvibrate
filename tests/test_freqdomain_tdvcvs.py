"""Tests for frequency-domain ConstantTimeDelayVCVS component.

TDVCVS = Time-Delay Voltage-Controlled Voltage Source

This is an ACTIVE element that implements a constant time delay with:
- Infinite input impedance (no loading)
- Zero output impedance (ideal voltage source)
- Frequency-dependent phase shift: phase = -omega * tau
- Can provide energy to the circuit
"""

import pytest
import jax.numpy as jnp
from jax import grad
import math


def test_tdvcvs_zero_delay():
    """With zero delay, output equals input."""
    from pyvibrate.frequencydomain import Network, R, ConstantTimeDelayVCVS, ACSource

    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, ps1 = ConstantTimeDelayVCVS(net, n1, net.gnd, n2, net.gnd, name="PS1", tau=0.0)
    net, r1 = R(net, n2, net.gnd, name="R1", value=100.0)

    solver = net.compile()

    freq = 1000.0
    omega = 2 * math.pi * freq
    sol = solver.solve_at(omega)

    v_in = solver.v(sol, n1)
    v_out = solver.v(sol, n2)

    # With zero delay, output should equal input
    assert abs(v_out - v_in) < 1e-6, f"V_out should equal V_in: {v_out} vs {v_in}"


def test_tdvcvs_quarter_wave():
    """At quarter wavelength delay, output is 90 degrees out of phase."""
    from pyvibrate.frequencydomain import Network, R, ConstantTimeDelayVCVS, ACSource

    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    freq = 1e6  # 1 MHz
    omega = 2 * math.pi * freq
    # Quarter wave delay: tau = T/4 = 1/(4*f) = 250 ns at 1 MHz
    tau_quarter = 1.0 / (4 * freq)

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, ps1 = ConstantTimeDelayVCVS(net, n1, net.gnd, n2, net.gnd, name="PS1", tau=tau_quarter)
    net, r1 = R(net, n2, net.gnd, name="R1", value=100.0)

    solver = net.compile()
    sol = solver.solve_at(omega)

    v_in = solver.v(sol, n1)
    v_out = solver.v(sol, n2)

    # Phase shift should be -90 degrees = -pi/2
    # exp(-j * omega * tau) = exp(-j * 2*pi*f * 1/(4f)) = exp(-j*pi/2) = -j
    phase_diff = jnp.angle(v_out) - jnp.angle(v_in)
    expected_phase_diff = -math.pi / 2

    assert abs(phase_diff - expected_phase_diff) < 0.01, \
        f"Phase diff should be -pi/2: {phase_diff} vs {expected_phase_diff}"


def test_tdvcvs_half_wave():
    """At half wavelength delay, output is 180 degrees out of phase (inverted)."""
    from pyvibrate.frequencydomain import Network, R, ConstantTimeDelayVCVS, ACSource

    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    freq = 1e6  # 1 MHz
    omega = 2 * math.pi * freq
    # Half wave delay: tau = T/2 = 1/(2*f) = 500 ns at 1 MHz
    tau_half = 1.0 / (2 * freq)

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, ps1 = ConstantTimeDelayVCVS(net, n1, net.gnd, n2, net.gnd, name="PS1", tau=tau_half)
    net, r1 = R(net, n2, net.gnd, name="R1", value=100.0)

    solver = net.compile()
    sol = solver.solve_at(omega)

    v_in = solver.v(sol, n1)
    v_out = solver.v(sol, n2)

    # Phase shift should be -180 degrees = -pi
    # V_out = V_in * exp(-j*pi) = -V_in
    # But phase wraps around, so we check magnitude and sign
    ratio = v_out / v_in
    assert abs(ratio + 1.0) < 0.01, f"V_out should be -V_in: ratio = {ratio}"


def test_tdvcvs_frequency_dependent():
    """Phase shift increases linearly with frequency for fixed delay."""
    from pyvibrate.frequencydomain import Network, R, ConstantTimeDelayVCVS, ACSource

    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    tau = 1e-6  # 1 us delay

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, ps1 = ConstantTimeDelayVCVS(net, n1, net.gnd, n2, net.gnd, name="PS1", tau=tau)
    net, r1 = R(net, n2, net.gnd, name="R1", value=100.0)

    solver = net.compile()

    # At different frequencies, phase shift should be -omega*tau
    for freq in [100e3, 200e3, 300e3]:
        omega = 2 * math.pi * freq
        sol = solver.solve_at(omega)

        v_in = solver.v(sol, n1)
        v_out = solver.v(sol, n2)

        phase_shift = jnp.angle(v_out / v_in)
        expected_phase = -omega * tau

        # Wrap expected phase to [-pi, pi]
        while expected_phase < -math.pi:
            expected_phase += 2 * math.pi
        while expected_phase > math.pi:
            expected_phase -= 2 * math.pi

        assert abs(phase_shift - expected_phase) < 0.05, \
            f"At {freq/1e3} kHz: phase = {phase_shift}, expected = {expected_phase}"


def test_tdvcvs_differentiable():
    """Test that delay parameter is differentiable."""
    from pyvibrate.frequencydomain import Network, R, ConstantTimeDelayVCVS, ACSource

    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, ps1 = ConstantTimeDelayVCVS(net, n1, net.gnd, n2, net.gnd, name="PS1")  # No default tau
    net, r1 = R(net, n2, net.gnd, name="R1", value=100.0)

    solver = net.compile()
    freq = 1e6
    omega = 2 * math.pi * freq

    def get_output_phase(tau_val):
        sol = solver.solve_at(omega, {"PS1_tau": tau_val})
        v_out = solver.v(sol, n2)
        return jnp.angle(v_out)

    # d(phase)/d(tau) = -omega
    tau_val = 100e-9  # 100 ns
    dphase_dtau = grad(get_output_phase)(tau_val)

    expected_deriv = -omega
    assert abs(dphase_dtau - expected_deriv) / abs(expected_deriv) < 0.01, \
        f"dphase/dtau should be -omega: {dphase_dtau} vs {expected_deriv}"


def test_tdvcvs_magnitude_unity():
    """ConstantTimeDelayVCVS should not change amplitude - only phase."""
    from pyvibrate.frequencydomain import Network, R, ConstantTimeDelayVCVS, ACSource

    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, ps1 = ConstantTimeDelayVCVS(net, n1, net.gnd, n2, net.gnd, name="PS1", tau=1e-6)
    net, r1 = R(net, n2, net.gnd, name="R1", value=100.0)

    solver = net.compile()

    for freq in [100e3, 500e3, 1e6]:
        omega = 2 * math.pi * freq
        sol = solver.solve_at(omega)

        v_in = solver.v(sol, n1)
        v_out = solver.v(sol, n2)

        # Magnitude ratio should be 1
        ratio = jnp.abs(v_out) / jnp.abs(v_in)
        assert abs(ratio - 1.0) < 0.01, f"Magnitude ratio should be 1: {ratio} at {freq/1e3} kHz"
