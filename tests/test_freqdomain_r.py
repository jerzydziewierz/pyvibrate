"""Tests for frequency-domain R component."""

import pytest
import jax.numpy as jnp
from jax import grad
import math


def test_r_voltage_divider():
    """Test simple resistive voltage divider - impedance is frequency-independent."""
    from pyvibrate.frequencydomain import Network, R

    # Build voltage divider: ACSource -- R1 -- R2 -- GND
    net = Network()
    net, n1 = net.node("n1")  # source output
    net, n2 = net.node("n2")  # divider midpoint

    net, r1 = R(net, n1, n2, name="R1", value=1000.0)
    net, r2 = R(net, n2, net.gnd, name="R2", value=1000.0)

    # Add an AC source - need to add ACSource component first
    # For now, just test that the solver works with R components
    solver = net.compile()

    # At any frequency, R1 = R2 should give V2 = V1/2
    # But we need a source to drive it - let's test the impedance calculation


def test_r_impedance_is_real():
    """Resistor impedance should be real at all frequencies."""
    from pyvibrate.frequencydomain import Network, R
    from pyvibrate.frequencydomain.components import ACSource

    net = Network()
    net, n1 = net.node("n1")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, r1 = R(net, n1, net.gnd, name="R1", value=100.0)

    solver = net.compile()

    # Test at various frequencies
    for freq in [100.0, 1000.0, 10000.0, 100000.0]:
        omega = 2 * math.pi * freq
        sol = solver.solve_at(omega)

        # Get impedance
        z = solver.z_in(sol, vs)

        # Resistor impedance should be purely real
        assert abs(z.imag) < 1e-6, f"Resistor has imaginary impedance at {freq} Hz"
        assert abs(z.real - 100.0) < 1e-3, f"Resistor impedance wrong at {freq} Hz"


def test_r_differentiable():
    """Test that resistor value is differentiable."""
    from pyvibrate.frequencydomain import Network, R
    from pyvibrate.frequencydomain.components import ACSource

    net = Network()
    net, n1 = net.node("n1")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, r1 = R(net, n1, net.gnd, name="R1")  # No default, provided via params

    solver = net.compile()
    omega = 2 * math.pi * 1000.0  # 1 kHz

    def get_impedance_magnitude(R_val):
        sol = solver.solve_at(omega, {"R1": R_val})
        z = solver.z_in(sol, vs)
        return jnp.abs(z)

    # dZ/dR = 1 for a single resistor
    dZ_dR = grad(get_impedance_magnitude)(100.0)
    assert abs(dZ_dR - 1.0) < 1e-3, f"dZ/dR should be 1.0, got {dZ_dR}"


def test_r_series():
    """Test two resistors in series."""
    from pyvibrate.frequencydomain import Network, R
    from pyvibrate.frequencydomain.components import ACSource

    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, r1 = R(net, n1, n2, name="R1", value=100.0)
    net, r2 = R(net, n2, net.gnd, name="R2", value=200.0)

    solver = net.compile()
    omega = 2 * math.pi * 1000.0

    sol = solver.solve_at(omega)
    z = solver.z_in(sol, vs)

    # Series: Z_total = R1 + R2 = 300 ohm
    assert abs(z.real - 300.0) < 1e-3
    assert abs(z.imag) < 1e-6


def test_r_parallel():
    """Test two resistors in parallel."""
    from pyvibrate.frequencydomain import Network, R
    from pyvibrate.frequencydomain.components import ACSource

    net = Network()
    net, n1 = net.node("n1")

    net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
    net, r1 = R(net, n1, net.gnd, name="R1", value=100.0)
    net, r2 = R(net, n1, net.gnd, name="R2", value=100.0)

    solver = net.compile()
    omega = 2 * math.pi * 1000.0

    sol = solver.solve_at(omega)
    z = solver.z_in(sol, vs)

    # Parallel: Z_total = R1*R2/(R1+R2) = 50 ohm
    assert abs(z.real - 50.0) < 1e-3
    assert abs(z.imag) < 1e-6
