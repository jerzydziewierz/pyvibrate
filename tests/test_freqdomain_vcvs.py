"""Tests for frequency-domain VCVS (Voltage-Controlled Voltage Source) component."""

import pytest
import jax.numpy as jnp
from jax import grad
import math


def test_vcvs_unity_gain():
    """VCVS with gain=1 should produce output equal to control voltage."""
    from pyvibrate.frequencydomain import Network, R, VCVS, ACSource

    net = Network()
    net, n_in = net.node("n_in")
    net, n_out = net.node("n_out")

    net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
    net, r_in = R(net, n_in, net.gnd, name="R_in", value=100.0)  # Input load
    net, e1 = VCVS(net, n_out, net.gnd, n_in, net.gnd, name="E1", gain=1.0)
    net, r_out = R(net, n_out, net.gnd, name="R_out", value=100.0)  # Output load

    solver = net.compile()
    omega = 2 * math.pi * 1000.0
    sol = solver.solve_at(omega)

    v_in = solver.v(sol, n_in)
    v_out = solver.v(sol, n_out)

    assert abs(v_out - v_in) < 1e-6, f"V_out should equal V_in: {v_out} vs {v_in}"


def test_vcvs_gain_10():
    """VCVS with gain=10 should amplify control voltage by 10."""
    from pyvibrate.frequencydomain import Network, R, VCVS, ACSource

    net = Network()
    net, n_in = net.node("n_in")
    net, n_out = net.node("n_out")

    net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
    net, r_in = R(net, n_in, net.gnd, name="R_in", value=100.0)
    net, e1 = VCVS(net, n_out, net.gnd, n_in, net.gnd, name="E1", gain=10.0)
    net, r_out = R(net, n_out, net.gnd, name="R_out", value=100.0)

    solver = net.compile()
    omega = 2 * math.pi * 1000.0
    sol = solver.solve_at(omega)

    v_in = solver.v(sol, n_in)
    v_out = solver.v(sol, n_out)

    expected_v_out = 10.0 * v_in
    assert abs(v_out - expected_v_out) < 1e-5, f"V_out should be 10*V_in: {v_out} vs {expected_v_out}"


def test_vcvs_negative_gain():
    """VCVS with negative gain should invert the signal."""
    from pyvibrate.frequencydomain import Network, R, VCVS, ACSource

    net = Network()
    net, n_in = net.node("n_in")
    net, n_out = net.node("n_out")

    net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
    net, r_in = R(net, n_in, net.gnd, name="R_in", value=100.0)
    net, e1 = VCVS(net, n_out, net.gnd, n_in, net.gnd, name="E1", gain=-1.0)
    net, r_out = R(net, n_out, net.gnd, name="R_out", value=100.0)

    solver = net.compile()
    omega = 2 * math.pi * 1000.0
    sol = solver.solve_at(omega)

    v_in = solver.v(sol, n_in)
    v_out = solver.v(sol, n_out)

    assert abs(v_out + v_in) < 1e-6, f"V_out should equal -V_in: {v_out} vs {-v_in}"


def test_vcvs_differential_input():
    """VCVS should respond to differential control voltage."""
    from pyvibrate.frequencydomain import Network, R, VCVS, ACSource

    net = Network()
    net, n_in_p = net.node("n_in_p")
    net, n_in_n = net.node("n_in_n")
    net, n_out = net.node("n_out")

    # Create a voltage divider for differential input
    net, vs = ACSource(net, n_in_p, net.gnd, name="vs", value=2.0)
    net, r1 = R(net, n_in_p, net.gnd, name="R1", value=100.0)
    # n_in_n at half the voltage
    net, vs2 = ACSource(net, n_in_n, net.gnd, name="vs2", value=1.0)
    net, r2 = R(net, n_in_n, net.gnd, name="R2", value=100.0)

    # VCVS with differential input: V_out = gain * (V_in_p - V_in_n)
    net, e1 = VCVS(net, n_out, net.gnd, n_in_p, n_in_n, name="E1", gain=2.0)
    net, r_out = R(net, n_out, net.gnd, name="R_out", value=100.0)

    solver = net.compile()
    omega = 2 * math.pi * 1000.0
    sol = solver.solve_at(omega)

    v_in_p = solver.v(sol, n_in_p)
    v_in_n = solver.v(sol, n_in_n)
    v_out = solver.v(sol, n_out)

    # V_out = 2.0 * (2.0 - 1.0) = 2.0
    expected_v_out = 2.0 * (v_in_p - v_in_n)
    assert abs(v_out - expected_v_out) < 1e-5, \
        f"V_out should be 2*(V_in_p - V_in_n): {v_out} vs {expected_v_out}"


def test_vcvs_frequency_independent():
    """VCVS gain should be constant across frequency."""
    from pyvibrate.frequencydomain import Network, R, VCVS, ACSource

    net = Network()
    net, n_in = net.node("n_in")
    net, n_out = net.node("n_out")

    net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
    net, r_in = R(net, n_in, net.gnd, name="R_in", value=100.0)
    net, e1 = VCVS(net, n_out, net.gnd, n_in, net.gnd, name="E1", gain=5.0)
    net, r_out = R(net, n_out, net.gnd, name="R_out", value=100.0)

    solver = net.compile()

    for freq in [100.0, 1000.0, 10000.0, 100000.0]:
        omega = 2 * math.pi * freq
        sol = solver.solve_at(omega)

        v_in = solver.v(sol, n_in)
        v_out = solver.v(sol, n_out)

        ratio = v_out / v_in
        assert abs(ratio - 5.0) < 1e-5, f"Gain should be 5.0 at {freq} Hz: {ratio}"


def test_vcvs_differentiable():
    """Test that VCVS gain is differentiable."""
    from pyvibrate.frequencydomain import Network, R, VCVS, ACSource

    net = Network()
    net, n_in = net.node("n_in")
    net, n_out = net.node("n_out")

    net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
    net, r_in = R(net, n_in, net.gnd, name="R_in", value=100.0)
    net, e1 = VCVS(net, n_out, net.gnd, n_in, net.gnd, name="E1")  # No default gain
    net, r_out = R(net, n_out, net.gnd, name="R_out", value=100.0)

    solver = net.compile()
    omega = 2 * math.pi * 1000.0

    def get_output_magnitude(gain_val):
        sol = solver.solve_at(omega, {"E1": gain_val})
        v_out = solver.v(sol, n_out)
        return jnp.abs(v_out)

    # With V_in fixed, d|V_out|/d(gain) = |V_in|
    gain_val = 5.0
    d_vout_d_gain = grad(get_output_magnitude)(gain_val)

    sol = solver.solve_at(omega, {"E1": gain_val})
    v_in_mag = jnp.abs(solver.v(sol, n_in))

    assert abs(d_vout_d_gain - v_in_mag) / v_in_mag < 0.01, \
        f"d|V_out|/d(gain) should equal |V_in|: {d_vout_d_gain} vs {v_in_mag}"
