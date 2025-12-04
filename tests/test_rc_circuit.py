"""
Test: RC circuit step response.

A voltage source charges a capacitor through a resistor.
V(t) = V0 * (1 - exp(-t / RC))

This validates:
- Network/Node construction (functional style)
- Resistor and Capacitor companion models
- MNA solver (JAX)
- Time stepping
"""
import math
import pytest


def test_rc_step_response():
    from pyvibrate.timedomain import Network, R, C, VSource

    # Circuit: Vs -- R -- C -- GND
    net = Network()
    net, n1 = net.node("n1")  # between Vs and R
    net, n2 = net.node("n2")  # between R and C

    net, vs = VSource(net, n1, net.gnd, name="vs")
    net, r1 = R(net, n1, n2, name="R1")
    net, c1 = C(net, n2, net.gnd, name="C1")

    # RC = 1ms, so at t=1ms we expect ~63.2% of final value
    # at t=5ms we expect ~99.3%
    R_val = 1000.0  # 1k ohm
    C_val = 1e-6    # 1 µF
    tau = R_val * C_val  # 1ms

    dt = 1e-6  # 1 µs timestep
    sim = net.compile(dt=dt)

    params = {"R1": R_val, "C1": C_val, "vs": 0.0}
    state = sim.init(params)

    # Step voltage to 5V
    controls = {"vs": 5.0}

    # Simulate for 5 tau
    n_steps = int(5 * tau / dt)

    v_at_1tau = None
    v_at_5tau = None

    for step in range(n_steps):
        state = sim.step(params, state, controls)
        t = float(state.time)

        if abs(t - tau) < dt / 2:
            v_at_1tau = float(sim.v(state, n2))
        if step == n_steps - 1:
            v_at_5tau = float(sim.v(state, n2))

    # Expected values
    expected_1tau = 5.0 * (1 - math.exp(-1))  # ~3.16V
    expected_5tau = 5.0 * (1 - math.exp(-5))  # ~4.97V

    # Allow 1% error (numerical integration isn't perfect)
    assert v_at_1tau is not None
    assert abs(v_at_1tau - expected_1tau) / expected_1tau < 0.01
    assert abs(v_at_5tau - expected_5tau) / expected_5tau < 0.01


def test_network_construction():
    """Basic smoke test for network building."""
    from pyvibrate.timedomain import Network, R, C, VSource

    net = Network()
    assert net.gnd is not None

    net, n1 = net.node("test_node")
    assert n1.name == "test_node"

    net, r = R(net, n1, net.gnd, name="R1")
    assert r.name == "R1"
    assert r.kind == "R"

    net, c = C(net, n1, net.gnd, name="C1")
    assert c.name == "C1"
    assert c.kind == "C"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
