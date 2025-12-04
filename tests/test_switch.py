"""
Test: Voltage-controlled switch.

Tests switch behavior in open and closed states.
"""
import pytest


def test_switch_open_blocks_current():
    """Open switch should have very high resistance, blocking current."""
    from pyvibrate.timedomain import Network, R, Switch, VSource

    # Circuit: Vs -- R -- Switch(open) -- GND
    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    net, vs = VSource(net, n1, net.gnd, name="vs")
    net, r1 = R(net, n1, n2, name="R1")
    net, sw = Switch(net, n2, net.gnd, name="sw")

    sim = net.compile(dt=1e-6)
    params = {"R1": 1000.0, "vs": 5.0}
    state = sim.init(params)
    controls = {"sw": False}  # open

    state = sim.step(params, state, controls)

    # With switch open, n2 should be at ~5V (tiny current through switch)
    v_n2 = float(sim.v(state, n2))
    assert v_n2 > 4.99, f"Expected ~5V with open switch, got {v_n2:.4f}V"


def test_switch_closed_conducts():
    """Closed switch should have very low resistance."""
    from pyvibrate.timedomain import Network, R, Switch, VSource

    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    net, vs = VSource(net, n1, net.gnd, name="vs")
    net, r1 = R(net, n1, n2, name="R1")
    net, sw = Switch(net, n2, net.gnd, name="sw")

    sim = net.compile(dt=1e-6)
    params = {"R1": 1000.0, "vs": 5.0}
    state = sim.init(params)
    controls = {"sw": True}  # closed

    state = sim.step(params, state, controls)

    # With switch closed, n2 should be at ~0V (grounded)
    v_n2 = float(sim.v(state, n2))
    assert v_n2 < 0.01, f"Expected ~0V with closed switch, got {v_n2:.4f}V"


def test_switch_control():
    """Switch state can be controlled via controls dict."""
    from pyvibrate.timedomain import Network, R, Switch, VSource

    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    net, vs = VSource(net, n1, net.gnd, name="vs")
    net, r1 = R(net, n1, n2, name="R1")
    net, sw = Switch(net, n2, net.gnd, name="sw")

    sim = net.compile(dt=1e-6)
    params = {"R1": 1000.0, "vs": 5.0}
    state = sim.init(params)

    # Initially open
    controls = {"sw": False}
    state = sim.step(params, state, controls)
    assert float(sim.v(state, n2)) > 4.5, "Switch should be open initially"

    # Close switch via control
    controls = {"sw": True}
    state = sim.step(params, state, controls)
    assert float(sim.v(state, n2)) < 0.5, "Switch should be closed after control"

    # Open switch via control
    controls = {"sw": False}
    state = sim.step(params, state, controls)
    assert float(sim.v(state, n2)) > 4.5, "Switch should be open after control"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
