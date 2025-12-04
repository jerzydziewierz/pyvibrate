"""
Test: Voltage-Controlled Voltage Source (VCVS).
"""
import pytest


def test_vcvs_unity_gain():
    """VCVS with gain=1 copies input voltage to output."""
    from pyvibrate.timedomain import Network, R, VSource, VCVS

    net = Network()
    net, v_in = net.node("v_in")
    net, v_out = net.node("v_out")

    # Input voltage source
    net, vs_in = VSource(net, v_in, net.gnd, name="v_in")

    # VCVS: output follows input with gain=1
    net, vcvs = VCVS(net, v_out, net.gnd, v_in, net.gnd, name="vcvs")

    # Load resistor (needed to define output current)
    net, r1 = R(net, v_out, net.gnd, name="R1")

    sim = net.compile(dt=1e-6)
    params = {"R1": 1000.0, "v_in": 3.0, "vcvs": 1.0}
    state = sim.init(params)
    controls = {"v_in": 3.0}

    # One-step delay for VCVS to read input
    state = sim.step(params, state, controls)
    state = sim.step(params, state, controls)

    v_output = float(sim.v(state, v_out))
    assert abs(v_output - 3.0) < 0.01, f"Expected 3.0V, got {v_output:.3f}V"


def test_vcvs_gain_10():
    """VCVS with gain=10 amplifies input."""
    from pyvibrate.timedomain import Network, R, VSource, VCVS

    net = Network()
    net, v_in = net.node("v_in")
    net, v_out = net.node("v_out")

    net, vs_in = VSource(net, v_in, net.gnd, name="v_in")
    net, vcvs = VCVS(net, v_out, net.gnd, v_in, net.gnd, name="vcvs")
    net, r1 = R(net, v_out, net.gnd, name="R1")

    sim = net.compile(dt=1e-6)
    params = {"R1": 1000.0, "v_in": 0.5, "vcvs": 10.0}
    state = sim.init(params)
    controls = {"v_in": 0.5}

    state = sim.step(params, state, controls)
    state = sim.step(params, state, controls)

    v_output = float(sim.v(state, v_out))
    assert abs(v_output - 5.0) < 0.01, f"Expected 5.0V, got {v_output:.3f}V"


def test_vcvs_differential_input():
    """VCVS reads differential voltage between two nodes."""
    from pyvibrate.timedomain import Network, R, VSource, VCVS

    net = Network()
    net, v_pos = net.node("v_pos")
    net, v_neg = net.node("v_neg")
    net, v_out = net.node("v_out")

    # Create differential input: 4V - 1V = 3V
    net, vs_pos = VSource(net, v_pos, net.gnd, name="v_pos")
    net, vs_neg = VSource(net, v_neg, net.gnd, name="v_neg")

    # VCVS with gain=2: output = 2 * (4 - 1) = 6V
    net, vcvs = VCVS(net, v_out, net.gnd, v_pos, v_neg, name="vcvs")
    net, r1 = R(net, v_out, net.gnd, name="R1")

    sim = net.compile(dt=1e-6)
    params = {"R1": 1000.0, "v_pos": 4.0, "v_neg": 1.0, "vcvs": 2.0}
    state = sim.init(params)
    controls = {"v_pos": 4.0, "v_neg": 1.0}

    state = sim.step(params, state, controls)
    state = sim.step(params, state, controls)

    v_output = float(sim.v(state, v_out))
    assert abs(v_output - 6.0) < 0.01, f"Expected 6.0V, got {v_output:.3f}V"


def test_vcvs_inverting():
    """VCVS with negative gain inverts input."""
    from pyvibrate.timedomain import Network, R, VSource, VCVS

    net = Network()
    net, v_in = net.node("v_in")
    net, v_out = net.node("v_out")

    net, vs_in = VSource(net, v_in, net.gnd, name="v_in")
    net, vcvs = VCVS(net, v_out, net.gnd, v_in, net.gnd, name="vcvs")
    net, r1 = R(net, v_out, net.gnd, name="R1")

    sim = net.compile(dt=1e-6)
    params = {"R1": 1000.0, "v_in": 2.0, "vcvs": -1.0}
    state = sim.init(params)
    controls = {"v_in": 2.0}

    state = sim.step(params, state, controls)
    state = sim.step(params, state, controls)

    v_output = float(sim.v(state, v_out))
    assert abs(v_output - (-2.0)) < 0.01, f"Expected -2.0V, got {v_output:.3f}V"


def test_vcvs_tracks_input_change():
    """VCVS output follows changing input (with one-step delay)."""
    from pyvibrate.timedomain import Network, R, VSource, VCVS

    net = Network()
    net, v_in = net.node("v_in")
    net, v_out = net.node("v_out")

    net, vs_in = VSource(net, v_in, net.gnd, name="v_in")
    net, vcvs = VCVS(net, v_out, net.gnd, v_in, net.gnd, name="vcvs")
    net, r1 = R(net, v_out, net.gnd, name="R1")

    sim = net.compile(dt=1e-6)
    params = {"R1": 1000.0, "v_in": 1.0, "vcvs": 2.0}
    state = sim.init(params)

    # Initial: 1V in -> 2V out
    controls = {"v_in": 1.0}
    state = sim.step(params, state, controls)
    state = sim.step(params, state, controls)
    assert abs(float(sim.v(state, v_out)) - 2.0) < 0.01

    # Change input to 3V -> output should become 6V
    controls = {"v_in": 3.0}
    state = sim.step(params, state, controls)
    state = sim.step(params, state, controls)
    assert abs(float(sim.v(state, v_out)) - 6.0) < 0.01, f"Expected 6.0V, got {sim.v(state, v_out):.3f}V"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
