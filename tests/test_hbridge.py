"""
Test: H-bridge subcircuit.
"""
import pytest


def test_hbridge_drive_a_high_b_low():
    """Drive A to high, B to low - should see full supply across load."""
    from pyvibrate.timedomain import Network, R, VSource, HBridge, hbridge_drive_a_high

    net = Network()
    net, vcc = net.node("vcc")
    net, out_a = net.node("out_a")
    net, out_b = net.node("out_b")

    net, vs = VSource(net, vcc, net.gnd, name="vcc")
    net, hb = HBridge(net, vcc, net.gnd, out_a, out_b, prefix="hb")

    # Load resistor between outputs
    net, r_load = R(net, out_a, out_b, name="R_load")

    sim = net.compile(dt=1e-6)
    params = {"vcc": 12.0, "R_load": 100.0}
    state = sim.init(params)

    # Drive A high, B low
    controls = {"vcc": 12.0, **hbridge_drive_a_high(hb)}
    state = sim.step(params, state, controls)

    v_a = float(sim.v(state, out_a))
    v_b = float(sim.v(state, out_b))

    # A should be near 12V, B near 0V
    assert v_a > 11.9, f"Expected out_a near 12V, got {v_a:.2f}V"
    assert v_b < 0.1, f"Expected out_b near 0V, got {v_b:.2f}V"


def test_hbridge_drive_a_low_b_high():
    """Drive A to low, B to high - reversed polarity."""
    from pyvibrate.timedomain import Network, R, VSource, HBridge, hbridge_drive_b_high

    net = Network()
    net, vcc = net.node("vcc")
    net, out_a = net.node("out_a")
    net, out_b = net.node("out_b")

    net, vs = VSource(net, vcc, net.gnd, name="vcc")
    net, hb = HBridge(net, vcc, net.gnd, out_a, out_b, prefix="hb")
    net, r_load = R(net, out_a, out_b, name="R_load")

    sim = net.compile(dt=1e-6)
    params = {"vcc": 12.0, "R_load": 100.0}
    state = sim.init(params)

    controls = {"vcc": 12.0, **hbridge_drive_b_high(hb)}
    state = sim.step(params, state, controls)

    v_a = float(sim.v(state, out_a))
    v_b = float(sim.v(state, out_b))

    assert v_a < 0.1, f"Expected out_a near 0V, got {v_a:.2f}V"
    assert v_b > 11.9, f"Expected out_b near 12V, got {v_b:.2f}V"


def test_hbridge_all_off():
    """All switches off - outputs float to intermediate voltage via leakage divider."""
    from pyvibrate.timedomain import Network, R, VSource, HBridge, hbridge_all_off

    net = Network()
    net, vcc = net.node("vcc")
    net, out_a = net.node("out_a")
    net, out_b = net.node("out_b")

    net, vs = VSource(net, vcc, net.gnd, name="vcc")
    net, hb = HBridge(net, vcc, net.gnd, out_a, out_b, prefix="hb")

    # Small load resistor
    net, r_load = R(net, out_a, out_b, name="R_load")

    sim = net.compile(dt=1e-6)
    params = {"vcc": 12.0, "R_load": 100.0}
    state = sim.init(params)

    # All switches off - outputs float via leakage resistances
    # With symmetric leakage, outputs should be near mid-supply
    controls = {"vcc": 12.0, **hbridge_all_off(hb)}
    state = sim.step(params, state, controls)

    v_a = float(sim.v(state, out_a))
    v_b = float(sim.v(state, out_b))

    # Both outputs should be at intermediate voltage (leakage divider)
    # Not at supply rails
    assert 2.0 < v_a < 10.0, f"out_a should float to intermediate, got {v_a:.2f}V"
    assert 2.0 < v_b < 10.0, f"out_b should float to intermediate, got {v_b:.2f}V"

    # Voltages should be similar (symmetric network)
    assert abs(v_a - v_b) < 1.0, f"Outputs should be similar: {v_a:.2f}V vs {v_b:.2f}V"


def test_hbridge_freewheel_low():
    """Freewheel through low-side - both outputs to ground."""
    from pyvibrate.timedomain import Network, R, VSource, HBridge, hbridge_freewheel_low

    net = Network()
    net, vcc = net.node("vcc")
    net, out_a = net.node("out_a")
    net, out_b = net.node("out_b")

    net, vs = VSource(net, vcc, net.gnd, name="vcc")
    net, hb = HBridge(net, vcc, net.gnd, out_a, out_b, prefix="hb")
    net, r_load = R(net, out_a, out_b, name="R_load")

    sim = net.compile(dt=1e-6)
    params = {"vcc": 12.0, "R_load": 100.0}
    state = sim.init(params)

    controls = {"vcc": 12.0, **hbridge_freewheel_low(hb)}
    state = sim.step(params, state, controls)

    v_a = float(sim.v(state, out_a))
    v_b = float(sim.v(state, out_b))

    # Both should be near ground
    assert v_a < 0.1, f"Expected out_a near 0V, got {v_a:.2f}V"
    assert v_b < 0.1, f"Expected out_b near 0V, got {v_b:.2f}V"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
