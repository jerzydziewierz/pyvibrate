"""
Test: Voltage-controlled switch.
"""
import pytest


def test_voltage_switch_basic():
    """VoltageSwitch closes when control voltage exceeds threshold."""
    from pyvibrate.timedomain import Network, R, VSource, VoltageSwitch

    net = Network()
    net, v_ctrl = net.node("v_ctrl")
    net, n_out = net.node("n_out")
    net, v_supply = net.node("v_supply")

    # Control voltage source
    net, vs_ctrl = VSource(net, v_ctrl, net.gnd, name="v_ctrl")

    # Supply voltage
    net, vs_supply = VSource(net, v_supply, net.gnd, name="v_supply")

    # Switch controlled by v_ctrl, threshold at 2.5V
    net, vsw = VoltageSwitch(net, v_supply, n_out, v_ctrl, net.gnd, name="vsw")

    # Pulldown resistor
    net, r1 = R(net, n_out, net.gnd, name="R1")

    sim = net.compile(dt=1e-6)
    params = {"R1": 10000.0, "v_ctrl": 0.0, "v_supply": 5.0, "vsw_threshold": 2.5}
    state = sim.init(params)

    # Control at 0V - switch open, output low
    controls = {"v_ctrl": 0.0, "v_supply": 5.0}
    state = sim.step(params, state, controls)
    assert float(sim.v(state, n_out)) < 0.1, "Switch should be open at 0V control"

    # Control at 3V - above threshold, switch closes
    # Note: VoltageSwitch has one-step delay (reads previous voltages)
    controls = {"v_ctrl": 3.0, "v_supply": 5.0}
    state = sim.step(params, state, controls)  # v_ctrl now at 3V, but switch sees old value
    state = sim.step(params, state, controls)  # now switch sees 3V, closes
    assert float(sim.v(state, n_out)) > 4.9, "Switch should be closed at 3V control"

    # Control at 2V - below threshold, switch opens (with delay)
    controls = {"v_ctrl": 2.0, "v_supply": 5.0}
    state = sim.step(params, state, controls)
    state = sim.step(params, state, controls)
    assert float(sim.v(state, n_out)) < 0.1, "Switch should be open at 2V control"


def test_voltage_switch_inverted():
    """Inverted VoltageSwitch opens when voltage exceeds threshold."""
    from pyvibrate.timedomain import Network, R, VSource, VoltageSwitch

    net = Network()
    net, v_ctrl = net.node("v_ctrl")
    net, n_out = net.node("n_out")
    net, v_supply = net.node("v_supply")

    net, vs_ctrl = VSource(net, v_ctrl, net.gnd, name="v_ctrl")
    net, vs_supply = VSource(net, v_supply, net.gnd, name="v_supply")

    # Inverted: opens when control > threshold
    net, vsw = VoltageSwitch(net, v_supply, n_out, v_ctrl, net.gnd, name="vsw")

    net, r1 = R(net, n_out, net.gnd, name="R1")

    sim = net.compile(dt=1e-6)
    params = {"R1": 10000.0, "v_ctrl": 0.0, "v_supply": 5.0,
              "vsw_threshold": 2.5, "vsw_inverted": True}
    state = sim.init(params)

    # Control at 0V - below threshold, switch closed (inverted)
    # Note: at t=0, all voltages are 0, so inverted switch is closed
    controls = {"v_ctrl": 0.0, "v_supply": 5.0}
    state = sim.step(params, state, controls)
    assert float(sim.v(state, n_out)) > 4.9, "Inverted switch should be closed at 0V"

    # Control at 3V - above threshold, switch opens (inverted)
    controls = {"v_ctrl": 3.0, "v_supply": 5.0}
    state = sim.step(params, state, controls)  # propagation delay
    state = sim.step(params, state, controls)
    assert float(sim.v(state, n_out)) < 0.1, "Inverted switch should be open at 3V"


def test_voltage_switch_comparator():
    """Use VoltageSwitch as a simple comparator."""
    from pyvibrate.timedomain import Network, R, VSource, VoltageSwitch

    # Compare two voltages: when v_in > v_ref, output goes high
    net = Network()
    net, v_in = net.node("v_in")
    net, v_ref = net.node("v_ref")
    net, v_supply = net.node("v_supply")
    net, n_out = net.node("n_out")

    net, vs_in = VSource(net, v_in, net.gnd, name="v_in")
    net, vs_ref = VSource(net, v_ref, net.gnd, name="v_ref")
    net, vs_supply = VSource(net, v_supply, net.gnd, name="v_supply")

    # Switch closes when v_in > v_ref (threshold=0 for differential)
    net, comp = VoltageSwitch(net, v_supply, n_out, v_in, v_ref, name="comp")

    net, r1 = R(net, n_out, net.gnd, name="R1")

    sim = net.compile(dt=1e-6)
    params = {"R1": 10000.0, "v_in": 0.0, "v_ref": 2.5, "v_supply": 5.0, "comp_threshold": 0.0}
    state = sim.init(params)

    # v_in = 1V < v_ref = 2.5V -> output low (with propagation delay)
    controls = {"v_in": 1.0, "v_ref": 2.5, "v_supply": 5.0}
    state = sim.step(params, state, controls)
    state = sim.step(params, state, controls)
    assert float(sim.v(state, n_out)) < 0.1, "Output should be low when v_in < v_ref"

    # v_in = 3V > v_ref = 2.5V -> output high
    controls = {"v_in": 3.0, "v_ref": 2.5, "v_supply": 5.0}
    state = sim.step(params, state, controls)
    state = sim.step(params, state, controls)
    assert float(sim.v(state, n_out)) > 4.9, "Output should be high when v_in > v_ref"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
