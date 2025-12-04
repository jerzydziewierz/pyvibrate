"""
Test: Delay line (pure voltage delay).
"""
import pytest


def test_delay_line_basic():
    """DelayLine delays voltage by N timesteps."""
    from pyvibrate.timedomain import Network, R, VSource, DelayLine

    net = Network()
    net, v_in = net.node("v_in")
    net, v_out = net.node("v_out")

    net, vs = VSource(net, v_in, net.gnd, name="v_in")
    net, dly = DelayLine(net, v_in, net.gnd, v_out, net.gnd, delay_samples=5, name="dly")

    # Load resistor on output
    net, r1 = R(net, v_out, net.gnd, name="R1")

    sim = net.compile(dt=1e-6)
    params = {"v_in": 0.0, "R1": 1000.0}
    state = sim.init(params)

    # Step a few times with input at 0V
    controls = {"v_in": 0.0}
    for _ in range(5):
        state = sim.step(params, state, controls)

    assert abs(float(sim.v(state, v_out))) < 0.01, "Output should be 0V initially"

    # Set input to 5V and step once to make it appear in node_voltages
    controls = {"v_in": 5.0}
    state = sim.step(params, state, controls)  # 5V now in node_voltages, this is "time 0" for the delay

    # Step 4 more times - output should still be 0 (delay=5, so need 4 more after first)
    for i in range(4):
        state = sim.step(params, state, controls)
        assert abs(float(sim.v(state, v_out))) < 0.01, f"Output should still be 0V at step {i+1}"

    # Step once more - now the 5V should appear (total 5 steps of delay)
    state = sim.step(params, state, controls)
    assert abs(float(sim.v(state, v_out)) - 5.0) < 0.01, f"Output should be 5V after delay, got {sim.v(state, v_out):.2f}V"


def test_delay_line_tracks_signal():
    """DelayLine output follows input with constant delay."""
    from pyvibrate.timedomain import Network, R, VSource, DelayLine

    net = Network()
    net, v_in = net.node("v_in")
    net, v_out = net.node("v_out")

    net, vs = VSource(net, v_in, net.gnd, name="v_in")
    net, dly = DelayLine(net, v_in, net.gnd, v_out, net.gnd, delay_samples=3, name="dly")
    net, r1 = R(net, v_out, net.gnd, name="R1")

    sim = net.compile(dt=1e-6)
    params = {"v_in": 0.0, "R1": 1000.0}
    state = sim.init(params)

    # Build up history
    input_sequence = [0, 0, 0, 1, 2, 3, 2, 1, 0]
    outputs = []

    for v in input_sequence:
        controls = {"v_in": float(v)}
        state = sim.step(params, state, controls)
        outputs.append(float(sim.v(state, v_out)))

    # Output should be input delayed by 3 samples
    # inputs:  [0, 0, 0, 1, 2, 3, 2, 1, 0]
    # outputs: [0, 0, 0, 0, 0, 0, 1, 2, 3] (delayed by 3)
    expected = [0, 0, 0, 0, 0, 0, 1, 2, 3]

    for i, (out, exp) in enumerate(zip(outputs, expected)):
        assert abs(out - exp) < 0.01, f"Step {i}: expected {exp}V, got {out:.2f}V"


def test_delay_line_zero_delay():
    """DelayLine with delay=0 passes through immediately (with 1-step VCVS delay)."""
    from pyvibrate.timedomain import Network, R, VSource, DelayLine

    net = Network()
    net, v_in = net.node("v_in")
    net, v_out = net.node("v_out")

    net, vs = VSource(net, v_in, net.gnd, name="v_in")
    net, dly = DelayLine(net, v_in, net.gnd, v_out, net.gnd, delay_samples=0, name="dly")
    net, r1 = R(net, v_out, net.gnd, name="R1")

    sim = net.compile(dt=1e-6)
    params = {"v_in": 3.0, "R1": 1000.0}
    state = sim.init(params)
    controls = {"v_in": 3.0}

    # With delay=0, behaves like VCVS (1-step delay)
    state = sim.step(params, state, controls)
    state = sim.step(params, state, controls)

    assert abs(float(sim.v(state, v_out)) - 3.0) < 0.01, f"Expected 3V, got {sim.v(state, v_out):.2f}V"


def test_delay_line_differential():
    """DelayLine can delay differential signals."""
    from pyvibrate.timedomain import Network, R, VSource, DelayLine

    net = Network()
    net, v_pos = net.node("v_pos")
    net, v_neg = net.node("v_neg")
    net, v_out = net.node("v_out")

    # Differential input: 4V - 1V = 3V
    net, vs_pos = VSource(net, v_pos, net.gnd, name="v_pos")
    net, vs_neg = VSource(net, v_neg, net.gnd, name="v_neg")

    # Delay the differential voltage
    net, dly = DelayLine(net, v_pos, v_neg, v_out, net.gnd, delay_samples=2, name="dly")
    net, r1 = R(net, v_out, net.gnd, name="R1")

    sim = net.compile(dt=1e-6)
    params = {"v_pos": 4.0, "v_neg": 1.0, "R1": 1000.0}
    state = sim.init(params)
    controls = {"v_pos": 4.0, "v_neg": 1.0}

    # Step through delay
    for _ in range(4):
        state = sim.step(params, state, controls)

    # Should output 3V (the differential)
    assert abs(float(sim.v(state, v_out)) - 3.0) < 0.01, f"Expected 3V differential, got {sim.v(state, v_out):.2f}V"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
