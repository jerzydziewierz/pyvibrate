"""
Test: Component values at construction time.

Demonstrates the hybrid API where values can be provided:
1. At construction time (as defaults)
2. At simulation time (overriding defaults)
"""
import pytest


def test_rc_with_construction_values():
    """Components with values specified at construction time."""
    from pyvibrate.timedomain import Network, R, C, VSource

    # Build circuit with explicit values at construction
    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)    # 5V
    net, r1 = R(net, n1, n2, name="R1", value=1000.0)            # 1 kΩ
    net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)         # 1 µF

    sim = net.compile(dt=1e-6)

    # No params needed - all values were provided at construction
    state = sim.init({})

    # Run for 1 time constant (1ms = 1000 steps)
    for _ in range(1000):
        state = sim.step({}, state, {})

    v_out = float(sim.v(state, n2))

    # At t = τ, V should be ~63.2% of 5V = ~3.16V
    expected = 5.0 * (1 - 2.718281828**-1)
    assert abs(v_out - expected) < 0.1, f"Expected ~{expected:.2f}V, got {v_out:.2f}V"


def test_rc_override_construction_values():
    """Override construction-time values at simulation time."""
    from pyvibrate.timedomain import Network, R, C, VSource

    # Build circuit with default values
    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)    # Default 5V
    net, r1 = R(net, n1, n2, name="R1", value=1000.0)            # Default 1 kΩ
    net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)         # Default 1 µF

    sim = net.compile(dt=1e-6)

    # Override R1 at simulation time to 2 kΩ (doubles τ)
    params = {"R1": 2000.0}  # Override: 2 kΩ instead of 1 kΩ
    state = sim.init(params)

    # Run for 2ms (which is now 1τ with R=2kΩ)
    for _ in range(2000):
        state = sim.step(params, state, {})

    v_out = float(sim.v(state, n2))

    # At t = τ (2ms with R=2kΩ, C=1µF), V should be ~63.2% of 5V
    expected = 5.0 * (1 - 2.718281828**-1)
    assert abs(v_out - expected) < 0.1, f"Expected ~{expected:.2f}V, got {v_out:.2f}V"


def test_mixed_construction_and_runtime_values():
    """Some values at construction, others at runtime."""
    from pyvibrate.timedomain import Network, R, C, VSource

    # Build circuit - only some values at construction
    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)  # 5V default
    net, r1 = R(net, n1, n2, name="R1")                        # No default - must provide
    net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)       # 1 µF default

    sim = net.compile(dt=1e-6)

    # Must provide R1, but vs and C1 use defaults
    params = {"R1": 1000.0}
    state = sim.init(params)

    # Run for 1τ
    for _ in range(1000):
        state = sim.step(params, state, {})

    v_out = float(sim.v(state, n2))
    expected = 5.0 * (1 - 2.718281828**-1)
    assert abs(v_out - expected) < 0.1, f"Expected ~{expected:.2f}V, got {v_out:.2f}V"


def test_voltage_source_override_via_controls():
    """VSource value can be overridden via controls dict."""
    from pyvibrate.timedomain import Network, R, C, VSource

    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)  # Default 5V
    net, r1 = R(net, n1, n2, name="R1", value=1000.0)
    net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)

    sim = net.compile(dt=1e-6)
    state = sim.init({})

    # Run with 10V instead of default 5V (via controls)
    controls = {"vs": 10.0}
    for _ in range(1000):
        state = sim.step({}, state, controls)

    v_out = float(sim.v(state, n2))

    # At t = τ, V should be ~63.2% of 10V = ~6.32V
    expected = 10.0 * (1 - 2.718281828**-1)
    assert abs(v_out - expected) < 0.2, f"Expected ~{expected:.2f}V, got {v_out:.2f}V"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
