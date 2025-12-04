"""
Test: RL circuit step response.

A voltage source drives current through an inductor and resistor in series.
I(t) = (V/R) * (1 - exp(-t / τ))  where τ = L/R

This validates:
- Inductor current buildup
- Time constant τ = L/R
"""
import math
import pytest


def test_rl_current_buildup():
    """Current through RL circuit should follow exponential rise with τ = L/R."""
    from pyvibrate.timedomain import Network, R, L, VSource

    # Circuit: Vs -- L -- R -- GND
    # Choose L=10mH, R=100Ω -> τ = L/R = 0.1ms
    # Final current = V/R = 5/100 = 50mA

    net = Network()
    net, n1 = net.node("n1")  # after Vs
    net, n2 = net.node("n2")  # between L and R

    net, vs = VSource(net, n1, net.gnd, name="vs")
    net, l1 = L(net, n1, n2, name="L1")
    net, r1 = R(net, n2, net.gnd, name="R1")

    L_val = 10e-3  # 10 mH
    R_val = 100.0  # 100 Ω
    tau = L_val / R_val  # 0.1ms
    V = 5.0
    I_final = V / R_val  # 50 mA

    dt = 1e-7  # 0.1 µs timestep (need fine resolution for τ=100µs)
    sim = net.compile(dt=dt)

    params = {"L1": L_val, "R1": R_val, "vs": 0.0}
    state = sim.init(params)
    controls = {"vs": V}

    # Simulate for 5τ
    n_steps = int(5 * tau / dt)

    i_at_1tau = None
    i_at_5tau = None

    for step in range(n_steps):
        state = sim.step(params, state, controls)
        t = float(state.time)

        # Current through inductor
        i_L = float(state.ind_currents[0])

        if abs(t - tau) < dt / 2:
            i_at_1tau = i_L
        if step == n_steps - 1:
            i_at_5tau = i_L

    # Expected values: I(t) = I_final * (1 - exp(-t/τ))
    expected_1tau = I_final * (1 - math.exp(-1))  # ~31.6 mA
    expected_5tau = I_final * (1 - math.exp(-5))  # ~49.7 mA

    # Allow 2% error (trapezoidal integration has some error)
    assert i_at_1tau is not None, "Should have captured current at t=τ"
    rel_error_1tau = abs(i_at_1tau - expected_1tau) / expected_1tau
    rel_error_5tau = abs(i_at_5tau - expected_5tau) / expected_5tau

    assert rel_error_1tau < 0.02, \
        f"At t=τ: got {i_at_1tau*1000:.3f}mA, expected {expected_1tau*1000:.3f}mA (error={rel_error_1tau*100:.1f}%)"
    assert rel_error_5tau < 0.02, \
        f"At t=5τ: got {i_at_5tau*1000:.3f}mA, expected {expected_5tau*1000:.3f}mA (error={rel_error_5tau*100:.1f}%)"


def test_rl_voltage_across_inductor():
    """Voltage across inductor should decay exponentially."""
    from pyvibrate.timedomain import Network, R, L, VSource

    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    V = 5.0
    net, vs = VSource(net, n1, net.gnd, name="vs")
    net, l1 = L(net, n1, n2, name="L1")
    net, r1 = R(net, n2, net.gnd, name="R1")

    L_val = 10e-3
    R_val = 100.0
    tau = L_val / R_val  # 0.1ms
    dt = 1e-7
    sim = net.compile(dt=dt)

    params = {"L1": L_val, "R1": R_val, "vs": 0.0}
    state = sim.init(params)
    controls = {"vs": V}

    # At t=0+, all voltage is across inductor (V_L = V)
    # At t=∞, no voltage across inductor (V_L = 0, all across R)
    # V_L(t) = V * exp(-t/τ)

    v_L_at_1tau = None

    for step in range(int(2 * tau / dt)):
        state = sim.step(params, state, controls)
        t = float(state.time)

        # Voltage across inductor: V(n1) - V(n2)
        v_L = float(sim.v(state, n1) - sim.v(state, n2))

        if abs(t - tau) < dt / 2:
            v_L_at_1tau = v_L

    expected_1tau = V * math.exp(-1)  # ~1.84V
    assert v_L_at_1tau is not None
    rel_error = abs(v_L_at_1tau - expected_1tau) / expected_1tau
    assert rel_error < 0.02, \
        f"V_L at t=τ: got {v_L_at_1tau:.3f}V, expected {expected_1tau:.3f}V (error={rel_error*100:.1f}%)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
