"""
Test: RLC circuit resonance.

A series RLC circuit has resonant frequency f0 = 1/(2*pi*sqrt(LC)).
With low R, it should oscillate. With high R, it should be overdamped.

This validates:
- Inductor companion model
- RLC interaction
"""
import math
import pytest


def test_rlc_underdamped_oscillation():
    """Underdamped RLC should oscillate at approximately the resonant frequency."""
    from pyvibrate.timedomain import Network, R, C, L, VSource

    # Circuit: Vs -- L -- R -- C -- GND
    # Choose L=1mH, C=1µF -> f0 = 1/(2*pi*sqrt(1e-3 * 1e-6)) ≈ 5033 Hz
    # Period T0 ≈ 199 µs
    # Use small R for underdamped response

    net = Network()
    net, n1 = net.node("n1")  # after Vs
    net, n2 = net.node("n2")  # between L and R
    net, n3 = net.node("n3")  # between R and C (output)

    net, vs = VSource(net, n1, net.gnd, name="vs")
    net, l1 = L(net, n1, n2, name="L1")
    net, r1 = R(net, n2, n3, name="R1")
    net, c1 = C(net, n3, net.gnd, name="C1")

    L_val = 1e-3   # 1 mH
    R_val = 10.0   # 10 ohms (low for underdamped)
    C_val = 1e-6   # 1 µF

    # Expected resonant frequency
    f0 = 1.0 / (2.0 * math.pi * math.sqrt(L_val * C_val))
    T0 = 1.0 / f0

    # Simulate for ~3 periods with fine timestep
    dt = 1e-7  # 0.1 µs
    sim = net.compile(dt=dt)

    params = {"L1": L_val, "R1": R_val, "C1": C_val, "vs": 0.0}
    state = sim.init(params)
    controls = {"vs": 5.0}

    # Record voltage at capacitor
    n_steps = int(3 * T0 / dt)
    voltages = []
    times = []

    for step in range(n_steps):
        state = sim.step(params, state, controls)
        voltages.append(float(sim.v(state, n3)))
        times.append(float(state.time))

    # Find zero crossings to estimate oscillation period
    # (voltage crosses the DC level of 5V in steady state)
    crossings = []
    dc_level = 5.0
    for i in range(1, len(voltages)):
        if (voltages[i - 1] < dc_level <= voltages[i]) or \
           (voltages[i - 1] >= dc_level > voltages[i]):
            # Linear interpolation for crossing time
            t_cross = times[i - 1] + (dc_level - voltages[i - 1]) / (voltages[i] - voltages[i - 1]) * dt
            crossings.append(t_cross)

    # Need at least 2 crossings to estimate period (half-period between consecutive)
    assert len(crossings) >= 4, f"Expected oscillation, got only {len(crossings)} crossings"

    # Period is 2x the half-period
    half_periods = [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]
    measured_period = 2 * sum(half_periods) / len(half_periods)

    # Allow 5% error (damping shifts frequency slightly)
    assert abs(measured_period - T0) / T0 < 0.05, \
        f"Period {measured_period*1e6:.1f}µs differs from expected {T0*1e6:.1f}µs"


def test_rlc_overdamped():
    """Overdamped RLC should not oscillate."""
    from pyvibrate.timedomain import Network, R, C, L, VSource

    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")
    net, n3 = net.node("n3")

    net, vs = VSource(net, n1, net.gnd, name="vs")
    net, l1 = L(net, n1, n2, name="L1")
    net, r1 = R(net, n2, n3, name="R1")
    net, c1 = C(net, n3, net.gnd, name="C1")

    L_val = 1e-3
    R_val = 1000.0  # High R for overdamped
    C_val = 1e-6

    dt = 1e-6
    sim = net.compile(dt=dt)

    params = {"L1": L_val, "R1": R_val, "C1": C_val, "vs": 0.0}
    state = sim.init(params)
    controls = {"vs": 5.0}

    # Simulate and check for monotonic approach to 5V (no oscillation)
    prev_v = 0.0
    overshoots = 0

    for step in range(1000):
        state = sim.step(params, state, controls)
        v = float(sim.v(state, n3))
        if v > 5.0:
            overshoots += 1
        if v < prev_v - 0.01:  # significant decrease
            overshoots += 1
        prev_v = v

    # Overdamped should have minimal overshoot
    assert overshoots < 10, f"Expected overdamped (monotonic), got {overshoots} overshoots"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
