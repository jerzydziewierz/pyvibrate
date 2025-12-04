"""
Test: JAX autodiff for circuit optimization.

Demonstrates:
1. Sensitivity analysis: dτ/dR for an RC circuit
2. Gradient descent: optimize R to achieve target τ
"""
import pytest
import jax
import jax.numpy as jnp
from jax import grad


def test_sensitivity_dV_dR():
    """
    Compute dV/dR using JAX autodiff.

    For an RC circuit at time t:
    V(t) = V0 * (1 - exp(-t / τ)) where τ = R*C

    At fixed time t, as R increases, τ increases, so V decreases.
    dV/dR = -V0 * exp(-t/τ) * t / (R^2 * C)
    """
    from pyvibrate.timedomain import Network, R, C, VSource

    # Circuit: Vs -- R1 -- C1 -- GND
    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    net, vs = VSource(net, n1, net.gnd, name="vs")
    net, r1 = R(net, n1, n2, name="R1")
    net, c1 = C(net, n2, net.gnd, name="C1")

    dt = 1e-6  # 1 µs timestep
    sim = net.compile(dt=dt)

    # Component values
    R_val = 1000.0    # 1 kΩ - this is the parameter we differentiate against
    C_val = 1e-6      # 1 µF
    V0 = 5.0          # 5 V input
    t_measure = 1e-3  # measure voltage at t = 1 ms

    def simulate_and_measure(R_param: float) -> float:
        """Run simulation with given R and return capacitor voltage at t=1ms."""
        # Explicit params dict showing all component values
        params = {
            "R1": R_param,  # Variable resistance (differentiation parameter)
            "C1": C_val,    # Fixed capacitance
        }
        state = sim.init(params)

        # Controls: voltage source value
        controls = {"vs": V0}

        n_steps = int(t_measure / dt)

        for _ in range(n_steps):
            state = sim.step(params, state, controls)

        return sim.v(state, n2)

    # Compute voltage and gradient at R = R_val
    V_measured = simulate_and_measure(R_val)
    dV_dR_fn = grad(simulate_and_measure)
    dV_dR = dV_dR_fn(R_val)

    # Analytical: dV/dR = -V0 * exp(-t/τ) * t / (R^2 * C)
    # As R increases, τ increases, charging is slower, V is lower => negative derivative
    tau = R_val * C_val
    dV_dR_analytical = -V0 * jnp.exp(-t_measure / tau) * t_measure / (R_val**2 * C_val)

    print(f"R = {R_val} Ω, C = {C_val*1e6:.1f} µF, t = {t_measure*1e3:.1f} ms")
    print(f"V(t) = {float(V_measured):.4f} V")
    print(f"dV/dR (JAX) = {float(dV_dR):.6e} V/Ω")
    print(f"dV/dR (analytical) = {float(dV_dR_analytical):.6e} V/Ω")

    # Check gradient is computable and correct
    assert jnp.isfinite(dV_dR), "Gradient should be finite"
    rel_error = abs(float(dV_dR) - float(dV_dR_analytical)) / abs(float(dV_dR_analytical))
    assert rel_error < 0.05, f"Gradient error too large: {rel_error*100:.1f}%"


def test_sensitivity_dTau_dR():
    """
    Compute dτ/dR using JAX autodiff.

    For an RC circuit, τ = R*C, so analytically dτ/dR = C.

    We measure τ by finding the time at which V reaches V0*(1-1/e) ≈ 0.632*V0.
    Since we simulate in discrete steps, we interpolate to find the crossing time.
    """
    from pyvibrate.timedomain import Network, R, C, VSource

    # Circuit: Vs -- R1 -- C1 -- GND
    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    net, vs = VSource(net, n1, net.gnd, name="vs")
    net, r1 = R(net, n1, n2, name="R1")
    net, c1 = C(net, n2, net.gnd, name="C1")

    dt = 1e-6  # 1 µs timestep
    sim = net.compile(dt=dt)

    # Component values
    R_val = 1000.0   # 1 kΩ - this is the parameter we differentiate against
    C_val = 1e-6     # 1 µF
    V0 = 5.0         # 5 V input

    target_fraction = 1.0 - jnp.exp(-1.0)  # ~0.632
    V_target = V0 * target_fraction

    def measure_tau(R_param: float) -> float:
        """
        Run simulation and estimate τ by finding when V crosses V_target.

        Returns estimated τ (time to reach 63.2% of V0).
        """
        # Explicit params dict showing all component values
        params = {
            "R1": R_param,  # Variable resistance (differentiation parameter)
            "C1": C_val,    # Fixed capacitance
        }
        state = sim.init(params)

        # Controls: voltage source value
        controls = {"vs": V0}

        # Run for 5*τ_expected to ensure we cross the threshold
        tau_expected = R_val * C_val
        n_steps = int(5 * tau_expected / dt)
        n_steps = max(n_steps, 100)  # at least 100 steps

        v_prev = 0.0
        t_prev = 0.0

        for step_i in range(n_steps):
            state = sim.step(params, state, controls)
            v_curr = sim.v(state, n2)
            t_curr = float(state.time)

            # Check if we crossed the threshold
            crossed = (v_prev < V_target) & (v_curr >= V_target)

            if crossed:
                # Linear interpolation to find crossing time
                # t_cross = t_prev + (V_target - v_prev) / (v_curr - v_prev) * dt
                frac = (V_target - v_prev) / (v_curr - v_prev + 1e-12)
                t_cross = t_prev + frac * dt
                return t_cross

            v_prev = v_curr
            t_prev = t_curr

        # If we didn't cross, return the last time (shouldn't happen)
        return t_curr

    # Compute tau and gradient at R = R_val
    tau_measured = measure_tau(R_val)
    dTau_dR_fn = grad(measure_tau)
    dTau_dR = dTau_dR_fn(R_val)

    # Analytical: τ = R*C, so dτ/dR = C
    tau_analytical = R_val * C_val
    dTau_dR_analytical = C_val

    print(f"R = {R_val} Ω, C = {C_val*1e6:.1f} µF")
    print(f"τ (measured) = {float(tau_measured)*1e3:.4f} ms")
    print(f"τ (analytical) = {tau_analytical*1e3:.4f} ms")
    print(f"dτ/dR (JAX) = {float(dTau_dR)*1e6:.4f} µs/Ω")
    print(f"dτ/dR (analytical) = {dTau_dR_analytical*1e6:.4f} µs/Ω")

    # Check tau is correct
    tau_error = abs(float(tau_measured) - tau_analytical) / tau_analytical
    assert tau_error < 0.02, f"Tau error too large: {tau_error*100:.1f}%"

    # Check gradient is correct
    assert jnp.isfinite(dTau_dR), "Gradient should be finite"
    grad_error = abs(float(dTau_dR) - dTau_dR_analytical) / dTau_dR_analytical
    assert grad_error < 0.05, f"Gradient error too large: {grad_error*100:.1f}%"


def test_gradient_descent_optimize_R():
    """
    Use gradient descent to find R that gives target τ = 2ms.

    Given: C = 1µF, target τ = 2ms
    Find: R such that measured τ = target τ
    Expected: R = τ/C = 2e-3 / 1e-6 = 2000 Ω

    Uses voltage-at-fixed-time as a differentiable proxy for τ optimization.
    At t = τ, V = V0*(1-1/e) ≈ 0.632*V0. So we target V(t_target) = V_target.
    """
    from pyvibrate.timedomain import Network, R, C, VSource

    # Circuit: Vs -- R1 -- C1 -- GND
    net = Network()
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")

    # Component values at construction time
    net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)  # 5V
    net, r1 = R(net, n1, n2, name="R1")                        # R to optimize
    net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)       # 1 µF

    dt = 1e-6  # 1 µs timestep
    sim = net.compile(dt=dt)

    # Optimization target
    V0 = 5.0
    C_val = 1e-6
    target_tau = 0.5e-3  # Target: τ = 0.5 ms (shorter for faster test)

    # At t = target_tau, if τ = target_tau, then V = V0*(1-1/e) ≈ 3.16V
    # If τ < target_tau (R too small), V > 3.16V (charged more)
    # If τ > target_tau (R too large), V < 3.16V (charged less)
    V_target = V0 * (1.0 - jnp.exp(-1.0))  # ~3.16V

    # Fixed number of steps to reach t = target_tau
    n_steps = int(target_tau / dt)  # 500 steps for 0.5ms

    def simulate_voltage_at_target_time(R_param: float) -> float:
        """
        Run simulation and return voltage at t = target_tau.

        This is differentiable w.r.t. R_param because:
        - Fixed number of steps (no data-dependent control flow)
        - All operations are JAX-compatible
        """
        params = {"R1": R_param}
        state = sim.init(params)

        for _ in range(n_steps):
            state = sim.step(params, state, {})

        return sim.v(state, n2)

    def loss(R_param: float) -> float:
        """
        Loss: (V_measured - V_target)^2

        When R is correct (τ = target_tau), V at t=target_tau equals V_target.
        """
        v = simulate_voltage_at_target_time(R_param)
        return (v - V_target) ** 2

    grad_loss = grad(loss)

    # Start at R = 1000 Ω (τ = 1ms, need to decrease to reach τ = 0.5ms)
    R_current = 1000.0

    # Learning rate tuning:
    # Use smaller learning rate for stable convergence
    learning_rate = 3e4

    print(f"Gradient descent to find R for τ = {target_tau*1e3:.1f}ms:")
    print(f"Target: V({target_tau*1e3:.1f}ms) = {float(V_target):.3f}V")
    print(f"Initial R = {R_current:.0f} Ω (τ = {R_current * C_val * 1e3:.1f} ms)")

    for iteration in range(15):
        v_measured = simulate_voltage_at_target_time(R_current)
        tau_effective = R_current * C_val
        l = loss(R_current)
        g = grad_loss(R_current)

        print(f"  Iter {iteration}: R = {R_current:.1f} Ω, τ = {tau_effective*1e3:.2f} ms, "
              f"V = {float(v_measured):.4f}V, loss = {float(l):.6f}, grad = {float(g):.2e}")

        if float(l) < 1e-6:  # V within ~1mV of target
            print("  Converged!")
            break

        R_current = R_current - learning_rate * float(g)

        # Keep R positive and bounded
        R_current = max(100.0, min(5000.0, R_current))

    R_expected = target_tau / C_val  # 500 Ω
    tau_achieved = R_current * C_val

    print(f"Final R = {R_current:.1f} Ω (expected: {R_expected:.0f} Ω)")
    print(f"Achieved τ = {tau_achieved*1e3:.4f} ms (target: {target_tau*1e3:.1f} ms)")

    # Check R within 10% of expected (relaxed for gradient descent convergence)
    R_error = abs(R_current - R_expected) / R_expected
    assert R_error < 0.10, f"R optimization error too large: {R_error*100:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
