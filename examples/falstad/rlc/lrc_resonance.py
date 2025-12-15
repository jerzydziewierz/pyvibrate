"""
Example: Series RLC Resonance
Converted from Falstad: lrc.txt

Demonstrates series RLC circuit behavior:
- Resonant frequency: f_0 = 1 / (2 * pi * sqrt(L * C))
- Quality factor: Q = (1/R) * sqrt(L/C)
- Damping behavior: underdamped, critically damped, overdamped

Original Falstad values: R=10 ohm, L=1H, C=15uF
This gives f_0 ~ 41 Hz, Q ~ 25.8 (underdamped)

Components used: R, L, C, VSource, Switch
"""
import math
import jax.numpy as jnp
from jax import grad
from pyvibrate.timedomain import Network, R, L, C, VSource, Switch


def build_rlc_circuit():
    """Build series RLC circuit with switch.

    Circuit (from Falstad lrc.txt):
        +--[R]--[Switch]--+
        |                 |
       [V]               [L]
        |                 |
        +------[C]--------+
    """
    net = Network()
    net, n1 = net.node("n1")  # After R
    net, n2 = net.node("n2")  # After switch, top of L
    net, n3 = net.node("n3")  # Bottom of L, top of C

    # Voltage source
    net, vs = VSource(net, n1, n3, name="vs")
    # Series R
    net, r1 = R(net, n1, n2, name="R1")
    # Switch (to enable/trigger the circuit)
    net, sw = Switch(net, n2, n3, name="sw")
    # Inductor and capacitor in the loop
    net, l1 = L(net, n2, n3, name="L1")
    net, c1 = C(net, n3, n1, name="C1")

    nodes = {"n1": n1, "n2": n2, "n3": n3}
    components = {"vs": vs, "R1": r1, "sw": sw, "L1": l1, "C1": c1}
    return net, nodes, components


def build_simple_rlc():
    """Build simpler series RLC for step response.

    Circuit:
        Vs ---[R]---[L]---+--- GND
                         [C]
                          |
                         GND
    """
    net = Network()
    net, n_in = net.node("in")
    net, n_rl = net.node("rl")   # Between R and L
    net, n_mid = net.node("mid")  # Between L and C

    net, vs = VSource(net, n_in, net.gnd, name="vs")
    net, r1 = R(net, n_in, n_rl, name="R1")
    net, l1 = L(net, n_rl, n_mid, name="L1")
    net, c1 = C(net, n_mid, net.gnd, name="C1")

    return net, {"in": n_in, "rl": n_rl, "mid": n_mid}, {"vs": vs, "R1": r1, "L1": l1, "C1": c1}


def simulate_step_response(R_val=10.0, L_val=0.1, C_val=1e-5, V_step=5.0, n_periods=10):
    """Simulate step response showing resonance/damping behavior."""
    net, nodes, components = build_simple_rlc()

    # Calculate circuit parameters
    omega_0 = 1 / math.sqrt(L_val * C_val)
    f_0 = omega_0 / (2 * math.pi)
    period = 1 / f_0

    # Damping factor
    alpha = R_val / (2 * L_val)
    omega_d = math.sqrt(max(0, omega_0**2 - alpha**2))  # Damped frequency

    dt = period / 50  # 50 samples per period
    n_steps = int(n_periods * period / dt)

    sim = net.compile(dt=dt)
    params = {"vs": 0.0, "R1": R_val, "L1": L_val, "C1": C_val}
    state = sim.init(params)

    times = []
    v_cap = []  # Voltage across capacitor (same as v_mid in this topology)
    i_ind = []  # Current through inductor

    controls = {"vs": V_step}

    for i in range(n_steps):
        state = sim.step(params, state, controls)
        times.append(float(state.time))
        v_cap.append(float(sim.v(state, nodes["mid"])))
        i_ind.append(float(sim.i(state, components["L1"])))

    return times, v_cap, i_ind, {"f_0": f_0, "omega_0": omega_0, "alpha": alpha, "omega_d": omega_d}


def analyze_damping(R_val, L_val, C_val):
    """Analyze damping characteristics."""
    omega_0 = 1 / math.sqrt(L_val * C_val)
    alpha = R_val / (2 * L_val)
    Q = (1 / R_val) * math.sqrt(L_val / C_val)

    if alpha < omega_0:
        damping = "underdamped"
    elif alpha == omega_0:
        damping = "critically damped"
    else:
        damping = "overdamped"

    return {
        "omega_0": omega_0,
        "f_0": omega_0 / (2 * math.pi),
        "alpha": alpha,
        "Q": Q,
        "damping": damping,
    }


def demo_differentiability(R_val=10.0, L_val=0.1, C_val=1e-5):
    """Demonstrate JAX differentiability of RLC circuit."""

    # Use fixed simulation parameters (can't depend on traced L_param)
    dt = 1e-5  # Fixed small timestep
    n_steps = 500  # Fixed number of steps

    def final_voltage(L_param):
        """Final capacitor voltage as function of L."""
        net, nodes, _ = build_simple_rlc()

        sim = net.compile(dt=dt)
        params = {"vs": 0.0, "R1": R_val, "L1": L_param, "C1": C_val}
        state = sim.init(params)

        for _ in range(n_steps):
            state = sim.step(params, state, {"vs": 5.0})

        return sim.v(state, nodes["mid"])

    dV_dL = grad(final_voltage)
    gradient = float(dV_dL(L_val))
    return gradient


def main():
    print("=" * 60)
    print("Series RLC Resonance Example")
    print("Converted from Falstad: lrc.txt")
    print("=" * 60)

    # Parameters (modified from Falstad for clearer demonstration)
    R_val = 10.0     # 10 ohm
    L_val = 0.1      # 100 mH
    C_val = 1e-5     # 10 uF

    analysis = analyze_damping(R_val, L_val, C_val)

    print(f"\nCircuit Parameters:")
    print(f"   R = {R_val:.1f} ohm")
    print(f"   L = {L_val*1000:.1f} mH")
    print(f"   C = {C_val*1e6:.1f} uF")
    print(f"\nDerived Parameters:")
    print(f"   f_0 = {analysis['f_0']:.2f} Hz (resonant frequency)")
    print(f"   Q = {analysis['Q']:.2f} (quality factor)")
    print(f"   Damping: {analysis['damping']}")

    # Step response
    print("\n1. Step Response (V_step = 5V)")
    print("-" * 40)
    times, v_cap, i_ind, params = simulate_step_response(R_val, L_val, C_val, V_step=5.0, n_periods=10)

    # Find peaks
    v_max = max(v_cap)
    v_max_idx = v_cap.index(v_max)
    t_peak = times[v_max_idx]

    print(f"   First peak: V = {v_max:.4f} V at t = {t_peak*1000:.2f} ms")
    print(f"   Overshoot: {((v_max/5.0) - 1)*100:.1f}%")

    # Count oscillations (zero crossings above 2.5V threshold)
    crossings = 0
    for i in range(1, len(v_cap)):
        if (v_cap[i-1] < 2.5 and v_cap[i] >= 2.5) or (v_cap[i-1] >= 2.5 and v_cap[i] < 2.5):
            crossings += 1
    print(f"   Oscillations (crossings): {crossings // 2}")

    # Final value
    print(f"   Final voltage: {v_cap[-1]:.4f} V (expected: ~0V for series RLC)")

    # Compare different damping scenarios
    print("\n2. Damping Comparison")
    print("-" * 40)

    scenarios = [
        ("Underdamped (Q=10)", 10.0, 0.1, 1e-5),
        ("Critically damped", 200.0, 0.1, 1e-5),
        ("Overdamped", 500.0, 0.1, 1e-5),
    ]

    for name, R, L, C in scenarios:
        analysis = analyze_damping(R, L, C)
        times, v_cap, _, _ = simulate_step_response(R, L, C, V_step=5.0, n_periods=5)
        v_max = max(v_cap)
        print(f"   {name:25s}: peak = {v_max:.3f} V, Q = {analysis['Q']:.2f}")

    # Differentiability
    print("\n3. JAX Differentiability")
    print("-" * 40)
    grad_val = demo_differentiability(R_val, L_val, C_val)
    print(f"   dV_peak/dL: {grad_val:.4f} V/H")
    print(f"   (Gradient is finite: demonstrates differentiability)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
