"""
Example: Inductor Kick (Back-EMF)
Converted from Falstad: inductkick.txt

Demonstrates the voltage spike that occurs when current through an
inductor is suddenly interrupted. This is known as "inductor kick"
or "back-EMF" and is described by:

    V = L * di/dt

When a switch opens, di/dt becomes very large (ideally infinite),
causing a large voltage spike that can damage components.

Original Falstad circuit: V=5V, L=1H, R=100 ohm
The circuit includes a small capacitor (500pF) which provides a
path for the spike energy (in reality, parasitic capacitance).

Components used: R, L, C, VSource, Switch
"""
import math
import jax.numpy as jnp
from jax import grad
from pyvibrate.timedomain import Network, R, L, C, VSource, Switch


def build_inductor_kick_circuit():
    """Build inductor kick circuit with switch.

    Circuit:
                    L
        Vs ---+---[===]---+
              |           |
             [S]         [R]
              |           |
              +-----+-----+
                    |
                   [C]
                    |
                   GND

    When switch S opens, the inductor current must continue flowing.
    The only path is through R and C, causing a voltage spike.
    """
    net = Network()
    net, n_vs = net.node("vs")      # Voltage source output
    net, n_top = net.node("top")    # Top of circuit (after L)
    net, n_bot = net.node("bot")    # Bottom (before R)

    # Voltage source
    net, vs = VSource(net, n_vs, net.gnd, name="vs")

    # Inductor from source to top
    net, l1 = L(net, n_vs, n_top, name="L1")

    # Switch in parallel path (allows current to bypass R when closed)
    net, sw = Switch(net, n_vs, n_bot, name="sw")

    # Resistor from top to bottom
    net, r1 = R(net, n_top, n_bot, name="R1")

    # Small capacitor to ground (represents parasitic/snubber capacitance)
    net, c1 = C(net, n_bot, net.gnd, name="C1")

    nodes = {"vs": n_vs, "top": n_top, "bot": n_bot}
    components = {"vs": vs, "L1": l1, "sw": sw, "R1": r1, "C1": c1}
    return net, nodes, components


def simulate_inductor_kick(V_supply=5.0, L_val=0.1, R_val=100.0, C_val=1e-9,
                           t_switch=1e-3):
    """Simulate inductor kick when switch opens.

    Args:
        V_supply: Supply voltage
        L_val: Inductance in henries
        R_val: Resistance in ohms
        C_val: Capacitance in farads (snubber/parasitic)
        t_switch: Time at which switch opens

    Returns:
        times: List of time values
        v_inductor: Voltage across inductor
        i_inductor: Current through inductor
        v_cap: Voltage across capacitor (spike voltage)
    """
    net, nodes, components = build_inductor_kick_circuit()

    # Simulation parameters
    # Time constant for current buildup: tau = L/R
    tau = L_val / R_val

    # Use small timestep to capture fast transients
    dt = min(tau / 100, 1e-6)
    t_total = t_switch + 5 * tau  # Run past switch opening
    n_steps = int(t_total / dt)

    sim = net.compile(dt=dt)
    params = {"vs": 0.0, "L1": L_val, "R1": R_val, "C1": C_val}
    state = sim.init(params)

    times = []
    v_inductor = []
    i_inductor = []
    v_cap = []

    for i in range(n_steps):
        t = float(state.time)

        # Switch is closed initially, opens at t_switch
        switch_closed = t < t_switch

        controls = {
            "vs": V_supply,
            "sw": switch_closed,
        }

        state = sim.step(params, state, controls)

        times.append(t)
        # Voltage across inductor = V_vs - V_top
        v_l = float(sim.v(state, nodes["vs"])) - float(sim.v(state, nodes["top"]))
        v_inductor.append(v_l)
        i_inductor.append(float(sim.i(state, components["L1"])))
        v_cap.append(float(sim.v(state, nodes["bot"])))

    return times, v_inductor, i_inductor, v_cap


def analyze_kick(times, v_cap, t_switch):
    """Analyze the voltage spike characteristics."""
    # Find values before and after switch opens
    switch_idx = min(range(len(times)), key=lambda i: abs(times[i] - t_switch))

    v_before = v_cap[switch_idx - 1] if switch_idx > 0 else v_cap[0]

    # Find peak after switch opens
    v_after = v_cap[switch_idx:]
    if v_after:
        v_peak = max(abs(v) for v in v_after)
        peak_idx = switch_idx + max(range(len(v_after)), key=lambda i: abs(v_after[i]))
        t_peak = times[peak_idx]
    else:
        v_peak = v_before
        t_peak = t_switch

    return {
        "v_before": v_before,
        "v_peak": v_peak,
        "t_peak": t_peak,
        "spike_ratio": v_peak / abs(v_before) if v_before != 0 else 0,
    }


def demo_differentiability(L_val=0.1, R_val=100.0):
    """Demonstrate JAX differentiability of inductor kick circuit."""

    def peak_voltage(L_param):
        """Peak capacitor voltage as function of L."""
        net, nodes, components = build_inductor_kick_circuit()

        dt = 1e-6
        n_steps = 500

        sim = net.compile(dt=dt)
        params = {"vs": 0.0, "L1": L_param, "R1": R_val, "C1": 1e-9}
        state = sim.init(params)

        # Build up current with switch closed
        for _ in range(200):
            state = sim.step(params, state, {"vs": 5.0, "sw": True})

        # Open switch and find peak
        v_max = 0.0
        for _ in range(300):
            state = sim.step(params, state, {"vs": 5.0, "sw": False})
            v = jnp.abs(sim.v(state, nodes["bot"]))
            v_max = jnp.maximum(v_max, v)

        return v_max

    dV_dL = grad(peak_voltage)
    gradient = float(dV_dL(L_val))
    return gradient


def main():
    print("=" * 60)
    print("Inductor Kick (Back-EMF) Example")
    print("Converted from Falstad: inductkick.txt")
    print("=" * 60)

    # Parameters
    V_supply = 5.0
    L_val = 0.1      # 100 mH
    R_val = 100.0    # 100 ohm
    C_val = 1e-9     # 1 nF (snubber capacitor)
    t_switch = 1e-3  # Switch opens at 1ms

    tau = L_val / R_val
    I_steady = V_supply / R_val  # Steady-state current before switch opens

    print(f"\nCircuit Parameters:")
    print(f"   V_supply = {V_supply:.1f} V")
    print(f"   L = {L_val*1000:.1f} mH")
    print(f"   R = {R_val:.0f} ohm")
    print(f"   C = {C_val*1e9:.1f} nF (snubber)")
    print(f"   tau = L/R = {tau*1000:.2f} ms")
    print(f"   I_steady = V/R = {I_steady*1000:.1f} mA")
    print(f"   Switch opens at t = {t_switch*1000:.1f} ms")

    # Simulate
    print("\n1. Simulation Results")
    print("-" * 40)
    times, v_ind, i_ind, v_cap = simulate_inductor_kick(
        V_supply, L_val, R_val, C_val, t_switch
    )

    # Analyze
    analysis = analyze_kick(times, v_cap, t_switch)

    print(f"   Voltage before switch opens: {analysis['v_before']:.2f} V")
    print(f"   Peak voltage after switch:   {analysis['v_peak']:.2f} V")
    print(f"   Spike ratio:                 {analysis['spike_ratio']:.1f}x")
    print(f"   Peak occurs at:              {analysis['t_peak']*1000:.3f} ms")

    # Current analysis
    switch_idx = min(range(len(times)), key=lambda i: abs(times[i] - t_switch))
    i_before = i_ind[switch_idx - 1] if switch_idx > 0 else 0
    i_after_peak = max(abs(i) for i in i_ind[switch_idx:switch_idx+100]) if switch_idx < len(i_ind)-100 else 0

    print(f"\n   Current before switch opens: {i_before*1000:.2f} mA")
    print(f"   Expected steady-state:       {I_steady*1000:.2f} mA")

    # Effect of different snubber capacitors
    print("\n2. Effect of Snubber Capacitor Size")
    print("-" * 40)
    for C_test in [1e-10, 1e-9, 1e-8, 1e-7]:
        times, v_ind, i_ind, v_cap = simulate_inductor_kick(
            V_supply, L_val, R_val, C_test, t_switch
        )
        analysis = analyze_kick(times, v_cap, t_switch)
        print(f"   C = {C_test*1e9:>5.1f} nF: peak = {analysis['v_peak']:>8.2f} V "
              f"(spike ratio: {analysis['spike_ratio']:.1f}x)")

    # Differentiability
    print("\n3. JAX Differentiability")
    print("-" * 40)
    grad_val = demo_differentiability(L_val, R_val)
    print(f"   dV_peak/dL: {grad_val:.4f} V/H")
    print(f"   (Gradient is finite: demonstrates differentiability)")

    # Physical explanation
    print("\n4. Physical Explanation")
    print("-" * 40)
    print("   When the switch opens, the inductor's magnetic field")
    print("   collapses, inducing a voltage V = L * di/dt to maintain")
    print("   current flow. Without a path, this voltage can be very")
    print("   high (theoretically infinite for ideal components).")
    print("   The snubber capacitor provides a path for the energy,")
    print("   limiting the voltage spike. Larger C = lower spike.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
