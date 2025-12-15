"""
Example: RLC Bandpass Filter
Converted from Falstad: bandpass.txt

Demonstrates series RLC bandpass filter behavior:
- Resonant frequency: f_0 = 1 / (2 * pi * sqrt(L * C))
- Quality factor: Q = (omega_0 * L) / R = (1/R) * sqrt(L/C)
- Bandwidth: BW = f_0 / Q = R / (2 * pi * L)
- Attenuation increases away from resonance

Original Falstad values:
  - Input: 10V AC at 150 Hz (offset 5V, phase 0.5)
  - Resistor: R = 250 ohm
  - Inductor: L = 0.5 H
  - Capacitor: C = 31.7 uF
  - Calculated f_0 ~ 40.1 Hz, Q ~ 1.0 (critically damped)

Components used: R, L, C, VSource
"""
import math
import jax.numpy as jnp
from jax import grad
from pyvibrate.timedomain import Network, R, L, C, VSource


def build_bandpass_filter():
    """Build RLC bandpass filter.

    Circuit:
        Vin ---[R]---+--- Vout
                     |
                    [L]
                     |
                    [C]
                     |
                    GND
    """
    net = Network()
    net, n_in = net.node("in")
    net, n_out = net.node("out")
    net, n_mid = net.node("mid")

    # Voltage source
    net, vs = VSource(net, n_in, net.gnd, name="vs")
    # Series resistor
    net, r1 = R(net, n_in, n_out, name="R1")
    # Series inductor
    net, l1 = L(net, n_out, n_mid, name="L1")
    # Capacitor to ground
    net, c1 = C(net, n_mid, net.gnd, name="C1")

    return net, {"in": n_in, "out": n_out, "mid": n_mid}, {"vs": vs, "R1": r1, "L1": l1, "C1": c1}


def simulate_step_response(R_val=250.0, L_val=0.5, C_val=3.17e-5, V_step=5.0, n_periods=10):
    """Simulate step response showing resonance behavior.

    Args:
        R_val: Resistance in ohms
        L_val: Inductance in henries
        C_val: Capacitance in farads
        V_step: Step voltage amplitude
        n_periods: Number of periods to simulate (based on f_0)

    Returns:
        times: Array of time values
        v_out: Array of output voltage (at junction between R and L)
        v_cap: Array of capacitor voltage (at node after L)
        analysis: Dict with derived parameters
    """
    net, nodes, components = build_bandpass_filter()

    # Calculate circuit parameters
    omega_0 = 1 / math.sqrt(L_val * C_val)
    f_0 = omega_0 / (2 * math.pi)
    period = 1 / f_0

    # Damping
    alpha = R_val / (2 * L_val)
    Q = omega_0 / (2 * alpha)  # Quality factor
    omega_d = math.sqrt(max(0, omega_0**2 - alpha**2))  # Damped frequency

    dt = period / 50  # 50 samples per period
    n_steps = int(n_periods * period / dt)

    sim = net.compile(dt=dt)
    params = {"vs": 0.0, "R1": R_val, "L1": L_val, "C1": C_val}
    state = sim.init(params)

    times = []
    v_out = []
    v_cap = []

    controls = {"vs": V_step}

    for i in range(n_steps):
        state = sim.step(params, state, controls)
        times.append(float(state.time))
        v_out.append(float(sim.v(state, nodes["out"])))
        v_cap.append(float(sim.v(state, nodes["mid"])))

    return times, v_out, v_cap, {
        "f_0": f_0,
        "omega_0": omega_0,
        "Q": Q,
        "alpha": alpha,
        "omega_d": omega_d,
    }


def simulate_ac_response(R_val=250.0, L_val=0.5, C_val=3.17e-5, frequencies=None):
    """Simulate AC frequency response at multiple frequencies.

    Returns magnitude and phase at each frequency for output impedance.
    """
    if frequencies is None:
        # Default: sweep from 10 Hz to 200 Hz around resonance
        f_0 = 1 / (2 * math.pi * math.sqrt(L_val * C_val))
        frequencies = [10, 20, 30, 40, 50, 60, 80, 100, 150, 200]

    net, nodes, components = build_bandpass_filter()

    results = []

    for freq in frequencies:
        period = 1.0 / freq
        dt = period / 40  # 40 samples per period
        n_cycles = 15  # Simulate enough cycles for steady state
        n_steps = int(n_cycles * period / dt)

        sim = net.compile(dt=dt)
        params = {"vs": 0.0, "R1": R_val, "L1": L_val, "C1": C_val}
        state = sim.init(params)

        V_amp = 1.0
        omega = 2 * math.pi * freq

        # Run simulation to reach steady state
        v_in_last = []
        v_out_last = []

        for i in range(n_steps):
            t = float(state.time)
            v_in = V_amp * math.sin(omega * t)
            state = sim.step(params, state, {"vs": v_in})

            # Collect last 3 cycles for analysis
            if i >= n_steps - int(3 * period / dt):
                v_in_last.append(v_in)
                v_out_last.append(float(sim.v(state, nodes["out"])))

        # Estimate amplitude ratio
        if v_out_last:
            v_out_max = max(v_out_last)
            v_out_min = min(v_out_last)
            v_out_amp = (v_out_max - v_out_min) / 2

            magnitude = v_out_amp / V_amp
            magnitude_db = 20 * math.log10(magnitude) if magnitude > 0 else -100
        else:
            magnitude = 0.0
            magnitude_db = -100

        results.append({
            "freq": freq,
            "magnitude": magnitude,
            "magnitude_db": magnitude_db,
        })

    return results


def analyze_bandpass(R_val, L_val, C_val):
    """Analyze bandpass filter characteristics."""
    omega_0 = 1 / math.sqrt(L_val * C_val)
    f_0 = omega_0 / (2 * math.pi)
    alpha = R_val / (2 * L_val)
    Q = omega_0 / (2 * alpha)
    bandwidth = R_val / (2 * math.pi * L_val)
    Z_L = 2 * math.pi * f_0 * L_val
    Z_C = 1 / (2 * math.pi * f_0 * C_val)

    return {
        "f_0": f_0,
        "omega_0": omega_0,
        "Q": Q,
        "bandwidth": bandwidth,
        "Z_L_at_f0": Z_L,
        "Z_C_at_f0": Z_C,
    }


def demo_differentiability(R_val=250.0, L_val=0.5, C_val=3.17e-5):
    """Demonstrate JAX differentiability of bandpass filter."""

    # Use fixed simulation parameters (can't depend on traced parameters)
    dt = 1e-4  # Fixed timestep
    n_steps = 1000  # Fixed number of steps
    omega_fixed = 2 * math.pi * 40  # Fixed frequency (resonance neighborhood)

    def output_voltage_at_fixed_freq(L_param):
        """Output voltage at fixed frequency as function of L."""
        net, nodes, _ = build_bandpass_filter()

        sim = net.compile(dt=dt)
        params = {"vs": 0.0, "R1": R_val, "L1": L_param, "C1": C_val}
        state = sim.init(params)

        for _ in range(n_steps):
            t = float(state.time)
            v_in = 5.0 * math.sin(omega_fixed * t)
            state = sim.step(params, state, {"vs": v_in})

        return sim.v(state, nodes["out"])

    dV_dL = grad(output_voltage_at_fixed_freq)
    gradient = float(dV_dL(L_val))
    return gradient


def main():
    print("=" * 60)
    print("RLC Bandpass Filter Example")
    print("Converted from Falstad: bandpass.txt")
    print("=" * 60)

    # Parameters (from Falstad original)
    R_val = 250.0      # 250 ohm
    L_val = 0.5        # 0.5 H
    C_val = 3.17e-5    # 31.7 uF

    analysis = analyze_bandpass(R_val, L_val, C_val)

    print(f"\nFilter Parameters:")
    print(f"   R = {R_val:.0f} ohm")
    print(f"   L = {L_val*1000:.1f} mH")
    print(f"   C = {C_val*1e6:.2f} uF")

    print(f"\nDerived Parameters:")
    print(f"   f_0 = {analysis['f_0']:.2f} Hz (resonant frequency)")
    print(f"   Q = {analysis['Q']:.3f} (quality factor)")
    print(f"   BW = {analysis['bandwidth']:.2f} Hz (bandwidth)")
    print(f"   |Z_L| at f_0 = {analysis['Z_L_at_f0']:.2f} ohm")
    print(f"   |Z_C| at f_0 = {analysis['Z_C_at_f0']:.2f} ohm")

    # Step response
    print("\n1. Step Response (V_step = 5V)")
    print("-" * 40)
    times, v_out, v_cap, params = simulate_step_response(R_val, L_val, C_val, V_step=5.0, n_periods=15)

    # Find peak and settling time
    v_out_max = max(v_out)
    v_out_max_idx = v_out.index(v_out_max)
    t_peak = times[v_out_max_idx]

    # Find settling time (to within 5%)
    settle_threshold = 0.05 * 5.0  # 5% of final value
    t_settle = None
    for i in range(len(v_out) - 1, -1, -1):
        if abs(v_out[i] - v_out[-1]) > settle_threshold:
            t_settle = times[i]
            break

    print(f"   Peak output voltage: {v_out_max:.4f} V")
    print(f"   Time to peak: {t_peak*1000:.2f} ms")
    print(f"   Final capacitor voltage: {v_cap[-1]:.4f} V")
    if t_settle:
        print(f"   Settling time (5%): {t_settle*1000:.2f} ms")

    # AC frequency response
    print("\n2. Frequency Response")
    print("-" * 40)
    print(f"   {'Freq':>8s}  {'f/f_0':>8s}  {'|H|':>8s}  {'dB':>8s}")
    ac_results = simulate_ac_response(R_val, L_val, C_val)
    for r in ac_results:
        f_ratio = r["freq"] / analysis["f_0"]
        print(f"   {r['freq']:>8.1f}  {f_ratio:>8.2f}  {r['magnitude']:>8.4f}  {r['magnitude_db']:>8.2f}")

    # Find peak response frequency
    peak_result = max(ac_results, key=lambda x: x["magnitude"])
    print(f"\n   Peak response: {peak_result['freq']:.1f} Hz ({peak_result['magnitude_db']:.2f} dB)")

    # Differentiability
    print("\n3. JAX Differentiability")
    print("-" * 40)
    grad_val = demo_differentiability(R_val, L_val, C_val)
    print(f"   dV_out/dL at resonance: {grad_val:.4f} V/H")
    print(f"   (Gradient is finite: demonstrates differentiability)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
