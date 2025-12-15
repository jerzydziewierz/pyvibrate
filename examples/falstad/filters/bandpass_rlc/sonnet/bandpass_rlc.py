"""
Example: RLC Bandpass Filter
Converted from Falstad: bandpass.txt

Demonstrates parallel-LC bandpass filter behavior:
- Center frequency: f_0 = 1 / (2 * pi * sqrt(L * C))
- At f_0: parallel LC has maximum impedance, maximum voltage transfer
- Quality factor: Q = R * sqrt(C/L)
- Bandwidth: BW = f_0 / Q

Original Falstad values: R=250 ohm, L=0.5H, C=31.7uF
This gives f_0 ~ 12.6 Hz, Q ~ 1.0 (moderately selective)

Components used: R, L, C, VSource
"""
import math
import jax.numpy as jnp
from jax import grad
from pyvibrate.timedomain import Network, R, L, C, VSource


def build_bandpass_filter():
    """Build parallel-LC bandpass filter.

    Circuit (from Falstad bandpass.txt):
        Vin ---[R]---+---+--- Vout
                     |   |
                    [L] [C]
                     |   |
                    GND GND

    The parallel LC tank has maximum impedance at resonance,
    allowing maximum voltage transfer from input to output.
    """
    net = Network()
    net, n_in = net.node("in")
    net, n_out = net.node("out")

    net, vs = VSource(net, n_in, net.gnd, name="vs")
    net, r1 = R(net, n_in, n_out, name="R1")
    net, l1 = L(net, n_out, net.gnd, name="L1")
    net, c1 = C(net, n_out, net.gnd, name="C1")

    return net, {"in": n_in, "out": n_out}, {"vs": vs, "R1": r1, "L1": l1, "C1": c1}


def calculate_filter_params(R_val, L_val, C_val):
    """Calculate bandpass filter parameters.

    Args:
        R_val: Series resistance in ohms
        L_val: Parallel inductance in henries
        C_val: Parallel capacitance in farads

    Returns:
        Dictionary with filter parameters
    """
    # Center frequency (resonance of parallel LC)
    omega_0 = 1 / math.sqrt(L_val * C_val)
    f_0 = omega_0 / (2 * math.pi)

    # Quality factor for parallel RLC bandpass
    # Q = R * sqrt(C/L) for this topology (series R, parallel LC)
    Q = R_val * math.sqrt(C_val / L_val)

    # Bandwidth
    BW = f_0 / Q

    return {
        "f_0": f_0,
        "omega_0": omega_0,
        "Q": Q,
        "BW": BW,
    }


def simulate_step_response(R_val=250.0, L_val=0.5, C_val=3.17e-5, V_step=5.0, n_periods=10):
    """Simulate step response showing transient behavior.

    Args:
        R_val: Series resistance in ohms
        L_val: Parallel inductance in henries
        C_val: Parallel capacitance in farads
        V_step: Step voltage amplitude
        n_periods: Number of periods at f_0 to simulate

    Returns:
        times: List of time values
        voltages: List of output voltage values
        analysis: Dict with filter parameters
    """
    net, nodes, _ = build_bandpass_filter()

    params = calculate_filter_params(R_val, L_val, C_val)
    period = 1 / params["f_0"]

    dt = period / 50  # 50 samples per period
    n_steps = int(n_periods * period / dt)

    sim = net.compile(dt=dt)
    sim_params = {"vs": 0.0, "R1": R_val, "L1": L_val, "C1": C_val}
    state = sim.init(sim_params)

    times = []
    voltages = []

    controls = {"vs": V_step}

    for i in range(n_steps):
        state = sim.step(sim_params, state, controls)
        times.append(float(state.time))
        voltages.append(float(sim.v(state, nodes["out"])))

    return times, voltages, params


def simulate_frequency_response(R_val=250.0, L_val=0.5, C_val=3.17e-5, frequencies=None):
    """Simulate AC frequency response (Bode plot data).

    Returns magnitude at each frequency.
    """
    params = calculate_filter_params(R_val, L_val, C_val)
    f_0 = params["f_0"]

    if frequencies is None:
        # Sweep from 0.1*f0 to 10*f0
        frequencies = [f_0 * mult for mult in [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 2.0, 5.0, 10.0]]

    net, nodes, _ = build_bandpass_filter()

    results = []

    for freq in frequencies:
        period = 1.0 / freq
        dt = period / 40  # 40 samples per period
        n_cycles = 10  # wait for steady state
        n_steps = int(n_cycles * period / dt)

        sim = net.compile(dt=dt)
        sim_params = {"vs": 0.0, "R1": R_val, "L1": L_val, "C1": C_val}
        state = sim.init(sim_params)

        V_amp = 1.0
        omega = 2 * math.pi * freq

        v_out_last = []

        for i in range(n_steps):
            t = float(state.time)
            v_in = V_amp * math.sin(omega * t)
            state = sim.step(sim_params, state, {"vs": v_in})

            # Collect last 2 cycles
            if i >= n_steps - int(2 * period / dt):
                v_out_last.append(float(sim.v(state, nodes["out"])))

        # Calculate magnitude
        v_out_amp = (max(v_out_last) - min(v_out_last)) / 2
        magnitude = v_out_amp / V_amp
        magnitude_db = 20 * math.log10(magnitude) if magnitude > 0 else -100

        results.append({
            "freq": freq,
            "magnitude": magnitude,
            "magnitude_db": magnitude_db,
        })

    return results


def demo_differentiability(R_val=250.0, L_val=0.5, C_val=3.17e-5):
    """Demonstrate JAX differentiability.

    Compute gradient of output voltage with respect to capacitance.
    """
    # Use fixed simulation parameters
    dt = 1e-4  # fixed
    n_steps = 500  # fixed

    def final_voltage(C_param):
        """Output voltage after fixed time as function of C."""
        net, nodes, _ = build_bandpass_filter()

        sim = net.compile(dt=dt)
        params = {"vs": 0.0, "R1": R_val, "L1": L_val, "C1": C_param}
        state = sim.init(params)

        for _ in range(n_steps):
            state = sim.step(params, state, {"vs": 5.0})

        return sim.v(state, nodes["out"])

    # Compute gradient
    grad_fn = grad(final_voltage)
    gradient = float(grad_fn(C_val))

    return gradient


def parameter_sweep(param_name, param_values, R_val=250.0, L_val=0.5, C_val=3.17e-5):
    """Sweep a parameter and collect center frequency and Q.

    Args:
        param_name: "R1", "L1", or "C1"
        param_values: List of values to test
        R_val, L_val, C_val: Default values for other parameters

    Returns:
        List of dicts with param value and results
    """
    results = []

    for val in param_values:
        if param_name == "R1":
            params = calculate_filter_params(val, L_val, C_val)
        elif param_name == "L1":
            params = calculate_filter_params(R_val, val, C_val)
        elif param_name == "C1":
            params = calculate_filter_params(R_val, L_val, val)
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        results.append({
            "param_value": val,
            "f_0": params["f_0"],
            "Q": params["Q"],
            "BW": params["BW"],
        })

    return results


def plot_results(times, outputs, title="Circuit Response"):
    """Generate and save plots."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, outputs)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(title)
        ax.grid(True)

        # Save to same directory as script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(script_dir, filename), dpi=150)
        plt.close()
        print(f"Plot saved: {filename}")

    except ImportError:
        print("matplotlib not available, skipping plots")


def plot_frequency_response(results, title="Frequency Response"):
    """Plot Bode magnitude plot."""
    try:
        import matplotlib.pyplot as plt

        freqs = [r["freq"] for r in results]
        mags_db = [r["magnitude_db"] for r in results]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogx(freqs, mags_db, 'b-o')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.axhline(y=-3, color='r', linestyle='--', alpha=0.5, label='-3dB')
        ax.legend()

        # Save to same directory as script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(script_dir, filename), dpi=150)
        plt.close()
        print(f"Plot saved: {filename}")

    except ImportError:
        print("matplotlib not available, skipping plots")


def main():
    print("=" * 60)
    print("RLC Bandpass Filter Example")
    print("Converted from Falstad: bandpass.txt")
    print("=" * 60)

    # Original Falstad parameters
    R_val = 250.0      # 250 ohm
    L_val = 0.5        # 0.5 H
    C_val = 3.17e-5    # 31.7 uF

    params = calculate_filter_params(R_val, L_val, C_val)

    print("\nCircuit Parameters:")
    print(f"   R = {R_val:.1f} ohm")
    print(f"   L = {L_val*1000:.1f} mH")
    print(f"   C = {C_val*1e6:.2f} uF")
    print("\nFilter Characteristics:")
    print(f"   f_0 = {params['f_0']:.2f} Hz (center frequency)")
    print(f"   Q = {params['Q']:.2f} (quality factor)")
    print(f"   BW = {params['BW']:.2f} Hz (bandwidth)")

    # 1. Step response
    print("\n1. Step Response")
    print("-" * 40)
    times, voltages, _ = simulate_step_response(R_val, L_val, C_val, V_step=5.0, n_periods=10)

    v_max = max(voltages)
    v_max_idx = voltages.index(v_max)
    t_peak = times[v_max_idx]
    v_final = voltages[-1]

    print(f"   Peak voltage: {v_max:.4f} V at t = {t_peak*1000:.2f} ms")
    print(f"   Final voltage: {v_final:.4f} V")
    print(f"   (Bandpass filters have zero DC gain)")

    # 2. Frequency response
    print("\n2. Frequency Response")
    print("-" * 40)
    ac_results = simulate_frequency_response(R_val, L_val, C_val)
    print(f"   {'Freq (Hz)':>12s}  {'f/f_0':>8s}  {'|H|':>8s}  {'dB':>8s}")

    for r in ac_results:
        f_ratio = r["freq"] / params["f_0"]
        print(f"   {r['freq']:>12.2f}  {f_ratio:>8.2f}  {r['magnitude']:>8.4f}  {r['magnitude_db']:>8.2f}")

    # Find peak gain
    peak_result = max(ac_results, key=lambda x: x["magnitude"])
    print(f"\n   Peak gain: {peak_result['magnitude']:.4f} ({peak_result['magnitude_db']:.2f} dB) at {peak_result['freq']:.2f} Hz")

    # 3. Parameter sweep (vary R to change Q)
    print("\n3. Parameter Sweep: Resistance (affects Q)")
    print("-" * 40)
    R_values = [50, 100, 250, 500, 1000]
    sweep_results = parameter_sweep("R1", R_values, R_val, L_val, C_val)

    print(f"   {'R (ohm)':>10s}  {'f_0 (Hz)':>10s}  {'Q':>8s}  {'BW (Hz)':>10s}")
    for r in sweep_results:
        print(f"   {r['param_value']:>10.0f}  {r['f_0']:>10.2f}  {r['Q']:>8.2f}  {r['BW']:>10.2f}")

    # 4. Differentiability demo
    print("\n4. JAX Differentiability")
    print("-" * 40)
    grad_val = demo_differentiability(R_val, L_val, C_val)
    print(f"   d(Vout)/d(C): {grad_val:.6f} V/F")
    print("   (Finite gradient demonstrates differentiability)")

    # 5. Generate plots
    print("\n5. Generating Plots")
    print("-" * 40)
    plot_results(times, voltages, "Bandpass Step Response")
    plot_frequency_response(ac_results, "Bandpass Frequency Response")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
