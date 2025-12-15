"""
Example: RLC Bandpass Filter (Parallel LC Tank)
Converted from Falstad: bandpass.txt

This circuit demonstrates a classic second-order bandpass filter using:
- Series resistor R (250 ohm)
- Parallel LC tank (L=0.5H, C=31.7uF) to ground

Circuit topology:

    Vin ---[R]---+------- Vout
                 |
             +---+---+
             |       |
            [L]     [C]
             |       |
             +---+---+
                 |
                GND

At resonance (f_0), the parallel LC tank has maximum impedance (ideally infinite
for lossless components). This creates maximum voltage transfer to Vout.

Key equations:
    f_0 = 1 / (2*pi*sqrt(L*C))           -- resonant frequency
    Z_tank(f_0) -> infinity (ideal)      -- tank impedance at resonance
    Q = R * sqrt(C/L)                    -- quality factor for this topology
    BW = f_0 / Q                         -- 3dB bandwidth

Why Q = R*sqrt(C/L)?
    At resonance, the parallel LC tank presents a real impedance Z_p = L/(R_loss*C)
    where R_loss is internal loss. For an external series R with ideal LC:
    Q = R / Z_0 where Z_0 = sqrt(L/C) is characteristic impedance
    Rearranging: Q = R * sqrt(C/L)

Original Falstad values:
    R = 250 ohm, L = 0.5 H, C = 31.7 uF
    f_0 = 39.98 Hz, Q = 1.99, BW = 20.08 Hz

The Falstad source uses 150Hz which is well above resonance, demonstrating
high-frequency attenuation (about -17 dB at 150 Hz).

Components used: R, L, C, VSource
"""
import math
from pyvibrate.timedomain import Network, R, L, C, VSource
from jax import grad


# Default component values from Falstad bandpass.txt
R_DEFAULT = 250.0       # ohm
L_DEFAULT = 0.5         # H
C_DEFAULT = 3.17e-5     # F (31.7 uF)


def build_circuit():
    """Build the parallel-LC bandpass filter.

    Returns:
        net: Network object
        nodes: Dict mapping node names to node references
        components: Dict mapping component names to component references
    """
    net = Network()
    net, n_in = net.node("in")
    net, n_out = net.node("out")

    # Series resistor from input to output
    net, vs = VSource(net, n_in, net.gnd, name="vs")
    net, r1 = R(net, n_in, n_out, name="R1")

    # Parallel LC tank from output to ground
    net, l1 = L(net, n_out, net.gnd, name="L1")
    net, c1 = C(net, n_out, net.gnd, name="C1")

    nodes = {"in": n_in, "out": n_out}
    components = {"vs": vs, "R1": r1, "L1": l1, "C1": c1}
    return net, nodes, components


def calculate_parameters(R_val, L_val, C_val):
    """Calculate bandpass filter parameters.

    For a series-R, parallel-LC topology:
        f_0 = 1/(2*pi*sqrt(LC)) - resonant frequency
        Z_0 = sqrt(L/C) - characteristic impedance
        Q = R/Z_0 = R*sqrt(C/L) - quality factor
        BW = f_0/Q - 3dB bandwidth

    Args:
        R_val: Series resistance (ohm)
        L_val: Inductance (H)
        C_val: Capacitance (F)

    Returns:
        Dict with f_0, omega_0, Z_0, Q, BW
    """
    omega_0 = 1.0 / math.sqrt(L_val * C_val)
    f_0 = omega_0 / (2 * math.pi)
    Z_0 = math.sqrt(L_val / C_val)
    Q = R_val / Z_0  # = R * sqrt(C/L)
    BW = f_0 / Q

    return {
        "f_0": f_0,
        "omega_0": omega_0,
        "Z_0": Z_0,
        "Q": Q,
        "BW": BW,
    }


def theoretical_transfer_function(freq, R_val, L_val, C_val):
    """Calculate theoretical transfer function H(f) = Vout/Vin.

    For series R with parallel LC to ground:
        Z_LC = j*omega*L || (1/(j*omega*C))
             = j*omega*L / (1 - omega^2*L*C)

        H(f) = Z_LC / (R + Z_LC)

    At resonance (omega = omega_0 = 1/sqrt(LC)):
        Z_LC -> infinity (denominator -> 0)
        H(f_0) -> 1

    Args:
        freq: Frequency in Hz
        R_val, L_val, C_val: Component values

    Returns:
        magnitude: |H(f)|
        phase_deg: Phase in degrees
    """
    omega = 2 * math.pi * freq
    omega_0 = 1.0 / math.sqrt(L_val * C_val)

    # Parallel LC impedance: Z_LC = jωL / (1 - ω²LC)
    # Let's compute in terms of real/imag
    # Z_L = jωL, Z_C = 1/(jωC) = -j/(ωC)
    # Z_LC = Z_L * Z_C / (Z_L + Z_C) = jωL * (-j/(ωC)) / (jωL - j/(ωC))
    #      = L/C / (jωL - j/(ωC))
    #      = (L/C) / (j(ωL - 1/(ωC)))
    #      = -j(L/C) / (ωL - 1/(ωC))

    X_L = omega * L_val
    X_C = 1.0 / (omega * C_val)
    denom = X_L - X_C  # = ωL - 1/(ωC)

    # Handle near-resonance carefully
    if abs(denom) < 1e-10:
        # At exact resonance, Z_LC -> infinity, H -> 1
        return 1.0, 0.0

    # Z_LC = -j * (L/C) / denom
    # Z_LC is purely imaginary: Z_LC = j * (L/C) / (1/(ωC) - ωL) = j * (L/C) / (-denom)
    Z_LC_imag = (L_val / C_val) / (-denom)
    # Z_LC = j * Z_LC_imag (purely imaginary)

    # H = Z_LC / (R + Z_LC) = (j*Z_LC_imag) / (R + j*Z_LC_imag)
    # |H|^2 = Z_LC_imag^2 / (R^2 + Z_LC_imag^2)
    H_mag_sq = Z_LC_imag**2 / (R_val**2 + Z_LC_imag**2)
    magnitude = math.sqrt(H_mag_sq)

    # Phase = arctan2(imag(H), real(H))
    # H = (j*a) / (R + j*a) where a = Z_LC_imag
    # H = (j*a * (R - j*a)) / ((R + j*a)(R - j*a))
    #   = (j*a*R + a^2) / (R^2 + a^2)
    #   = a^2/(R^2+a^2) + j*a*R/(R^2+a^2)
    H_real = Z_LC_imag**2 / (R_val**2 + Z_LC_imag**2)
    H_imag = Z_LC_imag * R_val / (R_val**2 + Z_LC_imag**2)
    phase_rad = math.atan2(H_imag, H_real)
    phase_deg = math.degrees(phase_rad)

    return magnitude, phase_deg


def simulate_step_response(R_val=R_DEFAULT, L_val=L_DEFAULT, C_val=C_DEFAULT,
                           V_step=5.0, n_periods=10):
    """Simulate step response of the bandpass filter.

    A step input contains broadband frequency content. The bandpass filter
    responds with a transient oscillation at f_0 that decays over time.
    The DC component is blocked (bandpass has zero DC gain).

    Args:
        R_val, L_val, C_val: Component values
        V_step: Step voltage amplitude
        n_periods: Number of periods at f_0 to simulate

    Returns:
        times: List of time values (s)
        voltages: List of output voltages (V)
        params: Dict of calculated circuit parameters
    """
    net, nodes, _ = build_circuit()
    params = calculate_parameters(R_val, L_val, C_val)

    period = 1.0 / params["f_0"]
    dt = period / 50  # 50 samples per period
    n_steps = int(n_periods * period / dt)

    sim = net.compile(dt=dt)
    sim_params = {"vs": 0.0, "R1": R_val, "L1": L_val, "C1": C_val}
    state = sim.init(sim_params)

    times = []
    voltages = []

    for _ in range(n_steps):
        state = sim.step(sim_params, state, {"vs": V_step})
        times.append(float(state.time))
        voltages.append(float(sim.v(state, nodes["out"])))

    return times, voltages, params


def simulate_frequency_response(R_val=R_DEFAULT, L_val=L_DEFAULT, C_val=C_DEFAULT,
                                frequencies=None):
    """Simulate AC frequency response with magnitude and phase.

    Applies a sinusoidal input at each frequency, waits for steady state,
    then measures output amplitude and estimates phase.

    Args:
        R_val, L_val, C_val: Component values
        frequencies: List of frequencies (Hz). If None, auto-generates.

    Returns:
        List of dicts with freq, magnitude, magnitude_db, phase_deg, theoretical_mag
    """
    params = calculate_parameters(R_val, L_val, C_val)
    f_0 = params["f_0"]

    if frequencies is None:
        # Logarithmic sweep centered on f_0
        freq_ratios = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0,
                       1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0]
        frequencies = [f_0 * r for r in freq_ratios]

    net, nodes, _ = build_circuit()
    results = []

    for freq in frequencies:
        period = 1.0 / freq
        dt = period / 40  # 40 samples per period
        n_cycles = 12  # Wait for steady state
        n_steps = int(n_cycles * period / dt)

        sim = net.compile(dt=dt)
        sim_params = {"vs": 0.0, "R1": R_val, "L1": L_val, "C1": C_val}
        state = sim.init(sim_params)

        V_amp = 1.0
        omega = 2 * math.pi * freq

        # Run simulation
        v_in_last = []
        v_out_last = []
        t_last = []

        for i in range(n_steps):
            t = float(state.time)
            v_in = V_amp * math.sin(omega * t)
            state = sim.step(sim_params, state, {"vs": v_in})

            # Collect last 3 cycles for analysis
            if i >= n_steps - int(3 * period / dt):
                t_last.append(t)
                v_in_last.append(v_in)
                v_out_last.append(float(sim.v(state, nodes["out"])))

        # Calculate magnitude from peak-to-peak
        v_out_pp = max(v_out_last) - min(v_out_last)
        v_out_amp = v_out_pp / 2
        magnitude = v_out_amp / V_amp
        magnitude_db = 20 * math.log10(magnitude) if magnitude > 1e-10 else -100

        # Estimate phase by finding zero crossings
        # (Simplified: find where output crosses zero going positive)
        phase_deg = estimate_phase(t_last, v_in_last, v_out_last, freq)

        # Theoretical comparison
        theo_mag, theo_phase = theoretical_transfer_function(freq, R_val, L_val, C_val)

        results.append({
            "freq": freq,
            "f_ratio": freq / f_0,
            "magnitude": magnitude,
            "magnitude_db": magnitude_db,
            "phase_deg": phase_deg,
            "theo_mag": theo_mag,
            "theo_phase": theo_phase,
        })

    return results


def estimate_phase(times, v_in, v_out, freq):
    """Estimate phase shift by comparing zero crossings.

    Finds positive-going zero crossings in both signals and calculates
    the time difference as a phase angle.

    Returns:
        phase_deg: Phase in degrees (positive = output leads)
    """
    def find_zero_crossing(t_vals, v_vals):
        """Find time of first positive-going zero crossing."""
        for i in range(1, len(v_vals)):
            if v_vals[i-1] <= 0 < v_vals[i]:
                # Linear interpolation
                frac = -v_vals[i-1] / (v_vals[i] - v_vals[i-1])
                return t_vals[i-1] + frac * (t_vals[i] - t_vals[i-1])
        return None

    t_in = find_zero_crossing(times, v_in)
    t_out = find_zero_crossing(times, v_out)

    if t_in is None or t_out is None:
        return 0.0

    period = 1.0 / freq
    dt = t_out - t_in
    phase_deg = 360.0 * dt / period

    # Normalize to [-180, 180]
    while phase_deg > 180:
        phase_deg -= 360
    while phase_deg < -180:
        phase_deg += 360

    return phase_deg


def demo_differentiability(R_val=R_DEFAULT, L_val=L_DEFAULT, C_val=C_DEFAULT):
    """Demonstrate JAX differentiability.

    Computes d(Vout)/dC at a fixed simulation time.
    This shows that the entire simulation is differentiable.

    Returns:
        Dict with gradient value and interpretation
    """
    # Fixed simulation parameters (must not depend on traced value)
    dt = 1e-4
    n_steps = 500

    def output_voltage(C_param):
        """Output voltage after fixed time as function of C."""
        net, nodes, _ = build_circuit()
        sim = net.compile(dt=dt)
        params = {"vs": 0.0, "R1": R_val, "L1": L_val, "C1": C_param}
        state = sim.init(params)

        for _ in range(n_steps):
            state = sim.step(params, state, {"vs": 5.0})

        return sim.v(state, nodes["out"])

    grad_fn = grad(output_voltage)
    gradient = float(grad_fn(C_val))

    return {
        "gradient": gradient,
        "description": "d(Vout)/d(C) in V/F",
        "interpretation": (
            f"Increasing C by 1nF changes Vout by {gradient * 1e-9:.6f} V "
            f"at t={n_steps * dt * 1000:.1f}ms"
        ),
    }


def parameter_sweep(sweep_param, values, R_val=R_DEFAULT, L_val=L_DEFAULT,
                    C_val=C_DEFAULT):
    """Sweep one parameter and observe effect on filter characteristics.

    Args:
        sweep_param: "R", "L", or "C"
        values: List of parameter values to test
        R_val, L_val, C_val: Default values for non-swept parameters

    Returns:
        List of dicts with parameter value and filter characteristics
    """
    results = []

    for val in values:
        if sweep_param == "R":
            params = calculate_parameters(val, L_val, C_val)
            params["param_name"] = "R"
            params["param_value"] = val
        elif sweep_param == "L":
            params = calculate_parameters(R_val, val, C_val)
            params["param_name"] = "L"
            params["param_value"] = val
        elif sweep_param == "C":
            params = calculate_parameters(R_val, L_val, val)
            params["param_name"] = "C"
            params["param_value"] = val
        else:
            raise ValueError(f"Unknown parameter: {sweep_param}")

        results.append(params)

    return results


def plot_results(times, voltages, title="Step Response"):
    """Generate and save step response plot."""
    try:
        import matplotlib.pyplot as plt
        import os

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, voltages, 'b-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Output Voltage (V)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(script_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Plot saved: {filename}")

    except ImportError:
        print("   matplotlib not available, skipping plot")


def plot_frequency_response(results, params, title="Frequency Response"):
    """Generate Bode plot (magnitude and phase)."""
    try:
        import matplotlib.pyplot as plt
        import os

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        freqs = [r["freq"] for r in results]
        mags_db = [r["magnitude_db"] for r in results]
        phases = [r["phase_deg"] for r in results]
        theo_mags = [20 * math.log10(r["theo_mag"]) if r["theo_mag"] > 0 else -100
                     for r in results]

        # Magnitude plot
        ax1.semilogx(freqs, mags_db, 'b-o', label='Simulated', markersize=4)
        ax1.semilogx(freqs, theo_mags, 'r--', label='Theoretical', linewidth=1)
        ax1.axhline(y=-3, color='gray', linestyle=':', alpha=0.7, label='-3dB')
        ax1.axvline(x=params["f_0"], color='g', linestyle=':', alpha=0.7,
                    label=f'f_0={params["f_0"]:.1f}Hz')
        ax1.set_ylabel("Magnitude (dB)")
        ax1.set_title(f"{title} (f_0={params['f_0']:.1f}Hz, Q={params['Q']:.2f})")
        ax1.grid(True, which='both', alpha=0.3)
        ax1.legend(loc='lower left')
        ax1.set_ylim([-40, 5])

        # Phase plot
        ax2.semilogx(freqs, phases, 'b-o', markersize=4)
        ax2.axvline(x=params["f_0"], color='g', linestyle=':', alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (degrees)")
        ax2.grid(True, which='both', alpha=0.3)
        ax2.set_ylim([-100, 100])

        plt.tight_layout()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(script_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Plot saved: {filename}")

    except ImportError:
        print("   matplotlib not available, skipping plot")


def main():
    print("=" * 70)
    print("RLC Bandpass Filter (Parallel LC Tank)")
    print("Converted from Falstad: bandpass.txt")
    print("=" * 70)

    # Component values from Falstad
    R_val = R_DEFAULT
    L_val = L_DEFAULT
    C_val = C_DEFAULT

    params = calculate_parameters(R_val, L_val, C_val)

    print("\nCircuit Parameters:")
    print(f"   R = {R_val:.1f} ohm")
    print(f"   L = {L_val*1000:.1f} mH ({L_val:.3f} H)")
    print(f"   C = {C_val*1e6:.2f} uF ({C_val:.2e} F)")

    print("\nFilter Characteristics:")
    print(f"   f_0 = {params['f_0']:.2f} Hz (resonant frequency)")
    print(f"   Z_0 = {params['Z_0']:.1f} ohm (characteristic impedance)")
    print(f"   Q   = {params['Q']:.2f} (quality factor = R/Z_0)")
    print(f"   BW  = {params['BW']:.2f} Hz (3dB bandwidth)")

    # 1. Step Response
    print("\n" + "-" * 70)
    print("1. STEP RESPONSE")
    print("-" * 70)
    times, voltages, _ = simulate_step_response(R_val, L_val, C_val,
                                                 V_step=5.0, n_periods=10)

    v_peak = max(voltages)
    t_peak = times[voltages.index(v_peak)]
    v_final = voltages[-1]

    print(f"   Input: 5V step")
    print(f"   Peak output: {v_peak:.4f} V at t = {t_peak*1000:.2f} ms")
    print(f"   Final output: {v_final:.6f} V (expected: 0V, DC blocked)")
    print(f"\n   Physical insight: The step contains broadband frequency content.")
    print(f"   The filter responds with a transient oscillation at f_0, which decays")
    print(f"   with time constant tau ~ 2*L*Q/R = {2*L_val*params['Q']/R_val*1000:.1f} ms")

    plot_results(times, voltages, "Bandpass Step Response")

    # 2. Frequency Response
    print("\n" + "-" * 70)
    print("2. FREQUENCY RESPONSE")
    print("-" * 70)
    ac_results = simulate_frequency_response(R_val, L_val, C_val)

    print(f"   {'f (Hz)':>8} {'f/f_0':>7} {'|H|':>8} {'dB':>8} {'Phase':>8} {'Theo':>8}")
    print("   " + "-" * 55)

    for r in ac_results:
        theo_db = 20 * math.log10(r["theo_mag"]) if r["theo_mag"] > 1e-10 else -100
        print(f"   {r['freq']:>8.1f} {r['f_ratio']:>7.2f} {r['magnitude']:>8.4f} "
              f"{r['magnitude_db']:>8.2f} {r['phase_deg']:>8.1f} {theo_db:>8.2f}")

    # Find -3dB points
    peak_mag = max(r["magnitude"] for r in ac_results)
    cutoff_mag = peak_mag / math.sqrt(2)
    in_band = [r for r in ac_results if r["magnitude"] >= cutoff_mag]
    if len(in_band) >= 2:
        f_low = min(r["freq"] for r in in_band)
        f_high = max(r["freq"] for r in in_band)
        bw_measured = f_high - f_low
        print(f"\n   Measured -3dB bandwidth: {bw_measured:.2f} Hz")
        print(f"   Theoretical bandwidth: {params['BW']:.2f} Hz")

    # Show Falstad source frequency response
    print(f"\n   At Falstad source frequency (150 Hz):")
    theo_150, _ = theoretical_transfer_function(150, R_val, L_val, C_val)
    print(f"   Theoretical |H(150Hz)| = {theo_150:.4f} ({20*math.log10(theo_150):.1f} dB)")
    print(f"   This is why Falstad shows attenuated output at 150 Hz.")

    plot_frequency_response(ac_results, params, "Bandpass Frequency Response")

    # 3. Parameter Sweep
    print("\n" + "-" * 70)
    print("3. PARAMETER SWEEP: Effect of R on Q")
    print("-" * 70)
    R_values = [50, 100, 250, 500, 1000, 2000]
    sweep_results = parameter_sweep("R", R_values, R_val, L_val, C_val)

    print(f"   {'R (ohm)':>10} {'f_0 (Hz)':>10} {'Q':>8} {'BW (Hz)':>10}")
    print("   " + "-" * 42)
    for r in sweep_results:
        print(f"   {r['param_value']:>10.0f} {r['f_0']:>10.2f} {r['Q']:>8.2f} {r['BW']:>10.2f}")

    print(f"\n   Note: f_0 is independent of R (determined by L and C only)")
    print(f"   Q increases linearly with R (Q = R/Z_0)")
    print(f"   Higher Q = narrower bandwidth = sharper selectivity")

    # 4. JAX Differentiability
    print("\n" + "-" * 70)
    print("4. JAX DIFFERENTIABILITY")
    print("-" * 70)
    diff_result = demo_differentiability(R_val, L_val, C_val)
    print(f"   {diff_result['description']}: {diff_result['gradient']:.4f}")
    print(f"   {diff_result['interpretation']}")
    print(f"\n   This finite gradient demonstrates that the simulation is")
    print(f"   fully differentiable, enabling gradient-based optimization.")

    print("\n" + "=" * 70)
    print("Summary: This bandpass filter passes signals near f_0 = {:.1f} Hz".format(
        params['f_0']))
    print(f"and attenuates frequencies outside the {params['BW']:.1f} Hz bandwidth.")
    print("=" * 70)


if __name__ == "__main__":
    main()
