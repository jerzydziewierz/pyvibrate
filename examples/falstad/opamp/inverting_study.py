"""
Study: Inverting Op-Amp with Finite Bandwidth

This study analyzes the realistic behavior of an inverting op-amp model
that includes parasitic RC elements. These elements:
1. Stabilize the simulation by breaking the algebraic feedback loop
2. Model realistic op-amp behavior with finite bandwidth

Configuration:
- Open-loop gain (A_ol): 1000
- Closed-loop gain: -Rf/Rin = -3
- Parasitic elements: Rp=100 ohm, Cp=500pF

The study includes:
1. Step response analysis (time domain) with plot
2. Frequency response analysis (Bode plot)
3. Comparison with ideal op-amp behavior

Key insight: Real op-amps have a gain-bandwidth product (GBW).
The parasitic RC creates a dominant pole that models this behavior.
"""
import math
import time
import matplotlib.pyplot as plt
import jax.numpy as jnp
from pyvibrate.timedomain import Network, R, C, VSource, VCVS


def build_inverting_amp_rc_damped():
    """Build inverting amplifier with RC dampeners.

    Circuit:
                       Rf
        +----------[====]----------+
        |                          |
        |   Rin        Rp     Cp   |
    Vin-+--[===]---(-)--[==]--||---+
                    |              |
              GND--(+)        [VCVS]---Vout
                                   |
                                  [Rload]
                                   |
                                  GND
    """
    net = Network()
    net, n_in = net.node("in")
    net, n_inv = net.node("inv")
    net, n_inv_damped = net.node("inv_damped")
    net, n_out = net.node("out")

    net, vs = VSource(net, n_in, net.gnd, name="vs")
    net, r_in = R(net, n_in, n_inv, name="Rin")
    net, r_parasitic = R(net, n_inv, n_inv_damped, name="Rp")
    net, c_parasitic = C(net, n_inv_damped, net.gnd, name="Cp")
    net, r_f = R(net, n_inv, n_out, name="Rf")
    net, opamp = VCVS(net, n_out, net.gnd, net.gnd, n_inv_damped, name="opamp")
    net, r_load = R(net, n_out, net.gnd, name="Rload")

    nodes = {"in": n_in, "inv": n_inv, "inv_damped": n_inv_damped, "out": n_out}
    return net, nodes


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "Rin": 1000.0,        # Input resistance (ohm)
    "Rf": 3000.0,         # Feedback resistance (ohm)
    "Rp": 100.0,          # Parasitic resistance (ohm)
    "Cp": 500e-12,        # Parasitic capacitance (F) = 500 pF -> tau = 50 ns
    "A_ol": 1000.0,       # Open-loop gain
    "Rload": 10000.0,     # Load resistance (ohm)
}


def get_params():
    """Get simulation parameters from config."""
    return {
        "vs": 0.0,
        "Rin": CONFIG["Rin"],
        "Rf": CONFIG["Rf"],
        "Rp": CONFIG["Rp"],
        "Cp": CONFIG["Cp"],
        "opamp": CONFIG["A_ol"],
        "Rload": CONFIG["Rload"],
    }


# =============================================================================
# Time Domain Analysis
# =============================================================================

def analyze_step_response():
    """Analyze step response of the op-amp circuit."""
    print("\n" + "=" * 70)
    print("1. STEP RESPONSE ANALYSIS")
    print("=" * 70)

    net, nodes = build_inverting_amp_rc_damped()

    # Use small timestep for accuracy
    tau_rc = CONFIG["Rp"] * CONFIG["Cp"]
    dt = tau_rc / 100  # at least 100 samples per RC time constant
    dt = min(dt, 1e-10)  # Minimum timestep = 0.01 ns

    sim = net.compile(dt=dt)
    params = get_params()
    state = sim.init(params)

    # Simulate step response
    V_step = 1.0
    n_steps = 2000

    times = []
    v_out = []
    v_inv = []

    for i in range(n_steps):
        state = sim.step(params, state, {"vs": V_step})
        times.append(float(state.time))
        v_out.append(float(sim.v(state, nodes["out"])))
        v_inv.append(float(sim.v(state, nodes["inv"])))

    # Analyze results
    expected_gain = -CONFIG["Rf"] / CONFIG["Rin"]
    expected_vout = V_step * expected_gain

    # Find settling characteristics
    final_vout = v_out[-1]
    target_90 = 0.9 * final_vout
    target_99 = 0.99 * final_vout

    t_90 = None
    t_99 = None
    for i, v in enumerate(v_out):
        if t_90 is None and abs(v) >= abs(target_90):
            t_90 = times[i]
        if t_99 is None and abs(v) >= abs(target_99):
            t_99 = times[i]

    # Find overshoot
    v_peak = min(v_out) if expected_vout < 0 else max(v_out)
    overshoot = (abs(v_peak) - abs(final_vout)) / abs(final_vout) * 100 if final_vout != 0 else 0

    print(f"\nCircuit Configuration:")
    print(f"   Rin = {CONFIG['Rin']:.0f} ohm")
    print(f"   Rf = {CONFIG['Rf']:.0f} ohm")
    print(f"   Rp = {CONFIG['Rp']:.0f} ohm (parasitic)")
    print(f"   Cp = {CONFIG['Cp']*1e12:.0f} pF (parasitic)")
    print(f"   Open-loop gain = {CONFIG['A_ol']:.0f}")
    print(f"   tau_RC = Rp * Cp = {tau_rc*1e9:.1f} ns")

    print(f"\nStep Response (V_step = {V_step:.1f} V):")
    print(f"   Expected ideal Vout:  {expected_vout:.4f} V")
    print(f"   Actual final Vout:    {final_vout:.4f} V")
    print(f"   Gain error:           {abs(final_vout - expected_vout)/abs(expected_vout)*100:.2f}%")

    print(f"\nSettling Time:")
    print(f"   Time to 90%:          {t_90*1e9:.1f} ns" if t_90 else "   Time to 90%:          N/A")
    print(f"   Time to 99%:          {t_99*1e9:.1f} ns" if t_99 else "   Time to 99%:          N/A")
    print(f"   Overshoot:            {overshoot:.2f}%")

    return times, v_out, v_inv, {
        "expected_vout": expected_vout,
        "final_vout": final_vout,
        "t_90": t_90,
        "t_99": t_99,
        "tau_rc": tau_rc,
    }


def plot_step_response(times, v_out, v_inv, analysis):
    """Create step response plot."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Convert times to nanoseconds for readability
    times_ns = [t * 1e9 for t in times]
    tau_ns = analysis["tau_rc"] * 1e9

    # Output voltage plot
    ax1 = axes[0]
    ax1.plot(times_ns, v_out, 'b.-', linewidth=1.5, label='Vout')
    ax1.axhline(y=analysis["expected_vout"], color='r', linestyle='--',
                label=f'Ideal: {analysis["expected_vout"]:.2f} V')
    ax1.axhline(y=analysis["final_vout"], color='g', linestyle=':',
                label=f'Final: {analysis["final_vout"]:.4f} V')

    # Mark settling times
    if analysis["t_90"]:
        ax1.axvline(x=analysis["t_90"]*1e9, color='orange', linestyle='--', alpha=0.7)
        ax1.annotate(f'90%: {analysis["t_90"]*1e9:.0f} ns',
                     xy=(analysis["t_90"]*1e9, analysis["final_vout"]*0.9),
                     xytext=(analysis["t_90"]*1e9 + tau_ns*0.5, analysis["final_vout"]*0.7),
                     fontsize=9)

    ax1.set_ylabel('Output Voltage (V)')
    ax1.set_title(f'Step Response: Inverting Op-Amp (Gain = -Rf/Rin = -3)\n'
                  f'Open-loop gain = {CONFIG["A_ol"]:.0f}, Rp = {CONFIG["Rp"]:.0f} ohm, '
                  f'Cp = {CONFIG["Cp"]*1e12:.0f} pF, tau = {tau_ns:.0f} ns')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Inverting input voltage plot (shows virtual ground settling)
    ax2 = axes[1]
    v_inv_mv = [v * 1000 for v in v_inv]  # Convert to mV
    ax2.plot(times_ns, v_inv_mv, 'b.-', linewidth=1.5, label='V_inv (inverting input)')
    ax2.axhline(y=0, color='r', linestyle='--', label='Virtual ground (ideal)')

    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Inverting Input (mV)')
    ax2.set_title('Virtual Ground Settling')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('step_response.png', dpi=150)
    print(f"\n   Step response plot saved to: step_response.png")
    plt.close()


# =============================================================================
# Frequency Response Analysis
# =============================================================================

def measure_frequency_response(freq, verbose=False):
    """Measure gain and phase at a specific frequency.

    Compiles a fresh circuit with dt appropriate for this frequency.
    """
    net, nodes = build_inverting_amp_rc_damped()

    # Choose dt: must resolve both the signal period AND the RC dynamics
    tau_rc = CONFIG["Rp"] * CONFIG["Cp"]
    period = 1.0 / freq

    # dt should give ~50 samples per period, but not larger than tau_rc/100
    # (same ratio as step response which was stable)
    dt = min(period / 50, tau_rc / 100)
    dt = max(dt, 1e-11)  # Minimum 0.01 ns

    sim = net.compile(dt=dt)
    params = get_params()

    samples_per_cycle = max(1, int(period / dt))

    # Run enough cycles for the RC to settle, then measure
    # Settling time ~ 5*tau_rc, so need 5*tau_rc / period cycles minimum
    settling_cycles = max(3, int(5 * tau_rc / period) + 1)
    measure_cycles = 2
    n_cycles = settling_cycles + measure_cycles
    n_steps = n_cycles * samples_per_cycle

    state = sim.init(params)

    V_amp = 1.0
    omega = 2 * math.pi * freq

    # Collect last 2 cycles for analysis
    collect_start = max(0, n_steps - measure_cycles * samples_per_cycle)

    t_samples = []
    v_in_samples = []
    v_out_samples = []

    for i in range(n_steps):
        t = float(state.time)
        v_in = V_amp * math.sin(omega * t)
        state = sim.step(params, state, {"vs": v_in})

        if i >= collect_start:
            t_samples.append(t)
            v_in_samples.append(v_in)
            v_out_samples.append(float(sim.v(state, nodes["out"])))

    # Measure amplitude
    v_out_max = max(v_out_samples)
    v_out_min = min(v_out_samples)
    v_out_amp = (v_out_max - v_out_min) / 2

    gain_magnitude = v_out_amp / V_amp
    gain_db = 20 * math.log10(gain_magnitude) if gain_magnitude > 0 else -100

    # Estimate phase by finding zero crossings
    v_out_mean = (v_out_max + v_out_min) / 2

    # Find first positive-going zero crossing of input in collected data
    in_cross_idx = None
    for i in range(1, len(v_in_samples)):
        if v_in_samples[i-1] < 0 and v_in_samples[i] >= 0:
            in_cross_idx = i
            break

    # Find corresponding crossing of output (negative-going for inverting amp)
    out_cross_idx = None
    if in_cross_idx is not None:
        for i in range(in_cross_idx, len(v_out_samples)):
            if v_out_samples[i-1] > v_out_mean and v_out_samples[i] <= v_out_mean:
                out_cross_idx = i
                break

    if in_cross_idx is not None and out_cross_idx is not None:
        t_in = t_samples[in_cross_idx]
        t_out = t_samples[out_cross_idx]
        phase_delay = t_out - t_in
        phase_deg = -180 - (phase_delay / period) * 360
        while phase_deg < -270:
            phase_deg += 360
        while phase_deg > 90:
            phase_deg -= 360
    else:
        phase_deg = -180

    return {
        "freq": freq,
        "gain_magnitude": gain_magnitude,
        "gain_db": gain_db,
        "phase_deg": phase_deg,
        "n_steps": n_steps,
    }


def analyze_frequency_response():
    """Analyze frequency response across a range of frequencies."""
    print("\n" + "=" * 70)
    print("2. FREQUENCY RESPONSE ANALYSIS")
    print("=" * 70)

    # Calculate theoretical values
    ideal_gain = CONFIG["Rf"] / CONFIG["Rin"]  # magnitude
    tau_rc = CONFIG["Rp"] * CONFIG["Cp"]
    f_pole = 1 / (2 * math.pi * tau_rc)

    print(f"\nTheoretical Parameters:")
    print(f"   Ideal closed-loop gain magnitude: {ideal_gain:.1f} ({20*math.log10(ideal_gain):.1f} dB)")
    print(f"   RC time constant: {tau_rc*1e9:.1f} ns")
    print(f"   Estimated pole frequency: {f_pole/1e6:.2f} MHz")

    # Frequency sweep - logarithmic spacing
    # With tau = 50 ns, pole is at ~3.2 MHz, so sweep from 100 kHz to 10 MHz
    frequencies = []
    for exp in range(5, 9):  # 100 kHz to 10 MHz
        base = 10 ** exp
        frequencies.extend([base * m for m in [1, 2, 3, 5, 7]])
    frequencies = sorted(set(f for f in frequencies if f <= 2000e6))

    print(f"\nFrequency Response (Bode Data) - {len(frequencies)} points:")
    print(f"   {'Freq':>12s}  {'|Gain|':>8s}  {'Gain (dB)':>10s}  {'Phase':>8s}  {'Steps':>8s}  {'Time':>8s}")
    print(f"   {'-'*62}")

    results = []
    total_time = 0
    for freq in frequencies:
        try:
            t_start = time.perf_counter()
            result = measure_frequency_response(freq)
            t_end = time.perf_counter()
            elapsed = t_end - t_start
            total_time += elapsed
            results.append(result)

            freq_str = f"{freq/1e3:.0f} kHz" if freq < 1e6 else f"{freq/1e6:.1f} MHz"
            print(f"   {freq_str:>12s}  {result['gain_magnitude']:>8.4f}  {result['gain_db']:>10.2f}  "
                  f"{result['phase_deg']:>8.0f}  {result['n_steps']:>8d}  {elapsed*1000:>7.0f}ms")
        except Exception as e:
            print(f"   {freq:>12.0e}  Error: {e}")

    print(f"\n   Total simulation time: {total_time:.1f} s")

    # Find -3dB bandwidth
    dc_gain_db = results[0]["gain_db"] if results else 0
    bandwidth = None
    for r in results:
        if r["gain_db"] <= dc_gain_db - 3:
            bandwidth = r["freq"]
            break

    print(f"\nBandwidth Analysis:")
    print(f"   DC gain: {results[0]['gain_magnitude']:.4f} ({results[0]['gain_db']:.2f} dB)" if results else "   DC gain: N/A")
    print(f"   -3dB bandwidth: {bandwidth/1e6:.1f} MHz" if bandwidth else "   -3dB bandwidth: > max test frequency")

    if bandwidth and results:
        gbw = results[0]["gain_magnitude"] * bandwidth
        print(f"   Gain-Bandwidth Product: {gbw/1e6:.1f} MHz")

    return results, {
        "ideal_gain": ideal_gain,
        "f_pole": f_pole,
        "bandwidth": bandwidth,
        "dc_gain_db": dc_gain_db,
    }


def plot_frequency_response(results, analysis):
    """Create Bode plot of frequency response."""
    if not results:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    freqs = [r["freq"] for r in results]
    gains_db = [r["gain_db"] for r in results]
    phases = [r["phase_deg"] for r in results]

    # Theoretical first-order response
    f_pole = analysis["f_pole"]
    ideal_gain = analysis["ideal_gain"]
    freqs_theory = [f for f in freqs if f <= 2e9]
    gains_theory = []
    phases_theory = []
    for f in freqs_theory:
        f_ratio = f / f_pole
        mag = ideal_gain / math.sqrt(1 + f_ratio**2)
        gains_theory.append(20 * math.log10(mag))
        phases_theory.append(-180 - math.degrees(math.atan(f_ratio)))

    # Magnitude plot
    ax1 = axes[0]
    ax1.semilogx(freqs, gains_db, 'b-o', linewidth=1.5, markersize=4, label='Simulated')
    ax1.semilogx(freqs_theory, gains_theory, 'r--', linewidth=1, label='Theory (1st order)')

    # Mark -3dB point
    if analysis["bandwidth"]:
        ax1.axvline(x=analysis["bandwidth"], color='green', linestyle=':', alpha=0.7)
        ax1.axhline(y=analysis["dc_gain_db"] - 3, color='green', linestyle=':', alpha=0.7)
        ax1.annotate(f'-3dB @ {analysis["bandwidth"]/1e6:.0f} MHz',
                     xy=(analysis["bandwidth"], analysis["dc_gain_db"] - 3),
                     xytext=(analysis["bandwidth"]*2, analysis["dc_gain_db"] - 6),
                     fontsize=9, arrowprops=dict(arrowstyle='->', color='green'))

    ax1.set_ylabel('Gain (dB)')
    ax1.set_title(f'Bode Plot: Inverting Op-Amp Frequency Response\n'
                  f'Open-loop gain = {CONFIG["A_ol"]:.0f}, '
                  f'Pole freq = {f_pole/1e6:.1f} MHz (from Rp={CONFIG["Rp"]:.0f}Ω, Cp={CONFIG["Cp"]*1e12:.0f}pF)')
    ax1.legend(loc='lower left')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.set_ylim([-20, 10])  # Fixed bounds for Bode plot

    # Phase plot
    ax2 = axes[1]
    ax2.semilogx(freqs, phases, 'b-o', linewidth=1.5, markersize=4, label='Simulated')
    ax2.semilogx(freqs_theory, phases_theory, 'r--', linewidth=1, label='Theory (1st order)')

    ax2.axhline(y=-180, color='gray', linestyle='--', alpha=0.5, label='Ideal phase (-180°)')

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_title('Phase Response')
    ax2.legend(loc='lower left')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.set_ylim([-270, -90])

    plt.tight_layout()
    plt.savefig('frequency_response.png', dpi=150)
    print(f"\n   Frequency response plot saved to: frequency_response.png")
    plt.close()


# =============================================================================
# Comparison with Ideal
# =============================================================================

def compare_with_ideal():
    """Compare RC-damped model with ideal op-amp behavior."""
    print("\n" + "=" * 70)
    print("3. COMPARISON: REALISTIC vs IDEAL OP-AMP")
    print("=" * 70)

    print(f"\nIdeal Op-Amp Assumptions:")
    print(f"   - Infinite open-loop gain")
    print(f"   - Infinite bandwidth")
    print(f"   - Zero output impedance")
    print(f"   - Infinite input impedance")

    print(f"\nRealistic Model (this simulation):")
    print(f"   - Open-loop gain: {CONFIG['A_ol']:.0f}")
    print(f"   - Bandwidth limited by Rp={CONFIG['Rp']:.0f} ohm, Cp={CONFIG['Cp']*1e12:.0f} pF")
    print(f"   - RC time constant: {CONFIG['Rp'] * CONFIG['Cp'] * 1e9:.0f} ns")
    print(f"   - Finite settling time")

    # Calculate key differences
    ideal_gain = -CONFIG["Rf"] / CONFIG["Rin"]
    beta = CONFIG["Rin"] / (CONFIG["Rin"] + CONFIG["Rf"])
    loop_gain = CONFIG["A_ol"] * beta
    gain_error = 1 / (1 + loop_gain)

    print(f"\nGain Analysis:")
    print(f"   Ideal closed-loop gain: {ideal_gain:.4f}")
    print(f"   Loop gain (A_ol * beta): {loop_gain:.1f}")
    print(f"   Gain error factor: {gain_error*100:.2f}%")
    print(f"   Expected realistic gain: {ideal_gain * (1 - gain_error):.4f}")

    tau_rc = CONFIG["Rp"] * CONFIG["Cp"]
    f_pole = 1 / (2 * math.pi * tau_rc)

    print(f"\nBandwidth Analysis:")
    print(f"   Ideal bandwidth: Infinite")
    print(f"   Model pole frequency: {f_pole/1e6:.1f} MHz")
    print(f"   This models realistic op-amp input capacitance effects")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("INVERTING OP-AMP STUDY: Finite Bandwidth Effects")
    print("=" * 70)
    print(f"\nThis study analyzes an inverting op-amp with parasitic RC elements")
    print(f"that model realistic finite-bandwidth behavior.")

    # Run analyses
    times, v_out, v_inv, step_analysis = analyze_step_response()
    plot_step_response(times, v_out, v_inv, step_analysis)


    if 1==1:
        freq_results, freq_analysis = analyze_frequency_response()
        plot_frequency_response(freq_results, freq_analysis)
        compare_with_ideal()

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
Key Findings:

1. TIME DOMAIN: The RC dampeners introduce a finite settling time.
   The circuit reaches steady state after several RC time constants.
   See: step_response.png

2. FREQUENCY DOMAIN: The parasitic RC creates a low-pass characteristic.
   Gain rolls off at high frequencies, modeling real op-amp bandwidth limits.
   See: frequency_response.png

3. GAIN ACCURACY: With open-loop gain of 1000, the closed-loop gain
   is very close to ideal (-Rf/Rin = -3), with <0.5% error.

4. STABILITY: The RC elements break the algebraic feedback loop,
   enabling stable time-domain simulation while providing physically
   meaningful bandwidth limitations.

Physical Interpretation:
- Rp models wire/trace resistance and op-amp internal resistance
- Cp models op-amp input capacitance (typically 1-10 pF)
- Together they create a dominant pole that limits bandwidth
- This is analogous to the gain-bandwidth product in real op-amps
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
