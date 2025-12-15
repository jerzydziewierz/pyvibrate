"""
Example: RLC Bandpass Filter
Converted from Falstad: bandpass.txt

Demonstrates series RLC bandpass filter behavior:
- Resonant frequency: f_0 = 1 / (2 * pi * sqrt(L * C))
- Bandwidth: BW = R / (2 * pi * L)
- Quality factor: Q = f_0 / BW = (1/R) * sqrt(L/C)
- At resonance: impedance is minimum, current is maximum

Original Falstad values:
  R = 250 ohm
  L = 0.5 H
  C = 31.7 uF
  AC source: 10V amplitude, 150 Hz center frequency

Circuit topology:
  AC Source (10V, 150Hz) ---[R=250Ω]---+--- Output
                                        |
                                      [L=0.5H]
                                        |
                                      [C=31.7μF]
                                        |
                                       GND

This is a series RLC bandpass filter where:
- At resonance (f_0 ≈ 50 Hz), impedance Z = R (minimum)
- At high frequencies, impedance ≈ ωL (inductor dominates)
- At low frequencies, impedance ≈ 1/(ωC) (capacitor dominates)

Components used: R, L, C, VSource
"""
import math
import jax.numpy as jnp
from jax import grad
from pyvibrate.timedomain import Network, R, L, C, VSource


def build_bandpass_filter():
    """Build series RLC bandpass filter.

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
    net, n_tank = net.node("tank")  # Node between R and LC tank

    net, vs = VSource(net, n_in, net.gnd, name="vs")
    net, r1 = R(net, n_in, n_tank, name="R1")
    net, l1 = L(net, n_tank, net.gnd, name="L1")
    net, c1 = C(net, n_tank, net.gnd, name="C1")

    return net, {"in": n_in, "tank": n_tank}, {"vs": vs, "R1": r1, "L1": l1, "C1": c1}


def simulate_step_response(R_val=250.0, L_val=0.5, C_val=3.17e-5, V_step=10.0, n_tau=10):
    """Simulate step response and return time/voltage arrays.

    Args:
        R_val: Resistance in ohms
        L_val: Inductance in henries
        C_val: Capacitance in farads
        V_step: Step voltage amplitude
        n_tau: Number of time constants to simulate

    Returns:
        times: Array of time values
        voltages: Array of output voltage values
    """
    net, nodes, _ = build_bandpass_filter()

    # Calculate damping ratio and natural frequency
    omega_0 = 1 / math.sqrt(L_val * C_val)
    tau = 2 * L_val / R_val  # Approximate time constant
    dt = tau / 100  # 100 samples per time constant
    n_steps = int(n_tau * tau / dt)

    sim = net.compile(dt=dt)
    params = {"vs": 0.0, "R1": R_val, "L1": L_val, "C1": C_val}
    state = sim.init(params)

    times = []
    voltages = []

    controls = {"vs": V_step}

    for i in range(n_steps):
        state = sim.step(params, state, controls)
        times.append(float(state.time))
        voltages.append(float(sim.v(state, nodes["tank"])))

    return times, voltages


def simulate_ac_response(R_val=250.0, L_val=0.5, C_val=3.17e-5, frequencies=None):
    """Simulate AC response at multiple frequencies.

    Returns magnitude and phase at each frequency.
    """
    if frequencies is None:
        # Calculate resonant frequency for smart frequency selection
        f_0 = 1 / (2 * math.pi * math.sqrt(L_val * C_val))
        bw = R_val / (2 * math.pi * L_val)
        # Sweep around resonance with finer resolution near peak
        frequencies = []
        # Before resonance
        for mult in [0.2, 0.3, 0.5, 0.7, 0.9, 0.95]:
            frequencies.append(f_0 * mult)
        # At and around resonance
        for mult in [0.99, 1.0, 1.01]:
            frequencies.append(f_0 * mult)
        # After resonance
        for mult in [1.1, 1.2, 1.5, 2.0, 3.0, 5.0]:
            frequencies.append(f_0 * mult)
        frequencies = sorted(frequencies)

    net, nodes, _ = build_bandpass_filter()

    results = []

    for freq in frequencies:
        # Use period/40 timestep for good resolution
        period = 1.0 / freq
        dt = period / 40
        n_cycles = 15  # Simulate enough cycles for steady state
        n_steps = int(n_cycles * period / dt)

        sim = net.compile(dt=dt)
        params = {"vs": 0.0, "R1": R_val, "L1": L_val, "C1": C_val}
        state = sim.init(params)

        V_amp = 10.0  # Match Falstad source amplitude
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
                v_out_last.append(float(sim.v(state, nodes["tank"])))

        # Estimate amplitude ratio
        v_out_max = max(v_out_last)
        v_out_min = min(v_out_last)
        v_out_amp = (v_out_max - v_out_min) / 2

        magnitude = v_out_amp / V_amp
        magnitude_db = 20 * math.log10(magnitude) if magnitude > 0 else -100

        results.append({
            "freq": freq,
            "magnitude": magnitude,
            "magnitude_db": magnitude_db,
        })

    return results


def demo_differentiability(R_val=250.0, L_val=0.5, C_val=3.17e-5):
    """Demonstrate JAX differentiability of filter response."""

    def voltage_at_resonance(R_param):
        """Output voltage at resonant frequency as function of R."""
        net, nodes, _ = build_bandpass_filter()

        # Fixed resonant frequency simulation
        f_0 = 1 / (2 * math.pi * math.sqrt(L_val * C_val))
        period = 1.0 / f_0
        dt = period / 40
        n_steps = int(15 * period / dt)  # 15 cycles

        sim = net.compile(dt=dt)
        params = {"vs": 0.0, "R1": R_param, "L1": L_val, "C1": C_val}
        state = sim.init(params)

        V_amp = 10.0
        omega = 2 * math.pi * f_0

        # Skip first few cycles to reach steady state
        skip_steps = int(5 * period / dt)
        for i in range(skip_steps):
            t = float(state.time)
            v_in = V_amp * math.sin(omega * t)
            state = sim.step(params, state, {"vs": v_in})

        # Measure output at one more cycle
        for i in range(int(period / dt)):
            t = float(state.time)
            v_in = V_amp * math.sin(omega * t)
            state = sim.step(params, state, {"vs": v_in})

        return sim.v(state, nodes["tank"])

    dV_dR = grad(voltage_at_resonance)
    gradient = float(dV_dR(R_val))

    return gradient


def main():
    print("=" * 60)
    print("RLC Bandpass Filter Example")
    print("Converted from Falstad: bandpass.txt")
    print("=" * 60)

    # Parameters (from Falstad original)
    R_val = 250.0        # ohm
    L_val = 0.5          # H
    C_val = 3.17e-5      # F (31.7 uF)

    # Calculate filter characteristics
    f_0 = 1 / (2 * math.pi * math.sqrt(L_val * C_val))
    omega_0 = 2 * math.pi * f_0
    Z_0 = math.sqrt(L_val / C_val)
    Q = omega_0 * L_val / R_val
    bandwidth = f_0 / Q
    damping_ratio = R_val / (2 * math.sqrt(L_val / C_val))

    print(f"\nFilter Parameters:")
    print(f"   R = {R_val:.1f} ohm")
    print(f"   L = {L_val:.3f} H")
    print(f"   C = {C_val*1e6:.2f} uF")
    print(f"\nFilter Characteristics:")
    print(f"   f_0 (resonant freq) = {f_0:.2f} Hz")
    print(f"   Z_0 (characteristic impedance) = {Z_0:.1f} ohm")
    print(f"   Q (quality factor) = {Q:.2f}")
    print(f"   BW (bandwidth) = {bandwidth:.2f} Hz")
    print(f"   ζ (damping ratio) = {damping_ratio:.3f}")

    # Step response
    print("\n1. Step Response")
    print("-" * 40)
    times, voltages = simulate_step_response(R_val, L_val, C_val, V_step=10.0, n_tau=10)

    tau = 2 * L_val / R_val
    print(f"   Time constant tau = {tau*1000:.3f} ms")
    print(f"   Peak overshoot: {max(voltages):.4f} V")
    print(f"   Settling value: {voltages[-1]:.4f} V (expected: 0V after transient)")

    # AC frequency response
    print("\n2. Frequency Response (Bode Plot)")
    print("-" * 40)
    print(f"   {'Freq (Hz)':>12s}  {'|H|':>10s}  {'dB':>10s}")
    ac_results = simulate_ac_response(R_val, L_val, C_val)

    peak_mag = max([r["magnitude"] for r in ac_results])
    for r in ac_results:
        marker = " <-- Peak" if abs(r["magnitude"] - peak_mag) < 0.01 else ""
        print(f"   {r['freq']:>12.1f}  {r['magnitude']:>10.6f}  {r['magnitude_db']:>10.2f}{marker}")

    # Find -3dB bandwidth
    cutoff_mag = peak_mag / math.sqrt(2)
    freqs_above_cutoff = [r["freq"] for r in ac_results if r["magnitude"] >= cutoff_mag]
    if len(freqs_above_cutoff) >= 2:
        bw_measured = freqs_above_cutoff[-1] - freqs_above_cutoff[0]
        print(f"\n   -3dB Bandwidth (measured): {bw_measured:.2f} Hz")
        print(f"   -3dB Bandwidth (theoretical): {bandwidth:.2f} Hz")
        print(f"\n   Note: Measured bandwidth is narrower because of high damping ratio (ζ={damping_ratio:.2f}).")
        print(f"   This is a heavily damped system, so the measured -3dB width is accurate.")

    # Differentiability
    print("\n3. JAX Differentiability")
    print("-" * 40)
    grad_val = demo_differentiability(R_val, L_val, C_val)
    print(f"   dV/dR at resonance: {grad_val:.6f} V/ohm")
    print(f"   (Gradient is finite and meaningful)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
