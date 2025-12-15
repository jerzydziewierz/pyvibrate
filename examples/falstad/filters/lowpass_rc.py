"""
Example: RC Low-Pass Filter
Converted from Falstad: filt-lopass.txt

Demonstrates first-order RC low-pass filter behavior:
- Cutoff frequency: f_c = 1 / (2 * pi * R * C)
- At f_c: output is -3dB (0.707x) and phase is -45 degrees
- Time constant: tau = R * C

Original Falstad values: R=187 ohm, C=10uF -> f_c ~ 85 Hz

Components used: R, C, VSource
"""
import math
import jax.numpy as jnp
from jax import grad
from pyvibrate.timedomain import Network, R, C, VSource


def build_lowpass_filter():
    """Build RC low-pass filter.

    Circuit:
        Vin ---[R]---+--- Vout
                     |
                    [C]
                     |
                    GND
    """
    net = Network()
    net, n_in = net.node("in")
    net, n_out = net.node("out")

    net, vs = VSource(net, n_in, net.gnd, name="vs")
    net, r1 = R(net, n_in, n_out, name="R1")
    net, c1 = C(net, n_out, net.gnd, name="C1")

    return net, {"in": n_in, "out": n_out}, {"vs": vs, "R1": r1, "C1": c1}


def simulate_step_response(R_val=1000.0, C_val=1e-6, V_step=5.0, n_tau=5):
    """Simulate step response and return time/voltage arrays.

    Args:
        R_val: Resistance in ohms
        C_val: Capacitance in farads
        V_step: Step voltage amplitude
        n_tau: Number of time constants to simulate

    Returns:
        times: Array of time values
        voltages: Array of output voltage values
    """
    net, nodes, _ = build_lowpass_filter()

    tau = R_val * C_val
    dt = tau / 100  # 100 samples per time constant
    n_steps = int(n_tau * tau / dt)

    sim = net.compile(dt=dt)
    params = {"vs": 0.0, "R1": R_val, "C1": C_val}
    state = sim.init(params)

    times = []
    voltages = []

    controls = {"vs": V_step}

    for i in range(n_steps):
        state = sim.step(params, state, controls)
        times.append(float(state.time))
        voltages.append(float(sim.v(state, nodes["out"])))

    return times, voltages


def simulate_ac_response(R_val=1000.0, C_val=1e-6, frequencies=None):
    """Simulate AC response at multiple frequencies.

    Returns magnitude and phase at each frequency.
    """
    if frequencies is None:
        # Default: sweep from 0.1*fc to 10*fc
        f_c = 1 / (2 * math.pi * R_val * C_val)
        frequencies = [f_c * mult for mult in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]

    net, nodes, _ = build_lowpass_filter()

    results = []

    for freq in frequencies:
        # Use period/20 timestep for good resolution
        period = 1.0 / freq
        dt = period / 40
        n_cycles = 10  # Simulate enough cycles for steady state
        n_steps = int(n_cycles * period / dt)

        sim = net.compile(dt=dt)
        params = {"vs": 0.0, "R1": R_val, "C1": C_val}
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

            # Collect last 2 cycles for analysis
            if i >= n_steps - int(2 * period / dt):
                v_in_last.append(v_in)
                v_out_last.append(float(sim.v(state, nodes["out"])))

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


def demo_differentiability(R_val=1000.0, C_val=1e-6):
    """Demonstrate JAX differentiability of filter response."""

    def voltage_at_tau(C_param):
        """Output voltage at t=tau as function of C."""
        net, nodes, _ = build_lowpass_filter()
        tau = R_val * C_param
        dt = tau / 100
        n_steps = 100  # Simulate for 1 tau

        sim = net.compile(dt=dt)
        params = {"vs": 0.0, "R1": R_val, "C1": C_param}
        state = sim.init(params)

        for _ in range(n_steps):
            state = sim.step(params, state, {"vs": 5.0})

        return sim.v(state, nodes["out"])

    dV_dC = grad(voltage_at_tau)
    gradient = float(dV_dC(C_val))

    return gradient


def main():
    print("=" * 60)
    print("RC Low-Pass Filter Example")
    print("Converted from Falstad: filt-lopass.txt")
    print("=" * 60)

    # Parameters (similar to Falstad original)
    R_val = 1000.0   # 1k ohm
    C_val = 1e-6     # 1 uF
    tau = R_val * C_val
    f_c = 1 / (2 * math.pi * tau)

    print(f"\nFilter Parameters:")
    print(f"   R = {R_val:.0f} ohm")
    print(f"   C = {C_val*1e6:.1f} uF")
    print(f"   tau = R*C = {tau*1000:.3f} ms")
    print(f"   f_c = 1/(2*pi*tau) = {f_c:.1f} Hz")

    # Step response
    print("\n1. Step Response")
    print("-" * 40)
    times, voltages = simulate_step_response(R_val, C_val, V_step=5.0, n_tau=5)

    # Find voltage at key times
    for target_tau in [1, 2, 3, 5]:
        target_t = target_tau * tau
        idx = min(range(len(times)), key=lambda i: abs(times[i] - target_t))
        v = voltages[idx]
        expected = 5.0 * (1 - math.exp(-target_tau))
        print(f"   At t={target_tau}*tau: V_out = {v:.4f} V (expected: {expected:.4f} V)")

    # AC frequency response
    print("\n2. Frequency Response")
    print("-" * 40)
    print(f"   {'Freq':>10s}  {'f/f_c':>8s}  {'|H|':>8s}  {'dB':>8s}  {'Expected dB':>12s}")
    ac_results = simulate_ac_response(R_val, C_val)
    for r in ac_results:
        f_ratio = r["freq"] / f_c
        # Expected: |H| = 1 / sqrt(1 + (f/fc)^2)
        expected_mag = 1 / math.sqrt(1 + f_ratio**2)
        expected_db = 20 * math.log10(expected_mag)
        print(f"   {r['freq']:>10.1f}  {f_ratio:>8.2f}  {r['magnitude']:>8.4f}  "
              f"{r['magnitude_db']:>8.2f}  {expected_db:>12.2f}")

    # Differentiability
    print("\n3. JAX Differentiability")
    print("-" * 40)
    grad_val = demo_differentiability(R_val, C_val)
    print(f"   dV/dC at t=tau: {grad_val:.2f} V/F")
    print(f"   (Gradient is finite and meaningful)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
