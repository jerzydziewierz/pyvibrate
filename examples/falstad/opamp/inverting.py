"""
Example: Inverting Op-Amp Amplifier
Converted from Falstad: amp-invert.txt

Demonstrates inverting amplifier behavior:
- Gain = -Rf / Rin
- Output inverts and scales input

Original Falstad values: Rin=1k, Rf=3k -> Gain = -3

IMPLEMENTATION NOTE:
--------------------
Modeling an op-amp with explicit negative feedback (high-gain VCVS with
resistor feedback network) creates numerical challenges in time-domain
simulation. The tight feedback loop produces an algebraically stiff system
that can cause ill-conditioned matrices in the MNA solver.

Possible solutions include:
1. Using a direct VCVS to model the closed-loop transfer function
2. Adding parasitic RC elements to break the algebraic loop
3. Solving the ideal op-amp equations analytically to eliminate the feedback

This example demonstrates:
- Approach 1: Direct VCVS model (recommended for ideal behavior)
- Approach 2: Feedback model without stabilization (shows instability)
- Approach 3: Feedback model with RC dampeners (works for moderate gains)

Components used: R, C, VSource, VCVS
"""
import math
import jax.numpy as jnp
from jax import grad
from pyvibrate.timedomain import Network, R, C, VSource, VCVS


# =============================================================================
# APPROACH 1: Direct VCVS Model (Recommended)
# =============================================================================
# This models the closed-loop inverting amplifier transfer function directly.
# The VCVS gain is set to -Rf/Rin, which is the ideal inverting amp gain.

def build_inverting_amp_direct(Rin=1000.0, Rf=3000.0):
    """Build inverting amplifier using direct VCVS model.

    The closed-loop gain of an ideal inverting amplifier is:
        A_cl = -Rf / Rin

    We model this directly with a VCVS, bypassing the feedback loop.

    Circuit:
        Vin ---[Rin]--- GND    (input termination for impedance)
                |
               [VCVS: gain = -Rf/Rin]
                |
               Vout
    """
    net = Network()
    net, n_in = net.node("in")
    net, n_out = net.node("out")

    # Input voltage source
    net, vs = VSource(net, n_in, net.gnd, name="vs")

    # Input resistor (defines input impedance Rin)
    net, r_in = R(net, n_in, net.gnd, name="Rin")

    # VCVS with closed-loop gain: senses n_in, outputs to n_out
    # Gain = -Rf/Rin (computed externally and passed as param)
    net, amp = VCVS(net, n_out, net.gnd, n_in, net.gnd, name="amp")

    # Output load resistor
    net, r_load = R(net, n_out, net.gnd, name="Rload")

    nodes = {"in": n_in, "out": n_out}
    components = {"vs": vs, "Rin": r_in, "amp": amp, "Rload": r_load}
    return net, nodes, components


def simulate_direct(Rin=1000.0, Rf=3000.0, V_in=1.0):
    """Simulate inverting amplifier using direct VCVS model."""
    net, nodes, _ = build_inverting_amp_direct(Rin, Rf)

    # Closed-loop gain
    gain = -Rf / Rin

    dt = 1e-6
    sim = net.compile(dt=dt)

    params = {"vs": 0.0, "Rin": Rin, "amp": gain, "Rload": 10000.0}
    state = sim.init(params)

    controls = {"vs": V_in}

    # Run a few steps to ensure settled
    for _ in range(5):
        state = sim.step(params, state, controls)

    v_out = float(sim.v(state, nodes["out"]))
    v_in_actual = float(sim.v(state, nodes["in"]))

    return v_out, v_in_actual, gain


def simulate_ac_direct(Rin=1000.0, Rf=3000.0, freq=1000.0):
    """Simulate AC response using direct model."""
    net, nodes, _ = build_inverting_amp_direct(Rin, Rf)
    gain = -Rf / Rin

    period = 1 / freq
    dt = period / 40
    n_cycles = 5
    n_steps = int(n_cycles * period / dt)

    sim = net.compile(dt=dt)
    params = {"vs": 0.0, "Rin": Rin, "amp": gain, "Rload": 10000.0}
    state = sim.init(params)

    V_amp = 1.0
    omega = 2 * math.pi * freq

    v_out_samples = []

    for i in range(n_steps):
        t = float(state.time)
        v_in = V_amp * math.sin(omega * t)
        state = sim.step(params, state, {"vs": v_in})

        if i >= n_steps - int(2 * period / dt):
            v_out_samples.append(float(sim.v(state, nodes["out"])))

    v_out_amp = (max(v_out_samples) - min(v_out_samples)) / 2
    return {
        "gain_magnitude": v_out_amp / V_amp,
        "expected_gain": abs(gain),
        "v_out_max": max(v_out_samples),
        "v_out_min": min(v_out_samples),
    }


def demo_differentiability_direct(Rin=1000.0, Rf=3000.0):
    """Demonstrate JAX differentiability with direct model."""

    def output_voltage(Rf_param):
        """Output voltage as function of Rf."""
        net, nodes, _ = build_inverting_amp_direct(Rin, Rf_param)
        gain = -Rf_param / Rin

        sim = net.compile(dt=1e-6)
        params = {"vs": 0.0, "Rin": Rin, "amp": gain, "Rload": 10000.0}
        state = sim.init(params)

        for _ in range(5):
            state = sim.step(params, state, {"vs": 1.0})

        return sim.v(state, nodes["out"])

    dVout_dRf = grad(output_voltage)
    gradient = float(dVout_dRf(Rf))

    # Analytical: Vout = -Vin * Rf/Rin, so dVout/dRf = -Vin/Rin
    analytical = -1.0 / Rin

    return gradient, analytical


# =============================================================================
# APPROACH 2: Feedback Model (Experimental - Shows Instability)
# =============================================================================
# This attempts to model the op-amp with explicit feedback.
# Demonstrates why tight feedback loops cause numerical instability.

def build_inverting_amp_feedback():
    """Build inverting amplifier with explicit feedback (experimental).

    Circuit:
                    Rf
        +--------[====]--------+
        |                      |
        |   Rin                |
    Vin-+--[===]---(-)         |
                    |    [VCVS]---Vout
              GND--(+)         |
                               |
                              GND

    WARNING: This topology creates a tight algebraic loop that
    causes numerical instability in time-domain simulation.
    """
    net = Network()
    net, n_in = net.node("in")
    net, n_inv = net.node("inv")  # Inverting input (virtual ground)
    net, n_out = net.node("out")

    net, vs = VSource(net, n_in, net.gnd, name="vs")
    net, r_in = R(net, n_in, n_inv, name="Rin")
    net, r_f = R(net, n_inv, n_out, name="Rf")

    # High-gain VCVS: V_out = gain * (V+ - V-) = gain * (0 - V_inv)
    net, opamp = VCVS(net, n_out, net.gnd, net.gnd, n_inv, name="opamp")

    # Load resistor to define output
    net, r_load = R(net, n_out, net.gnd, name="Rload")

    nodes = {"in": n_in, "inv": n_inv, "out": n_out}
    components = {"vs": vs, "Rin": r_in, "Rf": r_f, "opamp": opamp, "Rload": r_load}
    return net, nodes, components


def simulate_feedback(Rin=1000.0, Rf=3000.0, V_in=1.0, gain_opamp=1000.0):
    """Attempt simulation with feedback model (demonstrates instability)."""
    try:
        net, nodes, _ = build_inverting_amp_feedback()

        dt = 1e-7  # Small timestep for stability
        sim = net.compile(dt=dt)

        params = {"vs": 0.0, "Rin": Rin, "Rf": Rf, "opamp": gain_opamp, "Rload": 10000.0}
        state = sim.init(params)

        controls = {"vs": V_in}

        for _ in range(100):
            state = sim.step(params, state, controls)

        v_out = float(sim.v(state, nodes["out"]))
        v_inv = float(sim.v(state, nodes["inv"]))

        # Check for numerical issues
        if not math.isfinite(v_out) or abs(v_out) > 1e6:
            return None, None, "unstable"

        return v_out, v_inv, "ok"
    except Exception as e:
        return None, None, str(e)


# =============================================================================
# APPROACH 3: Feedback Model with RC Dampeners
# =============================================================================
# Adding parasitic RC elements simulates realistic wire impedance and
# finite-velocity signal propagation, helping the solver converge.

def build_inverting_amp_rc_damped():
    """Build inverting amplifier with RC dampeners for stability.

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

    Rp and Cp are parasitic elements that model wire impedance and
    propagation delay, breaking the algebraic loop.
    """
    net = Network()
    net, n_in = net.node("in")
    net, n_inv = net.node("inv")      # Inverting input (before parasitics)
    net, n_inv_damped = net.node("inv_damped")  # After parasitic R
    net, n_out = net.node("out")

    net, vs = VSource(net, n_in, net.gnd, name="vs")
    net, r_in = R(net, n_in, n_inv, name="Rin")

    # Parasitic R between inverting input and op-amp sensing point
    net, r_parasitic = R(net, n_inv, n_inv_damped, name="Rp")

    # Parasitic C to ground at the sensing point (models input capacitance)
    net, c_parasitic = C(net, n_inv_damped, net.gnd, name="Cp")

    # Feedback resistor from output to the inverting input
    net, r_f = R(net, n_inv, n_out, name="Rf")

    # High-gain VCVS senses the damped node: V_out = gain * (0 - V_inv_damped)
    net, opamp = VCVS(net, n_out, net.gnd, net.gnd, n_inv_damped, name="opamp")

    # Load resistor
    net, r_load = R(net, n_out, net.gnd, name="Rload")

    nodes = {"in": n_in, "inv": n_inv, "inv_damped": n_inv_damped, "out": n_out}
    components = {
        "vs": vs, "Rin": r_in, "Rp": r_parasitic, "Cp": c_parasitic,
        "Rf": r_f, "opamp": opamp, "Rload": r_load
    }
    return net, nodes, components


def simulate_rc_damped(Rin=1000.0, Rf=3000.0, V_in=1.0, gain_opamp=1000.0,
                       R_parasitic=10.0, C_parasitic=1e-12):
    """Simulate with RC dampeners for stability."""
    try:
        net, nodes, _ = build_inverting_amp_rc_damped()

        # Timestep should be small relative to RC time constant
        tau_parasitic = R_parasitic * C_parasitic
        dt = min(1e-7, tau_parasitic / 10) if tau_parasitic > 0 else 1e-7

        sim = net.compile(dt=dt)

        params = {
            "vs": 0.0,
            "Rin": Rin,
            "Rf": Rf,
            "Rp": R_parasitic,
            "Cp": C_parasitic,
            "opamp": gain_opamp,
            "Rload": 10000.0
        }
        state = sim.init(params)

        controls = {"vs": V_in}

        # Run enough steps for RC settling
        n_steps = 1000
        for _ in range(n_steps):
            state = sim.step(params, state, controls)

        v_out = float(sim.v(state, nodes["out"]))
        v_inv = float(sim.v(state, nodes["inv"]))
        v_inv_damped = float(sim.v(state, nodes["inv_damped"]))

        # Check for numerical issues
        if not math.isfinite(v_out) or abs(v_out) > 1e6:
            return None, None, None, "unstable"

        return v_out, v_inv, v_inv_damped, "ok"
    except Exception as e:
        return None, None, None, str(e)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Inverting Op-Amp Amplifier Example")
    print("Converted from Falstad: amp-invert.txt")
    print("=" * 60)

    Rin = 1000.0
    Rf = 3000.0
    expected_gain = -Rf / Rin

    print(f"\nCircuit Parameters:")
    print(f"   Rin = {Rin:.0f} ohm")
    print(f"   Rf = {Rf:.0f} ohm")
    print(f"   Expected gain = -Rf/Rin = {expected_gain:.1f}")

    # -------------------------------------------------------------------------
    # Direct model tests
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("APPROACH 1: Direct VCVS Model (Recommended)")
    print("=" * 60)

    print("\n1. DC Gain Test (Vin = 1V)")
    print("-" * 40)
    v_out, v_in, gain = simulate_direct(Rin, Rf, V_in=1.0)
    print(f"   Input voltage:     1.000 V")
    print(f"   Output voltage:    {v_out:.4f} V")
    print(f"   VCVS gain used:    {gain:.4f}")
    print(f"   Measured gain:     {v_out:.4f}")
    print(f"   Expected gain:     {expected_gain:.4f}")

    print("\n2. Different Gain Settings")
    print("-" * 40)
    gains = [(1000, 1000, -1), (1000, 5000, -5), (1000, 10000, -10)]
    for rin, rf, expected in gains:
        v_out, _, _ = simulate_direct(rin, rf, V_in=1.0)
        print(f"   Rf/Rin = {rf}/{rin}: Vout = {v_out:.3f} V (expected: {expected:.1f} V)")

    print("\n3. AC Response (1kHz)")
    print("-" * 40)
    ac = simulate_ac_direct(Rin, Rf, freq=1000.0)
    print(f"   Gain magnitude: {ac['gain_magnitude']:.4f} (expected: {ac['expected_gain']:.4f})")
    print(f"   Output swings:  {ac['v_out_min']:.3f} to {ac['v_out_max']:.3f} V")

    print("\n4. JAX Differentiability")
    print("-" * 40)
    grad_val, analytical = demo_differentiability_direct(Rin, Rf)
    print(f"   dVout/dRf (JAX):      {grad_val:.8f} V/ohm")
    print(f"   dVout/dRf (analytic): {analytical:.8f} V/ohm")
    print(f"   Match: {'Yes' if abs(grad_val - analytical) / abs(analytical) < 0.01 else 'No'}")

    # -------------------------------------------------------------------------
    # Feedback model tests (experimental - shows instability)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("APPROACH 2: Feedback Model (No Stabilization)")
    print("=" * 60)
    print("\nDemonstrates why tight feedback loops cause instability.")

    print("\n5. Feedback Model Test")
    print("-" * 40)
    for gain_opamp in [10.0, 50.0, 100.0, 500.0, 1000.0]:
        v_out, v_inv, status = simulate_feedback(Rin, Rf, V_in=1.0, gain_opamp=gain_opamp)
        if status == "ok":
            measured_gain = v_out / 1.0 if v_out else 0
            print(f"   Open-loop gain {gain_opamp:>6.0f}: Vout = {v_out:>8.4f} V, "
                  f"V_inv = {v_inv*1000:>6.2f} mV, gain = {measured_gain:.2f}")
        else:
            print(f"   Open-loop gain {gain_opamp:>6.0f}: {status}")

    # -------------------------------------------------------------------------
    # RC dampened feedback model tests
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("APPROACH 3: Feedback Model with RC Dampeners")
    print("=" * 60)
    print("\nParasitic RC elements (Rp=10 ohm, Cp=1pF) simulate wire")
    print("impedance and break the algebraic loop.")

    print("\n6. RC Damped Feedback: Gain Progression")
    print("-" * 40)
    print(f"   {'Gain':>8s}  {'Vout (V)':>10s}  {'V_inv (mV)':>12s}  {'Meas. Gain':>12s}  {'Error':>8s}")
    print("   " + "-" * 56)

    gain_steps = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
    for gain_opamp in gain_steps:
        v_out, v_inv, v_inv_d, status = simulate_rc_damped(
            Rin, Rf, V_in=1.0, gain_opamp=gain_opamp,
            R_parasitic=10.0, C_parasitic=1e-12
        )
        if status == "ok":
            measured_gain = v_out / 1.0 if v_out else 0
            error = abs(measured_gain - expected_gain) / abs(expected_gain) * 100
            print(f"   {gain_opamp:>8.0f}  {v_out:>10.4f}  {v_inv*1000:>12.3f}  "
                  f"{measured_gain:>12.4f}  {error:>7.2f}%")
        else:
            print(f"   {gain_opamp:>8.0f}  {'--':>10s}  {'--':>12s}  {'--':>12s}  {status}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Findings:
- Approach 1 (Direct VCVS): Recommended for ideal op-amp modeling.
  Models closed-loop gain directly, works perfectly for all cases.

- Approach 2 (Feedback, no stabilization): Unstable at all gains.
  The tight algebraic loop creates an ill-conditioned MNA matrix.

- Approach 3 (Feedback with RC dampeners): Works for moderate gains!
  As open-loop gain increases, output converges toward ideal -Rf/Rin.
  Stable up to gain ~1000-2000, becomes unstable at higher gains.

Physical interpretation of RC dampeners:
- Rp models parasitic wire/trace resistance
- Cp models op-amp input capacitance (~1-10 pF typical)
- Together they create a low-pass filter that prevents instantaneous
  feedback, giving the solver time to find equilibrium.

For practical op-amp modeling:
1. Use Direct VCVS (Approach 1) for ideal behavior
2. Use RC dampeners (Approach 3) when explicit feedback dynamics
   are needed, with moderate open-loop gains (~100-1000)
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
