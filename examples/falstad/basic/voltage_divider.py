"""
Example: Voltage Divider
Converted from Falstad: voltdivide.txt

Demonstrates the voltage divider rule: V_out = V_in * R2 / (R1 + R2)

Two examples:
1. Simple 2-resistor divider (50% division with equal resistors)
2. 4-resistor divider chain showing multiple tap points

Components used: R, VSource
"""
import jax.numpy as jnp
from jax import grad
from pyvibrate.timedomain import Network, R, VSource


def build_simple_divider():
    """Build a simple 2-resistor voltage divider.

    Circuit:
        Vs ---[R1]---+---[R2]--- GND
                     |
                    Vout
    """
    net = Network()
    net, n_top = net.node("top")      # Top of divider (Vs output)
    net, n_mid = net.node("mid")      # Middle tap point

    net, vs = VSource(net, n_top, net.gnd, name="vs")
    net, r1 = R(net, n_top, n_mid, name="R1")
    net, r2 = R(net, n_mid, net.gnd, name="R2")

    return net, {"top": n_top, "mid": n_mid}, {"vs": vs, "R1": r1, "R2": r2}


def build_chain_divider():
    """Build a 4-resistor chain divider with multiple taps.

    Circuit:
        Vs ---[R1]---+---[R2]---+---[R3]---+---[R4]--- GND
                     |          |          |
                   tap1       tap2       tap3
    """
    net = Network()
    net, n_top = net.node("top")
    net, tap1 = net.node("tap1")
    net, tap2 = net.node("tap2")
    net, tap3 = net.node("tap3")

    net, vs = VSource(net, n_top, net.gnd, name="vs")
    net, r1 = R(net, n_top, tap1, name="R1")
    net, r2 = R(net, tap1, tap2, name="R2")
    net, r3 = R(net, tap2, tap3, name="R3")
    net, r4 = R(net, tap3, net.gnd, name="R4")

    nodes = {"top": n_top, "tap1": tap1, "tap2": tap2, "tap3": tap3}
    components = {"vs": vs, "R1": r1, "R2": r2, "R3": r3, "R4": r4}
    return net, nodes, components


def simulate_simple_divider(V_in=10.0, R1=10000.0, R2=10000.0):
    """Simulate simple voltage divider and return output voltage."""
    net, nodes, _ = build_simple_divider()

    dt = 1e-6
    sim = net.compile(dt=dt)

    params = {"vs": 0.0, "R1": R1, "R2": R2}
    state = sim.init(params)

    # Apply input voltage
    controls = {"vs": V_in}
    state = sim.step(params, state, controls)

    v_out = float(sim.v(state, nodes["mid"]))
    return v_out


def simulate_chain_divider(V_in=10.0, R_val=10000.0):
    """Simulate 4-resistor chain divider and return tap voltages."""
    net, nodes, _ = build_chain_divider()

    dt = 1e-6
    sim = net.compile(dt=dt)

    params = {"vs": 0.0, "R1": R_val, "R2": R_val, "R3": R_val, "R4": R_val}
    state = sim.init(params)

    controls = {"vs": V_in}
    state = sim.step(params, state, controls)

    v_tap1 = float(sim.v(state, nodes["tap1"]))
    v_tap2 = float(sim.v(state, nodes["tap2"]))
    v_tap3 = float(sim.v(state, nodes["tap3"]))

    return {"tap1": v_tap1, "tap2": v_tap2, "tap3": v_tap3}


def demo_differentiability():
    """Demonstrate JAX differentiability of voltage divider."""

    def v_out_vs_r2(R2_val):
        """Output voltage as function of R2."""
        net, nodes, _ = build_simple_divider()
        sim = net.compile(dt=1e-6)
        params = {"vs": 0.0, "R1": 10000.0, "R2": R2_val}
        state = sim.init(params)
        state = sim.step(params, state, {"vs": 10.0})
        return sim.v(state, nodes["mid"])

    # Compute gradient dVout/dR2
    dVout_dR2 = grad(v_out_vs_r2)

    R2_test = 10000.0
    gradient = float(dVout_dR2(R2_test))

    # Analytical: Vout = Vin * R2/(R1+R2)
    # dVout/dR2 = Vin * R1 / (R1+R2)^2
    R1 = 10000.0
    V_in = 10.0
    analytical = V_in * R1 / (R1 + R2_test)**2

    return gradient, analytical


def main():
    print("=" * 60)
    print("Voltage Divider Example")
    print("Converted from Falstad: voltdivide.txt")
    print("=" * 60)

    # Simple divider test
    print("\n1. Simple Voltage Divider (R1 = R2 = 10k)")
    print("-" * 40)
    V_in = 10.0
    v_out = simulate_simple_divider(V_in=V_in, R1=10000.0, R2=10000.0)
    expected = V_in * 0.5  # Equal resistors = 50% division
    print(f"   Input voltage:    {V_in:.2f} V")
    print(f"   Output voltage:   {v_out:.4f} V")
    print(f"   Expected (50%):   {expected:.2f} V")
    print(f"   Error:            {abs(v_out - expected):.6f} V")

    # Unequal resistors
    print("\n2. Unequal Resistors (R1=10k, R2=20k)")
    print("-" * 40)
    v_out_2 = simulate_simple_divider(V_in=V_in, R1=10000.0, R2=20000.0)
    expected_2 = V_in * 20000 / (10000 + 20000)  # 2/3
    print(f"   Input voltage:    {V_in:.2f} V")
    print(f"   Output voltage:   {v_out_2:.4f} V")
    print(f"   Expected (2/3):   {expected_2:.4f} V")

    # Chain divider
    print("\n3. 4-Resistor Chain (equal 10k resistors)")
    print("-" * 40)
    taps = simulate_chain_divider(V_in=V_in, R_val=10000.0)
    print(f"   Input voltage:    {V_in:.2f} V")
    print(f"   Tap 1 (75%):      {taps['tap1']:.4f} V (expected: {V_in*0.75:.2f})")
    print(f"   Tap 2 (50%):      {taps['tap2']:.4f} V (expected: {V_in*0.50:.2f})")
    print(f"   Tap 3 (25%):      {taps['tap3']:.4f} V (expected: {V_in*0.25:.2f})")

    # Differentiability demo
    print("\n4. JAX Differentiability")
    print("-" * 40)
    grad_val, analytical = demo_differentiability()
    print(f"   dVout/dR2 (JAX):      {grad_val:.10f}")
    print(f"   dVout/dR2 (analytic): {analytical:.10f}")
    print(f"   Match: {'Yes' if abs(grad_val - analytical) < 1e-8 else 'No'}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
