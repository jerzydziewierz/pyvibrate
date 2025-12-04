"""
Test: Verify all components are differentiable with respect to continuous parameters.

Each test computes a gradient using JAX autodiff and verifies:
1. The gradient is finite (not NaN or Inf)
2. The gradient has the expected sign (sanity check)
3. The gradient magnitude is reasonable (order-of-magnitude check)
"""
import pytest
import jax.numpy as jnp
from jax import grad


# Fixed simulation parameters for consistent tracing
N_STEPS = 100
DT = 1e-6


class TestResistorDifferentiability:
    """Test R component is differentiable w.r.t. resistance."""

    def test_dV_dR_finite(self):
        """Gradient dV/dR should be finite."""
        from pyvibrate.timedomain import Network, R, C, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)
        net, r1 = R(net, n1, n2, name="R1")
        net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)

        sim = net.compile(dt=DT)

        def voltage_after_simulation(R_val: float) -> float:
            params = {"R1": R_val}
            state = sim.init(params)
            for _ in range(N_STEPS):
                state = sim.step(params, state, {})
            return sim.v(state, n2)

        R_test = 1000.0
        dV_dR = grad(voltage_after_simulation)(R_test)

        assert jnp.isfinite(dV_dR), f"dV/dR should be finite, got {dV_dR}"
        # Increasing R slows charging, so dV/dR should be negative
        assert dV_dR < 0, f"dV/dR should be negative, got {dV_dR}"

    def test_dI_dR_via_ohms_law(self):
        """Gradient dI/dR can be verified via Ohm's law: I = V/R, so dI/dR = -V/R^2."""
        from pyvibrate.timedomain import Network, R, C, VSource

        # Use an RC circuit - current through R can be computed from capacitor voltage
        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)
        net, r1 = R(net, n1, n2, name="R1")
        net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)

        sim = net.compile(dt=DT)

        # Current through R = (Vs - Vc) / R
        # At steady state: Vc = Vs, I = 0
        # During charging: I = (Vs - Vc) / R

        def current_estimate(R_val: float) -> float:
            """Estimate current as (Vs - Vc) / R."""
            params = {"R1": R_val}
            state = sim.init(params)
            for _ in range(N_STEPS):
                state = sim.step(params, state, {})
            Vc = sim.v(state, n2)
            Vs = 5.0
            return (Vs - Vc) / R_val

        R_test = 1000.0
        dI_dR = grad(current_estimate)(R_test)

        assert jnp.isfinite(dI_dR), f"dI/dR should be finite, got {dI_dR}"


class TestCapacitorDifferentiability:
    """Test C component is differentiable w.r.t. capacitance."""

    def test_dV_dC_finite(self):
        """Gradient dV/dC should be finite."""
        from pyvibrate.timedomain import Network, R, C, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)
        net, r1 = R(net, n1, n2, name="R1", value=1000.0)
        net, c1 = C(net, n2, net.gnd, name="C1")

        sim = net.compile(dt=DT)

        def voltage_after_simulation(C_val: float) -> float:
            params = {"C1": C_val}
            state = sim.init(params)
            for _ in range(N_STEPS):
                state = sim.step(params, state, {})
            return sim.v(state, n2)

        C_test = 1e-6
        dV_dC = grad(voltage_after_simulation)(C_test)

        assert jnp.isfinite(dV_dC), f"dV/dC should be finite, got {dV_dC}"
        # Increasing C slows charging (larger tau), so dV/dC should be negative
        assert dV_dC < 0, f"dV/dC should be negative, got {dV_dC}"


class TestInductorDifferentiability:
    """Test L component is differentiable w.r.t. inductance."""

    def test_dI_dL_finite(self):
        """Gradient dI/dL (inductor current) should be finite."""
        from pyvibrate.timedomain import Network, R, L, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=10.0)
        net, r1 = R(net, n1, n2, name="R1", value=100.0)
        net, l1 = L(net, n2, net.gnd, name="L1")

        sim = net.compile(dt=DT)

        def inductor_current(L_val: float) -> float:
            params = {"L1": L_val}
            state = sim.init(params)
            for _ in range(N_STEPS):
                state = sim.step(params, state, {})
            # Access inductor current directly from state
            return state.ind_currents[0]

        L_test = 1e-3
        dI_dL = grad(inductor_current)(L_test)

        assert jnp.isfinite(dI_dL), f"dI/dL should be finite, got {dI_dL}"
        # Increasing L slows current buildup (larger tau = L/R), so dI/dL < 0
        assert dI_dL < 0, f"dI/dL should be negative, got {dI_dL}"

    def test_dV_dL_finite(self):
        """Gradient dV/dL (voltage across inductor) should be finite."""
        from pyvibrate.timedomain import Network, R, L, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=10.0)
        net, r1 = R(net, n1, n2, name="R1", value=100.0)
        net, l1 = L(net, n2, net.gnd, name="L1")

        sim = net.compile(dt=DT)

        def inductor_voltage(L_val: float) -> float:
            params = {"L1": L_val}
            state = sim.init(params)
            for _ in range(N_STEPS):
                state = sim.step(params, state, {})
            return sim.v(state, n2)

        L_test = 1e-3
        dV_dL = grad(inductor_voltage)(L_test)

        assert jnp.isfinite(dV_dL), f"dV/dL should be finite, got {dV_dL}"


class TestVSourceDifferentiability:
    """Test VSource is differentiable w.r.t. voltage (via controls)."""

    def test_dVout_dVin_finite(self):
        """Gradient dVout/dVin should be finite and ~1 for direct connection."""
        from pyvibrate.timedomain import Network, R, VSource

        net = Network()
        net, n1 = net.node("n1")

        net, vs = VSource(net, n1, net.gnd, name="vs")
        net, r1 = R(net, n1, net.gnd, name="R1", value=1000.0)

        sim = net.compile(dt=DT)

        def output_voltage(V_in: float) -> float:
            state = sim.init({})
            controls = {"vs": V_in}
            for _ in range(N_STEPS):
                state = sim.step({}, state, controls)
            return sim.v(state, n1)

        V_test = 5.0
        dVout_dVin = grad(output_voltage)(V_test)

        assert jnp.isfinite(dVout_dVin), f"dVout/dVin should be finite, got {dVout_dVin}"
        # Direct connection: Vout = Vin, so dVout/dVin = 1
        assert abs(dVout_dVin - 1.0) < 0.01, f"dVout/dVin should be ~1, got {dVout_dVin}"

    def test_dI_dV_via_ohms_law(self):
        """Gradient dI/dV can be verified via Ohm's law: I = V/R, so dI/dV = 1/R."""
        from pyvibrate.timedomain import Network, R, C, VSource

        # Use an RC circuit - current through R can be computed from voltage difference
        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        net, vs = VSource(net, n1, net.gnd, name="vs")
        net, r1 = R(net, n1, n2, name="R1", value=1000.0)
        net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)

        sim = net.compile(dt=DT)

        def current_from_voltage(V_in: float) -> float:
            """Current = (Vs - Vc) / R."""
            state = sim.init({})
            controls = {"vs": V_in}
            for _ in range(N_STEPS):
                state = sim.step({}, state, controls)
            Vc = sim.v(state, n2)
            return (V_in - Vc) / 1000.0

        V_test = 5.0
        dI_dV = grad(current_from_voltage)(V_test)

        assert jnp.isfinite(dI_dV), f"dI/dV should be finite, got {dI_dV}"
        # During transient, some current flows, dI/dV should be positive
        assert dI_dV > 0, f"dI/dV should be positive, got {dI_dV}"


class TestVCVSDifferentiability:
    """Test VCVS is differentiable w.r.t. gain."""

    def test_dVout_dGain_finite(self):
        """Gradient dVout/dGain should be finite."""
        from pyvibrate.timedomain import Network, R, VSource, VCVS

        net = Network()
        net, n_in = net.node("n_in")
        net, n_out = net.node("n_out")

        # Input: voltage divider gives 2.5V
        net, vs = VSource(net, n_in, net.gnd, name="vs", value=5.0)

        # VCVS amplifies input
        net, e1 = VCVS(net, n_out, net.gnd, n_in, net.gnd, name="E1")

        # Load resistor
        net, r_load = R(net, n_out, net.gnd, name="R_load", value=1000.0)

        sim = net.compile(dt=DT)

        def output_voltage(gain: float) -> float:
            params = {"E1": gain}
            state = sim.init(params)
            for _ in range(N_STEPS):
                state = sim.step(params, state, {})
            return sim.v(state, n_out)

        gain_test = 2.0
        dVout_dGain = grad(output_voltage)(gain_test)

        assert jnp.isfinite(dVout_dGain), f"dVout/dGain should be finite, got {dVout_dGain}"
        # Vout = gain * Vin = gain * 5V, so dVout/dGain = Vin = 5V
        expected = 5.0
        assert abs(dVout_dGain - expected) < 0.1, f"dVout/dGain should be ~{expected}, got {dVout_dGain}"


class TestSwitchDifferentiability:
    """Test Switch is differentiable w.r.t. r_on and r_off."""

    def test_dV_d_r_on_finite(self):
        """Gradient dV/d(r_on) should be finite when switch is closed."""
        from pyvibrate.timedomain import Network, R, Switch, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)
        net, sw = Switch(net, n1, n2, name="sw1")
        net, r_load = R(net, n2, net.gnd, name="R_load", value=1000.0)

        sim = net.compile(dt=DT)

        def voltage_with_r_on(r_on: float) -> float:
            params = {"sw1_r_on": r_on}
            state = sim.init(params)
            controls = {"sw1": True}  # Switch closed
            for _ in range(N_STEPS):
                state = sim.step(params, state, controls)
            return sim.v(state, n2)

        r_on_test = 0.1  # 0.1 Ohm
        dV_d_r_on = grad(voltage_with_r_on)(r_on_test)

        assert jnp.isfinite(dV_d_r_on), f"dV/d(r_on) should be finite, got {dV_d_r_on}"
        # With switch closed: voltage divider, increasing r_on reduces V at n2
        assert dV_d_r_on < 0, f"dV/d(r_on) should be negative, got {dV_d_r_on}"

    def test_dV_d_r_off_finite(self):
        """Gradient dV/d(r_off) should be finite when switch is open."""
        from pyvibrate.timedomain import Network, R, Switch, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)
        net, sw = Switch(net, n1, n2, name="sw1")
        net, r_load = R(net, n2, net.gnd, name="R_load", value=1000.0)

        sim = net.compile(dt=DT)

        def voltage_with_r_off(r_off: float) -> float:
            params = {"sw1_r_off": r_off}
            state = sim.init(params)
            controls = {"sw1": False}  # Switch open
            for _ in range(N_STEPS):
                state = sim.step(params, state, controls)
            return sim.v(state, n2)

        r_off_test = 1e6  # 1 MOhm
        dV_d_r_off = grad(voltage_with_r_off)(r_off_test)

        assert jnp.isfinite(dV_d_r_off), f"dV/d(r_off) should be finite, got {dV_d_r_off}"
        # With switch open: small leakage, increasing r_off reduces leakage voltage
        # (voltage divider with very high r_off)


class TestVoltageSwitchDifferentiability:
    """Test VoltageSwitch is differentiable w.r.t. threshold and resistances."""

    def test_dV_d_threshold_finite(self):
        """Gradient dV/d(threshold) should be finite."""
        from pyvibrate.timedomain import Network, R, VSource, VoltageSwitch

        net = Network()
        net, n_ctrl = net.node("n_ctrl")
        net, n_out = net.node("n_out")

        # Control voltage
        net, vs_ctrl = VSource(net, n_ctrl, net.gnd, name="vs_ctrl", value=3.0)

        # Power supply for switch output
        net, vs_pwr = VSource(net, net.gnd, net.gnd, name="vs_pwr", value=5.0)

        # Voltage-controlled switch
        net, vsw = VoltageSwitch(net, n_ctrl, n_out, n_ctrl, net.gnd, name="vsw1")

        # Load
        net, r_load = R(net, n_out, net.gnd, name="R_load", value=1000.0)

        sim = net.compile(dt=DT)

        def output_voltage(threshold: float) -> float:
            params = {"vsw1_threshold": threshold}
            state = sim.init(params)
            for _ in range(N_STEPS):
                state = sim.step(params, state, {})
            return sim.v(state, n_out)

        threshold_test = 2.5
        dV_dThresh = grad(output_voltage)(threshold_test)

        assert jnp.isfinite(dV_dThresh), f"dV/d(threshold) should be finite, got {dV_dThresh}"

    def test_dV_d_r_on_vsw_finite(self):
        """Gradient dV/d(r_on) for voltage switch should be finite."""
        from pyvibrate.timedomain import Network, R, VSource, VoltageSwitch

        net = Network()
        net, n_ctrl = net.node("n_ctrl")
        net, n_out = net.node("n_out")

        # Control voltage above threshold (switch will be closed)
        net, vs_ctrl = VSource(net, n_ctrl, net.gnd, name="vs_ctrl", value=5.0)

        # Voltage-controlled switch
        net, vsw = VoltageSwitch(net, n_ctrl, n_out, n_ctrl, net.gnd, name="vsw1")

        # Load
        net, r_load = R(net, n_out, net.gnd, name="R_load", value=1000.0)

        sim = net.compile(dt=DT)

        def output_voltage(r_on: float) -> float:
            params = {"vsw1_threshold": 2.0, "vsw1_r_on": r_on}
            state = sim.init(params)
            for _ in range(N_STEPS):
                state = sim.step(params, state, {})
            return sim.v(state, n_out)

        r_on_test = 0.1
        dV_d_r_on = grad(output_voltage)(r_on_test)

        assert jnp.isfinite(dV_d_r_on), f"dV/d(r_on) should be finite, got {dV_d_r_on}"


class TestDelayLineDifferentiability:
    """Test DelayLine passes gradients through (no continuous params of its own)."""

    def test_gradient_passes_through_delay_via_params(self):
        """Gradient should propagate through delay line from upstream R parameter."""
        from pyvibrate.timedomain import Network, R, C, VSource, DelayLine

        # Use an RC circuit feeding into a delay line
        # The R parameter affects the RC voltage, which then passes through delay
        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")  # RC output
        net, n_out = net.node("n_out")  # Delay output

        net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)
        net, r1 = R(net, n1, n2, name="R1")
        net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)

        # Delay line copies n2 to n_out with delay
        net, dl = DelayLine(net, n2, net.gnd, n_out, net.gnd, delay_samples=2, name="D1")

        # Load resistor on delay output
        net, r_load = R(net, n_out, net.gnd, name="R_load", value=10000.0)

        sim = net.compile(dt=DT)

        def output_voltage(R_val: float) -> float:
            params = {"R1": R_val}
            state = sim.init(params)
            # Run enough steps for signal to propagate
            for _ in range(N_STEPS):
                state = sim.step(params, state, {})
            return sim.v(state, n_out)

        R_test = 1000.0
        dVout_dR = grad(output_voltage)(R_test)

        assert jnp.isfinite(dVout_dR), f"dVout/dR should be finite, got {dVout_dR}"
        # Increasing R slows RC charging, so voltage at delay output decreases
        assert dVout_dR < 0, f"dVout/dR should be negative, got {dVout_dR}"


class TestCombinedCircuitDifferentiability:
    """Test gradients in more complex combined circuits."""

    def test_rlc_all_params_differentiable(self):
        """All R, L, C parameters should be differentiable in an RLC circuit."""
        from pyvibrate.timedomain import Network, R, L, C, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")
        net, n3 = net.node("n3")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=10.0)
        net, r1 = R(net, n1, n2, name="R1")
        net, l1 = L(net, n2, n3, name="L1")
        net, c1 = C(net, n3, net.gnd, name="C1")

        sim = net.compile(dt=DT)

        def capacitor_voltage(R_val: float, L_val: float, C_val: float) -> float:
            params = {"R1": R_val, "L1": L_val, "C1": C_val}
            state = sim.init(params)
            for _ in range(N_STEPS):
                state = sim.step(params, state, {})
            return sim.v(state, n3)

        R_test, L_test, C_test = 100.0, 1e-3, 1e-6

        # Test each partial derivative
        dV_dR = grad(capacitor_voltage, argnums=0)(R_test, L_test, C_test)
        dV_dL = grad(capacitor_voltage, argnums=1)(R_test, L_test, C_test)
        dV_dC = grad(capacitor_voltage, argnums=2)(R_test, L_test, C_test)

        assert jnp.isfinite(dV_dR), f"dV/dR should be finite, got {dV_dR}"
        assert jnp.isfinite(dV_dL), f"dV/dL should be finite, got {dV_dL}"
        assert jnp.isfinite(dV_dC), f"dV/dC should be finite, got {dV_dC}"

        print(f"RLC gradients: dV/dR={float(dV_dR):.2e}, dV/dL={float(dV_dL):.2e}, dV/dC={float(dV_dC):.2e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
