"""
Test: Voltage-Controlled Resistor (VCR) component.

Tests:
1. VCR with k=0 behaves like a regular resistor
2. VCR resistance changes with control voltage (R = R0 + k * V_ctrl)
3. VCR is differentiable w.r.t. r0, k, and control voltage
4. Current probe raises NotImplementedError
"""

import math
import pytest
import jax.numpy as jnp
from jax import grad

from pyvibrate.timedomain import Network, R, C, VSource, VCR


class TestVCRBasicOperation:
    """Test VCR with k=0 behaves like a regular resistor."""

    def test_vcr_k_zero_behaves_like_resistor(self):
        """With k=0, VCR should behave exactly like R with same resistance."""
        # Circuit with VCR: Vs -- VCR -- GND
        # Control node must be driven (floating nodes cause singular matrix)
        net = Network()
        net, n1 = net.node("n1")
        net, n_ctrl = net.node("n_ctrl")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=10.0)
        net, v_ctrl = VSource(net, n_ctrl, net.gnd, name="v_ctrl", value=0.0)  # Drive control to 0V
        net, vcr1 = VCR(net, n1, net.gnd, n_ctrl, net.gnd, name="VCR1", r0=1000.0, k=0.0)

        dt = 1e-6
        sim = net.compile(dt=dt)
        state = sim.init({})
        state = sim.step({}, state, {})

        # With Vs=10V and R=1000Ω, current should be 10mA
        # V at n1 should be 10V (directly from voltage source)
        v_n1 = float(sim.v(state, n1))
        assert abs(v_n1 - 10.0) < 0.01

    def test_vcr_with_load_k_zero(self):
        """VCR in voltage divider with k=0."""
        # Circuit: Vs -- R1 -- VCR -- GND
        # With R1=1k and VCR(r0=1k, k=0), voltage at middle should be 5V
        net = Network()
        net, n_src = net.node("n_src")
        net, n_mid = net.node("n_mid")
        net, n_ctrl = net.node("n_ctrl")

        net, vs = VSource(net, n_src, net.gnd, name="vs", value=10.0)
        net, v_ctrl = VSource(net, n_ctrl, net.gnd, name="v_ctrl", value=0.0)  # Drive control
        net, r1 = R(net, n_src, n_mid, name="R1", value=1000.0)
        net, vcr1 = VCR(net, n_mid, net.gnd, n_ctrl, net.gnd, name="VCR1", r0=1000.0, k=0.0)

        sim = net.compile(dt=1e-6)
        state = sim.init({})
        state = sim.step({}, state, {})

        v_mid = float(sim.v(state, n_mid))
        # Voltage divider: 10V * 1k/(1k+1k) = 5V
        assert abs(v_mid - 5.0) < 0.01


class TestVCRVoltageControl:
    """Test VCR resistance varies with control voltage.

    Note: VCR uses the control voltage from the PREVIOUS state,
    creating a 1-step delay. Tests account for this by running 2 steps.
    """

    def test_vcr_positive_k(self):
        """With positive k, increasing V_ctrl should increase R."""
        # Circuit: Vs -- R1 -- VCR -- GND
        # V_ctrl comes from a separate voltage source
        net = Network()
        net, n_src = net.node("n_src")
        net, n_mid = net.node("n_mid")
        net, n_ctrl = net.node("n_ctrl")

        net, vs = VSource(net, n_src, net.gnd, name="vs", value=10.0)
        net, v_ctrl = VSource(net, n_ctrl, net.gnd, name="v_ctrl", value=0.0)
        net, r1 = R(net, n_src, n_mid, name="R1", value=1000.0)
        # VCR: R = 1000 + 100*V_ctrl
        net, vcr1 = VCR(net, n_mid, net.gnd, n_ctrl, net.gnd, name="VCR1", r0=1000.0, k=100.0)

        sim = net.compile(dt=1e-6)

        # Case 1: V_ctrl = 0V -> R_vcr = 1000Ω
        # Need 2 steps: first step populates state, second uses that state
        state = sim.init({})
        state = sim.step({}, state, {"v_ctrl": 0.0})
        state = sim.step({}, state, {"v_ctrl": 0.0})
        v_mid_0 = float(sim.v(state, n_mid))

        # Case 2: V_ctrl = 10V -> R_vcr = 2000Ω
        state = sim.init({})
        state = sim.step({}, state, {"v_ctrl": 10.0})
        state = sim.step({}, state, {"v_ctrl": 10.0})
        v_mid_10 = float(sim.v(state, n_mid))

        # With V_ctrl=0: divider 10V * 1k/(1k+1k) = 5V
        assert abs(v_mid_0 - 5.0) < 0.1

        # With V_ctrl=10: divider 10V * 2k/(1k+2k) = 6.67V
        expected_v_mid_10 = 10.0 * 2000.0 / (1000.0 + 2000.0)
        assert abs(v_mid_10 - expected_v_mid_10) < 0.1

    def test_vcr_negative_k(self):
        """With negative k, increasing V_ctrl should decrease R."""
        net = Network()
        net, n_src = net.node("n_src")
        net, n_mid = net.node("n_mid")
        net, n_ctrl = net.node("n_ctrl")

        net, vs = VSource(net, n_src, net.gnd, name="vs", value=10.0)
        net, v_ctrl = VSource(net, n_ctrl, net.gnd, name="v_ctrl", value=0.0)
        net, r1 = R(net, n_src, n_mid, name="R1", value=1000.0)
        # VCR: R = 2000 - 100*V_ctrl (negative k)
        net, vcr1 = VCR(net, n_mid, net.gnd, n_ctrl, net.gnd, name="VCR1", r0=2000.0, k=-100.0)

        sim = net.compile(dt=1e-6)

        # Case 1: V_ctrl = 0V -> R_vcr = 2000Ω (2 steps for delay)
        state = sim.init({})
        state = sim.step({}, state, {"v_ctrl": 0.0})
        state = sim.step({}, state, {"v_ctrl": 0.0})
        v_mid_0 = float(sim.v(state, n_mid))

        # Case 2: V_ctrl = 10V -> R_vcr = 1000Ω
        state = sim.init({})
        state = sim.step({}, state, {"v_ctrl": 10.0})
        state = sim.step({}, state, {"v_ctrl": 10.0})
        v_mid_10 = float(sim.v(state, n_mid))

        # With V_ctrl=0: divider 10V * 2k/(1k+2k) = 6.67V
        expected_v_mid_0 = 10.0 * 2000.0 / 3000.0
        assert abs(v_mid_0 - expected_v_mid_0) < 0.1

        # With V_ctrl=10: divider 10V * 1k/(1k+1k) = 5V
        assert abs(v_mid_10 - 5.0) < 0.1

    def test_vcr_dynamic_control(self):
        """VCR responds to changing control voltage over multiple steps."""
        net = Network()
        net, n_src = net.node("n_src")
        net, n_mid = net.node("n_mid")
        net, n_ctrl = net.node("n_ctrl")

        net, vs = VSource(net, n_src, net.gnd, name="vs", value=10.0)
        net, v_ctrl = VSource(net, n_ctrl, net.gnd, name="v_ctrl", value=0.0)
        net, r1 = R(net, n_src, n_mid, name="R1", value=1000.0)
        net, vcr1 = VCR(net, n_mid, net.gnd, n_ctrl, net.gnd, name="VCR1", r0=1000.0, k=100.0)

        sim = net.compile(dt=1e-6)
        state = sim.init({})

        # Sweep control voltage and verify resistance changes
        # Run 2 steps per control value to let it settle (due to 1-step delay)
        v_mids = []
        for v_c in [0.0, 5.0, 10.0, 5.0, 0.0]:
            state = sim.step({}, state, {"v_ctrl": v_c})
            state = sim.step({}, state, {"v_ctrl": v_c})
            v_mids.append(float(sim.v(state, n_mid)))

        # Voltages should change with control
        assert v_mids[0] < v_mids[1] < v_mids[2]  # Increasing
        assert v_mids[2] > v_mids[3] > v_mids[4]  # Decreasing


class TestVCRDifferentiability:
    """Test VCR gradients work correctly.

    Note: VCR uses control voltage from the PREVIOUS state, so we need
    multiple steps for the control voltage effect to propagate.
    """

    def test_vcr_differentiable_wrt_r0(self):
        """Gradient of output voltage w.r.t. r0 should be finite."""
        net = Network()
        net, n_src = net.node("n_src")
        net, n_mid = net.node("n_mid")
        net, n_ctrl = net.node("n_ctrl")

        net, vs = VSource(net, n_src, net.gnd, name="vs", value=10.0)
        net, v_ctrl = VSource(net, n_ctrl, net.gnd, name="v_ctrl", value=5.0)
        net, r1 = R(net, n_src, n_mid, name="R1", value=1000.0)
        net, vcr1 = VCR(net, n_mid, net.gnd, n_ctrl, net.gnd, name="VCR1")

        sim = net.compile(dt=1e-6)

        def get_v_mid(r0_val):
            params = {"VCR1_r0": r0_val, "VCR1_k": 100.0}
            state = sim.init(params)
            # Run 2 steps to let control voltage settle
            state = sim.step(params, state, {})
            state = sim.step(params, state, {})
            return sim.v(state, n_mid)

        grad_fn = grad(get_v_mid)
        g = grad_fn(1000.0)

        assert jnp.isfinite(g)
        # With higher r0, v_mid increases (more voltage drop across VCR)
        assert g > 0

    def test_vcr_differentiable_wrt_k(self):
        """Gradient of output voltage w.r.t. k should be finite."""
        net = Network()
        net, n_src = net.node("n_src")
        net, n_mid = net.node("n_mid")
        net, n_ctrl = net.node("n_ctrl")

        net, vs = VSource(net, n_src, net.gnd, name="vs", value=10.0)
        net, v_ctrl = VSource(net, n_ctrl, net.gnd, name="v_ctrl", value=5.0)
        net, r1 = R(net, n_src, n_mid, name="R1", value=1000.0)
        net, vcr1 = VCR(net, n_mid, net.gnd, n_ctrl, net.gnd, name="VCR1")

        sim = net.compile(dt=1e-6)

        def get_v_mid(k_val):
            params = {"VCR1_r0": 1000.0, "VCR1_k": k_val}
            state = sim.init(params)
            # Run 2 steps to let control voltage settle
            state = sim.step(params, state, {})
            state = sim.step(params, state, {})
            return sim.v(state, n_mid)

        grad_fn = grad(get_v_mid)
        g = grad_fn(100.0)

        assert jnp.isfinite(g)
        # With V_ctrl=5V and positive k, increasing k increases R, increases v_mid
        assert g > 0

    def test_vcr_differentiable_wrt_control_voltage(self):
        """Gradient flows through control voltage."""
        net = Network()
        net, n_src = net.node("n_src")
        net, n_mid = net.node("n_mid")
        net, n_ctrl = net.node("n_ctrl")

        net, vs = VSource(net, n_src, net.gnd, name="vs", value=10.0)
        net, v_ctrl = VSource(net, n_ctrl, net.gnd, name="v_ctrl")
        net, r1 = R(net, n_src, n_mid, name="R1", value=1000.0)
        net, vcr1 = VCR(net, n_mid, net.gnd, n_ctrl, net.gnd, name="VCR1", r0=1000.0, k=100.0)

        sim = net.compile(dt=1e-6)

        def get_v_mid(v_ctrl_val):
            params = {}
            state = sim.init(params)
            # Run 2 steps so control voltage from step 1 affects step 2
            state = sim.step(params, state, {"v_ctrl": v_ctrl_val})
            state = sim.step(params, state, {"v_ctrl": v_ctrl_val})
            return sim.v(state, n_mid)

        grad_fn = grad(get_v_mid)
        g = grad_fn(5.0)

        assert jnp.isfinite(g)
        # With positive k, increasing V_ctrl increases R, increases v_mid
        assert g > 0

    def test_vcr_differentiable_multi_step(self):
        """Gradient propagates through multiple simulation steps."""
        net = Network()
        net, n_src = net.node("n_src")
        net, n_mid = net.node("n_mid")
        net, n_ctrl = net.node("n_ctrl")
        net, n_cap = net.node("n_cap")

        net, vs = VSource(net, n_src, net.gnd, name="vs", value=10.0)
        net, v_ctrl = VSource(net, n_ctrl, net.gnd, name="v_ctrl", value=5.0)
        net, r1 = R(net, n_src, n_mid, name="R1", value=1000.0)
        net, vcr1 = VCR(net, n_mid, n_cap, n_ctrl, net.gnd, name="VCR1", r0=1000.0, k=100.0)
        net, c1 = C(net, n_cap, net.gnd, name="C1", value=1e-6)

        sim = net.compile(dt=1e-6)

        def get_final_v(r0_val):
            params = {"VCR1_r0": r0_val, "VCR1_k": 100.0}
            state = sim.init(params)
            for _ in range(100):
                state = sim.step(params, state, {})
            return sim.v(state, n_cap)

        grad_fn = grad(get_final_v)
        g = grad_fn(1000.0)

        assert jnp.isfinite(g)


class TestVCRCurrentProbe:
    """Test current probing for VCR."""

    def test_vcr_current_not_implemented(self):
        """sim.i() on VCR should raise NotImplementedError."""
        net = Network()
        net, n1 = net.node("n1")
        net, n_ctrl = net.node("n_ctrl")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=10.0)
        net, v_ctrl = VSource(net, n_ctrl, net.gnd, name="v_ctrl", value=0.0)  # Drive control
        net, vcr1 = VCR(net, n1, net.gnd, n_ctrl, net.gnd, name="VCR1", r0=1000.0, k=0.0)

        sim = net.compile(dt=1e-6)
        state = sim.init({})
        state = sim.step({}, state, {})

        with pytest.raises(NotImplementedError) as exc_info:
            sim.i(state, vcr1)

        assert "VCR" in str(exc_info.value)


class TestVCRParamsOverride:
    """Test that params can override construction-time defaults.

    Note: VCR uses control voltage from the PREVIOUS state, so tests
    run 2 steps to let the control voltage effect propagate.
    """

    def test_params_override_r0(self):
        """r0 from params overrides construction-time default."""
        net = Network()
        net, n_src = net.node("n_src")
        net, n_mid = net.node("n_mid")
        net, n_ctrl = net.node("n_ctrl")

        net, vs = VSource(net, n_src, net.gnd, name="vs", value=10.0)
        net, v_ctrl = VSource(net, n_ctrl, net.gnd, name="v_ctrl", value=0.0)  # Drive control
        net, r1 = R(net, n_src, n_mid, name="R1", value=1000.0)
        # Default r0=1000, but we'll override
        net, vcr1 = VCR(net, n_mid, net.gnd, n_ctrl, net.gnd, name="VCR1", r0=1000.0, k=0.0)

        sim = net.compile(dt=1e-6)

        # Use default r0=1000 (2 steps to settle)
        state = sim.init({})
        state = sim.step({}, state, {})
        state = sim.step({}, state, {})
        v_default = float(sim.v(state, n_mid))

        # Override r0=2000 (2 steps to settle)
        state = sim.init({"VCR1_r0": 2000.0})
        state = sim.step({"VCR1_r0": 2000.0}, state, {})
        state = sim.step({"VCR1_r0": 2000.0}, state, {})
        v_override = float(sim.v(state, n_mid))

        # With r0=1000: 10V * 1k/(1k+1k) = 5V
        # With r0=2000: 10V * 2k/(1k+2k) = 6.67V
        assert abs(v_default - 5.0) < 0.1
        assert abs(v_override - 6.67) < 0.1

    def test_params_override_k(self):
        """k from params overrides construction-time default."""
        net = Network()
        net, n_src = net.node("n_src")
        net, n_mid = net.node("n_mid")
        net, n_ctrl = net.node("n_ctrl")

        net, vs = VSource(net, n_src, net.gnd, name="vs", value=10.0)
        net, v_ctrl = VSource(net, n_ctrl, net.gnd, name="v_ctrl", value=10.0)
        net, r1 = R(net, n_src, n_mid, name="R1", value=1000.0)
        # Default k=0
        net, vcr1 = VCR(net, n_mid, net.gnd, n_ctrl, net.gnd, name="VCR1", r0=1000.0, k=0.0)

        sim = net.compile(dt=1e-6)

        # Use default k=0 (R = 1000), 2 steps
        state = sim.init({})
        state = sim.step({}, state, {})
        state = sim.step({}, state, {})
        v_k0 = float(sim.v(state, n_mid))

        # Override k=100 (R = 1000 + 100*10 = 2000), 2 steps
        state = sim.init({"VCR1_k": 100.0})
        state = sim.step({"VCR1_k": 100.0}, state, {})
        state = sim.step({"VCR1_k": 100.0}, state, {})
        v_k100 = float(sim.v(state, n_mid))

        # With k=0: 10V * 1k/(1k+1k) = 5V
        # With k=100: 10V * 2k/(1k+2k) = 6.67V
        assert abs(v_k0 - 5.0) < 0.1
        assert abs(v_k100 - 6.67) < 0.1
