"""
Test: Current probing via sim.i() for all components.

Tests:
1. sim.i() returns correct current for Inductors (L) and Capacitors (C)
2. sim.i() raises NotImplementedError for unsupported components (R, VSource, Switch, etc.)
3. sim.i() is differentiable for L and C components
"""
import math
import pytest
import jax.numpy as jnp
from jax import grad


# Fixed simulation parameters
N_STEPS = 100
DT = 1e-6


class TestInductorCurrentProbe:
    """Test sim.i() for inductors."""

    def test_inductor_current_matches_state(self):
        """sim.i(l1) should match state.ind_currents[0]."""
        from pyvibrate.timedomain import Network, R, L, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=10.0)
        net, l1 = L(net, n1, n2, name="L1", value=10e-3)
        net, r1 = R(net, n2, net.gnd, name="R1", value=100.0)

        sim = net.compile(dt=DT)
        state = sim.init({})

        for _ in range(N_STEPS):
            state = sim.step({}, state, {})

        # sim.i() should return same value as direct state access
        i_from_probe = float(sim.i(state, l1))
        i_from_state = float(state.ind_currents[0])

        assert i_from_probe == i_from_state, \
            f"sim.i() returned {i_from_probe}, but state.ind_currents[0] is {i_from_state}"

    def test_inductor_current_physical_behavior(self):
        """Inductor current should follow RL step response: I(t) = (V/R)(1 - e^(-t/τ))."""
        from pyvibrate.timedomain import Network, R, L, VSource

        # RL circuit: Vs -- L -- R -- GND
        # τ = L/R, I_final = V/R
        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        V = 10.0
        L_val = 10e-3  # 10 mH
        R_val = 100.0  # 100 Ω
        tau = L_val / R_val  # 0.1 ms
        I_final = V / R_val  # 0.1 A

        net, vs = VSource(net, n1, net.gnd, name="vs", value=V)
        net, l1 = L(net, n1, n2, name="L1", value=L_val)
        net, r1 = R(net, n2, net.gnd, name="R1", value=R_val)

        dt = 1e-7  # Fine timestep for accurate τ=100µs
        sim = net.compile(dt=dt)
        state = sim.init({})

        # Simulate to t = τ
        n_steps = int(tau / dt)
        for _ in range(n_steps):
            state = sim.step({}, state, {})

        i_at_tau = float(sim.i(state, l1))
        expected_at_tau = I_final * (1 - math.exp(-1))  # ~63.2% of final

        rel_error = abs(i_at_tau - expected_at_tau) / expected_at_tau
        assert rel_error < 0.02, \
            f"At t=τ: got {i_at_tau*1e3:.3f}mA, expected {expected_at_tau*1e3:.3f}mA (error={rel_error*100:.1f}%)"

    def test_inductor_current_differentiable(self):
        """Gradient of inductor current w.r.t. L should be finite and negative."""
        from pyvibrate.timedomain import Network, R, L, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=10.0)
        net, l1 = L(net, n1, n2, name="L1")
        net, r1 = R(net, n2, net.gnd, name="R1", value=100.0)

        sim = net.compile(dt=DT)

        def inductor_current_via_probe(L_val: float) -> float:
            params = {"L1": L_val}
            state = sim.init(params)
            for _ in range(N_STEPS):
                state = sim.step(params, state, {})
            return sim.i(state, l1)

        L_test = 1e-3
        dI_dL = grad(inductor_current_via_probe)(L_test)

        assert jnp.isfinite(dI_dL), f"dI/dL should be finite, got {dI_dL}"
        # Increasing L slows current buildup (larger τ = L/R), so dI/dL < 0
        assert dI_dL < 0, f"dI/dL should be negative, got {dI_dL}"


class TestCapacitorCurrentProbe:
    """Test sim.i() for capacitors."""

    def test_capacitor_current_matches_state(self):
        """sim.i(c1) should match state.cap_currents[0]."""
        from pyvibrate.timedomain import Network, R, C, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)
        net, r1 = R(net, n1, n2, name="R1", value=1000.0)
        net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)

        sim = net.compile(dt=DT)
        state = sim.init({})

        for _ in range(N_STEPS):
            state = sim.step({}, state, {})

        # sim.i() should return same value as direct state access
        i_from_probe = float(sim.i(state, c1))
        i_from_state = float(state.cap_currents[0])

        assert i_from_probe == i_from_state, \
            f"sim.i() returned {i_from_probe}, but state.cap_currents[0] is {i_from_state}"

    def test_capacitor_current_physical_behavior(self):
        """Capacitor current should follow RC charging: I(t) = (V/R) * e^(-t/τ)."""
        from pyvibrate.timedomain import Network, R, C, VSource

        # RC circuit: Vs -- R -- C -- GND
        # τ = RC, I_initial = V/R
        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        V = 5.0
        R_val = 1000.0  # 1k Ω
        C_val = 1e-6  # 1 µF
        tau = R_val * C_val  # 1 ms
        I_initial = V / R_val  # 5 mA

        net, vs = VSource(net, n1, net.gnd, name="vs", value=V)
        net, r1 = R(net, n1, n2, name="R1", value=R_val)
        net, c1 = C(net, n2, net.gnd, name="C1", value=C_val)

        dt = 1e-6  # 1 µs timestep
        sim = net.compile(dt=dt)
        state = sim.init({})

        # Run a few steps to get past initial transient
        # Check current at t = τ (should be ~36.8% of initial)
        n_steps = int(tau / dt)
        for _ in range(n_steps):
            state = sim.step({}, state, {})

        i_at_tau = float(sim.i(state, c1))
        expected_at_tau = I_initial * math.exp(-1)  # ~36.8% of initial

        # Allow larger error for capacitor current (numerical effects)
        rel_error = abs(i_at_tau - expected_at_tau) / expected_at_tau
        assert rel_error < 0.05, \
            f"At t=τ: got {i_at_tau*1e3:.3f}mA, expected {expected_at_tau*1e3:.3f}mA (error={rel_error*100:.1f}%)"

    def test_capacitor_current_differentiable(self):
        """Gradient of capacitor current w.r.t. C should be finite."""
        from pyvibrate.timedomain import Network, R, C, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)
        net, r1 = R(net, n1, n2, name="R1", value=1000.0)
        net, c1 = C(net, n2, net.gnd, name="C1")

        sim = net.compile(dt=DT)

        def capacitor_current_via_probe(C_val: float) -> float:
            params = {"C1": C_val}
            state = sim.init(params)
            for _ in range(N_STEPS):
                state = sim.step(params, state, {})
            return sim.i(state, c1)

        C_test = 1e-6
        dI_dC = grad(capacitor_current_via_probe)(C_test)

        assert jnp.isfinite(dI_dC), f"dI/dC should be finite, got {dI_dC}"


class TestUnsupportedComponentCurrentProbe:
    """Test that sim.i() raises NotImplementedError for unsupported components."""

    def test_resistor_current_not_implemented(self):
        """sim.i() on a resistor should raise NotImplementedError."""
        from pyvibrate.timedomain import Network, R, VSource

        net = Network()
        net, n1 = net.node("n1")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)
        net, r1 = R(net, n1, net.gnd, name="R1", value=1000.0)

        sim = net.compile(dt=DT)
        state = sim.init({})
        state = sim.step({}, state, {})

        with pytest.raises(NotImplementedError) as exc_info:
            sim.i(state, r1)

        assert "R current probing" in str(exc_info.value)

    def test_vsource_current_not_implemented(self):
        """sim.i() on a voltage source should raise NotImplementedError."""
        from pyvibrate.timedomain import Network, R, VSource

        net = Network()
        net, n1 = net.node("n1")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)
        net, r1 = R(net, n1, net.gnd, name="R1", value=1000.0)

        sim = net.compile(dt=DT)
        state = sim.init({})
        state = sim.step({}, state, {})

        with pytest.raises(NotImplementedError) as exc_info:
            sim.i(state, vs)

        assert "VSource current probing" in str(exc_info.value)

    def test_switch_current_not_implemented(self):
        """sim.i() on a switch should raise NotImplementedError."""
        from pyvibrate.timedomain import Network, R, Switch, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)
        net, sw = Switch(net, n1, n2, name="sw1")
        net, r1 = R(net, n2, net.gnd, name="R_load", value=1000.0)

        sim = net.compile(dt=DT)
        state = sim.init({})
        controls = {"sw1": True}
        state = sim.step({}, state, controls)

        with pytest.raises(NotImplementedError) as exc_info:
            sim.i(state, sw)

        assert "Switch current probing" in str(exc_info.value)

    def test_voltage_switch_current_not_implemented(self):
        """sim.i() on a voltage switch should raise NotImplementedError."""
        from pyvibrate.timedomain import Network, R, VSource, VoltageSwitch

        net = Network()
        net, n_ctrl = net.node("n_ctrl")
        net, n_out = net.node("n_out")

        net, vs = VSource(net, n_ctrl, net.gnd, name="vs", value=5.0)
        net, vsw = VoltageSwitch(net, n_ctrl, n_out, n_ctrl, net.gnd, name="vsw1")
        net, r1 = R(net, n_out, net.gnd, name="R_load", value=1000.0)

        sim = net.compile(dt=DT)
        state = sim.init({})
        state = sim.step({}, state, {})

        with pytest.raises(NotImplementedError) as exc_info:
            sim.i(state, vsw)

        assert "VoltageSwitch current probing" in str(exc_info.value)

    def test_vcvs_current_not_implemented(self):
        """sim.i() on a VCVS should raise NotImplementedError."""
        from pyvibrate.timedomain import Network, R, VSource, VCVS

        net = Network()
        net, n_in = net.node("n_in")
        net, n_out = net.node("n_out")

        net, vs = VSource(net, n_in, net.gnd, name="vs", value=5.0)
        net, e1 = VCVS(net, n_out, net.gnd, n_in, net.gnd, name="E1")
        net, r1 = R(net, n_out, net.gnd, name="R_load", value=1000.0)

        sim = net.compile(dt=DT)
        params = {"E1": 2.0}  # Gain is provided via params
        state = sim.init(params)
        state = sim.step(params, state, {})

        with pytest.raises(NotImplementedError) as exc_info:
            sim.i(state, e1)

        assert "VCVS current probing" in str(exc_info.value)


class TestMultipleInductorsCapacitors:
    """Test sim.i() with multiple inductors and capacitors."""

    def test_multiple_inductors(self):
        """sim.i() should return correct current for each inductor."""
        from pyvibrate.timedomain import Network, R, L, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")
        net, n3 = net.node("n3")

        net, vs = VSource(net, n1, net.gnd, name="vs", value=10.0)
        net, l1 = L(net, n1, n2, name="L1", value=10e-3)
        net, l2 = L(net, n2, n3, name="L2", value=20e-3)
        net, r1 = R(net, n3, net.gnd, name="R1", value=100.0)

        sim = net.compile(dt=1e-7)
        state = sim.init({})

        for _ in range(1000):
            state = sim.step({}, state, {})

        # In series, both inductors carry the same current
        i_l1 = float(sim.i(state, l1))
        i_l2 = float(sim.i(state, l2))

        # They should be approximately equal (series connection)
        assert abs(i_l1 - i_l2) / abs(i_l1) < 0.01, \
            f"Series inductors should have same current: L1={i_l1:.6f}A, L2={i_l2:.6f}A"

    def test_multiple_capacitors(self):
        """sim.i() should return correct current for each capacitor."""
        from pyvibrate.timedomain import Network, R, C, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")
        net, n3 = net.node("n3")

        # Parallel capacitors: each gets same voltage, different current
        net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)
        net, r1 = R(net, n1, n2, name="R1", value=1000.0)
        # C1 and C2 in parallel
        net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)  # 1 µF
        net, c2 = C(net, n2, net.gnd, name="C2", value=2e-6)  # 2 µF

        sim = net.compile(dt=DT)
        state = sim.init({})

        # Run a few steps while charging
        for _ in range(10):
            state = sim.step({}, state, {})

        i_c1 = float(sim.i(state, c1))
        i_c2 = float(sim.i(state, c2))

        # For parallel capacitors with same dV/dt, I = C * dV/dt
        # So I2/I1 should be approximately C2/C1 = 2
        if abs(i_c1) > 1e-9:  # Avoid division by zero
            ratio = abs(i_c2 / i_c1)
            assert 1.8 < ratio < 2.2, \
                f"Current ratio should be ~2 (C2/C1), got {ratio:.2f}"


class TestCurrentProbeDifferentiabilityAdvanced:
    """Advanced differentiability tests for current probing."""

    def test_inductor_current_differentiable_wrt_voltage(self):
        """Gradient of inductor current w.r.t. source voltage should be finite."""
        from pyvibrate.timedomain import Network, R, L, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        net, vs = VSource(net, n1, net.gnd, name="vs")
        net, l1 = L(net, n1, n2, name="L1", value=1e-3)
        net, r1 = R(net, n2, net.gnd, name="R1", value=100.0)

        sim = net.compile(dt=DT)

        def inductor_current_vs_voltage(V: float) -> float:
            state = sim.init({})
            controls = {"vs": V}
            for _ in range(N_STEPS):
                state = sim.step({}, state, controls)
            return sim.i(state, l1)

        V_test = 5.0
        dI_dV = grad(inductor_current_vs_voltage)(V_test)

        assert jnp.isfinite(dI_dV), f"dI/dV should be finite, got {dI_dV}"
        # More voltage = more current, so dI/dV > 0
        assert dI_dV > 0, f"dI/dV should be positive, got {dI_dV}"

    def test_capacitor_current_differentiable_wrt_voltage(self):
        """Gradient of capacitor current w.r.t. source voltage should be finite."""
        from pyvibrate.timedomain import Network, R, C, VSource

        net = Network()
        net, n1 = net.node("n1")
        net, n2 = net.node("n2")

        net, vs = VSource(net, n1, net.gnd, name="vs")
        net, r1 = R(net, n1, n2, name="R1", value=1000.0)
        net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)

        sim = net.compile(dt=DT)

        def capacitor_current_vs_voltage(V: float) -> float:
            state = sim.init({})
            controls = {"vs": V}
            for _ in range(N_STEPS):
                state = sim.step({}, state, controls)
            return sim.i(state, c1)

        V_test = 5.0
        dI_dV = grad(capacitor_current_vs_voltage)(V_test)

        assert jnp.isfinite(dI_dV), f"dI/dV should be finite, got {dI_dV}"

    def test_rlc_inductor_current_differentiable(self):
        """Inductor current in RLC circuit should be differentiable w.r.t. all params."""
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

        def inductor_current(R_val: float, L_val: float, C_val: float) -> float:
            params = {"R1": R_val, "L1": L_val, "C1": C_val}
            state = sim.init(params)
            for _ in range(N_STEPS):
                state = sim.step(params, state, {})
            return sim.i(state, l1)

        R_test, L_test, C_test = 100.0, 1e-3, 1e-6

        dI_dR = grad(inductor_current, argnums=0)(R_test, L_test, C_test)
        dI_dL = grad(inductor_current, argnums=1)(R_test, L_test, C_test)
        dI_dC = grad(inductor_current, argnums=2)(R_test, L_test, C_test)

        assert jnp.isfinite(dI_dR), f"dI/dR should be finite, got {dI_dR}"
        assert jnp.isfinite(dI_dL), f"dI/dL should be finite, got {dI_dL}"
        assert jnp.isfinite(dI_dC), f"dI/dC should be finite, got {dI_dC}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
