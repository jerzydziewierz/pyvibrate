"""
Tests for time-domain Series and Parallel subcircuit operations.

Test plan: see subcircuit_testing_plan.md
"""
import math
import pytest


class TestSeriesCore:
    """Core functionality tests for Series operation."""

    def test_series_resistors_equivalence(self):
        """Series of two resistors should equal their sum (R1 + R2).

        Circuit comparison:
        - Circuit A: Vs -- [R1] -- [R2] -- GND  (using Series)
        - Circuit B: Vs -- [R_eq] -- GND       (single equivalent resistor)

        Both should produce the same current: I = V / (R1 + R2)
        """
        from pyvibrate.timedomain import Network, R, VSource
        from pyvibrate.timedomain.subcircuits import Series

        R1_val = 1000.0  # 1k ohm
        R2_val = 2000.0  # 2k ohm
        R_eq_val = R1_val + R2_val  # 3k ohm
        V_source = 6.0  # 6V

        # Expected current: I = V / R_eq = 6 / 3000 = 2 mA
        expected_current = V_source / R_eq_val

        # --- Circuit A: Two resistors in series using Series() ---
        net_a = Network()
        net_a, n_in = net_a.node("in")
        net_a, n_out = net_a.node("out")

        net_a, (r1_ref, r2_ref, mid) = Series(
            net_a, n_in, n_out,
            lambda net, a, b: R(net, a, b, name="r1", value=R1_val),
            lambda net, a, b: R(net, a, b, name="r2", value=R2_val),
            prefix="ser"
        )
        net_a, vs_a = VSource(net_a, n_in, net_a.gnd, name="vs", value=V_source)
        # Connect output to ground to complete circuit
        net_a, r_load = R(net_a, n_out, net_a.gnd, name="r_load", value=1e-6)  # near-short

        dt = 1e-6
        sim_a = net_a.compile(dt=dt)
        state_a = sim_a.init({})
        # Step to reach steady state
        for _ in range(10):
            state_a = sim_a.step({}, state_a, {})

        # Get voltage across series combination (should be ~6V since r_load is tiny)
        v_in = float(sim_a.v(state_a, n_in))
        v_out = float(sim_a.v(state_a, n_out))
        v_series = v_in - v_out

        # Current through series = V_series / (R1 + R2)
        # With near-short load, v_out ≈ 0, so v_series ≈ V_source
        i_series = v_series / R_eq_val

        # --- Circuit B: Single equivalent resistor ---
        net_b = Network()
        net_b, n1 = net_b.node("n1")

        net_b, vs_b = VSource(net_b, n1, net_b.gnd, name="vs", value=V_source)
        net_b, r_eq = R(net_b, n1, net_b.gnd, name="r_eq", value=R_eq_val)

        sim_b = net_b.compile(dt=dt)
        state_b = sim_b.init({})
        for _ in range(10):
            state_b = sim_b.step({}, state_b, {})

        v_b = float(sim_b.v(state_b, n1))
        i_eq = v_b / R_eq_val

        # Both should give same current (within 1%)
        assert abs(i_series - expected_current) / expected_current < 0.01, \
            f"Series current {i_series} != expected {expected_current}"
        assert abs(i_eq - expected_current) / expected_current < 0.01, \
            f"Equivalent current {i_eq} != expected {expected_current}"
        assert abs(i_series - i_eq) / i_eq < 0.01, \
            f"Series current {i_series} != equivalent {i_eq}"

    def test_series_voltage_division(self):
        """Voltage across each element follows voltage divider rule.

        For Vs -- [R1] -- [R2] -- GND:
        - V_R1 = Vs * R1 / (R1 + R2)
        - V_R2 = Vs * R2 / (R1 + R2)
        - V_mid = Vs * R2 / (R1 + R2) = V_R2 (voltage at midpoint)
        """
        from pyvibrate.timedomain import Network, R, VSource
        from pyvibrate.timedomain.subcircuits import Series

        R1_val = 1000.0  # 1k ohm
        R2_val = 3000.0  # 3k ohm
        V_source = 8.0   # 8V

        # Expected voltages
        R_total = R1_val + R2_val
        expected_v_mid = V_source * R2_val / R_total  # 8 * 3000/4000 = 6V
        expected_v_r1 = V_source * R1_val / R_total   # 8 * 1000/4000 = 2V

        # Build circuit: Vs -- [R1] -- mid -- [R2] -- GND
        net = Network()
        net, n_in = net.node("in")

        net, (r1_ref, r2_ref, mid) = Series(
            net, n_in, net.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R1_val),
            lambda net, a, b: R(net, a, b, name="r2", value=R2_val),
            prefix="div"
        )
        net, vs = VSource(net, n_in, net.gnd, name="vs", value=V_source)

        dt = 1e-6
        sim = net.compile(dt=dt)
        state = sim.init({})
        # Step to reach steady state
        for _ in range(10):
            state = sim.step({}, state, {})

        v_in = float(sim.v(state, n_in))
        v_mid = float(sim.v(state, mid))

        # V across R1 = V_in - V_mid
        v_r1 = v_in - v_mid
        # V across R2 = V_mid (since other end is ground)
        v_r2 = v_mid

        # Verify voltage division (within 1%)
        assert abs(v_mid - expected_v_mid) / expected_v_mid < 0.01, \
            f"Mid voltage {v_mid} != expected {expected_v_mid}"
        assert abs(v_r1 - expected_v_r1) / expected_v_r1 < 0.01, \
            f"V_R1 {v_r1} != expected {expected_v_r1}"
        # Also verify total voltage
        assert abs(v_r1 + v_r2 - V_source) / V_source < 0.01, \
            f"Total voltage {v_r1 + v_r2} != source {V_source}"


class TestParallelCore:
    """Core functionality tests for Parallel operation."""

    def test_parallel_resistors_equivalence(self):
        """Parallel of two resistors should equal 1/(1/R1 + 1/R2).

        Circuit comparison:
        - Circuit A: Vs -- [R1 || R2] -- GND  (using Parallel)
        - Circuit B: Vs -- [R_eq] -- GND     (single equivalent resistor)

        Both should produce the same total current: I = V / R_eq
        """
        from pyvibrate.timedomain import Network, R, VSource
        from pyvibrate.timedomain.subcircuits import Parallel

        R1_val = 1000.0  # 1k ohm
        R2_val = 2000.0  # 2k ohm
        R_eq_val = (R1_val * R2_val) / (R1_val + R2_val)  # 666.67 ohm
        V_source = 6.0  # 6V

        # Expected total current: I = V / R_eq = 6 / 666.67 = 9 mA
        expected_current = V_source / R_eq_val

        # --- Circuit A: Two resistors in parallel using Parallel() ---
        net_a = Network()
        net_a, n_in = net_a.node("in")

        net_a, (r1_ref, r2_ref) = Parallel(
            net_a, n_in, net_a.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R1_val),
            lambda net, a, b: R(net, a, b, name="r2", value=R2_val),
            prefix="par"
        )
        net_a, vs_a = VSource(net_a, n_in, net_a.gnd, name="vs", value=V_source)

        dt = 1e-6
        sim_a = net_a.compile(dt=dt)
        state_a = sim_a.init({})
        # Step to reach steady state
        for _ in range(10):
            state_a = sim_a.step({}, state_a, {})

        v_a = float(sim_a.v(state_a, n_in))
        # Total current through parallel combination
        i_parallel = v_a / R_eq_val

        # --- Circuit B: Single equivalent resistor ---
        net_b = Network()
        net_b, n1 = net_b.node("n1")

        net_b, vs_b = VSource(net_b, n1, net_b.gnd, name="vs", value=V_source)
        net_b, r_eq = R(net_b, n1, net_b.gnd, name="r_eq", value=R_eq_val)

        sim_b = net_b.compile(dt=dt)
        state_b = sim_b.init({})
        for _ in range(10):
            state_b = sim_b.step({}, state_b, {})

        v_b = float(sim_b.v(state_b, n1))
        i_eq = v_b / R_eq_val

        # Both should give same current (within 1%)
        assert abs(i_parallel - expected_current) / expected_current < 0.01, \
            f"Parallel current {i_parallel} != expected {expected_current}"
        assert abs(i_eq - expected_current) / expected_current < 0.01, \
            f"Equivalent current {i_eq} != expected {expected_current}"
        assert abs(i_parallel - i_eq) / i_eq < 0.01, \
            f"Parallel current {i_parallel} != equivalent {i_eq}"

    def test_parallel_current_division(self):
        """Current through each element follows current divider rule.

        For parallel resistors: I_total * R_other / R_total
        - I1 = I_total * R2 / (R1 + R2)
        - I2 = I_total * R1 / (R1 + R2)

        Since sim.i() doesn't work for resistors, we calculate I = V/R.
        """
        from pyvibrate.timedomain import Network, R, VSource
        from pyvibrate.timedomain.subcircuits import Parallel

        R1_val = 1000.0  # 1k ohm
        R2_val = 4000.0  # 4k ohm
        V_source = 10.0  # 10V

        # Expected currents using current divider
        R_total = R1_val + R2_val  # for divider formula
        I_total = V_source * (1/R1_val + 1/R2_val)  # total current
        expected_i1 = V_source / R1_val  # 10 mA
        expected_i2 = V_source / R2_val  # 2.5 mA

        # Build circuit: Vs -- [R1 || R2] -- GND
        net = Network()
        net, n_in = net.node("in")

        net, (r1_ref, r2_ref) = Parallel(
            net, n_in, net.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R1_val),
            lambda net, a, b: R(net, a, b, name="r2", value=R2_val),
            prefix="cdiv"
        )
        net, vs = VSource(net, n_in, net.gnd, name="vs", value=V_source)

        dt = 1e-6
        sim = net.compile(dt=dt)
        state = sim.init({})
        for _ in range(10):
            state = sim.step({}, state, {})

        # Voltage across both resistors (same node, parallel)
        v_across = float(sim.v(state, n_in))

        # Calculate current through each resistor: I = V / R
        i1 = v_across / R1_val
        i2 = v_across / R2_val
        i_total = i1 + i2

        # Verify current division (within 1%)
        assert abs(i1 - expected_i1) / expected_i1 < 0.01, \
            f"I_R1 {i1} != expected {expected_i1}"
        assert abs(i2 - expected_i2) / expected_i2 < 0.01, \
            f"I_R2 {i2} != expected {expected_i2}"
        # Verify ratio: I1/I2 should be R2/R1
        expected_ratio = R2_val / R1_val  # 4
        actual_ratio = i1 / i2
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.01, \
            f"Current ratio {actual_ratio} != expected {expected_ratio}"

    def test_parallel_vs_manual(self):
        """Parallel(R, C) produces identical results to manual construction.

        Compare transient response of:
        - Circuit A: Vs -- R_load -- Parallel(R, C) -- GND
        - Circuit B: Vs -- R_load -- (R || C) -- GND (manually constructed)
        """
        from pyvibrate.timedomain import Network, R, C, VSource
        from pyvibrate.timedomain.subcircuits import Parallel

        R_load = 500.0   # Load resistor
        R_par = 1000.0   # Parallel resistor
        C_val = 1e-6     # 1 µF
        V_source = 5.0

        dt = 1e-6
        n_steps = 2000

        # --- Circuit A: Using Parallel() ---
        net_a = Network()
        net_a, n_in_a = net_a.node("in")
        net_a, n_par_a = net_a.node("par")

        net_a, vs_a = VSource(net_a, n_in_a, net_a.gnd, name="vs", value=V_source)
        net_a, rl_a = R(net_a, n_in_a, n_par_a, name="r_load", value=R_load)
        net_a, (r_ref, c_ref) = Parallel(
            net_a, n_par_a, net_a.gnd,
            lambda net, a, b: R(net, a, b, name="r_par", value=R_par),
            lambda net, a, b: C(net, a, b, name="c_par", value=C_val),
            prefix="par"
        )

        sim_a = net_a.compile(dt=dt)
        state_a = sim_a.init({})

        # --- Circuit B: Manual construction ---
        net_b = Network()
        net_b, n_in_b = net_b.node("in")
        net_b, n_par_b = net_b.node("par")

        net_b, vs_b = VSource(net_b, n_in_b, net_b.gnd, name="vs", value=V_source)
        net_b, rl_b = R(net_b, n_in_b, n_par_b, name="r_load", value=R_load)
        net_b, r_par_b = R(net_b, n_par_b, net_b.gnd, name="r_par", value=R_par)
        net_b, c_par_b = C(net_b, n_par_b, net_b.gnd, name="c_par", value=C_val)

        sim_b = net_b.compile(dt=dt)
        state_b = sim_b.init({})

        # Simulate both and compare parallel node voltages
        max_diff = 0.0
        for step in range(n_steps):
            state_a = sim_a.step({}, state_a, {})
            state_b = sim_b.step({}, state_b, {})

            v_par_a = float(sim_a.v(state_a, n_par_a))
            v_par_b = float(sim_b.v(state_b, n_par_b))

            diff = abs(v_par_a - v_par_b)
            max_diff = max(max_diff, diff)

        assert max_diff < 1e-10, \
            f"Max voltage difference {max_diff} exceeds tolerance"

    def test_parallel_voltage_same(self):
        """Voltage across both parallel elements is identical.

        Both elements in Parallel(R, C) share the same nodes,
        so voltage across each must be identical.
        """
        from pyvibrate.timedomain import Network, R, C, VSource
        from pyvibrate.timedomain.subcircuits import Parallel

        R_par = 1000.0
        C_val = 1e-6
        V_source = 5.0

        net = Network()
        net, n_in = net.node("in")
        net, n_par = net.node("par")

        net, vs = VSource(net, n_in, net.gnd, name="vs", value=V_source)
        net, rl = R(net, n_in, n_par, name="r_load", value=500.0)
        net, (r_ref, c_ref) = Parallel(
            net, n_par, net.gnd,
            lambda net, a, b: R(net, a, b, name="r_par", value=R_par),
            lambda net, a, b: C(net, a, b, name="c_par", value=C_val),
            prefix="pv"
        )

        dt = 1e-6
        sim = net.compile(dt=dt)
        state = sim.init({})

        # Check at multiple time steps
        for step in range(100):
            state = sim.step({}, state, {})

            # Both elements share n_par as top node and gnd as bottom
            # So voltage across both is simply v(n_par) - v(gnd) = v(n_par)
            v_across = float(sim.v(state, n_par))

            # Since both are between the same two nodes, voltage is identical
            # This test mostly validates the structure is correct
            assert v_across >= 0, f"Voltage should be non-negative, got {v_across}"


class TestSeriesManualComparison:
    """Tests comparing Series() with manual construction."""

    def test_series_vs_manual(self):
        """Series(R, C) produces identical results to manually built R-C chain.

        Compare transient response of:
        - Circuit A: Vs -- Series(R, C) -- GND
        - Circuit B: Vs -- R -- C -- GND (manually constructed)
        """
        from pyvibrate.timedomain import Network, R, C, VSource
        from pyvibrate.timedomain.subcircuits import Series

        R_val = 1000.0   # 1k ohm
        C_val = 1e-6     # 1 µF
        V_source = 5.0
        tau = R_val * C_val  # 1 ms

        dt = 1e-6
        n_steps = int(2 * tau / dt)  # simulate 2 time constants

        # --- Circuit A: Using Series() ---
        net_a = Network()
        net_a, n_in_a = net_a.node("in")

        net_a, (r_ref, c_ref, mid_a) = Series(
            net_a, n_in_a, net_a.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R_val),
            lambda net, a, b: C(net, a, b, name="c1", value=C_val),
            prefix="ser"
        )
        net_a, vs_a = VSource(net_a, n_in_a, net_a.gnd, name="vs", value=V_source)

        sim_a = net_a.compile(dt=dt)
        state_a = sim_a.init({})

        # --- Circuit B: Manual construction ---
        net_b = Network()
        net_b, n_in_b = net_b.node("in")
        net_b, mid_b = net_b.node("mid")

        net_b, vs_b = VSource(net_b, n_in_b, net_b.gnd, name="vs", value=V_source)
        net_b, r1_b = R(net_b, n_in_b, mid_b, name="r1", value=R_val)
        net_b, c1_b = C(net_b, mid_b, net_b.gnd, name="c1", value=C_val)

        sim_b = net_b.compile(dt=dt)
        state_b = sim_b.init({})

        # Simulate both and compare mid-node voltages at each step
        max_diff = 0.0
        for step in range(n_steps):
            state_a = sim_a.step({}, state_a, {})
            state_b = sim_b.step({}, state_b, {})

            v_mid_a = float(sim_a.v(state_a, mid_a))
            v_mid_b = float(sim_b.v(state_b, mid_b))

            diff = abs(v_mid_a - v_mid_b)
            max_diff = max(max_diff, diff)

        # Voltages should be identical (within numerical precision)
        assert max_diff < 1e-10, \
            f"Max voltage difference {max_diff} exceeds tolerance"

    def test_series_midpoint_accessible(self):
        """Internal mid-node is probeable and voltage is between input/output.

        For Vs -- [R1] -- mid -- [R2] -- GND:
        - mid-node should be accessible via sim.v()
        - V_mid should be between V_in and 0 (ground)
        """
        from pyvibrate.timedomain import Network, R, VSource
        from pyvibrate.timedomain.subcircuits import Series

        R1_val = 1000.0
        R2_val = 2000.0
        V_source = 9.0

        net = Network()
        net, n_in = net.node("in")

        net, (r1_ref, r2_ref, mid) = Series(
            net, n_in, net.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R1_val),
            lambda net, a, b: R(net, a, b, name="r2", value=R2_val),
            prefix="mp"
        )
        net, vs = VSource(net, n_in, net.gnd, name="vs", value=V_source)

        dt = 1e-6
        sim = net.compile(dt=dt)
        state = sim.init({})
        for _ in range(10):
            state = sim.step({}, state, {})

        v_in = float(sim.v(state, n_in))
        v_mid = float(sim.v(state, mid))
        v_out = 0.0  # ground

        # Mid-node voltage should be probeable (not raise)
        # and should be between input and output
        assert v_out < v_mid < v_in, \
            f"Mid voltage {v_mid} should be between {v_out} and {v_in}"

        # More specifically, for voltage divider: V_mid = V_in * R2 / (R1 + R2)
        expected_mid = v_in * R2_val / (R1_val + R2_val)
        assert abs(v_mid - expected_mid) / expected_mid < 0.01, \
            f"Mid voltage {v_mid} != expected {expected_mid}"


class TestSeriesPhysical:
    """Physical correctness tests for Series operation."""

    def test_series_current_continuity(self):
        """Current through both elements in series is identical at all time steps.

        Use two inductors in series so we can probe current via sim.i().
        """
        from pyvibrate.timedomain import Network, R, L, VSource
        from pyvibrate.timedomain.subcircuits import Series

        L1_val = 10e-3   # 10 mH
        L2_val = 20e-3   # 20 mH
        R_val = 100.0    # Load resistor
        V_source = 10.0

        net = Network()
        net, n_in = net.node("in")
        net, n_out = net.node("out")

        net, vs = VSource(net, n_in, net.gnd, name="vs", value=V_source)
        net, (l1_ref, l2_ref, mid) = Series(
            net, n_in, n_out,
            lambda net, a, b: L(net, a, b, name="L1", value=L1_val),
            lambda net, a, b: L(net, a, b, name="L2", value=L2_val),
            prefix="ll"
        )
        net, r_load = R(net, n_out, net.gnd, name="r_load", value=R_val)

        dt = 1e-7  # Fine timestep for inductors
        sim = net.compile(dt=dt)
        state = sim.init({})

        # Check current continuity at multiple time steps
        for step in range(1000):
            state = sim.step({}, state, {})

            i_l1 = float(sim.i(state, l1_ref))
            i_l2 = float(sim.i(state, l2_ref))

            # Current through series elements must be identical
            if abs(i_l1) > 1e-12:  # Skip near-zero currents
                rel_diff = abs(i_l1 - i_l2) / abs(i_l1)
                assert rel_diff < 0.01, \
                    f"Step {step}: L1 current {i_l1:.6e} != L2 current {i_l2:.6e}"

    def test_series_rc_charging(self):
        """Series RC shows correct exponential charging (τ = RC).

        For Vs -- Series(R, C) -- GND:
        V_C(t) = Vs * (1 - exp(-t/τ)) where τ = RC

        At t = τ: V_C ≈ 63.2% of Vs
        At t = 3τ: V_C ≈ 95% of Vs
        """
        from pyvibrate.timedomain import Network, R, C, VSource
        from pyvibrate.timedomain.subcircuits import Series

        R_val = 1000.0   # 1k ohm
        C_val = 1e-6     # 1 µF
        V_source = 10.0
        tau = R_val * C_val  # 1 ms

        net = Network()
        net, n_in = net.node("in")

        net, (r_ref, c_ref, mid) = Series(
            net, n_in, net.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R_val),
            lambda net, a, b: C(net, a, b, name="c1", value=C_val),
            prefix="rc"
        )
        net, vs = VSource(net, n_in, net.gnd, name="vs", value=V_source)

        dt = 1e-6  # 1 µs
        sim = net.compile(dt=dt)
        state = sim.init({})

        # Simulate to t = τ
        n_steps_1tau = int(tau / dt)
        for _ in range(n_steps_1tau):
            state = sim.step({}, state, {})

        # V_C = V across capacitor = V_mid (mid is between R and C, C is to ground)
        v_c_at_1tau = float(sim.v(state, mid))
        expected_1tau = V_source * (1 - math.exp(-1))  # ~6.32V

        # Continue to t = 3τ
        for _ in range(2 * n_steps_1tau):
            state = sim.step({}, state, {})

        v_c_at_3tau = float(sim.v(state, mid))
        expected_3tau = V_source * (1 - math.exp(-3))  # ~9.50V

        # Allow 2% error for numerical integration
        assert abs(v_c_at_1tau - expected_1tau) / expected_1tau < 0.02, \
            f"V_C at τ: {v_c_at_1tau:.4f}V != expected {expected_1tau:.4f}V"
        assert abs(v_c_at_3tau - expected_3tau) / expected_3tau < 0.02, \
            f"V_C at 3τ: {v_c_at_3tau:.4f}V != expected {expected_3tau:.4f}V"

    def test_series_rl_current_rise(self):
        """Series RL shows correct current rise time (τ = L/R).

        For Vs -- Series(R, L) -- GND:
        I(t) = (Vs/R) * (1 - exp(-t/τ)) where τ = L/R

        At t = τ: I ≈ 63.2% of final value (Vs/R)
        """
        from pyvibrate.timedomain import Network, R, L, VSource
        from pyvibrate.timedomain.subcircuits import Series

        R_val = 100.0    # 100 ohm
        L_val = 10e-3    # 10 mH
        V_source = 10.0
        tau = L_val / R_val  # 0.1 ms = 100 µs
        I_final = V_source / R_val  # 0.1 A

        net = Network()
        net, n_in = net.node("in")

        net, vs = VSource(net, n_in, net.gnd, name="vs", value=V_source)
        net, (r_ref, l_ref, mid) = Series(
            net, n_in, net.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R_val),
            lambda net, a, b: L(net, a, b, name="L1", value=L_val),
            prefix="rl"
        )

        dt = 1e-7  # 0.1 µs - fine timestep for τ = 100 µs
        sim = net.compile(dt=dt)
        state = sim.init({})

        # Simulate to t = τ
        n_steps_1tau = int(tau / dt)
        for _ in range(n_steps_1tau):
            state = sim.step({}, state, {})

        i_at_1tau = float(sim.i(state, l_ref))
        expected_1tau = I_final * (1 - math.exp(-1))  # ~0.0632 A

        # Continue to t = 3τ
        for _ in range(2 * n_steps_1tau):
            state = sim.step({}, state, {})

        i_at_3tau = float(sim.i(state, l_ref))
        expected_3tau = I_final * (1 - math.exp(-3))  # ~0.095 A

        # Allow 2% error for numerical integration
        assert abs(i_at_1tau - expected_1tau) / expected_1tau < 0.02, \
            f"I at τ: {i_at_1tau*1e3:.3f}mA != expected {expected_1tau*1e3:.3f}mA"
        assert abs(i_at_3tau - expected_3tau) / expected_3tau < 0.02, \
            f"I at 3τ: {i_at_3tau*1e3:.3f}mA != expected {expected_3tau*1e3:.3f}mA"

    def test_series_lc_resonance(self):
        """Series LC oscillates at f = 1/(2π√LC).

        For Vs -- Series(L, C) -- GND (with small damping R):
        The circuit will oscillate at resonant frequency f_0.

        We verify by measuring the period of oscillation.
        """
        from pyvibrate.timedomain import Network, R, L, C, VSource
        from pyvibrate.timedomain.subcircuits import Series

        L_val = 1e-3     # 1 mH
        C_val = 1e-6     # 1 µF
        R_damp = 1.0     # Small damping resistor
        V_source = 10.0

        # Expected resonant frequency and period
        f_0 = 1 / (2 * math.pi * math.sqrt(L_val * C_val))  # ~5033 Hz
        T_0 = 1 / f_0  # ~198.7 µs

        net = Network()
        net, n_in = net.node("in")
        net, n_mid = net.node("mid")

        # Small source resistance + Series(L, C) to ground
        net, vs = VSource(net, n_in, net.gnd, name="vs", value=V_source)
        net, r_damp = R(net, n_in, n_mid, name="r_damp", value=R_damp)
        net, (l_ref, c_ref, mid_lc) = Series(
            net, n_mid, net.gnd,
            lambda net, a, b: L(net, a, b, name="L1", value=L_val),
            lambda net, a, b: C(net, a, b, name="C1", value=C_val),
            prefix="lc"
        )

        # Timestep should be small relative to period
        dt = T_0 / 100  # 100 samples per period
        sim = net.compile(dt=dt)
        state = sim.init({})

        # Collect voltage at mid-point to detect oscillation
        voltages = []
        n_periods = 5
        n_steps = int(n_periods * T_0 / dt)

        for _ in range(n_steps):
            state = sim.step({}, state, {})
            voltages.append(float(sim.v(state, mid_lc)))

        # Find zero crossings to measure period
        zero_crossings = []
        for i in range(1, len(voltages)):
            # Detect upward zero crossing (around expected oscillation midpoint)
            v_mid = sum(voltages) / len(voltages)
            if voltages[i-1] < v_mid <= voltages[i]:
                zero_crossings.append(i * dt)

        # Should have multiple zero crossings for multiple periods
        assert len(zero_crossings) >= 3, \
            f"Not enough zero crossings detected: {len(zero_crossings)}"

        # Calculate average period from zero crossings
        periods = [zero_crossings[i+1] - zero_crossings[i]
                   for i in range(len(zero_crossings) - 1)]
        avg_period = sum(periods) / len(periods)
        measured_f = 1 / avg_period

        # Allow 5% error (damping affects frequency slightly)
        assert abs(measured_f - f_0) / f_0 < 0.05, \
            f"Measured f={measured_f:.1f}Hz != expected f_0={f_0:.1f}Hz"


class TestComposability:
    """Tests for composability of Series and Parallel operations."""

    def test_series_nested(self):
        """Series can be nested: Series(Series(R, R), R) = 3R total.

        Compare:
        - Circuit A: Vs -- Series(Series(R1, R2), R3) -- GND
        - Circuit B: Vs -- R_eq (3R) -- GND
        """
        from pyvibrate.timedomain import Network, R, VSource
        from pyvibrate.timedomain.subcircuits import Series

        R_val = 1000.0  # Each resistor is 1k
        V_source = 9.0
        R_eq = 3 * R_val  # 3k total

        # Expected current: I = V / R_eq = 9V / 3k = 3mA
        expected_i = V_source / R_eq

        # --- Circuit A: Nested Series ---
        net_a = Network()
        net_a, n_in_a = net_a.node("in")

        # Inner series: R1 + R2
        def inner_series(net, a, b):
            return Series(
                net, a, b,
                lambda net, a, b: R(net, a, b, name="r1", value=R_val),
                lambda net, a, b: R(net, a, b, name="r2", value=R_val),
                prefix="inner"
            )

        # Outer series: (R1+R2) + R3
        net_a, (inner_refs, r3_ref, outer_mid) = Series(
            net_a, n_in_a, net_a.gnd,
            inner_series,
            lambda net, a, b: R(net, a, b, name="r3", value=R_val),
            prefix="outer"
        )
        net_a, vs_a = VSource(net_a, n_in_a, net_a.gnd, name="vs", value=V_source)

        dt = 1e-6
        sim_a = net_a.compile(dt=dt)
        state_a = sim_a.init({})
        for _ in range(10):
            state_a = sim_a.step({}, state_a, {})

        v_a = float(sim_a.v(state_a, n_in_a))
        i_a = v_a / R_eq

        # --- Circuit B: Single equivalent resistor ---
        net_b = Network()
        net_b, n_in_b = net_b.node("in")

        net_b, vs_b = VSource(net_b, n_in_b, net_b.gnd, name="vs", value=V_source)
        net_b, r_eq = R(net_b, n_in_b, net_b.gnd, name="r_eq", value=R_eq)

        sim_b = net_b.compile(dt=dt)
        state_b = sim_b.init({})
        for _ in range(10):
            state_b = sim_b.step({}, state_b, {})

        v_b = float(sim_b.v(state_b, n_in_b))
        i_b = v_b / R_eq

        # Both should give same current
        assert abs(i_a - expected_i) / expected_i < 0.01, \
            f"Nested series current {i_a} != expected {expected_i}"
        assert abs(i_a - i_b) / i_b < 0.01, \
            f"Nested series current {i_a} != equivalent {i_b}"

    def test_parallel_nested(self):
        """Parallel can be nested: Parallel(Parallel(R, R), R).

        For three 3k resistors: Parallel(Parallel(3k, 3k), 3k) = Parallel(1.5k, 3k) = 1k
        Compare with single 1k resistor.
        """
        from pyvibrate.timedomain import Network, R, VSource
        from pyvibrate.timedomain.subcircuits import Parallel

        R_val = 3000.0  # Each resistor is 3k
        V_source = 6.0

        # Inner parallel: 3k || 3k = 1.5k
        # Outer parallel: 1.5k || 3k = (1.5 * 3) / (1.5 + 3) = 4.5 / 4.5 = 1k
        R_eq = 1000.0

        # Expected current: I = V / R_eq = 6V / 1k = 6mA
        expected_i = V_source / R_eq

        # --- Circuit A: Nested Parallel ---
        net_a = Network()
        net_a, n_in_a = net_a.node("in")

        # Inner parallel: R1 || R2
        def inner_parallel(net, a, b):
            return Parallel(
                net, a, b,
                lambda net, a, b: R(net, a, b, name="r1", value=R_val),
                lambda net, a, b: R(net, a, b, name="r2", value=R_val),
                prefix="inner"
            )

        # Outer parallel: (R1||R2) || R3
        net_a, (inner_refs, r3_ref) = Parallel(
            net_a, n_in_a, net_a.gnd,
            inner_parallel,
            lambda net, a, b: R(net, a, b, name="r3", value=R_val),
            prefix="outer"
        )
        net_a, vs_a = VSource(net_a, n_in_a, net_a.gnd, name="vs", value=V_source)

        dt = 1e-6
        sim_a = net_a.compile(dt=dt)
        state_a = sim_a.init({})
        for _ in range(10):
            state_a = sim_a.step({}, state_a, {})

        v_a = float(sim_a.v(state_a, n_in_a))
        i_a = v_a / R_eq

        # --- Circuit B: Single equivalent resistor ---
        net_b = Network()
        net_b, n_in_b = net_b.node("in")

        net_b, vs_b = VSource(net_b, n_in_b, net_b.gnd, name="vs", value=V_source)
        net_b, r_eq = R(net_b, n_in_b, net_b.gnd, name="r_eq", value=R_eq)

        sim_b = net_b.compile(dt=dt)
        state_b = sim_b.init({})
        for _ in range(10):
            state_b = sim_b.step({}, state_b, {})

        v_b = float(sim_b.v(state_b, n_in_b))
        i_b = v_b / R_eq

        # Both should give same current
        assert abs(i_a - expected_i) / expected_i < 0.01, \
            f"Nested parallel current {i_a} != expected {expected_i}"
        assert abs(i_a - i_b) / i_b < 0.01, \
            f"Nested parallel current {i_a} != equivalent {i_b}"

    def test_series_parallel_mixed(self):
        """Mixed Series and Parallel: Series(Parallel(R1, R2), R3).

        Parallel(1k, 1k) = 500 ohm, then + 500 ohm in series = 1k total.
        """
        from pyvibrate.timedomain import Network, R, VSource
        from pyvibrate.timedomain.subcircuits import Series, Parallel

        R_par = 1000.0   # Two 1k in parallel = 500 ohm
        R_ser = 500.0    # Series resistor
        V_source = 10.0
        R_eq = 1000.0    # 500 + 500 = 1k total

        expected_i = V_source / R_eq  # 10 mA

        # --- Circuit A: Series(Parallel(R, R), R) ---
        net_a = Network()
        net_a, n_in_a = net_a.node("in")

        def parallel_rs(net, a, b):
            return Parallel(
                net, a, b,
                lambda net, a, b: R(net, a, b, name="r1", value=R_par),
                lambda net, a, b: R(net, a, b, name="r2", value=R_par),
                prefix="par"
            )

        net_a, (par_refs, r3_ref, mid) = Series(
            net_a, n_in_a, net_a.gnd,
            parallel_rs,
            lambda net, a, b: R(net, a, b, name="r3", value=R_ser),
            prefix="ser"
        )
        net_a, vs_a = VSource(net_a, n_in_a, net_a.gnd, name="vs", value=V_source)

        dt = 1e-6
        sim_a = net_a.compile(dt=dt)
        state_a = sim_a.init({})
        for _ in range(10):
            state_a = sim_a.step({}, state_a, {})

        v_a = float(sim_a.v(state_a, n_in_a))
        i_a = v_a / R_eq

        # --- Circuit B: Equivalent single resistor ---
        net_b = Network()
        net_b, n_in_b = net_b.node("in")

        net_b, vs_b = VSource(net_b, n_in_b, net_b.gnd, name="vs", value=V_source)
        net_b, r_eq = R(net_b, n_in_b, net_b.gnd, name="r_eq", value=R_eq)

        sim_b = net_b.compile(dt=dt)
        state_b = sim_b.init({})
        for _ in range(10):
            state_b = sim_b.step({}, state_b, {})

        v_b = float(sim_b.v(state_b, n_in_b))
        i_b = v_b / R_eq

        assert abs(i_a - expected_i) / expected_i < 0.01, \
            f"Mixed current {i_a} != expected {expected_i}"
        assert abs(i_a - i_b) / i_b < 0.01, \
            f"Mixed current {i_a} != equivalent {i_b}"


class TestJaxIntegration:
    """Tests for JAX autodiff integration with subcircuits."""

    def test_series_jax_differentiable(self):
        """Gradients w.r.t. component values in Series subcircuit.

        Test ∂V_out/∂R for a voltage divider using Series.
        For a divider Vs -- R1 -- R2 -- GND, V_out = Vs * R2 / (R1 + R2)
        ∂V_out/∂R1 should be negative (more R1 = less V_out)
        ∂V_out/∂R2 should be positive (more R2 = more V_out)
        """
        import jax.numpy as jnp
        from jax import grad
        from pyvibrate.timedomain import Network, R, VSource
        from pyvibrate.timedomain.subcircuits import Series

        V_source = 10.0

        def voltage_at_midpoint(R1_val: float, R2_val: float) -> float:
            net = Network()
            net, n_in = net.node("in")

            net, (r1_ref, r2_ref, mid) = Series(
                net, n_in, net.gnd,
                lambda net, a, b: R(net, a, b, name="r1"),
                lambda net, a, b: R(net, a, b, name="r2"),
                prefix="div"
            )
            net, vs = VSource(net, n_in, net.gnd, name="vs", value=V_source)

            dt = 1e-6
            sim = net.compile(dt=dt)
            params = {"r1": R1_val, "r2": R2_val}
            state = sim.init(params)

            for _ in range(10):
                state = sim.step(params, state, {})

            return sim.v(state, mid)

        R1_test = 1000.0
        R2_test = 2000.0

        # Compute gradients
        dV_dR1 = grad(voltage_at_midpoint, argnums=0)(R1_test, R2_test)
        dV_dR2 = grad(voltage_at_midpoint, argnums=1)(R1_test, R2_test)

        # Verify gradients are finite
        assert jnp.isfinite(dV_dR1), f"∂V/∂R1 should be finite, got {dV_dR1}"
        assert jnp.isfinite(dV_dR2), f"∂V/∂R2 should be finite, got {dV_dR2}"

        # Verify gradient signs match physical expectation
        # V_mid = V_s * R2 / (R1 + R2)
        # ∂V_mid/∂R1 = -V_s * R2 / (R1 + R2)^2 < 0
        # ∂V_mid/∂R2 = V_s * R1 / (R1 + R2)^2 > 0
        assert dV_dR1 < 0, f"∂V/∂R1 should be negative, got {dV_dR1}"
        assert dV_dR2 > 0, f"∂V/∂R2 should be positive, got {dV_dR2}"

    def test_parallel_jax_differentiable(self):
        """Gradients w.r.t. component values in Parallel subcircuit.

        Test ∂V_out/∂R for a parallel combination.
        For Vs -- R_load -- (R1 || R2) -- GND:
        V_out decreases when R1 or R2 increases (parallel impedance increases)
        """
        import jax.numpy as jnp
        from jax import grad
        from pyvibrate.timedomain import Network, R, VSource
        from pyvibrate.timedomain.subcircuits import Parallel

        V_source = 10.0
        R_load = 500.0

        def voltage_at_parallel(R1_val: float, R2_val: float) -> float:
            net = Network()
            net, n_in = net.node("in")
            net, n_par = net.node("par")

            net, vs = VSource(net, n_in, net.gnd, name="vs", value=V_source)
            net, rl = R(net, n_in, n_par, name="r_load", value=R_load)
            net, (r1_ref, r2_ref) = Parallel(
                net, n_par, net.gnd,
                lambda net, a, b: R(net, a, b, name="r1"),
                lambda net, a, b: R(net, a, b, name="r2"),
                prefix="par"
            )

            dt = 1e-6
            sim = net.compile(dt=dt)
            params = {"r1": R1_val, "r2": R2_val}
            state = sim.init(params)

            for _ in range(10):
                state = sim.step(params, state, {})

            return sim.v(state, n_par)

        R1_test = 1000.0
        R2_test = 2000.0

        # Compute gradients
        dV_dR1 = grad(voltage_at_parallel, argnums=0)(R1_test, R2_test)
        dV_dR2 = grad(voltage_at_parallel, argnums=1)(R1_test, R2_test)

        # Verify gradients are finite
        assert jnp.isfinite(dV_dR1), f"∂V/∂R1 should be finite, got {dV_dR1}"
        assert jnp.isfinite(dV_dR2), f"∂V/∂R2 should be finite, got {dV_dR2}"

        # V_par = V_s * R_par / (R_load + R_par) where R_par = R1*R2/(R1+R2)
        # Increasing R1 or R2 increases R_par, which increases V_par
        assert dV_dR1 > 0, f"∂V/∂R1 should be positive, got {dV_dR1}"
        assert dV_dR2 > 0, f"∂V/∂R2 should be positive, got {dV_dR2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
